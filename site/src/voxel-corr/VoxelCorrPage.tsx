import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { intParam, useUrlState } from 'use-prms'
import { themedHoverlabel, ThemeToggle } from '../theme'
import { API_BASE } from '../runs/api'

/** Empirical MP charge-density grid resolution (~0.065 Å/voxel ≈ uniform
 *  across the dataset; see scripts/analyze_voxel_resolution.py). Used to
 *  surface a secondary distance axis in Å on the dist-corr plot. */
const ANGSTROMS_PER_VOXEL = 0.065

/** JSON sidecar shape — matches `tomat analyze voxel-corr-blob`. */
interface CorrMetadata {
  run_label: string
  kind: 'pred' | 'true' | 'residual'
  patch_size: number
  n_voxels: number
  n_patches: number
  n_mats: number
  /** Decoded float = int8 * scale → in [-1, 1]. */
  int8_scale: number
  /** 1D collapse curve: avg corr(ρ_i, ρ_{i+k}) over offset k. Length P³. */
  curve: number[]
  /** Optional: present if blob is served gzipped. CLI sets it. */
  blob_compression?: 'gzip' | null
  /** Compressed-blob byte length (for the "Loading…" message). */
  blob_size?: number
  /**
   * How the int8 buffer is laid out:
   *   - `'upper_triangle'` (current): P³(P³+1)/2 entries, row-major over
   *     the upper triangle (i <= j). Halves the wire size; the FE mirrors
   *     into the full matrix on load.
   *   - `'full'` (legacy, omitted in older blobs): P³ × P³ entries,
   *     row-major. Treated as the default when the field is absent.
   */
  blob_format?: 'upper_triangle' | 'full'
}

interface CorrData {
  meta: CorrMetadata
  /** Row-major P³×P³ int8 corr matrix. Length P³ × P³. */
  matrix: Int8Array
}

// Color-LUT for int8 (-128..127) → packed RGBA uint32. Build once. Maps
// matplotlib RdBu_r endpoints: red=high, white=zero, blue=low.
function buildColorLUT(): Uint32Array {
  const lut = new Uint32Array(256)
  for (let i = 0; i < 256; i++) {
    const sint = i < 128 ? i : i - 256
    const v = Math.max(-1, Math.min(1, sint / 127))
    let r: number, g: number, b: number
    if (v >= 0) {
      const t = v
      r = Math.round(255 - t * (255 - 180))
      g = Math.round(255 - t * (255 - 30))
      b = Math.round(255 - t * (255 - 30))
    } else {
      const t = -v
      r = Math.round(255 - t * (255 - 30))
      g = Math.round(255 - t * (255 - 80))
      b = Math.round(255 - t * (255 - 200))
    }
    // RGBA little-endian: A << 24 | B << 16 | G << 8 | R
    lut[i] = (255 << 24) | (b << 16) | (g << 8) | r
  }
  return lut
}

const COLOR_LUT = buildColorLUT()

/** Continuous viewport in source-matrix coords. (x0,y0) top-left, (x1,y1) bottom-right. */
interface Viewport {
  x0: number
  y0: number
  x1: number
  y1: number
}

const CANVAS_DIM = 1024

function isZoomedIn(vp: Viewport): boolean {
  return Math.max(vp.x1 - vp.x0, vp.y1 - vp.y0) < CANVAS_DIM
}

interface RenderOptions {
  showCanonicalDiagonals: boolean
  patchSize: number
  nVoxels: number
}

function renderViewport(
  ctx: CanvasRenderingContext2D,
  matrix: Int8Array,
  vp: Viewport,
  opts: RenderOptions,
): void {
  const dim = CANVAS_DIM
  const N = opts.nVoxels
  const w = vp.x1 - vp.x0
  const h = vp.y1 - vp.y0
  const sx = dim / w
  const sy = dim / h

  const imgData = ctx.createImageData(dim, dim)
  const u32 = new Uint32Array(imgData.data.buffer)

  const rowSrcIdx = new Int32Array(dim)
  for (let py = 0; py < dim; py++) {
    const sy_src = Math.floor(vp.y0 + (py + 0.5) / sy)
    rowSrcIdx[py] = sy_src >= 0 && sy_src < N ? sy_src * N : -1
  }
  const colSrcIdx = new Int32Array(dim)
  for (let px = 0; px < dim; px++) {
    const sx_src = Math.floor(vp.x0 + (px + 0.5) / sx)
    colSrcIdx[px] = sx_src >= 0 && sx_src < N ? sx_src : -1
  }

  for (let py = 0; py < dim; py++) {
    const rowOff = rowSrcIdx[py]
    const dstRow = py * dim
    if (rowOff < 0) {
      const grey = (255 << 24) | (60 << 16) | (60 << 8) | 60
      for (let px = 0; px < dim; px++) u32[dstRow + px] = grey
      continue
    }
    for (let px = 0; px < dim; px++) {
      const col = colSrcIdx[px]
      if (col < 0) {
        u32[dstRow + px] = (255 << 24) | (60 << 16) | (60 << 8) | 60
        continue
      }
      const sint = matrix[rowOff + col]
      const lutIdx = sint < 0 ? sint + 256 : sint
      u32[dstRow + px] = COLOR_LUT[lutIdx]
    }
  }
  ctx.putImageData(imgData, 0, 0)

  if (opts.showCanonicalDiagonals) {
    drawDiagonals(ctx, vp, opts.patchSize, opts.nVoxels)
  }
}

function drawDiagonals(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  P: number,
  N: number,
): void {
  const dim = CANVAS_DIM
  const xToCanvas = (x: number): number => ((x - vp.x0) / (vp.x1 - vp.x0)) * dim
  const yToCanvas = (y: number): number => ((y - vp.y0) / (vp.y1 - vp.y0)) * dim
  // Pixels of separation between this diag and the main diag at the current
  // zoom. If <2, the line will visually merge with the main diagonal — bump
  // line width to keep it visible.
  const pxPerSrc = dim / (vp.x1 - vp.x0)
  const lines: { offset: number; color: string; label: string }[] = [
    { offset: P, color: '#ff8a65', label: `+P=${P}` },
    { offset: P * P, color: '#ffd54f', label: `+P²=${P * P}` },
    { offset: 2 * P, color: '#ff8a65', label: `+2P` },
    { offset: 2 * P * P, color: '#ffd54f', label: `+2P²` },
  ]
  ctx.save()
  ctx.font = '11px -apple-system, sans-serif'
  for (const { offset, color, label } of lines) {
    if (offset >= N) continue
    const sepPx = offset * pxPerSrc
    const lw = sepPx < 2 ? 2.5 : sepPx < 6 ? 1.5 : 1
    ctx.lineWidth = lw
    ctx.globalAlpha = sepPx < 2 ? 0.9 : 0.55
    ctx.strokeStyle = color
    ctx.fillStyle = color
    // line j = i + offset (above main diag): from (0, offset) to (N-offset, N)
    ctx.beginPath()
    ctx.moveTo(xToCanvas(0), yToCanvas(offset))
    ctx.lineTo(xToCanvas(N - offset), yToCanvas(N))
    ctx.stroke()
    // mirror diag i = j + offset
    ctx.beginPath()
    ctx.moveTo(xToCanvas(offset), yToCanvas(0))
    ctx.lineTo(xToCanvas(N), yToCanvas(N - offset))
    ctx.stroke()
    // Label near top-right corner of the upper diagonal, if in view.
    const lx = xToCanvas(N - offset - 5)
    const ly = yToCanvas(N - 5)
    if (lx > 0 && lx < dim && ly > 0 && ly < dim) {
      ctx.globalAlpha = 0.9
      ctx.fillText(label, lx, ly)
    }
  }
  ctx.restore()
}

/** Reset viewport to full-extent square (covers entire [0, N) × [0, N)). */
function fullViewport(N: number): Viewport {
  return { x0: 0, y0: 0, x1: N, y1: N }
}

/** Clamp viewport so the matrix never slides fully off canvas. When the
 *  viewport is larger than N (zoomed out past full extent), allow the matrix
 *  region to remain centered; when smaller (zoomed in), keep [x0,x1] ⊆ [0,N]
 *  so we can't pan past the edges. */
function clampViewport(vp: Viewport, N: number): Viewport {
  const w = vp.x1 - vp.x0
  const h = vp.y1 - vp.y0
  let { x0, y0 } = vp
  // X axis
  if (w >= N) {
    // Zoomed out — clamp so the [0,N] matrix region stays within the canvas;
    // i.e. don't let x0 > 0 (would push matrix off left), don't let
    // x1 < N (would push matrix off right).
    x0 = Math.min(0, Math.max(N - w, x0))
  } else {
    // Zoomed in — keep [x0,x1] ⊆ [0,N].
    x0 = Math.max(0, Math.min(N - w, x0))
  }
  if (h >= N) {
    y0 = Math.min(0, Math.max(N - h, y0))
  } else {
    y0 = Math.max(0, Math.min(N - h, y0))
  }
  return { x0, y0, x1: x0 + w, y1: y0 + h }
}

function zoomViewport(vp: Viewport, factor: number, anchorCx: number, anchorCy: number, N: number): Viewport {
  const w = vp.x1 - vp.x0
  const h = vp.y1 - vp.y0
  const ax = vp.x0 + (anchorCx / CANVAS_DIM) * w
  const ay = vp.y0 + (anchorCy / CANVAS_DIM) * h
  let nw = w * factor
  let nh = h * factor
  // Min 4 source pixels (so we never zoom past usefulness); max = N so the
  // fully-zoomed-out viewport is exactly the matrix's [0,N]×[0,N] extent
  // (no dark margin around the matrix). Previously was `2 * N` which let
  // the user zoom out into empty space.
  const minW = 4
  const maxW = N
  nw = Math.max(minW, Math.min(maxW, nw))
  nh = Math.max(minW, Math.min(maxW, nh))
  const x0 = ax - (anchorCx / CANVAS_DIM) * nw
  const y0 = ay - (anchorCy / CANVAS_DIM) * nh
  return clampViewport({ x0, y0, x1: x0 + nw, y1: y0 + nh }, N)
}

interface AxisTick {
  /** Source-matrix coord. */
  v: number
  /** 'pp' = multiple of P² (major, always labeled). 'p' = multiple of P
   *  between majors (minor; labeled only when zoomed in enough). */
  kind: 'pp' | 'p'
  /** When undefined, the tick is drawn without a number label. */
  label?: string
}

/** Pick ticks for one axis. Always emits P² (or 2P², 4P², …) majors at a
 *  spacing of ≥ MIN_PP_PX so labels never collide. Adds P minors between
 *  them, labeled only when P-spacing is wide enough to fit a number; skipped
 *  entirely when P-spacing would be sub-pixel clutter. */
function buildTicks(v0: number, v1: number, P: number, N: number): AxisTick[] {
  const MIN_PP_PX = 60  // P² (or multiple) major spacing in canvas px
  const MIN_P_LABEL_PX = 40  // require this much room before labeling P minors
  const MIN_P_TICK_PX = 5   // below this, suppress P minors entirely (clutter)
  const range = v1 - v0
  const pxPerSrc = CANVAS_DIM / range
  const ticks: AxisTick[] = []

  // ── Major: P² ticks, doubling the multiple until spacing ≥ MIN_PP_PX ──
  const P2 = P * P
  let ppMult = 1
  while (P2 * ppMult * pxPerSrc < MIN_PP_PX && P2 * ppMult * 2 < N) ppMult *= 2
  const ppStep = P2 * ppMult
  for (let v = Math.ceil(v0 / ppStep) * ppStep; v <= v1; v += ppStep) {
    if (v >= 0 && v <= N) ticks.push({ v, kind: 'pp', label: String(v) })
  }

  // ── Minor: P ticks. Suppress when too dense; label only when spacious ──
  const pSpacingPx = P * pxPerSrc
  if (pSpacingPx >= MIN_P_TICK_PX) {
    const labelP = pSpacingPx >= MIN_P_LABEL_PX
    for (let v = Math.ceil(v0 / P) * P; v <= v1; v += P) {
      if (v < 0 || v > N) continue
      if (v % ppStep === 0) continue  // already drawn as major
      ticks.push({ v, kind: 'p', label: labelP ? String(v) : undefined })
    }
  }

  return ticks
}

/** Aggregated stats (mean, σ, count) for one distance bin. */
interface DistBin {
  /** Lower edge of the bin, in voxel-index units. */
  lo: number
  /** Upper edge of the bin, in voxel-index units. */
  hi: number
  /** Bin midpoint = (lo + hi) / 2, voxel-index units. */
  mid: number
  /** Number of (i, j) pairs that fell in this bin (i < j only). */
  count: number
  /** Mean correlation across the bin. NaN if count = 0. */
  mean: number
  /** Population standard deviation across the bin. NaN if count = 0. */
  std: number
}

interface DistCorrHist {
  /** All non-empty bins, ordered by ascending distance. */
  bins: DistBin[]
  /** Total pairs aggregated (i < j, with d > 0). */
  totalPairs: number
  /** Max distance encountered = √3 · (P − 1). */
  maxDist: number
}

/** Compute the (3D-Euclidean-distance) vs (decoded correlation) histogram
 *  for every (i, j) with i < j in the P³×P³ matrix. Distances are in
 *  voxel-index units; positions decode as pos = x + y·P + z·P². The d=0
 *  diagonal is excluded. Bin edges are linear in [0, √3·(P−1)] across
 *  `nBins` buckets. */
function computeDistCorrHist(matrix: Int8Array, P: number, int8Scale: number, nBins: number): DistCorrHist {
  const N = P * P * P
  // Pre-decode each linear index into (x, y, z) integer coords. Avoids
  // doing 3 divides+mods per pair (would be ~70M ops at P=19).
  const xs = new Int8Array(N)
  const ys = new Int8Array(N)
  const zs = new Int8Array(N)
  for (let i = 0; i < N; i++) {
    const z = Math.floor(i / (P * P))
    const y = Math.floor((i - z * P * P) / P)
    const x = i - z * P * P - y * P
    xs[i] = x
    ys[i] = y
    zs[i] = z
  }

  const maxDist = Math.sqrt(3) * (P - 1)
  // Linear bins in [0, maxDist]. We skip d == 0 (diagonal) entirely.
  const binW = maxDist / nBins
  const sums = new Float64Array(nBins)
  const sumSqs = new Float64Array(nBins)
  const counts = new Uint32Array(nBins)

  let totalPairs = 0
  for (let i = 0; i < N; i++) {
    const xi = xs[i], yi = ys[i], zi = zs[i]
    const row = i * N
    for (let j = i + 1; j < N; j++) {
      const dx = xs[j] - xi
      const dy = ys[j] - yi
      const dz = zs[j] - zi
      const d2 = dx * dx + dy * dy + dz * dz
      if (d2 === 0) continue
      const d = Math.sqrt(d2)
      // Clamp the bin index for the d = maxDist corner pair.
      let b = Math.floor(d / binW)
      if (b >= nBins) b = nBins - 1
      const c = matrix[row + j] * int8Scale
      sums[b] += c
      sumSqs[b] += c * c
      counts[b] += 1
      totalPairs += 1
    }
  }

  const bins: DistBin[] = []
  for (let b = 0; b < nBins; b++) {
    if (counts[b] === 0) continue
    const lo = b * binW
    const hi = (b + 1) * binW
    const n = counts[b]
    const mean = sums[b] / n
    // Population variance — clamp to ≥ 0 to handle catastrophic
    // cancellation on near-uniform bins.
    const var_ = Math.max(0, sumSqs[b] / n - mean * mean)
    bins.push({ lo, hi, mid: (lo + hi) / 2, count: n, mean, std: Math.sqrt(var_) })
  }
  return { bins, totalPairs, maxDist }
}

/** Decode a row-major voxel index → (x, y, z) integer coords. */
function decodePos(i: number, P: number): { x: number; y: number; z: number } {
  const z = Math.floor(i / (P * P))
  const y = Math.floor((i - z * P * P) / P)
  const x = i - z * P * P - y * P
  return { x, y, z }
}

/** 3D Euclidean distance between two voxel indices, in voxel-index units. */
function voxelDistance(i: number, j: number, P: number): number {
  const a = decodePos(i, P)
  const b = decodePos(j, P)
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

async function loadCorrData(runLabel: string): Promise<CorrData> {
  // Blob + sidecar live on R2 (`tomat/voxel-corr/<label>.{json,bin.gzip}`),
  // served via the runs-API worker so the static site stays free of the ~12MB
  // matrix files. See worker route `/api/voxel-corr/:label.{json,bin.gzip}`.
  const jsonUrl = `${API_BASE}/api/voxel-corr/${runLabel}.json`
  const metaRes = await fetch(jsonUrl)
  if (!metaRes.ok) throw new Error(`fetch ${jsonUrl}: ${metaRes.status}`)
  const meta = (await metaRes.json()) as CorrMetadata
  // Prefer the .gz blob (smaller); fall back to .bin if the CLI hasn't
  // re-emitted (older blobs). The .json's blob_compression field tells us
  // which the CLI wrote.
  const compressed = meta.blob_compression === 'gzip'
  // `.gzip` extension (not `.gz`): see CLI comment in `tomat analyze
  // voxel-corr-blob` — `.gz` triggers an auto Content-Encoding setter on
  // some static hosts and would double-decompress through our
  // DecompressionStream. We keep the `.gzip` extension end-to-end.
  const binUrl = `${API_BASE}/api/voxel-corr/${runLabel}.bin${compressed ? '.gzip' : ''}`
  const binRes = await fetch(binUrl)
  if (!binRes.ok) throw new Error(`fetch ${binUrl}: ${binRes.status}`)
  let buf: ArrayBuffer
  if (compressed) {
    // Stream the gzipped bytes through DecompressionStream so we never hold
    // both compressed + decompressed copies. Supported in all modern browsers
    // (Chrome 80+, FF 113+, Safari 16.4+).
    if (!binRes.body) throw new Error('response has no body to decompress')
    const decompressed = binRes.body.pipeThrough(new DecompressionStream('gzip'))
    buf = await new Response(decompressed).arrayBuffer()
  } else {
    buf = await binRes.arrayBuffer()
  }
  const N = meta.n_voxels
  const format = meta.blob_format ?? 'full'
  if (format === 'upper_triangle') {
    // Upper triangle: N*(N+1)/2 int8 entries, row-major over (i <= j).
    // Reflect into a full N×N matrix on the client. ~50% wire-size savings.
    const expectedHalf = (N * (N + 1)) / 2
    if (buf.byteLength !== expectedHalf) {
      throw new Error(
        `corr blob size mismatch (upper_triangle): got ${buf.byteLength} bytes, ` +
        `expected ${expectedHalf} (N=${N}, N(N+1)/2)`,
      )
    }
    const upper = new Int8Array(buf)
    const full = new Int8Array(N * N)
    let k = 0
    for (let i = 0; i < N; i++) {
      for (let j = i; j < N; j++) {
        const v = upper[k++]
        full[i * N + j] = v
        if (i !== j) full[j * N + i] = v
      }
    }
    return { meta, matrix: full }
  }
  const expected = N * N
  if (buf.byteLength !== expected) {
    throw new Error(
      `corr blob size mismatch: got ${buf.byteLength} bytes, expected ${expected} ` +
      `(N=${N})`,
    )
  }
  return { meta, matrix: new Int8Array(buf) }
}

const KNOWN_RUNS = [
  { label: 'cont33k', desc: 'cont33k TF predictions (200 M, P=19, 9 mats × ~200 patches)' },
]

interface Props {
  parts: string[]
}

interface CursorState {
  /** Source-matrix coords (integer). */
  i: number
  j: number
  /** Decoded corr in [-1, 1]. */
  v: number
  /** Client coords (px) of the mouse, for the floating tooltip. */
  clientX: number
  clientY: number
}

const DIST_BINS_DEFAULT = 40
const DIST_BINS_MIN = 5
const DIST_BINS_MAX = 200

export function VoxelCorrPage({ parts }: Props) {
  const runLabel = parts[0] ?? KNOWN_RUNS[0].label
  const { isDark } = useTheme()

  const [data, setData] = useState<CorrData | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)
  const [showDiagonals, setShowDiagonals] = useState(true)
  const [viewport, setViewport] = useState<Viewport | null>(null)
  const [cursor, setCursor] = useState<CursorState | null>(null)
  const [distBins, setDistBins] = useUrlState('distBins', intParam(DIST_BINS_DEFAULT))
  const nBins = Math.max(DIST_BINS_MIN, Math.min(DIST_BINS_MAX, distBins))

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ startVp: Viewport; startMx: number; startMy: number } | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setData(null)
    setViewport(null)
    loadCorrData(runLabel)
      .then(d => {
        setData(d)
        setViewport(fullViewport(d.meta.n_voxels))
        setLoading(false)
      })
      .catch(e => {
        setError(e instanceof Error ? e : new Error(String(e)))
        setLoading(false)
      })
  }, [runLabel])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data || !viewport) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    renderViewport(ctx, data.matrix, viewport, {
      showCanonicalDiagonals: showDiagonals,
      patchSize: data.meta.patch_size,
      nVoxels: data.meta.n_voxels,
    })
  }, [data, viewport, showDiagonals])

  // Native non-passive wheel listener. React's synthetic onWheel is passive
  // since React 17, so e.preventDefault() is silently ignored and the page
  // scrolls under the matrix as the user zooms. addEventListener with
  // {passive:false} restores the ability to swallow the wheel event.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const cx = ((e.clientX - rect.left) / rect.width) * CANVAS_DIM
      const cy = ((e.clientY - rect.top) / rect.height) * CANVAS_DIM
      const factor = Math.exp(e.deltaY * 0.0015)
      setViewport(vp => (vp ? zoomViewport(vp, factor, cx, cy, data.meta.n_voxels) : vp))
    }
    canvas.addEventListener('wheel', handler, { passive: false })
    return () => canvas.removeEventListener('wheel', handler)
  }, [data])

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!viewport) return
    const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect()
    const cx = ((e.clientX - rect.left) / rect.width) * CANVAS_DIM
    const cy = ((e.clientY - rect.top) / rect.height) * CANVAS_DIM
    dragRef.current = { startVp: viewport, startMx: cx, startMy: cy }
  }, [viewport])

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !viewport) return
    const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect()
    const cx = ((e.clientX - rect.left) / rect.width) * CANVAS_DIM
    const cy = ((e.clientY - rect.top) / rect.height) * CANVAS_DIM
    if (dragRef.current) {
      const { startVp, startMx, startMy } = dragRef.current
      const w = startVp.x1 - startVp.x0
      const h = startVp.y1 - startVp.y0
      const dx = ((cx - startMx) / CANVAS_DIM) * w
      const dy = ((cy - startMy) / CANVAS_DIM) * h
      setViewport(clampViewport({
        x0: startVp.x0 - dx,
        y0: startVp.y0 - dy,
        x1: startVp.x1 - dx,
        y1: startVp.y1 - dy,
      }, data.meta.n_voxels))
    }
    const w = viewport.x1 - viewport.x0
    const h = viewport.y1 - viewport.y0
    const j = Math.floor(viewport.x0 + (cx / CANVAS_DIM) * w)
    const i = Math.floor(viewport.y0 + (cy / CANVAS_DIM) * h)
    if (i >= 0 && i < data.meta.n_voxels && j >= 0 && j < data.meta.n_voxels) {
      const sint = data.matrix[i * data.meta.n_voxels + j]
      setCursor({
        i, j,
        v: sint * data.meta.int8_scale,
        clientX: e.clientX,
        clientY: e.clientY,
      })
    } else {
      setCursor(null)
    }
  }, [data, viewport])

  const onMouseUp = useCallback(() => { dragRef.current = null }, [])
  const onMouseLeave = useCallback(() => {
    dragRef.current = null
    setCursor(null)
  }, [])

  const resetView = useCallback(() => {
    if (data) setViewport(fullViewport(data.meta.n_voxels))
  }, [data])

  /** Memoized distance-vs-correlation histogram. Recomputes when the
   *  underlying matrix or bin count changes. ~23M pair iterations at
   *  P=19 → ~300 ms one-shot on a typical laptop; the user can re-bin
   *  via the URL knob below without re-loading. */
  const distHist = useMemo(() => {
    if (!data) return null
    return computeDistCorrHist(data.matrix, data.meta.patch_size, data.meta.int8_scale, nBins)
  }, [data, nBins])

  const cursorLabel = useMemo(() => {
    if (!cursor || !data) return ''
    const P = data.meta.patch_size
    const decode = (n: number) => {
      const z = Math.floor(n / (P * P))
      const y = Math.floor((n - z * P * P) / P)
      const x = n - z * P * P - y * P
      return `(${x},${y},${z})`
    }
    const d = voxelDistance(cursor.i, cursor.j, P)
    const dA = d * ANGSTROMS_PER_VOXEL
    return `i=${cursor.i} ${decode(cursor.i)} · j=${cursor.j} ${decode(cursor.j)} · ` +
      `d=${d.toFixed(2)} (${dA.toFixed(2)} Å) · corr=${cursor.v.toFixed(3)}`
  }, [cursor, data])

  // Tick rulers (top + left) — HTML-rendered for crisp text. P² (and
  // higher multiples) are major + always labeled; P is minor + labeled
  // only when zoomed in enough; below ~5 px-per-P the minors are
  // suppressed entirely to avoid clutter.
  const ticks = useMemo(() => {
    if (!data || !viewport) return null
    const N = data.meta.n_voxels
    const P = data.meta.patch_size
    const xs = buildTicks(viewport.x0, viewport.x1, P, N)
    const ys = buildTicks(viewport.y0, viewport.y1, P, N)
    const xPct = (x: number) => `${((x - viewport.x0) / (viewport.x1 - viewport.x0)) * 100}%`
    const yPct = (y: number) => `${((y - viewport.y0) / (viewport.y1 - viewport.y0)) * 100}%`
    return { xs, ys, xPct, yPct, P }
  }, [data, viewport])

  return (
    <>
      <header>
        <h1>tomat 🍅 voxel-position corr</h1>
        <ThemeToggle />
      </header>
      <p className="meta">
        Pearson correlation between voxel-positions across patches: each
        cell (i, j) is corr(ρ at row-major pos i, ρ at row-major pos j),
        averaged over all patches in <code>{runLabel}</code>. Diagonals at
        offset k=P and k=P² are the spatial neighbors in raster order — if
        the model has learned local 3-D structure, these should be visibly
        brighter than the off-diagonal background. The <em>3-D distance</em>
        {' '}plot below decodes each (i, j) → (x, y, z) positions and asks
        the same question more directly: does prediction-correlation decay
        with 3-D Euclidean distance? Background on the analysis:{' '}
        <a href="https://github.com/Open-Athena/tomat/blob/main/scripts/voxel_corr.py">
          scripts/voxel_corr.py</a>.
        {' '}<a href="#/">← home</a> · <a href="#/runs">runs dashboard</a>
      </p>

      <div className="voxel-corr-controls">
        <label>
          run:{' '}
          <select
            value={runLabel}
            onChange={e => { window.location.hash = `#/voxel-corr/${e.target.value}` }}
          >
            {KNOWN_RUNS.map(r => (
              <option key={r.label} value={r.label}>{r.label} — {r.desc}</option>
            ))}
          </select>
        </label>
        {data && (
          <span className="meta">
            {data.meta.n_patches.toLocaleString()} patches ·{' '}
            {data.meta.n_mats} mats · P={data.meta.patch_size} ·{' '}
            kind={data.meta.kind}
          </span>
        )}
      </div>

      {loading && <p className="note">Loading correlation blob…</p>}
      {error && <p className="note" style={{ color: 'crimson' }}>Error: {error.message}</p>}

      {data && (
        <CurvePlot curve={data.meta.curve} patchSize={data.meta.patch_size} isDark={isDark} />
      )}

      {data && distHist && (
        <>
          <div className="voxel-corr-controls">
            <label>
              dist bins:{' '}
              <input
                type="number"
                min={DIST_BINS_MIN}
                max={DIST_BINS_MAX}
                value={distBins}
                onChange={e => {
                  const v = Number(e.target.value)
                  if (Number.isFinite(v)) setDistBins(v)
                }}
                style={{ width: '5em' }}
              />
            </label>
            <span className="meta">
              {distHist.totalPairs.toLocaleString()} pairs (i &lt; j, d &gt; 0) ·{' '}
              max d = {distHist.maxDist.toFixed(2)} voxels ({(distHist.maxDist * ANGSTROMS_PER_VOXEL).toFixed(2)} Å)
            </span>
          </div>
          <DistCorrPlot
            hist={distHist}
            patchSize={data.meta.patch_size}
            hoverDist={cursor ? voxelDistance(cursor.i, cursor.j, data.meta.patch_size) : null}
            isDark={isDark}
          />
        </>
      )}

      {data && viewport && (
        <>
          <div className="voxel-corr-controls">
            <button type="button" onClick={resetView}>Reset view</button>
            <label>
              <input
                type="checkbox"
                checked={showDiagonals}
                onChange={e => setShowDiagonals(e.target.checked)}
              />{' '}
              Diagonal markers (P, P², 2P, 2P²)
            </label>
            <span className="meta">
              viewport: x ∈ [{Math.round(viewport.x0)}, {Math.round(viewport.x1)}] ·{' '}
              y ∈ [{Math.round(viewport.y0)}, {Math.round(viewport.y1)}] ·{' '}
              {isZoomedIn(viewport) ? 'zoomed in' : 'zoomed out'}
            </span>
          </div>

          <div className="voxel-corr-canvas-wrap" ref={wrapRef}>
            <div className="voxel-corr-axes">
              {/* Top tick ruler */}
              <div className="voxel-corr-ruler voxel-corr-ruler-x">
                {ticks && ticks.xs.map(t => (
                  <div
                    key={`xt-${t.v}`}
                    className={`voxel-corr-tick is-${t.kind}${t.label ? '' : ' is-unlabeled'}`}
                    style={{ left: ticks.xPct(t.v) }}
                  >
                    {t.label && <span>{t.label}</span>}
                  </div>
                ))}
              </div>
              {/* Left tick ruler + canvas */}
              <div className="voxel-corr-ruler voxel-corr-ruler-y">
                {ticks && ticks.ys.map(t => (
                  <div
                    key={`yt-${t.v}`}
                    className={`voxel-corr-tick is-${t.kind}${t.label ? '' : ' is-unlabeled'}`}
                    style={{ top: ticks.yPct(t.v) }}
                  >
                    {t.label && <span>{t.label}</span>}
                  </div>
                ))}
              </div>
              <canvas
                ref={canvasRef}
                width={CANVAS_DIM}
                height={CANVAS_DIM}
                className="voxel-corr-canvas"
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseLeave}
              />
            </div>
          </div>

          <p className="meta voxel-corr-cursor">
            {cursorLabel || 'Hover over the matrix to see (i, j) → (x, y, z) decodes + correlation.'}
          </p>
          <p className="note">
            Mouse wheel zooms (centered on cursor); drag to pan. The
            color scale runs blue (corr = −1) → white (0) → red (+1),
            int8-quantized. Each axis indexes the row-major voxel
            position within a patch: pos = x + y·P + z·P². Diagonals at
            k=P (orange) and k=P² (yellow) are the model's spatial
            neighbors.
          </p>
        </>
      )}

      {/* Floating cursor tooltip — follows the mouse with edge-aware flip. */}
      {cursor && (
        <CursorTooltip cursor={cursor} label={cursorLabel} />
      )}
    </>
  )
}

function CursorTooltip({ cursor, label }: { cursor: CursorState; label: string }) {
  // Flip to left/above if too close to right/bottom edge of viewport.
  const OFFSET = 14
  const W_GUESS = 280
  const H_GUESS = 36
  const winW = typeof window !== 'undefined' ? window.innerWidth : 1024
  const winH = typeof window !== 'undefined' ? window.innerHeight : 768
  const flipX = cursor.clientX + OFFSET + W_GUESS > winW
  const flipY = cursor.clientY + OFFSET + H_GUESS > winH
  const left = flipX ? cursor.clientX - OFFSET - W_GUESS : cursor.clientX + OFFSET
  const top = flipY ? cursor.clientY - OFFSET - H_GUESS : cursor.clientY + OFFSET
  return (
    <div
      className="voxel-corr-floating-tt"
      style={{ position: 'fixed', left, top, pointerEvents: 'none' }}
    >
      {label}
    </div>
  )
}

function CurvePlot({ curve, patchSize, isDark }: { curve: number[]; patchSize: number; isDark: boolean }) {
  const P = patchSize
  const P3 = P * P * P
  const xs = useMemo(() => Array.from({ length: curve.length }, (_, i) => i), [curve.length])
  return (
    <div className="plot-card" style={{ marginTop: '0.5rem' }}>
      <Plot
        data={[
          {
            x: xs,
            y: curve,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#1976d2', width: 0.7 },
            hovertemplate: 'k=%{x}<br>avg corr=%{y:.4f}<extra></extra>',
            name: 'avg corr at offset k',
          },
        ]}
        layout={{
          autosize: true,
          height: 260,
          margin: { t: 24, r: 30, b: 40, l: 60 },
          title: { text: `1D collapse: avg corr(ρ_i, ρ_{i+k})` },
          xaxis: {
            title: { text: 'row-major position offset k' },
            range: [0, P3],
          },
          yaxis: { title: { text: 'avg corr' } },
          hoverlabel: themedHoverlabel(isDark),
          shapes: [
            { type: 'line', x0: 1, x1: 1, yref: 'paper', y0: 0, y1: 1, line: { color: '#e91e63', width: 1, dash: 'dash' } },
            { type: 'line', x0: P, x1: P, yref: 'paper', y0: 0, y1: 1, line: { color: '#e91e63', width: 1, dash: 'dash' } },
            { type: 'line', x0: P * P, x1: P * P, yref: 'paper', y0: 0, y1: 1, line: { color: '#e91e63', width: 1, dash: 'dash' } },
          ],
          annotations: [
            { x: 1, yref: 'paper', y: 1.0, text: 'k=1', showarrow: false, font: { color: '#e91e63', size: 10 } },
            { x: P, yref: 'paper', y: 1.0, text: `k=P=${P}`, showarrow: false, font: { color: '#e91e63', size: 10 } },
            { x: P * P, yref: 'paper', y: 1.0, text: `k=P²=${P * P}`, showarrow: false, font: { color: '#e91e63', size: 10 } },
          ],
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  )
}

/** 3D-distance vs prediction-correlation plot. Tests the smoothness
 *  hypothesis: pairs of voxels close in 3D space should have higher
 *  prediction-correlation than pairs at long distance. A monotone-decay
 *  curve = the model is producing locally smooth fields. A flat curve =
 *  it isn't. The shaded band shows ±σ of correlation within each
 *  distance bin. Anchor lines mark unit distance (nearest-neighbor) and
 *  the patch-size scales. */
function DistCorrPlot({
  hist,
  patchSize,
  hoverDist,
  isDark,
}: {
  hist: DistCorrHist
  patchSize: number
  hoverDist: number | null
  isDark: boolean
}) {
  const P = patchSize
  const mids = hist.bins.map(b => b.mid)
  const means = hist.bins.map(b => b.mean)
  const upper = hist.bins.map(b => b.mean + b.std)
  const lower = hist.bins.map(b => b.mean - b.std)
  const counts = hist.bins.map(b => b.count)
  const xMax = hist.maxDist
  // Optional pink hover-marker if the user is hovering a cell on the matrix.
  const hoverShapes = hoverDist != null && hoverDist > 0
    ? [{
        type: 'line' as const,
        x0: hoverDist,
        x1: hoverDist,
        yref: 'paper' as const,
        y0: 0,
        y1: 1,
        line: { color: '#26c6da', width: 1.5, dash: 'dot' as const },
      }]
    : []
  return (
    <div className="plot-card" style={{ marginTop: '0.5rem' }}>
      <Plot
        data={[
          // ±σ band — drawn first so the mean line sits on top. Use a
          // 'tonexty' fill between the upper and lower trace.
          {
            x: mids,
            y: upper,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(25, 118, 210, 0)', width: 0 },
            hoverinfo: 'skip',
            showlegend: false,
            name: '+σ',
          },
          {
            x: mids,
            y: lower,
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(25, 118, 210, 0.18)',
            line: { color: 'rgba(25, 118, 210, 0)', width: 0 },
            hoverinfo: 'skip',
            showlegend: false,
            name: '−σ',
          },
          {
            x: mids,
            y: means,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#1976d2', width: 1.5 },
            marker: { size: 4, color: '#1976d2' },
            customdata: counts.map((c, i) => [c, hist.bins[i].std, mids[i] * ANGSTROMS_PER_VOXEL] as [number, number, number]),
            hovertemplate:
              'd=%{x:.2f} voxels (%{customdata[2]:.2f} Å)' +
              '<br>mean corr=%{y:.4f} ± %{customdata[1]:.4f}' +
              '<br>n=%{customdata[0]:,} pairs<extra></extra>',
            name: 'mean ± σ corr',
          },
        ]}
        layout={{
          autosize: true,
          height: 280,
          margin: { t: 24, r: 30, b: 44, l: 60 },
          title: { text: 'corr(ρ_i, ρ_j) vs 3-D voxel distance' },
          xaxis: {
            title: { text: '3-D distance ‖pos(i) − pos(j)‖₂ (voxel units; ~0.065 Å each)' },
            range: [0, xMax],
            zeroline: false,
          },
          yaxis: {
            title: { text: 'mean corr (±σ band)' },
            zeroline: true,
          },
          hoverlabel: themedHoverlabel(isDark),
          shapes: [
            { type: 'line', x0: 1, x1: 1, yref: 'paper', y0: 0, y1: 1, line: { color: '#e91e63', width: 1, dash: 'dash' } },
            { type: 'line', x0: P, x1: P, yref: 'paper', y0: 0, y1: 1, line: { color: '#ff8a65', width: 1, dash: 'dash' } },
            ...hoverShapes,
          ],
          annotations: [
            { x: 1, yref: 'paper', y: 1.0, text: 'd=1 (NN)', showarrow: false, font: { color: '#e91e63', size: 10 } },
            { x: P, yref: 'paper', y: 1.0, text: `d=P=${P}`, showarrow: false, font: { color: '#ff8a65', size: 10 } },
          ],
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  )
}
