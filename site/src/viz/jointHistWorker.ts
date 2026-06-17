// Web Worker: 2D joint histogram of two coregistered voxel arrays.
//
// Receives two Float32Arrays of identical length (one per source), plus
// histogram parameters (bins, scale, vmin, vmax). Returns the count matrix
// `H` + bin edges + per-axis summary stats + on-diagonal fraction. The
// caller spawns one worker per render and tears it down after the result
// posts back; if we ever want to support a million-voxel re-bin loop, we'd
// keep the worker around and use transferable buffers.

export interface JointHistRequest {
  xData: Float32Array
  yData: Float32Array
  vmin: number
  vmax: number
  bins: number
  scale: 'log' | 'linear'
}

export interface AxisStats {
  min: number
  max: number
  mean: number
  p50: number
  p99: number
}

export interface JointHistResponse {
  H: Float32Array
  xEdges: Float64Array
  yEdges: Float64Array
  xStats: AxisStats
  yStats: AxisStats
  /** Fraction of voxels where |y/x − 1| ≤ 0.1 (i.e. predicted within 10%
   *  of ground truth). High on a well-fit model; collapses on a model that
   *  predicts constant ρ regardless of structure. */
  onDiagFrac: number
  /** Total voxel count (= xData.length == yData.length). */
  total: number
}

// Worker entry. Vite picks up `?worker` imports and transpiles this file as
// a module-worker; the `self` handler is the standard worker contract.
self.onmessage = (e: MessageEvent<JointHistRequest>) => {
  const result = computeJointHist(e.data)
  // Transfer the typed-array buffers back to the main thread (zero-copy).
  // The worker no longer touches `H` / `xEdges` / `yEdges` after this point,
  // so the postMessage transfer-list is safe.
  ;(self as unknown as Worker).postMessage(result, [
    result.H.buffer,
    result.xEdges.buffer,
    result.yEdges.buffer,
  ])
}

function computeJointHist(req: JointHistRequest): JointHistResponse {
  const { xData, yData, bins, scale } = req
  const n = xData.length
  if (yData.length !== n) {
    throw new Error(`joint-hist: array length mismatch (${xData.length} vs ${yData.length})`)
  }
  const xEdges = buildEdges(req.vmin, req.vmax, bins, scale)
  const yEdges = buildEdges(req.vmin, req.vmax, bins, scale)
  const H = new Float32Array(bins * bins)

  // Hot loop: edge lookup is a binary search (log-spaced edges aren't
  // uniformly invertible). For the small bin counts we use (≤ 1024), the
  // ~10-step binary search beats hashing each value through `Math.log`.
  let onDiag = 0
  for (let i = 0; i < n; i++) {
    const x = xData[i]
    const y = yData[i]
    const bx = binIndex(xEdges, x, bins)
    const by = binIndex(yEdges, y, bins)
    if (bx < 0 || by < 0) continue
    H[by * bins + bx] += 1
    if (x > 0 && Math.abs(y / x - 1) <= 0.1) onDiag += 1
  }

  return {
    H,
    xEdges,
    yEdges,
    xStats: summarize(xData),
    yStats: summarize(yData),
    onDiagFrac: n > 0 ? onDiag / n : 0,
    total: n,
  }
}

function buildEdges(vmin: number, vmax: number, bins: number, scale: 'log' | 'linear'): Float64Array {
  const out = new Float64Array(bins + 1)
  if (scale === 'log') {
    // Floor the lower edge above zero — log(0) blows up. The caller is
    // expected to pass a positive vmin (we ensure that in JointHistPage).
    const safeMin = Math.max(vmin, 1e-6)
    const safeMax = Math.max(vmax, safeMin * 10)
    const lo = Math.log(safeMin)
    const hi = Math.log(safeMax)
    const dx = (hi - lo) / bins
    for (let i = 0; i <= bins; i++) out[i] = Math.exp(lo + i * dx)
  } else {
    const dx = (vmax - vmin) / bins
    for (let i = 0; i <= bins; i++) out[i] = vmin + i * dx
  }
  return out
}

/** Binary search: returns the bin index in [0, bins). Returns -1 when v
 *  falls outside [edges[0], edges[bins]]. */
function binIndex(edges: Float64Array, v: number, bins: number): number {
  if (!(v >= edges[0]) || !(v <= edges[bins])) return -1
  let lo = 0
  let hi = bins
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1
    if (v >= edges[mid]) lo = mid
    else hi = mid
  }
  return lo
}

function summarize(arr: Float32Array): AxisStats {
  const n = arr.length
  if (n === 0) return { min: 0, max: 0, mean: 0, p50: 0, p99: 0 }
  let min = arr[0], max = arr[0], sum = 0
  for (let i = 0; i < n; i++) {
    const v = arr[i]
    if (v < min) min = v
    if (v > max) max = v
    sum += v
  }
  // Copy-and-sort for percentiles. arr itself is owned by the worker (it
  // arrived via structured clone in `e.data`), so mutating would be OK,
  // but sorting in place wastes work — we're done with the worker after
  // this point, but the function should remain side-effect-free.
  const sorted = arr.slice().sort()
  return {
    min, max,
    mean: sum / n,
    p50: sorted[Math.floor(n * 0.5)],
    p99: sorted[Math.floor(n * 0.99)],
  }
}
