// Standalone joint-histogram route: `#/viz/joint-hist?mp=…&x=…&y=…`.
//
// Loads the level-0 voxel arrays for two sources, dispatches a worker to
// compute the 2D histogram, renders via `JointHistPlot`. URL params drive
// everything; the page is shareable end-to-end.
//
// Phase A only: every load re-downloads + re-computes. Phase B (CLI
// precompute → R2) lands later.

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { enumParam, intParam, stringParam, useUrlState } from 'use-prms'
import { boolTrueParam } from '../lib/urlParams'
import { fetchGrids, type GridRow } from '../mp/api'
import { shortenRun } from '../lib/runNames'
import { parseSpec, shortenSpec, zarrUrlFor } from './specs'
import { loadZarrLevel0 } from './zarrLoader'
import { JointHistPlot, type JointHistRange } from './JointHistPlot'
import type { JointHistRequest, JointHistResponse } from './jointHistWorker'

// Vite `?worker` import → bundled worker constructor.
import JointHistWorker from './jointHistWorker?worker'

type Scale = 'log' | 'linear'
const SCALES: readonly Scale[] = ['log', 'linear'] as const

/** Optional `[lo, hi]` URL param encoded as `lo,hi`. Null when absent. */
function rangeParam() {
  return {
    encode: (v: [number, number] | null) =>
      v ? `${v[0].toPrecision(6)},${v[1].toPrecision(6)}` : undefined,
    decode: (enc: string | undefined) => {
      if (!enc) return null
      const parts = enc.split(',')
      if (parts.length !== 2) return null
      const lo = parseFloat(parts[0])
      const hi = parseFloat(parts[1])
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null
      return [lo, hi] as [number, number]
    },
  }
}

interface LoadState {
  status: 'idle' | 'loading' | 'ready' | 'error'
  message?: string
  hist?: JointHistResponse
  dims?: [number, number, number]
}

export function JointHistPage() {
  const [mp] = useUrlState('mp', stringParam(''))
  const [xSpec, setXSpec] = useUrlState('x', stringParam(''))
  const [ySpec, setYSpec] = useUrlState('y', stringParam(''))
  const [bins] = useUrlState('bins', intParam(256))
  const [scale, setScale] = useUrlState('scale', enumParam<Scale>('log', SCALES))
  // `?M` — present (any value) → marginals HIDDEN; absent → marginals shown
  // (the default). Uppercase single-letter name signals the inverted-default
  // convention (see `boolTrueParam` jsdoc).
  const [showMarg, setShowMarg] = useUrlState('M', boolTrueParam)
  const [xRange, setXRange] = useUrlState('xRange', rangeParam(), { debounce: 250 })
  const [yRange, setYRange] = useUrlState('yRange', rangeParam(), { debounce: 250 })

  // BC for legacy `?marg=…` URLs that pre-date the `?M` rename. Migrate on
  // mount: `marg=0` → marginals OFF (write `?M`); anything else (`marg=1`,
  // valueless `marg`, etc.) → marginals ON (default, no `M`). Drop the
  // legacy key from the URL so the canonical form takes over. We rewrite
  // the hash directly here (rather than via use-prms) because `?marg` is
  // a key the page no longer owns — there's no `useUrlState` consumer for
  // it that would auto-encode `undefined` → drop, so an unmanaged param
  // would otherwise linger forever.
  const migrateRef = useRef(false)
  useEffect(() => {
    if (migrateRef.current) return
    migrateRef.current = true
    const hash = window.location.hash
    const qIdx = hash.indexOf('?')
    if (qIdx < 0) return
    const usp = new URLSearchParams(hash.slice(qIdx + 1))
    if (!usp.has('marg')) return
    const margOff = usp.get('marg') === '0'
    usp.delete('marg')
    // Bake the migrated `M` (marginals-off) signal into the same hash
    // rewrite. Doing this via use-prms' setShowMarg would race with the
    // replaceState below: `writeToUrl` reads the current URL (still has
    // `marg`) and emits its own replaceState; our replaceState would then
    // overwrite it and strip `M` too. Use URLSearchParams for correct
    // percent-encoding of the surviving keys, then convert any
    // `key=` (empty-value) entries to the valueless `key` form to match
    // use-prms' on-disk encoding.
    if (margOff) usp.set('M', '')
    const search = usp.toString().replace(/(^|&)([^=&]+)=(?=&|$)/g, '$1$2')
    const newHash = hash.slice(0, qIdx) + (search ? `?${search}` : '')
    window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${newHash}`)
    // Nudge listeners (use-prms subscribes to popstate) so the snapshot
    // re-parses and the migrated `M` value flows into `showMarg`.
    window.dispatchEvent(new PopStateEvent('popstate'))
  }, [])

  const xParsed = useMemo(() => xSpec ? parseSpec(xSpec) : null, [xSpec])
  const yParsed = useMemo(() => ySpec ? parseSpec(ySpec) : null, [ySpec])

  // Per-mat grid index — powers the per-axis source/step dropdowns. Same
  // endpoint the /mp/<id> page consumes; cache hits across the two views.
  const gridsQ = useQuery({
    queryKey: ['mp-grids', mp],
    queryFn: () => fetchGrids(mp ?? ''),
    enabled: !!mp,
    staleTime: 30 * 60 * 1000,
  })

  const [state, setState] = useState<LoadState>({ status: 'idle' })

  useEffect(() => {
    if (!mp || !xParsed || !yParsed) {
      setState({ status: 'idle' })
      return
    }
    let cancelled = false
    setState({ status: 'loading', message: `Fetching level-0 voxel arrays for ${mp}…` })

    const xUrl = zarrUrlFor(xParsed, mp)
    const yUrl = zarrUrlFor(yParsed, mp)

    void (async () => {
      try {
        const [xLoad, yLoad] = await Promise.all([
          loadZarrLevel0(xUrl),
          loadZarrLevel0(yUrl),
        ])
        if (cancelled) return
        const xs = xLoad.meta.shape
        const ys = yLoad.meta.shape
        if (xs[0] !== ys[0] || xs[1] !== ys[1] || xs[2] !== ys[2]) {
          throw new Error(
            `shape mismatch: ${xs.join('×')} (x) vs ${ys.join('×')} (y) — sources must be on the same grid`,
          )
        }
        setState({
          status: 'loading',
          message: `Computing histogram (${bins}×${bins} bins, ${xLoad.data.length.toLocaleString()} voxels)…`,
        })
        // Compute vmin/vmax from data extents (positive subset for log).
        const { vmin, vmax } = computeRange(xLoad.data, yLoad.data, scale)
        const result = await runHistWorker({
          xData: xLoad.data,
          yData: yLoad.data,
          vmin,
          vmax,
          bins,
          scale,
        })
        if (cancelled) return
        setState({ status: 'ready', hist: result, dims: xs })
      } catch (err) {
        if (cancelled) return
        setState({
          status: 'error',
          message: err instanceof Error ? err.message : String(err),
        })
      }
    })()
    return () => { cancelled = true }
    // bins + scale changes also trigger re-load (re-binning logically
    // requires re-running the worker; could optimize later to keep voxel
    // arrays in memory and only re-bin, but Phase A keeps it simple).
  }, [mp, xParsed, yParsed, bins, scale])

  const xLabel = useMemo(() => xSpec ? shortenSpec(xSpec) : '?', [xSpec])
  const yLabel = useMemo(() => ySpec ? shortenSpec(ySpec) : '?', [ySpec])
  const title = useMemo(() => {
    const dims = state.dims ? state.dims.join('×') : '…'
    return `${mp || '<no mp>'} — ${dims} voxels — ${xLabel} vs ${yLabel}`
  }, [mp, xLabel, yLabel, state.dims])

  const onRangeChange = (r: JointHistRange) => {
    setXRange(r.xRange)
    setYRange(r.yRange)
  }

  return (
    <>
      <header>
        <h1>tomat 🍅 joint-histogram</h1>
      </header>
      <p className="meta">
        Voxel-wise joint distribution of two charge-density grids on the
        same material. Diagonal = perfect prediction; vertical/horizontal
        stripes = mode collapse; banding parallel to the diagonal = blurry
        prediction. Source specs accept <code>gt:val</code>,{' '}
        <code>gt:train</code>, or{' '}
        <code>pred:&lt;run&gt;:&lt;setMode&gt;:&lt;step&gt;</code>.{' '}
        <a href="#/">← home</a>
      </p>

      <div className="voxel-corr-controls">
        <label>
          mp: <code>{mp || '—'}</code>
          {(() => {
            // Show the dataset split (val/train) once per page as a
            // property OF THE MAT, derived from which GT zarr exists
            // for this mp. Was previously qualifying every `GT` /
            // `bin5 · val · K=12` label — Yael flagged the
            // redundancy.
            const gtSet = gridsQ.data?.find((g) => g.role === 'gt')?.set
            const split = gtSet?.startsWith('val') ? 'val' : gtSet?.startsWith('train') ? 'train' : null
            if (!split) return null
            return <span className="meta" style={{ marginLeft: 4 }}>({split})</span>
          })()}
        </label>
        <AxisPicker axis="x" spec={xSpec ?? ''} setSpec={setXSpec} grids={gridsQ.data ?? []} />
        <AxisPicker axis="y" spec={ySpec ?? ''} setSpec={setYSpec} grids={gridsQ.data ?? []} />
        <label>
          scale:{' '}
          <select value={scale} onChange={e => setScale(e.target.value as Scale)}>
            <option value="log">log</option>
            <option value="linear">linear</option>
          </select>
        </label>
        <label>
          <input type="checkbox" checked={showMarg} onChange={e => setShowMarg(e.target.checked)} />
          {' '}marginals
        </label>
        <span className="meta">bins={bins}</span>
      </div>

      {state.status === 'idle' && (
        <p className="note">
          Set <code>?mp=…&amp;x=…&amp;y=…</code> in the URL. Example:{' '}
          <a href="#/viz/joint-hist?mp=mp-1797712&x=gt:val&y=pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000">
            mp-1797712, GT vs bin5@30k
          </a>
        </p>
      )}
      {state.status === 'loading' && (
        <p className="note">{state.message ?? 'Loading…'}</p>
      )}
      {state.status === 'error' && (
        <p className="note" style={{ color: 'crimson' }}>Error: {state.message}</p>
      )}

      {state.status === 'ready' && state.hist && (
        <JointHistPlot
          hist={state.hist}
          title={title}
          xLabel={xLabel}
          yLabel={yLabel}
          scale={scale}
          showMarginals={showMarg}
          xRange={xRange}
          yRange={yRange}
          onRangeChange={onRangeChange}
        />
      )}
    </>
  )
}

/** Per-axis source picker — flat "source" dropdown (GT (val), GT (train),
 *  pred run × setmode flattened) + a step dropdown that activates when
 *  source is a pred. URL spec format unchanged (`gt:val` /
 *  `pred:<run>:<setmode>:<step>`), so deep-links keep working.
 *
 *  Source key encoding:
 *    - `gt:val`, `gt:train` for ground truth
 *    - `pred:<run>:<setmode>` for a per-(run, setmode) source — step is a
 *      second picker
 *  Available steps per source come from the grids index. */
interface SourceKey {
  /** Stable string key used in the source <select>. */
  key: string
  /** Display label, e.g. `GT (val)`, `bin5 · val · K=12`. */
  label: string
  /** Steps available for this source (empty for GT). Sorted ascending. */
  steps: number[]
}

/** Parse the current spec into `(sourceKey, step | null)`. Returns null
 *  for empty/malformed specs. */
function specToSourceKeyStep(spec: string): { sourceKey: string; step: number | null } | null {
  if (!spec) return null
  const gt = /^gt:(val|train)$/.exec(spec)
  if (gt) return { sourceKey: `gt:${gt[1]}`, step: null }
  const pred = /^pred:([^:]+):([^:]+):(\d+)$/.exec(spec)
  if (pred) return { sourceKey: `pred:${pred[1]}:${pred[2]}`, step: parseInt(pred[3], 10) }
  return null
}

/** Compose (sourceKey, step) back into a canonical spec string. */
function sourceKeyStepToSpec(sourceKey: string, step: number | null): string {
  if (sourceKey.startsWith('gt:')) return sourceKey
  if (sourceKey.startsWith('pred:') && step != null) {
    return `${sourceKey}:${step}`
  }
  return ''
}

/** Build the set of available source keys + labels + steps from the grids
 *  index. Order: GT first, then pred sources sorted by (run, setmode).
 *
 *  The set token (`val` / `train`) is dropped from labels when only ONE
 *  set is available for this mat — by construction every mat is in
 *  either val_200 OR train_200 (never both), so showing it would be
 *  redundant noise on every option. The dropping logic falls back to
 *  including the set token if (somehow) both sets appear, so future-
 *  proofing isn't lost. */
function sourcesFromGrids(grids: GridRow[]): SourceKey[] {
  const out: SourceKey[] = []
  // GT: collect distinct sets seen across gt rows.
  const gtSets = new Set<string>()
  for (const g of grids) if (g.role === 'gt' && g.set) gtSets.add(g.set)
  // Normalize MP-side `validation`/`train` set names to the URL-spec
  // shorthand `val`/`train`. The /api/mp/<id>/grids rows use the full
  // GCS subdir name; the joint-hist URL spec uses the abbreviated form.
  const gtNorm = (s: string) => (s.startsWith('val') ? 'val' : s.startsWith('train') ? 'train' : null)
  // Collect distinct (run, set-without-mode) pairs across BOTH gt and
  // pred. If exactly one set token appears across all options, drop it
  // from the labels for ambient clarity. Otherwise keep it to
  // disambiguate.
  const setTokens = new Set<string>()
  for (const s of gtSets) {
    const n = gtNorm(s)
    if (n) setTokens.add(n)
  }
  for (const g of grids) {
    if (g.role !== 'pred' || !g.set) continue
    if (g.set.startsWith('val')) setTokens.add('val')
    else if (g.set.startsWith('train')) setTokens.add('train')
  }
  const dropSet = setTokens.size <= 1
  const seenGt = new Set<string>()
  for (const s of gtSets) {
    const n = gtNorm(s)
    if (!n || seenGt.has(n)) continue
    seenGt.add(n)
    out.push({
      key: `gt:${n}`,
      label: dropSet ? 'GT' : `GT (${n})`,
      steps: [],
    })
  }
  // Pred: group by (run_id, set) and synthesize the setMode the URL
  // spec expects. The grids API returns `set` values like `val_200` and
  // `val_200-maskgit` — these are themselves the setMode token used in
  // `pred:<run>:<setMode>:<step>`. So we use `g.set` directly as the
  // setMode in the spec, and label the picker with a friendlier form.
  type Bucket = { run: string; setMode: string; steps: Set<number> }
  const buckets = new Map<string, Bucket>()
  for (const g of grids) {
    if (g.role !== 'pred' || !g.run_id || !g.set || g.step == null) continue
    const key = `${g.run_id}|${g.set}`
    const b = buckets.get(key) ?? { run: g.run_id, setMode: g.set, steps: new Set<number>() }
    b.steps.add(g.step)
    buckets.set(key, b)
  }
  const friendlyLabel = (run: string, setMode: string): string => {
    const runShort = shortenRun(run)
    // setMode is e.g. `val_200`, `val_200-maskgit`, `train_200`,
    // `train_200-maskgit`. Mode is the K=1/K=12 distinction; set is
    // val/train (omitted when there's only one).
    const set = setMode.startsWith('val') ? 'val' : 'train'
    const mode = setMode.endsWith('-maskgit') ? 'K=12' : 'K=1'
    return dropSet ? `${runShort} · ${mode}` : `${runShort} · ${set} · ${mode}`
  }
  const sorted = [...buckets.values()].sort((a, b) =>
    a.run.localeCompare(b.run) || a.setMode.localeCompare(b.setMode))
  for (const b of sorted) {
    out.push({
      key: `pred:${b.run}:${b.setMode}`,
      label: friendlyLabel(b.run, b.setMode),
      steps: [...b.steps].sort((a, b) => a - b),
    })
  }
  return out
}

/** Format a step int as a compact label (e.g. 89999 → `90k`, 30000 → `30k`).
 *  Mirrors the run-cards convention; we don't carry the asterisk semantics
 *  here since the joint-hist viewer just needs a picker label. */
function shortStepLabel(step: number): string {
  if (step >= 1_000_000 && step % 100_000 === 0) {
    const v = step / 1_000_000
    return v === Math.floor(v) ? `${v}M` : `${v.toFixed(1)}M`
  }
  if (step >= 1000) {
    // Snap to nearest round k for the legacy `-1` force-saves (`89999 →
    // 90k`). Anything else → `Math.round(step/1000)k`.
    const round = Math.round(step / 1000)
    const expected = round * 1000
    if (Math.abs(expected - step) <= 1) return `${round}k`
    return step.toLocaleString()
  }
  return String(step)
}

function AxisPicker({
  axis, spec, setSpec, grids,
}: {
  axis: 'x' | 'y'
  spec: string
  setSpec: (s: string) => void
  grids: GridRow[]
}) {
  const sources = useMemo(() => sourcesFromGrids(grids), [grids])
  const current = useMemo(() => specToSourceKeyStep(spec), [spec])
  const activeSource = sources.find((s) => s.key === current?.sourceKey)

  const onSourceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const nextKey = e.target.value
    const next = sources.find((s) => s.key === nextKey)
    if (!next) return
    // GT sources have no step. For pred, pick the LATEST step by default
    // when switching sources (the most-recent ckpt is usually what you
    // want; the user can pull the step dropdown back to override).
    const nextStep = next.steps.length > 0 ? next.steps[next.steps.length - 1] : null
    setSpec(sourceKeyStepToSpec(nextKey, nextStep))
  }
  const onStepChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const nextStep = parseInt(e.target.value, 10)
    if (!current || Number.isNaN(nextStep)) return
    setSpec(sourceKeyStepToSpec(current.sourceKey, nextStep))
  }
  return (
    <label>
      {axis}:{' '}
      <select value={current?.sourceKey ?? ''} onChange={onSourceChange}>
        {!current && <option value="">—</option>}
        {sources.map((s) => (
          <option key={s.key} value={s.key}>{s.label}</option>
        ))}
      </select>
      {activeSource && activeSource.steps.length > 0 && current?.step != null && (
        <>
          {' '}@{' '}
          <select value={current.step} onChange={onStepChange}>
            {activeSource.steps.map((s) => (
              <option key={s} value={s}>{shortStepLabel(s)}</option>
            ))}
          </select>
        </>
      )}
    </label>
  )
}

/** Pick a (vmin, vmax) for log-binning. Uses the union of finite-positive
 *  values across both sources to cover both distributions in one shared
 *  range, then bounds the upper limit at p99.9 to avoid a long thin tail
 *  swallowing the bin grid. */
function computeRange(
  x: Float32Array,
  y: Float32Array,
  scale: Scale,
): { vmin: number; vmax: number } {
  let minPos = Infinity
  let max = -Infinity
  let minAny = Infinity
  for (let i = 0; i < x.length; i++) {
    const a = x[i], b = y[i]
    if (a < minAny) minAny = a
    if (b < minAny) minAny = b
    if (a > max) max = a
    if (b > max) max = b
    if (a > 0 && a < minPos) minPos = a
    if (b > 0 && b < minPos) minPos = b
  }
  if (!Number.isFinite(max)) max = 1
  if (scale === 'log') {
    if (!Number.isFinite(minPos)) minPos = 1e-3
    // Drop the bottom decade of fp16 noise — fp16 underflow → very tiny
    // positive values blow up the range without adding signal.
    const floor = Math.max(minPos, max * 1e-6)
    return { vmin: floor, vmax: max }
  }
  return { vmin: Math.min(0, minAny), vmax: max }
}

/** Run the worker once + collect its single response. Tears the worker down
 *  on completion so a stale instance can't post a late result over our
 *  next render. */
function runHistWorker(req: JointHistRequest): Promise<JointHistResponse> {
  return new Promise((resolve, reject) => {
    const w = new JointHistWorker()
    w.onmessage = (e: MessageEvent<JointHistResponse>) => {
      resolve(e.data)
      w.terminate()
    }
    w.onerror = (e: ErrorEvent) => {
      reject(new Error(`worker error: ${e.message || 'unknown'}`))
      w.terminate()
    }
    // Transfer the Float32 buffers — saves the structured-clone copy on the
    // way over (~5MB at 80×80×64 × 2 arrays).
    w.postMessage(req, [req.xData.buffer, req.yData.buffer])
  })
}
