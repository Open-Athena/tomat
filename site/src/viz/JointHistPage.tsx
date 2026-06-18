// Standalone joint-histogram route: `#/viz/joint-hist?mp=…&x=…&y=…`.
//
// Loads the level-0 voxel arrays for two sources, dispatches a worker to
// compute the 2D histogram, renders via `JointHistPlot`. URL params drive
// everything; the page is shareable end-to-end.
//
// Phase A only: every load re-downloads + re-computes. Phase B (CLI
// precompute → R2) lands later.

import { useEffect, useMemo, useRef, useState } from 'react'
import { enumParam, intParam, stringParam, useUrlState } from 'use-prms'
import { boolTrueParam } from '../lib/urlParams'
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
  const [xSpec] = useUrlState('x', stringParam(''))
  const [ySpec] = useUrlState('y', stringParam(''))
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
        </label>
        <SourceLabel axis="x" spec={xSpec} short={xLabel} />
        <SourceLabel axis="y" spec={ySpec} short={yLabel} />
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

/** Render one axis's source label as `<axis>: <short>`, linking pred specs
 *  to their run page and surfacing the full canonical spec as a `title`
 *  tooltip. GT specs (no run page) render plain. Empty/unparseable specs
 *  render as `—`. */
function SourceLabel({ axis, spec, short }: { axis: 'x' | 'y'; spec: string; short: string }) {
  if (!spec) {
    return <label>{axis}: <code>—</code></label>
  }
  const m = /^pred:([^:]+):[^:]+:\d+$/.exec(spec)
  const runId = m?.[1]
  if (runId) {
    return (
      <label>
        {axis}: <a href={`#/runs/${runId}`} title={spec}>{short}</a>
      </label>
    )
  }
  return <label title={spec}>{axis}: {short}</label>
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
