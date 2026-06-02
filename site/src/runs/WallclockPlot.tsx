// Per-run plot — 2 stacked subplots over a shared x-axis. x-axis modes:
//   • wallclock — local time (the viewer's zone)
//   • elapsed   — hours since the run's first log
//   • step      — Levanter `global_step`
// Panels:
//   1. step (running max of global_step) — top, 28%. Hidden in step mode
//      (it would degenerate to y = x).
//   2. losses + mat-NMAE + mat-NEMD on a single log y-axis (bottom).
//
// Eval bands (MV = val_200, MT = train_200) come from the canonical per-step
// eval.json — NOT the parquet's collapsed harvested points. Per (set × metric)
// the plot draws a median center line plus two shaded spread bands across the
// 200 mats: p25–p75 (IQR) and a fainter p1–p99. NMAE is green, NEMD teal; MV
// solid, MT dashed. Eval points are keyed by checkpoint step; on the
// time/elapsed axes they're placed at the wallclock of that step, recovered
// from the parquet's (timestamp, global_step) rows.
//
// Lifecycle events render as vertical lines via `shapes` (yref='paper') so
// they span both panels.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { rolling } from 'pltly/core'
import { enumParam, useUrlState } from 'use-prms'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'
import type { IrisAttempts, RunEval, RunManifest } from './api'
import { classifyDeath, DEATH_COLORS, type DeathCause } from './deathEvents'
import { SmoothingChips, useBandsToggle, useSmoothMode } from './RunsTimelinePlot'
import { epochOfStep } from './runMeta'
import { applySmoothing, type SmoothMode } from './smoothing'

interface Props {
  history: RunHistory
  evalSeries: RunEval | null
  runId: string
  /** Initial x-axis mode when the URL has no `?x=…`. Defaults to `'step'`
   *  for the run-detail page (training-progress is the obvious x for a
   *  single-run view); callers using this on a multi-run context can
   *  override to `'wallclock'` or `'elapsed'`. */
  defaultXMode?: UrlXMode
  /** Per-task attempt history (death events). Drives the death-cause vlines
   *  + legend entries that augment (but don't replace) the existing
   *  trainer_started / sigterm / cluster_preempt overlays. */
  attempts?: IrisAttempts | null
  /** Run manifest — used for the `epoch` x-axis mode (fractional epoch =
   *  `step · train_batch_size / epoch_sequences`). When null or missing the
   *  data needed (data label not in `EPOCH_SEQUENCES`, `train_batch_size`
   *  missing), the `epoch` button is hidden. */
  manifest?: RunManifest | null
}

type XMode = 'time' | 'elapsed' | 'step' | 'epoch'

// URL-facing x-axis mode names (`?x=wallclock|elapsed|step|epoch`). The
// internal XMode uses `'time'` for the wallclock axis; `wallclock` reads
// better in shared links.
type UrlXMode = 'wallclock' | 'elapsed' | 'step' | 'epoch'
const URL_X_MODES = ['wallclock', 'elapsed', 'step', 'epoch'] as const
const X_TO_URL: Record<XMode, UrlXMode> = {
  time: 'wallclock', elapsed: 'elapsed', step: 'step', epoch: 'epoch',
}
const URL_TO_X: Record<UrlXMode, XMode> = {
  wallclock: 'time', elapsed: 'elapsed', step: 'step', epoch: 'epoch',
}

const COLORS = {
  step: '#2196f3',
  TL: '#ef5350',     // train/loss
  VL: '#ffa726',     // eval/loss
  nmae: '#43a047',   // mat-NMAE — green
  nemd: '#00acc1',   // mat-NEMD — teal
  start: '#ffa726',
  sigterm: '#bdbdbd',
  preempt: '#ba68c8',
} as const

/** Translucent fill for a metric's spread band. */
const bandFill = (metric: 'nmae' | 'nemd', alpha: number): string =>
  metric === 'nmae' ? `rgba(67,160,71,${alpha})` : `rgba(0,172,193,${alpha})`

/** `#rrggbb` → `"R, G, B"` for use inside `rgba(…)` strings. Used by the
 *  smoothing ±σ bands to colour-match their parent line. */
function hexToRgbTuple(hex: string): string {
  let h = hex.startsWith('#') ? hex.slice(1) : hex
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  if (h.length !== 6) return '128, 128, 128'
  const n = parseInt(h, 16)
  if (!Number.isFinite(n)) return '128, 128, 128'
  return `${(n >> 16) & 0xff}, ${(n >> 8) & 0xff}, ${n & 0xff}`
}

/** Local-time `YYYY-MM-DD HH:MM:SS` (no tz suffix → a Plotly date axis renders
 *  it verbatim, i.e. in the viewer's local zone rather than UTC). */
function toLocal(ts: number): string {
  const d = new Date(ts * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} `
    + `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** Short local-timezone label for the x-axis title, e.g. "EDT". */
const TZ_LABEL: string = (() => {
  try {
    const parts = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' })
      .formatToParts(new Date())
    return parts.find((p) => p.type === 'timeZoneName')?.value ?? 'local'
  } catch {
    return 'local'
  }
})()

export function WallclockPlot({ history, evalSeries, runId, defaultXMode = 'step', attempts, manifest = null }: Props) {
  const { isDark } = useTheme()
  // Wrapper around <Plot> so we can DOM-walk to the `.js-plotly-plot` element
  // and call `Plotly.restyle` on the band traces directly. Bands have
  // `showlegend: false`; pltly's built-in `applyFadeSolo` skips those (since
  // they don't appear in the legend) — so without this fix the teal NEMD bands
  // stay full opacity when MV NMAE is hovered.
  const plotWrapperRef = useRef<HTMLDivElement | null>(null)
  const [activeTraceName, setActiveTraceName] = useState<string | null>(null)
  type PlotlyDiv = HTMLElement & {
    data?: Array<Record<string, unknown>>
    _Plotly?: { restyle: (el: HTMLElement, attrs: Record<string, unknown>, indices?: number[]) => Promise<void> }
  }
  // Fade bands by `legendgroup` whenever the active trace changes. We walk
  // the Plotly trace list to find:
  //   1. activeLG = legendgroup of the active (legend-visible) trace
  //   2. bandIndices = indices of all `showlegend: false` traces
  // Then `Plotly.restyle(plotDiv, { opacity: vals }, bandIndices)` flips each
  // band to 1 if it shares the active legendgroup, 0.3 otherwise. When
  // activeTraceName is null, all bands → 1.
  //
  // Idempotency check: every `Plotly.restyle` re-emits `plotly_afterplot`,
  // which we listen for to re-apply fade after Plotly.react resets defaults.
  // Without this no-op guard the afterplot → restyle → afterplot recurses
  // until the call stack blows.
  const applyBandFade = useCallback(() => {
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as PlotlyDiv | null
    const P = plotDiv?._Plotly
    if (!plotDiv?.data || !P) return
    let activeLG: string | null = null
    if (activeTraceName) {
      for (const t of plotDiv.data) {
        if (t.name === activeTraceName && t.showlegend !== false) {
          activeLG = (t.legendgroup as string | undefined) ?? null
          break
        }
      }
    }
    const indices: number[] = []
    const opacities: number[] = []
    let changed = false
    for (let i = 0; i < plotDiv.data.length; i++) {
      const t = plotDiv.data[i]
      // Only touch the band-EDGE traces (eval p1/p99/p25/p75, smoothing ±σ
      // edges). They all set `hoverinfo: 'skip'` to suppress their own
      // tooltips. Restart-segment traces (showlegend: false but
      // hoverinfo undefined) must NOT be touched — they carry their own
      // recency-ramp opacity, and forcing them to 1 here would flatten the
      // ramp.
      if (t.showlegend !== false) continue
      if (t.hoverinfo !== 'skip') continue
      indices.push(i)
      const want = activeTraceName == null || (activeLG != null && (t.legendgroup as string | undefined) === activeLG) ? 1 : 0.3
      opacities.push(want)
      const current = (t.opacity as number | undefined) ?? 1
      if (Math.abs(current - want) > 1e-9) changed = true
    }
    if (indices.length === 0 || !changed) return
    P.restyle(plotDiv, { opacity: opacities }, indices)
  }, [activeTraceName])
  useEffect(applyBandFade, [applyBandFade])
  // Re-apply on every Plotly redraw — `Plotly.react` (triggered by xMode swap,
  // data refetch, etc.) resets band opacity defaults to 1. Without this, bands
  // would briefly snap to full opacity after a re-render even with a hover
  // active. The no-op guard in `applyBandFade` breaks the otherwise-infinite
  // afterplot → restyle → afterplot loop (restyle re-emits afterplot).
  useEffect(() => {
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as (HTMLElement & {
      on?: (evt: string, fn: () => void) => void
      removeListener?: (evt: string, fn: () => void) => void
    }) | null
    if (!plotDiv?.on) return
    plotDiv.on('plotly_afterplot', applyBandFade)
    return () => plotDiv.removeListener?.('plotly_afterplot', applyBandFade)
  }, [applyBandFade])

  // `?x=wallclock|elapsed|step|epoch` — URL-persisted so deep-links carry
  // the view choice. The run-detail page defaults to `'step'` (training
  // progress is the obvious x for a single run); callers can override.
  const [urlXMode, setUrlXMode] = useUrlState('x', enumParam<UrlXMode>(defaultXMode, URL_X_MODES))
  // Whether the `epoch` axis is available — requires `train_batch_size`
  // from the run config + an `EPOCH_SEQUENCES` entry for the run's data
  // label. Hides the button (and silently falls back to `step` if the URL
  // asks for `?x=epoch` on a run whose manifest can't compute it).
  const epochAvailable = epochOfStep(0, manifest) != null
  const rawXMode: XMode = URL_TO_X[urlXMode]
  const xMode: XMode = rawXMode === 'epoch' && !epochAvailable ? 'step' : rawXMode
  const setXMode = (m: XMode) => setUrlXMode(X_TO_URL[m])

  // User-set x-range captured from box-zoom (`plotly_relayout`). Persists
  // across smoothing-chip clicks and poll-driven re-renders that would
  // otherwise reset the axis to autorange. `null` = no user range; render
  // with autorange. Double-click on the plot clears the range (plotly emits
  // `xaxis.autorange: true`); we mirror that into local state.
  //
  // Values come back as `number` for linear axes (`x=step|elapsed`) and as
  // date-string for the `date`-type wallclock axis (`x=wallclock`). Accept
  // both — initially we filtered on `typeof === 'number'`, which silently
  // dropped every wallclock zoom and caused the range to reset on every
  // 30 s react-query poll.
  type AxisVal = number | string
  const [userXRange, setUserXRange] = useState<[AxisVal, AxisVal] | null>(null)
  // Capture box-zoom into local state via `<Plot onRelayout=…>` (not a
  // post-mount DOM `plot.on('plotly_relayout', …)` listener). pltly's <Plot>
  // attaches the prop INSIDE its own `bindEvents` — which runs AFTER
  // `Plotly.react` creates the `.js-plotly-plot` element. A useEffect with
  // `[]` deps that tries to grab `.js-plotly-plot` at mount races against
  // <Plot>'s async first render: querySelector returns null, we bail, and
  // the listener is silently never attached — so box-zoom never updates
  // `userXRange`, and the next render (e.g. legend-hover triggering a
  // layout rebuild via `eventShapes` colors) drops `xaxis.range` from the
  // prop, letting plotly autorange. From the user's seat: hovering a
  // legend item resets their zoom.
  const onRelayout = useCallback((ev: Record<string, unknown>) => {
    if (ev['xaxis.autorange'] === true) {
      setUserXRange(null)
      return
    }
    const lo = ev['xaxis.range[0]']
    const hi = ev['xaxis.range[1]']
    const validLo = typeof lo === 'number' || typeof lo === 'string'
    const validHi = typeof hi === 'number' || typeof hi === 'string'
    if (validLo && validHi) {
      setUserXRange([lo as AxisVal, hi as AxisVal])
    }
  }, [])
  // x-mode changes are unit changes (step ↔ elapsed ↔ wallclock) — preserving
  // a numeric range across them would land you somewhere meaningless. Clear.
  useEffect(() => { setUserXRange(null) }, [xMode])
  const { timestamps, cols } = history

  const ordered = useMemo(
    () => timestamps
      .map((ts, i) => ({ ts, i }))
      .filter((r) => r.ts !== null)
      .sort((a, b) => (a.ts as number) - (b.ts as number)),
    [timestamps],
  )
  const t0 = ordered.length > 0 ? (ordered[0].ts as number) : 0

  // (ts, gstep) pairs from every row carrying a `global_step` — sorted by ts
  // and, since training is monotonic, effectively by gstep too. Drives both
  // ts→gstep (eval/lifecycle back-fill) and gstep→ts (eval-point placement).
  const tsGstep = useMemo(() => {
    const globalStep = cols.get('global_step') ?? []
    const pairs: { ts: number; gstep: number }[] = []
    for (const { ts, i } of ordered) {
      const g = globalStep[i]
      if (g === null || g === undefined) continue
      pairs.push({ ts: ts as number, gstep: g as number })
    }
    return pairs
  }, [ordered, cols])

  /** gstep of the latest logged row at or before `ts`. */
  function gstepAtTs(ts: number): number | null {
    if (tsGstep.length === 0) return null
    let lo = 0, hi = tsGstep.length - 1, best = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (tsGstep[mid].ts <= ts) { best = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    return best < 0 ? (tsGstep[0]?.gstep ?? null) : tsGstep[best].gstep
  }

  /** Wallclock ts when the run first reached `step` (gstep is monotonic in
   *  ts order). Used to place per-step eval points on the time axes. */
  function tsAtGstep(step: number): number | null {
    if (tsGstep.length === 0) return null
    let lo = 0, hi = tsGstep.length - 1, best = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (tsGstep[mid].gstep >= step) { best = mid; hi = mid - 1 }
      else lo = mid + 1
    }
    return best < 0 ? tsGstep[tsGstep.length - 1].ts : tsGstep[best].ts
  }

  /** x-coordinate for a wallclock ts, per the current x-mode. */
  function xOfTs(ts: number): string | number {
    if (xMode === 'step') return gstepAtTs(ts) ?? NaN
    if (xMode === 'epoch') {
      const s = gstepAtTs(ts)
      if (s == null) return NaN
      const ep = epochOfStep(s, manifest)
      return ep ?? NaN
    }
    if (xMode === 'elapsed') return (ts - t0) / 3600
    return toLocal(ts)
  }

  /** x-coordinate for an eval point at checkpoint `step`. */
  function xOfStep(step: number): string | number | null {
    if (xMode === 'step') return step
    if (xMode === 'epoch') return epochOfStep(step, manifest)
    const ts = tsAtGstep(step)
    if (ts === null) return null
    return xMode === 'elapsed' ? (ts - t0) / 3600 : toLocal(ts)
  }

  // Per-metric parquet series (TL/VL): xs, ys, gsteps (gstep for the tooltip).
  type Series = { xs: (string | number)[]; ys: number[]; gsteps: (number | null)[] }

  // Restart-segment boundaries: indices into `ordered` keyed on
  // `lifecycle/trainer_started` events (each one marks a new trainer
  // process). Splitting at these boundaries lets us render older restart
  // trajectories at low opacity so they don't deface the plot, while the
  // latest segment stays fully visible.
  //
  // Per-row `global_step` regression OVER-segments — interleaved train/eval
  // rows in async timestamp order can show gstep flipping back and forth
  // within a single restart, especially when an eval log lands after the
  // next train step. `lifecycle/trainer_started` is the unambiguous signal:
  // one entry per trainer process boot.
  //
  // With one start, this returns `[0]` and the whole run renders as a
  // single trace as before.
  const segmentStarts = useMemo<number[]>(() => {
    const startedCol = cols.get('lifecycle/trainer_started') ?? []
    const startTsRaw: number[] = []
    for (let i = 0; i < startedCol.length; i++) {
      if (startedCol[i] === 1 && timestamps[i] != null) {
        startTsRaw.push(timestamps[i] as number)
      }
    }
    startTsRaw.sort((a, b) => a - b)
    if (startTsRaw.length === 0) return [0]
    // First segment always starts at index 0. For each subsequent
    // trainer_started ts, find the first `ordered` index whose ts >= that ts.
    const out: number[] = [0]
    let k = 0
    for (let s = 1; s < startTsRaw.length; s++) {
      while (k < ordered.length && (ordered[k].ts as number) < startTsRaw[s]) k++
      if (out[out.length - 1] !== k && k < ordered.length) out.push(k)
    }
    return out
  }, [ordered, cols, timestamps])
  const numSegments = segmentStarts.length

  // Segment index for a given index into `ordered`. Binary search over
  // `segmentStarts` (monotonic).
  function segmentOf(orderedIdx: number): number {
    let lo = 0, hi = segmentStarts.length - 1, best = 0
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (segmentStarts[mid] <= orderedIdx) { best = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    return best
  }

  // Returns one Series per restart segment. Empty segments are preserved at
  // their index so opacity / naming line up with `numSegments`.
  function series(key: string): Series[] {
    const col = cols.get(key) ?? []
    const out: Series[] = Array.from({ length: numSegments },
      () => ({ xs: [], ys: [], gsteps: [] }))
    for (let k = 0; k < ordered.length; k++) {
      const { ts, i } = ordered[k]
      const v = col[i]
      if (v === null || v === undefined) continue
      const seg = out[segmentOf(k)]
      seg.gsteps.push(gstepAtTs(ts as number))
      seg.xs.push(xOfTs(ts as number))
      seg.ys.push(v as number)
    }
    return out
  }

  // Step (top-panel) trace: running-max of global_step.
  const stepTrace = useMemo(() => {
    const globalStep = cols.get('global_step') ?? []
    const xs: (string | number)[] = []
    const ys: number[] = []
    let runningMax = -Infinity
    for (const { ts, i } of ordered) {
      const s = globalStep[i]
      if (s === null) continue
      runningMax = Math.max(runningMax, s)
      xs.push(xOfTs(ts as number))
      ys.push(runningMax)
    }
    return { xs, ys }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ordered, cols, xMode])

  const customGsteps = (s: Series) => s.gsteps.map((g) => (g === null ? '?' : g))

  const TL = series('train/loss')
  const VL = series('eval/loss')

  // Smoothing (shared URL state with the cross-run timeline). When raw the
  // TL/VL traces render unchanged; otherwise replace y, and optionally emit
  // ±σ fill bands in the same `legendgroup` so the existing legendgroup-fade
  // machinery (above) brushes the bands with their parent line.
  //
  // `rolling` mode goes through pltly's `rolling()` (Welford's O(n)) so the
  // ±σ band is the real within-window stddev. `ema` keeps tomat's local
  // implementation (pltly ships rolling but not EMA); since EMA has no natural
  // σ companion, bands are unavailable in that mode (matching the chip rail's
  // disabled-state).
  const [smooth, setSmooth] = useSmoothMode()
  const [bandsOn, setBandsOn] = useBandsToggle()

  /** Smoothed mean + (rolling-only) σ for the supplied y-array. Sample-index
   *  window — matches the legacy `?smooth=rolling:N` semantics where N counts
   *  log rows, not x-units. */
  function smoothedMeanStd(
    ys: (number | null)[], mode: SmoothMode,
  ): { mean: (number | null)[]; std: (number | null)[] | null } {
    if (mode.kind !== 'rolling') {
      return { mean: applySmoothing(ys, mode), std: null }
    }
    const idx = ys.map((_, i) => i)
    const sm = rolling(idx, {
      getX: (i) => i,
      metrics: ['y'],
      getValue: (i) => ys[i],
      windowSize: mode.window,
    })
    const mean: (number | null)[] = []
    const std: (number | null)[] = []
    for (const pt of sm) {
      const m = pt.smoothed.y
      mean.push(Number.isNaN(m.mean) ? null : m.mean)
      std.push(Number.isNaN(m.stddev) ? null : m.stddev)
    }
    return { mean, std }
  }

  // Gap detection in wallclock / elapsed modes: when a run is paused
  // (e.g. a Modal-side respawn between two real data points without an
  // intervening `lifecycle/trainer_started`), the parquet rows that bookend
  // the pause would otherwise be connected by a solid line indistinguishable
  // from a normal training step — especially misleading once rolling
  // smoothing is on. Detect "gaps" as consecutive-x deltas ≥ `GAP_THRESHOLD`
  // × the segment's median delta, split the main line at those points (insert
  // `null` y to break it), and emit a separate dotted bridge trace (same
  // color/legendgroup/opacity, `showlegend: false`) covering only the gap
  // span. Step-mode is exempt: restart-segment splitting already breaks at
  // step regressions and intra-segment step deltas are unit-1 monotone.
  const GAP_THRESHOLD = 10
  // Numeric distance between consecutive xs. `NaN` if either is null/mixed.
  // For date-typed `wallclock` xs (ISO strings) → ms; for `elapsed` xs
  // (hours, numbers) → hours. Units are irrelevant for the median-based
  // threshold — only the ratio matters.
  function xDelta(prev: string | number, curr: string | number): number {
    if (typeof prev === 'number' && typeof curr === 'number') return curr - prev
    if (typeof prev === 'string' && typeof curr === 'string') {
      return new Date(curr).getTime() - new Date(prev).getTime()
    }
    return NaN
  }
  // Indices `i` such that `xs[i-1] → xs[i]` is a gap (delta ≥ threshold ×
  // median). Empty in step-mode (caller short-circuits) and for short series.
  function findGapEndIndices(xs: (string | number)[]): number[] {
    if (xMode === 'step' || xs.length < 3) return []
    const deltas: number[] = []
    for (let i = 1; i < xs.length; i++) {
      const d = xDelta(xs[i - 1], xs[i])
      if (Number.isFinite(d) && d > 0) deltas.push(d)
    }
    if (deltas.length === 0) return []
    const sorted = [...deltas].sort((a, b) => a - b)
    const median = sorted[Math.floor(sorted.length / 2)]
    if (!(median > 0)) return []
    const cutoff = median * GAP_THRESHOLD
    const out: number[] = []
    for (let i = 1; i < xs.length; i++) {
      const d = xDelta(xs[i - 1], xs[i])
      if (Number.isFinite(d) && d > cutoff) out.push(i)
    }
    return out
  }

  // Smooth a parquet-derived Series in place + (optionally) build paired
  // ±σ band traces. Bands inherit the line's `name` (so the closest-trace
  // matching logic upstream remains correct) and a shared `legendgroup` so
  // pltly's solo/fade + our `applyBandFade` desaturate them together.
  //
  // For runs with restart segments (`segments.length > 1`), older segments
  // render at low opacity so the latest segment dominates visually. Only the
  // latest segment gets a legend entry; the older segments share its
  // `legendgroup` so the legend toggle hides/shows the whole stack.
  type SmoothedTrace = Record<string, unknown>
  function smoothedSeriesTraces(
    segments: Series[], name: string, color: string, lineWidth: number, lg: string,
  ): SmoothedTrace[] {
    const N = segments.length
    // Most-recent segment with data — gets the legend entry + full opacity.
    // Without this anchor (just using `N - 1`), a fresh resume that hasn't
    // logged any rows yet would render no legend entry and no full-opacity
    // trace — the entire plot would be older segments at fade.
    let lastNonEmpty = -1
    for (let i = N - 1; i >= 0; i--) {
      if (segments[i].xs.length > 0) { lastNonEmpty = i; break }
    }
    const out: SmoothedTrace[] = []
    segments.forEach((s, segIdx) => {
      if (s.xs.length === 0) return
      const isLatest = segIdx === lastNonEmpty
      // Linear ramp anchored on lastNonEmpty: oldest = 0.40, latest = 1.0.
      // Floor at 0.40 (was 0.18) so older trajectories stay readable as
      // individual lines, not just an overlapping smear. Aggregate alpha
      // from many overlapping segments is still bounded by 1.0.
      const opacity = lastNonEmpty <= 0 ? 1
        : 0.40 + 0.60 * (segIdx / lastNonEmpty)
      // Label per-segment as `<name> #k/N` (1-based, matches the
      // restart-segment header count). The hovertemplate uses `segName`
      // too so the x-unified tooltip shows distinct lines per segment
      // instead of three identical "TL (train/loss)" rows.
      const segName = N > 1 && !isLatest ? `${name} #${segIdx + 1}/${N}` : name
      const { mean: ySmoothed, std: yStd } = smoothedMeanStd(s.ys, smooth)
      const segGsteps = customGsteps(s)
      // Gap-break the main-line arrays so plotly doesn't connect across pauses.
      // We insert one extra `null` y entry between the gap-start and gap-end
      // indices (plotly drops the null point but breaks the line on either
      // side). The bridge trace gets only the (start, end, null) triples for
      // each gap so consecutive bridges render as discrete dotted spans
      // rather than fusing into one polyline.
      const gapEnds = findGapEndIndices(s.xs)
      const gapSet = new Set(gapEnds)
      const bridgeX: (string | number | null)[] = []
      const bridgeY: (number | null)[] = []
      let lineX: (string | number)[] = s.xs
      let lineY: (number | null)[] = ySmoothed
      let lineG: (number | string | null)[] = segGsteps
      let lineYStd: (number | null)[] | null = yStd
      if (gapEnds.length > 0) {
        const newX: (string | number)[] = []
        const newY: (number | null)[] = []
        const newG: (number | string | null)[] = []
        const newYStd: (number | null)[] | null = yStd ? [] : null
        for (let i = 0; i < s.xs.length; i++) {
          if (gapSet.has(i)) {
            // Insert a null-y break-marker BEFORE the post-gap point. Reuse
            // the gap-start x as the marker's x — plotly drops the null
            // anyway; its x value never renders.
            newX.push(s.xs[i - 1])
            newY.push(null)
            newG.push(null)
            newYStd?.push(null)
            // Bridge: (start, end, null) so consecutive bridges don't fuse.
            bridgeX.push(s.xs[i - 1], s.xs[i], null)
            bridgeY.push(ySmoothed[i - 1], ySmoothed[i], null)
          }
          newX.push(s.xs[i])
          newY.push(ySmoothed[i])
          newG.push(segGsteps[i])
          newYStd?.push(yStd ? yStd[i] : null)
        }
        lineX = newX; lineY = newY; lineG = newG; lineYStd = newYStd
      }
      const lineTrace: SmoothedTrace = {
        x: lineX, y: lineY, name: segName,
        type: 'scatter', mode: 'lines',
        line: { color, width: lineWidth },
        opacity,
        yaxis: 'y2',
        legendgroup: lg,
        showlegend: isLatest,
        customdata: lineG,
        hovertemplate: `${segName} %{y:.3f}<br>gstep %{customdata}<extra></extra>`,
      }
      // Dotted bridge connecting each gap's endpoints. Same color/opacity/
      // legendgroup as the main line so a legend-toggle hides them with the
      // rest of the segment; `showlegend: false` so it doesn't add its own
      // LI. We use `hoverinfo: 'none'` (NOT 'skip') intentionally: the
      // `applyBandFade` callback above force-sets opacity on all
      // `showlegend:false + hoverinfo:'skip'` traces (the eval / ±σ band
      // edges) to either 1 or 0.3, which would flatten the older-segment
      // opacity ramp on our bridges. `'none'` lets the configured opacity
      // pass through while still suppressing the bridge's tooltip.
      const bridgeTrace: SmoothedTrace | null = bridgeX.length > 0 ? {
        x: bridgeX, y: bridgeY, name: segName,
        type: 'scatter', mode: 'lines',
        line: { color, width: lineWidth, dash: 'dot' },
        opacity,
        yaxis: 'y2',
        legendgroup: lg,
        showlegend: false,
        hoverinfo: 'none',
      } : null
      // Only show ±σ bands on the latest segment — drawing 28 overlapping
      // bands would just be noise, and the older trajectories already smear
      // visually via opacity.
      if (!bandsOn || !lineYStd || !isLatest) {
        out.push(lineTrace)
        if (bridgeTrace) out.push(bridgeTrace)
        return
      }
      const yLower: (number | null)[] = []
      const yUpper: (number | null)[] = []
      for (let i = 0; i < lineY.length; i++) {
        const m = lineY[i]
        const sd = lineYStd[i]
        if (m == null || sd == null) { yLower.push(null); yUpper.push(null); continue }
        // y2 is log-scaled — keep band edges strictly positive so plotly doesn't
        // drop them (`log(<=0) = NaN`). 1e-6 is well below any real loss value.
        yLower.push(Math.max(1e-6, m - sd))
        yUpper.push(m + sd)
      }
      const edge = (y: (number | null)[], fillKey: string | null): SmoothedTrace => ({
        x: lineX, y, name: segName,
        type: 'scatter', mode: 'lines',
        line: { width: 0, color: 'rgba(0,0,0,0)' },
        yaxis: 'y2', legendgroup: lg,
        showlegend: false,
        hoverinfo: 'skip',
        ...(fillKey ? { fill: 'tonexty', fillcolor: fillKey } : {}),
      })
      const rgb = hexToRgbTuple(color)
      const fillcolor = `rgba(${rgb}, 0.18)`
      out.push(edge(yLower, null), edge(yUpper, fillcolor), lineTrace)
      if (bridgeTrace) out.push(bridgeTrace)
    })
    return out
  }

  // ── eval bands from eval.json ──
  // Per (set × metric): a median center line + a p25–p75 IQR band + a fainter
  // p1–p99 band. Band edges are `showlegend:false`; the median carries the
  // legend entry, and a shared `legendgroup` makes a legend-click toggle the
  // whole group (median + both bands). `fill:'tonexty'` fills to the
  // immediately-preceding trace, so the per-group order is
  // [p1, p99, p25, p75, median].
  const evalBandGroup = (
    setKey: string, mvmt: string, dash: 'solid' | 'dash',
    metric: 'nmae' | 'nemd',
  ) => {
    const pts = evalSeries?.sets[setKey] ?? []
    const xs: (string | number)[] = []
    const steps: number[] = []
    const pct: Record<string, (number | null)[]> = {
      p1: [], p25: [], median: [], p75: [], p99: [],
    }
    for (const pt of pts) {
      const x = xOfStep(pt.step)
      if (x === null) continue
      const rec = pt as unknown as Record<string, number | null>
      xs.push(x); steps.push(pt.step)
      for (const k of ['p1', 'p25', 'median', 'p75', 'p99']) {
        const v = rec[`${metric}_${k}`]
        pct[k].push(typeof v === 'number' ? v * 100 : null)  // fraction → %
      }
    }
    const color = COLORS[metric]
    const name = `${mvmt} ${metric.toUpperCase()}`
    const lg = `${mvmt}-${metric}`
    const edge = (key: string, fill: number | null) => ({
      x: xs, y: pct[key], name,
      type: 'scatter' as const, mode: 'lines' as const,
      line: { width: 0, color },
      ...(fill !== null
        ? { fill: 'tonexty' as const, fillcolor: bandFill(metric, fill) }
        : {}),
      yaxis: 'y2', legendgroup: lg, showlegend: false,
      hoverinfo: 'skip' as const,
    })
    // Smooth the median (the eval points themselves still carry the IQR /
    // p1–p99 spread bands across the 200 mats, so the user's `bands=1` ±σ
    // toggle is NOT applied here — adding rolling σ on top of inter-mat IQR
    // would conflate two different sources of spread and confuse the reader).
    const medianSmoothed = smoothedMeanStd(pct.median, smooth).mean
    return [
      edge('p1', null),     // lower edge of the p1–p99 band
      edge('p99', 0.09),    // upper edge → fills down to p1
      edge('p25', null),    // lower edge of the IQR band
      edge('p75', 0.20),    // upper edge → fills down to p25
      {                     // median — the legend-bearing center line
        x: xs, y: medianSmoothed, name,
        type: 'scatter' as const, mode: 'lines+markers' as const,
        line: { color, width: 1.6, dash },
        marker: { color, size: 3 },
        yaxis: 'y2', legendgroup: lg,
        customdata: steps,
        hovertemplate: `${name} median %{y:.2f}%%<br>step %{customdata}<extra></extra>`,
      },
    ]
  }
  const evalTraces = useMemo(() => {
    const out: Record<string, unknown>[] = []
    for (const [setKey, mvmt, dash] of [
      ['val_200', 'MV', 'solid'], ['train_200', 'MT', 'dash'],
    ] as [string, string, 'solid' | 'dash'][]) {
      for (const metric of ['nmae', 'nemd'] as const) {
        out.push(...evalBandGroup(setKey, mvmt, dash, metric))
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalSeries, xMode, smooth])

  // Lifecycle event timestamps.
  const eventTimes = (key: string): number[] => {
    const ts: number[] = []
    const col = cols.get(key) ?? []
    for (let i = 0; i < col.length; i++) {
      if (col[i] === 1 && timestamps[i] !== null) ts.push(timestamps[i] as number)
    }
    return ts.sort((a, b) => a - b)
  }
  const startTs = eventTimes('lifecycle/trainer_started')
  const sigtermTs = eventTimes('lifecycle/sigterm_received')

  const preemptCol = cols.get('cluster/preemptions') ?? []
  const preemptTs: number[] = []
  let prev: number | null = null
  for (const { ts, i } of ordered) {
    const v = preemptCol[i]
    if (v === null) continue
    if (prev !== null && v > prev) preemptTs.push(ts as number)
    prev = v
  }

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'

  // Event vlines spanning both panels via paper-coord shapes.
  //
  // Shapes also fade on legend hover: if the user hovers a real trace (TL /
  // VL / etc.), all event vlines dim to alpha 0.18 so the trace stands out.
  // If the user hovers an event LI ("trainer_started", "sigterm", "cluster
  // preempt", "death: preempt/cascade/failed"), shapes matching that event's
  // color stay full opacity; the rest dim. With no hover, all shapes render
  // at full opacity.
  type Shape = {
    type: 'line'
    xref: 'x'
    yref: 'paper'
    x0: string | number
    x1: string | number
    y0: number
    y1: number
    line: { color: string; width: number; dash: 'dash' | 'dot' | 'solid' }
  }
  // Map active-trace legend-item name → base color for the shapes it
  // represents. Anything not in this map (TL, VL, NMAE, NEMD, step, etc.)
  // means "user hovered a non-event trace" → fade ALL shapes.
  function activeShapeColor(activeName: string | null): string | null {
    if (activeName === null) return null  // no hover → full opacity for all
    if (activeName.startsWith('trainer_started')) return COLORS.start
    if (activeName.startsWith('sigterm')) return COLORS.sigterm
    if (activeName.startsWith('cluster preempt')) return COLORS.preempt
    if (activeName.startsWith('death: preempt')) return DEATH_COLORS.preempt
    if (activeName.startsWith('death: cascade')) return DEATH_COLORS.cascade
    if (activeName.startsWith('death: failed')) return DEATH_COLORS.failed
    return ''  // hovered a non-event trace → fade everything
  }
  const activeColor = activeShapeColor(activeTraceName)
  function shapeColor(baseColor: string): string {
    if (activeColor === null) return baseColor
    if (activeColor === baseColor) return baseColor
    // Convert hex → rgba(…, 0.18) so non-matching shapes dim out.
    const rgb = hexToRgbTuple(baseColor)
    return `rgba(${rgb}, 0.18)`
  }
  const eventShapes: Shape[] = []
  const addShapes = (ts: number[], color: string, dash: 'dash' | 'dot' | 'solid') => {
    for (const t of ts) {
      const x = xOfTs(t)
      if (typeof x === 'number' && Number.isNaN(x)) continue
      eventShapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color: shapeColor(color), width: dash === 'solid' ? 1 : 1.2, dash },
      })
    }
  }
  addShapes(startTs, COLORS.start, 'dash')
  addShapes(sigtermTs, COLORS.sigterm, 'dot')
  addShapes(preemptTs, COLORS.preempt, 'solid')

  // Death-cause vlines, sourced from the iris-attempts sidecar. These layer on
  // top of (do NOT replace) the trainer_started / sigterm / cluster_preempt
  // overlays above: those come from wandb-logged lifecycle counters, these
  // come from iris's per-task per-attempt bug-report data and surface the
  // CAUSE of each attempt's death (cascade vs preempt vs other failure).
  type DeathBucket = { ts: number; cause: DeathCause; error: string; taskId: string }
  const deathBuckets: Record<DeathCause, DeathBucket[]> = {
    preempt: [], cascade: [], failed: [], completed: [],
  }
  if (attempts) {
    for (const t of attempts.tasks) {
      for (const a of t.attempts) {
        if (a.finished_at_ms == null) continue
        const cause = classifyDeath(a)
        if (cause === 'completed') continue
        deathBuckets[cause].push({
          ts: a.finished_at_ms / 1000, cause, error: a.error ?? '', taskId: t.task_id,
        })
      }
    }
  }
  for (const cause of ['preempt', 'cascade', 'failed'] as const) {
    addShapes(deathBuckets[cause].map((b) => b.ts), DEATH_COLORS[cause], 'solid')
  }

  // Legend-only invisible point so the event vlines show up in the legend
  // (real lines are `shapes`, which don't legend-ify).
  const legendOnly = (name: string, color: string, dash: 'dash' | 'dot' | 'solid') => ({
    x: [null], y: [null],
    mode: 'lines' as const,
    type: 'scatter' as const,
    line: { color, width: 1.5, dash },
    name,
    showlegend: true,
    hoverinfo: 'skip' as const,
  })

  // The top-panel (running-max global_step) is degenerate in both `step` and
  // `epoch` modes (epoch is a pure rescaling of step → also y = x there).
  const showTopPanel = xMode !== 'step' && xMode !== 'epoch'

  const logType: 'log' = 'log'
  const lossDomain: [number, number] = showTopPanel ? [0.0, 0.66] : [0.0, 1.0]
  const stepDomain: [number, number] = [0.72, 1.0]

  const xTitle = xMode === 'time' ? TZ_LABEL
    : xMode === 'elapsed' ? 'elapsed (h)'
    : xMode === 'epoch' ? 'epoch (fractional)'
    : 'global_step'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#888', alignSelf: 'center' }}>x-axis:</span>
        {(['time', 'elapsed', 'step', 'epoch'] as XMode[])
          // Hide `epoch` when the manifest can't compute it (data label not
          // in EPOCH_SEQUENCES, or `train_batch_size` missing) — a useless
          // button would just confuse.
          .filter((m) => m !== 'epoch' || epochAvailable)
          .map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setXMode(m)}
            style={{
              fontSize: '0.75rem',
              padding: '0.15rem 0.5rem',
              borderRadius: 4,
              border: `1px solid ${xMode === m ? '#4a8aff' : (isDark ? '#444' : '#ccc')}`,
              background: xMode === m ? 'rgba(74,138,255,0.15)' : 'transparent',
              color: 'inherit',
              cursor: 'pointer',
            }}
          >
            {m === 'time' ? 'wallclock' : m}
          </button>
        ))}
        <SmoothingChips
          mode={smooth} setMode={setSmooth}
          bandsOn={bandsOn} setBandsOn={setBandsOn}
          isDark={isDark}
          fg={isDark ? '#bbb' : '#444'}
          muted={isDark ? '#888' : '#666'}
        />
      </div>
      <div ref={plotWrapperRef}>
      <Plot
        onActiveTraceChange={setActiveTraceName}
        onRelayout={onRelayout as (ev: unknown) => void}
        data={[
          // 1. step (top panel) — only when not in step mode
          ...(showTopPanel ? [{
            x: stepTrace.xs, y: stepTrace.ys, name: 'step',
            type: 'scatter' as const, mode: 'lines' as const,
            line: { color: COLORS.step, width: 2, shape: 'hv' as const },
            yaxis: 'y',
            hovertemplate: 'step %{y}<extra></extra>',
          }] : []),
          // 2. losses (shared log y2) — smoothing-aware. Bands (when on) share
          //    each line's `legendgroup` so `applyBandFade` desats them with
          //    their parent on hover.
          ...smoothedSeriesTraces(TL, 'TL (train/loss)', COLORS.TL, 1.2, 'TL'),
          ...smoothedSeriesTraces(VL, 'VL (eval/loss)', COLORS.VL, 1.4, 'VL'),
          // 3. mat-NMAE + mat-NEMD (from eval.json)
          ...evalTraces,
          // Hide lifecycle LIs whose count is 0 — for a Modal run with no
          // iris-style restarts / preempts / sigterms, three permanently
          // greyed (0) rows just clutter the legend.
          ...(startTs.length > 0 ? [legendOnly(`trainer_started (${startTs.length})`, COLORS.start, 'dash')] : []),
          ...(sigtermTs.length > 0 ? [legendOnly(`sigterm (${sigtermTs.length})`, COLORS.sigterm, 'dot')] : []),
          ...(preemptTs.length > 0 ? [legendOnly(`cluster preempt (${preemptTs.length})`, COLORS.preempt, 'solid')] : []),
          // Death-cause legend entries — one row per non-empty bucket. Hidden
          // when the sidecar hasn't loaded yet (`attempts == null`) so the
          // legend doesn't grow until the data's in.
          ...(attempts ? (['preempt', 'cascade', 'failed'] as const)
            .filter((c) => deathBuckets[c].length > 0)
            .map((c) => legendOnly(
              `death: ${c} (${deathBuckets[c].length})`,
              DEATH_COLORS[c], 'solid',
            )) : []),
        ]}
        layout={{
          title: {
            text: (() => {
              const segChunk = numSegments > 1 ? ` (${numSegments} seg)` : ''
              const head = `${runId}  ·  ${startTs.length} starts${segChunk}, ${sigtermTs.length} sigterms, ${preemptTs.length} preempts`
              if (!attempts) return head
              const parts = (['preempt', 'cascade', 'failed'] as const)
                .filter((c) => deathBuckets[c].length > 0)
                .map((c) => `${deathBuckets[c].length} ${c}`)
              return parts.length > 0 ? `${head}  ·  ${parts.join(', ')}` : head
            })(),
            font: { size: 14 },
          },
          autosize: true,
          height: 640,
          xaxis: {
            title: { text: xTitle },
            type: xMode === 'time' ? 'date' : 'linear',
            ...(xMode === 'time' ? { tickformat: '%-m/%-d %H:%M' } : {}),
            gridcolor, zerolinecolor, linecolor: gridcolor,
            // Restore user-picked range across re-renders triggered by
            // smoothing / bands toggles. `null` → omit → plotly autoranges.
            ...(userXRange ? { range: userXRange, autorange: false } : {}),
          },
          yaxis: {
            title: { text: 'step', font: { color: COLORS.step } },
            tickfont: { color: COLORS.step },
            domain: stepDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
            visible: showTopPanel,
            // Lock y-zoom: click-drag should zoom x-only (constraint imposed
            // by `fixedrange: true` on every y-axis; plotly's box-zoom then
            // spans the full y range automatically).
            fixedrange: true,
          },
          yaxis2: {
            title: { text: 'loss / NMAE·NEMD % (log)' },
            type: logType,
            domain: lossDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
            fixedrange: true,
          },
          shapes: eventShapes,
          margin: { t: 50, l: 70, r: 170, b: 50 },
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          legend: { x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)' },
        }}
      />
      </div>
    </div>
  )
}
