// Per-run plot — up to 3 stacked subplots over a shared x-axis. x-axis modes:
//   • wallclock — local time (the viewer's zone)
//   • elapsed   — hours since the run's first log
//   • step      — Levanter `global_step`
// Panels (top → bottom):
//   1. step (running max of global_step) — ~17%. Hidden in step/epoch modes
//      (it would degenerate to y = x).
//   2. TL + VL on a log y-axis — ~45% (the "tall, watch the loss decrease"
//      panel).
//   3. MT/MV (mat-NMAE + mat-NEMD on train_200 / val_200) on a separate log
//      y-axis — ~35%. Hidden when the run has no eval.json yet (the plot
//      falls back to the legacy 2-panel layout).
//
// Eval points (MV = val_200, MT = train_200) come from the canonical per-step
// eval.json — NOT the parquet's collapsed harvested points. Per (set × metric)
// the plot draws the per-step median as a connected line+markers trace; the
// p25–p75 / p1–p99 spread bands were dropped (4-timepoint horizontal smears
// were confusing rather than informative — see spec 25 follow-up). NMAE is
// green, NEMD teal; MV solid, MT dashed. Eval points are keyed by checkpoint
// step; on the time/elapsed axes they're placed at the wallclock of that
// step, recovered from the parquet's (timestamp, global_step) rows.
// Non-teacher-mode points (maskgit, free) come from `val/train_200-<mode>`
// set keys and render with the same visual style (legend disambiguates via
// `MV/maskgit`).
//
// Lifecycle events render as vertical lines via `shapes` (yref='paper') so
// they span all panels.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { enumParam, useUrlState } from 'use-prms'
import { themedHoverlabel } from '../theme'
import { Tooltip } from '../Tooltip'
import type { RunHistory, RunHistoryRow } from './parquet'
import type { IrisAttempts, RunEval, RunManifest } from './api'
import { classifyDeath, DEATH_COLORS, type DeathCause } from './deathEvents'
import { SmoothingChips, useBandsToggle, useSmoothMode } from './RunsTimelinePlot'
import { epochOfStep } from './runMeta'
import { annotationsFor, type RunAnnotation } from './annotations'
import { computeSmoothedSeries } from './smoothing'
import { FlopUnitChips, formatFlops, useFlopUnit } from './flops'
import { ancestorRelation } from './lineage'

/** Ancestor metadata for a lineage-glued history. Each entry corresponds to
 *  one part of the concatenated `history` (root → parent order, the order
 *  `concatHistories` was called with — minus the current run, which is the
 *  tail). `rowCount` is the number of rows that ancestor contributed; the
 *  plot uses these to map rows back to their source run, override the trace
 *  color to `color`, and tag the segment in tooltips with `name`. */
export interface LineageAncestor {
  name: string
  rowCount: number
  color: string
}

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
  /** Ancestor lineage (root → parent). When non-null, the plot splits each
   *  metric's trace into per-ancestor + current-run sub-series so each
   *  ancestor renders in its own color (full opacity, no recency-ramp).
   *  The current run keeps the existing restart-segment opacity-ramp. */
  lineageInfo?: LineageAncestor[] | null
}

type XMode = 'time' | 'elapsed' | 'step' | 'epoch' | 'flop'

// URL-facing x-axis mode names (`?x=wallclock|elapsed|step|epoch|flop`). The
// internal XMode uses `'time'` for the wallclock axis; `wallclock` reads
// better in shared links.
type UrlXMode = 'wallclock' | 'elapsed' | 'step' | 'epoch' | 'flop'
const URL_X_MODES = ['wallclock', 'elapsed', 'step', 'epoch', 'flop'] as const
const X_TO_URL: Record<XMode, UrlXMode> = {
  time: 'wallclock', elapsed: 'elapsed', step: 'step', epoch: 'epoch', flop: 'flop',
}
const URL_TO_X: Record<UrlXMode, XMode> = {
  wallclock: 'time', elapsed: 'elapsed', step: 'step', epoch: 'epoch', flop: 'flop',
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

export function WallclockPlot({ history, evalSeries, runId, defaultXMode = 'step', attempts, manifest = null, lineageInfo = null }: Props) {
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
  // Fade bands by trace `name` whenever the active trace changes. Each band
  // edge trace shares its parent line's `name` (e.g. `'TL (train/loss)'`), so
  // matching by name brushes the band with its parent. We previously matched
  // by `legendgroup`, but TL and VL now share `legendgroup: 'losses'` so
  // their legend items sit flush (no flicker-inducing duplicate group-title
  // row + gap between them), and group-based matching can no longer
  // distinguish them.
  //
  // Walks the Plotly trace list to find all `showlegend: false` band-edge
  // traces. `Plotly.restyle(plotDiv, { opacity: vals }, bandIndices)` flips
  // each band to 1 if its name matches the active trace's name, 0.3 otherwise.
  // When activeTraceName is null, all bands → 1.
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
      const want = activeTraceName == null || t.name === activeTraceName ? 1 : 0.3
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
  // Whether the `flop` axis is available — requires `throughput/total_gflops`
  // in the parquet (added 2026-06-07; older runs don't have it). Hide the
  // button + fall back to `step` from a `?x=flop` URL when missing. Cheap to
  // probe: the parquet reader skips absent columns, so the Map will not have
  // the key at all (vs the `epoch` test which evaluates a formula).
  const flopColAvailable = (history.cols.get('throughput/total_gflops')?.length ?? 0) > 0
  const rawXMode: XMode = URL_TO_X[urlXMode]
  const xMode: XMode = (
    rawXMode === 'epoch' && !epochAvailable ? 'step'
    : rawXMode === 'flop' && !flopColAvailable ? 'step'
    : rawXMode
  )
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

  // (ts, flops) pairs from every row carrying `throughput/total_gflops`.
  // The wandb column logs cumulative GFLOPs; we store raw FLOPs (×1e9) so
  // `formatFlops(x, unit)` lands a value in the unit the user chose. Sorted by
  // ts (matches `ordered`). Drives `flopAtTs` for the `flop` x-axis mode.
  // Empty for parquets logged before the column was added to the schema —
  // the `flop` button is hidden in that case.
  const tsFlop = useMemo(() => {
    const totalGflops = cols.get('throughput/total_gflops') ?? []
    const pairs: { ts: number; flop: number }[] = []
    for (const { ts, i } of ordered) {
      const g = totalGflops[i]
      if (g === null || g === undefined) continue
      pairs.push({ ts: ts as number, flop: (g as number) * 1e9 })
    }
    return pairs
  }, [ordered, cols])
  const flopAvailable = tsFlop.length > 0

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

  /** Cumulative FLOPs at the latest logged row at or before `ts`. Returns NaN
   *  if the parquet doesn't carry the column (older runs); callers should fall
   *  back gracefully (the index plot just skips that point; per-run plot
   *  renders NaN gaps). */
  function flopAtTs(ts: number): number {
    if (tsFlop.length === 0) return NaN
    let lo = 0, hi = tsFlop.length - 1, best = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (tsFlop[mid].ts <= ts) { best = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    return best < 0 ? NaN : tsFlop[best].flop
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
    if (xMode === 'flop') return flopAtTs(ts)
    if (xMode === 'elapsed') return (ts - t0) / 3600
    return toLocal(ts)
  }

  /** x-coordinate for an eval point at checkpoint `step`. */
  function xOfStep(step: number): string | number | null {
    if (xMode === 'step') return step
    if (xMode === 'epoch') return epochOfStep(step, manifest)
    if (xMode === 'flop') {
      const ts = tsAtGstep(step)
      if (ts === null) return null
      const f = flopAtTs(ts)
      return Number.isFinite(f) ? f : null
    }
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
  // When the history is lineage-glued (`history.partRowCounts` from
  // `concatHistories`), we ALSO split at every ancestor boundary so each
  // ancestor's portion can render in its own color (full opacity, no
  // recency-ramp). Each segment tracks which lineage part it belongs to
  // (`partIdx`): 0..ancestors.length-1 → that ancestor, ancestors.length →
  // current run.
  //
  // With one start and no lineage, this returns `[{ start: 0, partIdx: 0 }]`
  // and the whole run renders as a single trace as before.
  type SegMeta = { start: number; partIdx: number }
  const segments = useMemo<SegMeta[]>(() => {
    // partOfRaw: raw-row-index → which concat-part the row came from. The
    // last part (index `partRowCounts.length - 1`) is the current run; the
    // rest are ancestors in root→parent order.
    const partRowCounts = history.partRowCounts
    const numParts = partRowCounts?.length ?? 1
    const currentPartIdx = numParts - 1
    const partOfRaw = (rawIdx: number): number => {
      if (!partRowCounts || numParts === 1) return 0
      let acc = 0
      for (let p = 0; p < partRowCounts.length; p++) {
        acc += partRowCounts[p]
        if (rawIdx < acc) return p
      }
      return numParts - 1
    }
    // Trainer_started timestamps from the CURRENT run's portion only —
    // ancestor parts get one segment each (so they each get one solid color,
    // no intra-ancestor restart-fade noise on top of the cross-part color
    // story).
    const startedCol = cols.get('lifecycle/trainer_started') ?? []
    const startTsCurrent: number[] = []
    for (let i = 0; i < startedCol.length; i++) {
      if (startedCol[i] === 1 && timestamps[i] != null) {
        if (partOfRaw(i) === currentPartIdx) {
          startTsCurrent.push(timestamps[i] as number)
        }
      }
    }
    startTsCurrent.sort((a, b) => a - b)
    if (ordered.length === 0) return [{ start: 0, partIdx: currentPartIdx }]
    // Walk `ordered` in ts order; insert seams at part-changes (for lineage
    // glue) and at the first `ordered` index whose ts is ≥ each
    // trainer_started ts within the current run's portion.
    const out: SegMeta[] = []
    let nextStartTsIdx = 0
    let lastPartIdx = -1
    for (let k = 0; k < ordered.length; k++) {
      const { ts, i } = ordered[k]
      const p = partOfRaw(i)
      // Advance the trainer_started pointer past any starts whose ts we've
      // now caught up to. Each such advance marks "this k is the first row
      // at-or-after the start". We push a seam for each pending start (after
      // the first one — the very first start aligns with the first current-
      // run row, which is already a seam from the part transition or from
      // k===0).
      let restartHere = false
      while (nextStartTsIdx < startTsCurrent.length
             && startTsCurrent[nextStartTsIdx] <= (ts as number)) {
        // Skip the very first trainer_started — it coincides with the
        // current run's start (already a seam via partChanged / k===0). Only
        // 2nd+ starts indicate intra-run restarts.
        if (nextStartTsIdx > 0) restartHere = true
        nextStartTsIdx++
      }
      const partChanged = p !== lastPartIdx
      if (k === 0 || partChanged || restartHere) {
        if (out.length === 0 || out[out.length - 1].start !== k) {
          out.push({ start: k, partIdx: p })
        }
      }
      lastPartIdx = p
    }
    return out
  }, [ordered, cols, timestamps, history.partRowCounts])
  const numSegments = segments.length
  const segmentStarts = useMemo(() => segments.map((s) => s.start), [segments])

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
  // partIdx → ancestor name, or null when partIdx is the current-run tail.
  // `lineageInfo` is root→parent; its length is `partRowCounts.length - 1`,
  // so any partIdx === lineageInfo.length is the current run.
  function ancestorOf(partIdx: number): LineageAncestor | null {
    if (!lineageInfo) return null
    return lineageInfo[partIdx] ?? null
  }

  // Returns one Series per restart segment. Empty segments are preserved at
  // their index so opacity / naming line up with `numSegments`.
  function series(key: keyof RunHistoryRow): Series[] {
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
  // machinery (above) brushes the bands with their parent line. `rolling`
  // mode also returns a real within-window σ (Welford-equivalent); `ema` has
  // no natural σ companion so its ±σ chip is disabled.
  //
  // Window is SAMPLE-INDEX (`window` samples wide, ±window/2 around each i),
  // NOT current-x-axis units. We tried x-axis units (`pltly.rolling`'s
  // `getX/windowSize` API) to make `rolling:50` keep its meaning for sparse
  // MT/MV — but that made `?x=wallclock&smooth=rolling:N` interpret N as
  // ±N/2 ms, capturing only the point itself for typical 5-10 s training
  // cadences, so rolling silently passed through. The fix is to keep
  // N → sample-index everywhere (matches `RunsTimelinePlot.tsx`) and bypass
  // smoothing entirely for sparse traces (see `evalMedianTrace`). Smoothing
  // is computed once over the full series; plotly's `xaxis.range` then clips
  // on display, so box-zoom is a pure display concern. See
  // `WallclockPlot.test.ts` for the contract.
  const [smooth, setSmooth] = useSmoothMode()
  const [bandsOn, setBandsOn] = useBandsToggle()
  // FLOP-unit display preference (`?fopu=`). Drives the axis title + tooltip
  // formatting whenever the user is in `?x=flop` mode.
  const [flopUnit, setFlopUnit] = useFlopUnit()

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
  // For runs with restart segments (`seriesPerSeg.length > 1`), older
  // current-run segments render at low opacity so the latest segment
  // dominates visually. Only the latest current-run segment gets a legend
  // entry; older current-run segments share its `legendgroup` so the
  // legend toggle hides/shows the whole stack.
  //
  // Ancestor (lineage-glued) segments instead render in their ancestor's
  // OWN color (looked up via `ancestorOf(segments[segIdx].partIdx)`) at full
  // opacity, in a per-ancestor legendgroup so each ancestor has its own
  // legend row + a hover-fade story independent from the current run's
  // metric stack.
  type SmoothedTrace = Record<string, unknown>
  function smoothedSeriesTraces(
    seriesPerSeg: Series[], name: string, color: string, lineWidth: number, lg: string,
  ): SmoothedTrace[] {
    const N = seriesPerSeg.length
    // Index of the most-recent CURRENT-RUN segment with data — gets the
    // legend entry + full opacity. Ancestor segments do NOT use this anchor
    // (each ancestor stands on its own). Without it, a fresh resume that
    // hasn't logged any rows yet would render no legend entry and no
    // full-opacity trace — the entire plot would be older segments at fade.
    let lastNonEmptyCurrent = -1
    // Total count of current-run segments (drives the in-run `#k/N` label).
    // Includes empty ones for index stability, matching pre-lineage behavior
    // for non-glued plots.
    let firstCurrentSeg = -1
    for (let i = 0; i < N; i++) {
      const sm = segments[i]
      const isCurrent = !ancestorOf(sm.partIdx)
      if (isCurrent && firstCurrentSeg === -1) firstCurrentSeg = i
      if (isCurrent && seriesPerSeg[i].xs.length > 0) lastNonEmptyCurrent = i
    }
    const numCurrentSegs = firstCurrentSeg === -1 ? 0 : N - firstCurrentSeg
    const out: SmoothedTrace[] = []
    seriesPerSeg.forEach((s, segIdx) => {
      if (s.xs.length === 0) return
      const segMeta = segments[segIdx]
      const ancestor = ancestorOf(segMeta.partIdx)
      const isCurrent = !ancestor
      const isLatest = isCurrent && segIdx === lastNonEmptyCurrent
      // Color override + opacity / legend strategy:
      //   - ancestor segments: full opacity, ancestor's color, per-ancestor
      //     legendgroup so each ancestor has its own legend row;
      //   - current-run segments: existing ramp anchored on lastNonEmptyCurrent.
      let traceColor: string
      let opacity: number
      let legendGroup: string
      let groupTitle: string
      let showLegend: boolean
      let segName: string
      // Full descriptive label used in the per-trace hovertemplate (the legend
      // text itself stays short — `name` or `${name} #k/N`). For ancestor
      // segments, this surfaces the full ancestor name on hover so the
      // discoverability cost of the short group title is paid back.
      let hoverLabel: string
      if (ancestor) {
        traceColor = ancestor.color
        opacity = 1
        // Each ancestor gets its own legend group so hover-fade and
        // legend-toggle isolate the ancestor's metric stack from the
        // current run's `losses` group. Group key is unique per ancestor.
        legendGroup = `lineage:${ancestor.name}`
        // Short relationship label (parent / grandparent / great-grandparent
        // / ancestor #N) — the previous full-name group title `ancestor:
        // <full-name>` dominated the plot width on multi-ancestor runs.
        // Full name lives on hover (see `hoverLabel`). `lineageInfo` is
        // non-null here because `ancestor` is.
        groupTitle = ancestorRelation(segMeta.partIdx, lineageInfo!.length)
        showLegend = true  // one row per ancestor per metric
        // Trace name matches the current-run's `name` (e.g. `TL (train/loss)`)
        // — the legend group title disambiguates which ancestor it is, and
        // color further distinguishes. Same-name traces still work for the
        // band-fade logic: ancestor segments don't draw ±σ bands (see
        // `isLatest` gate below), so the only `name`-matched band edges are
        // the current run's, which fade together with their parent line.
        segName = name
        hoverLabel = `${name} · ${groupTitle} (${ancestor.name})`
      } else {
        traceColor = color
        // Current-run restart segments are sequential in x (step, wallclock,
        // and epoch all monotonic across resumes), so they never visually
        // overlap — a recency fade adds no information, only readability cost.
        // The dashed gap-bridge between segments already shows where restarts
        // happened.
        opacity = 1
        const relIdx = segIdx - firstCurrentSeg
        legendGroup = lg
        groupTitle = 'losses (log)'
        showLegend = isLatest
        // Label per-segment as `<name> #k/N` (1-based, matches the
        // restart-segment header count). The hovertemplate uses `segName`
        // too so the x-unified tooltip shows distinct lines per segment
        // instead of three identical "TL (train/loss)" rows.
        segName = numCurrentSegs > 1 && !isLatest
          ? `${name} #${relIdx + 1}/${numCurrentSegs}` : name
        hoverLabel = segName
      }
      // Bypass smoothing for sparse traces — VL on a long run carries ~5-20
      // points across 10k+ steps, and sample-index rolling with window >
      // len(ys) collapses every output to the global mean → flat line. Same
      // rationale as the MT/MV exemption at `evalMedianTrace` below; VL ends
      // up here because it's logically grouped with TL into one
      // `smoothedSeriesTraces` call. Threshold 30 is well above typical VL
      // counts and well below TL's 1000s, so TL still smooths normally.
      const SPARSE_THRESHOLD = 30
      const nonNullYs = s.ys.reduce((acc, y) => acc + (y == null ? 0 : 1), 0)
      const isSparse = nonNullYs < SPARSE_THRESHOLD
      const { mean: ySmoothed, std: yStd } = isSparse
        ? { mean: s.ys, std: null as (number | null)[] | null }
        : computeSmoothedSeries(s.ys, smooth)
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
        line: { color: traceColor, width: lineWidth },
        opacity,
        yaxis: 'y2',
        legendgroup: legendGroup,
        legendgrouptitle: { text: groupTitle },
        showlegend: showLegend,
        customdata: lineG,
        hovertemplate: `${hoverLabel} %{y:.3f}<br>gstep %{customdata}<extra></extra>`,
      }
      // Dotted bridge connecting each gap's endpoints. Same color/opacity/
      // legendgroup as the main line so a legend-toggle hides them with the
      // rest of the segment; `showlegend: false` so it doesn't add its own
      // LI. `hoverinfo: 'none'` (NOT 'skip') keeps `applyBandFade` from
      // sweeping the bridge into its `showlegend:false + hoverinfo:'skip'`
      // bucket (which force-sets opacity to 1 or 0.3 on every band edge);
      // 'none' suppresses the bridge's tooltip without joining that bucket.
      const bridgeTrace: SmoothedTrace | null = bridgeX.length > 0 ? {
        x: bridgeX, y: bridgeY, name: segName,
        type: 'scatter', mode: 'lines',
        line: { color: traceColor, width: lineWidth, dash: 'dot' },
        opacity,
        yaxis: 'y2',
        legendgroup: legendGroup,
        showlegend: false,
        hoverinfo: 'none',
      } : null
      // Only show ±σ bands on the latest current-run segment — drawing 28
      // overlapping bands would just be noise, and the older trajectories
      // already smear visually via opacity. Ancestor segments are skipped
      // here too: their context is "look at the previous color smoothly
      // joining ours", not "what was that ancestor's intra-window σ".
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
        yaxis: 'y2', legendgroup: legendGroup,
        showlegend: false,
        hoverinfo: 'skip',
        ...(fillKey ? { fill: 'tonexty', fillcolor: fillKey } : {}),
      })
      const rgb = hexToRgbTuple(traceColor)
      const fillcolor = `rgba(${rgb}, 0.18)`
      out.push(edge(yLower, null), edge(yUpper, fillcolor), lineTrace)
      if (bridgeTrace) out.push(bridgeTrace)
    })
    return out
  }

  // ── eval points from eval.json ──
  // Per (set × metric): per-step median rendered as a `lines+markers` trace
  // on the dedicated MT/MV panel (yaxis: 'y3'). Markers are size 6 so the
  // typically-handful (~4) of timepoints are clearly visible (was: median +
  // p25–p75 + p1–p99 bands, dropped — see header comment).
  const evalMedianTrace = (
    setKey: string, mvmt: string, dash: 'solid' | 'dash',
    metric: 'nmae' | 'nemd',
  ) => {
    const pts = evalSeries?.sets[setKey] ?? []
    const xs: (string | number)[] = []
    const ys: (number | null)[] = []
    const customdata: [number, number | string][] = []
    for (const pt of pts) {
      const x = xOfStep(pt.step)
      if (x === null) continue
      const rec = pt as unknown as Record<string, number | null>
      const v = rec[`${metric}_median`]
      const nMats = (pt as unknown as { n_mats?: number }).n_mats
      xs.push(x)
      customdata.push([pt.step, nMats ?? '?'])
      ys.push(typeof v === 'number' ? v * 100 : null)  // fraction → %
    }
    const color = COLORS[metric]
    const name = `${mvmt} ${metric.toUpperCase()}`
    // Bypass smoothing for MT/MV — these are sparse eval-point traces (~5-30
    // timepoints per run, 1-10k steps between them). Applying a sample-index
    // rolling window of e.g. N=50 to a 5-point series collapses each output
    // to the mean of all points → flat line; EMA across so few samples also
    // doesn't carry useful information. The earlier x-units workaround that
    // made `rolling:N` mean "N steps wide" silently broke wallclock-mode
    // rolling for TL/VL, so we went back to sample-index everywhere and
    // exempt sparse traces here instead.
    return {
      x: xs, y: ys, name,
      type: 'scatter' as const, mode: 'lines+markers' as const,
      line: { color, width: 1.6, dash },
      marker: { color, size: 6 },
      yaxis: 'y3',
      // Single shared legendgroup so the "MT/MV …" group title only renders
      // once; per-trace click-to-toggle still works (plotly's default click
      // toggles the individual trace, not the whole legendgroup).
      legendgroup: 'mtmv',
      legendgrouptitle: { text: 'MT/MV (mat-NMAE / mat-NEMD %)' },
      customdata,
      hovertemplate: `${name} %{y:.2f}%%<br>step %{customdata[0]} · n_mats %{customdata[1]}<extra></extra>`,
    }
  }
  const evalTraces = useMemo(() => {
    const out: Record<string, unknown>[] = []
    // val/train_200 are the teacher-mode bare keys (cont33k-era eval.json);
    // val/train_200-<mode> are non-teacher modes (maskgit, free) — same shape,
    // same line style for now (spec 25 follow-up: color-code per mode).
    const setKeys = Object.keys(evalSeries?.sets ?? {})
    const labelFor = (setKey: string): [string, 'solid' | 'dash'] => {
      const [base, mode] = setKey.split('-', 2) as [string, string | undefined]
      const mvmt = base === 'val_200' ? 'MV' : 'MT'
      const dash = base === 'val_200' ? 'solid' : 'dash'
      return [mode ? `${mvmt}/${mode}` : mvmt, dash]
    }
    for (const setKey of setKeys) {
      const [mvmt, dash] = labelFor(setKey)
      for (const metric of ['nmae', 'nemd'] as const) {
        out.push(evalMedianTrace(setKey, mvmt, dash, metric))
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalSeries, xMode])
  // Whether to render the third (MT/MV) panel: true iff the run has at least
  // one eval set with at least one point. Runs without eval.json (or with an
  // empty sets dict) fall back to the legacy 2-panel layout.
  const hasEvalData = useMemo(() => {
    const sets = evalSeries?.sets
    if (!sets) return false
    for (const k of Object.keys(sets)) {
      if ((sets[k]?.length ?? 0) > 0) return true
    }
    return false
  }, [evalSeries])

  // Lifecycle event timestamps.
  const eventTimes = (key: keyof RunHistoryRow): number[] => {
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

  // Base color for hand-curated annotation vlines + the icon-stroke for the
  // overlay icons. Declared here (not inline at the annotation block below)
  // because `activeShapeColor` needs to reference it.
  const ANNOTATION_COLOR = isDark ? 'rgba(190,190,210,0.6)' : 'rgba(80,80,100,0.55)'

  // Event vlines spanning both panels via paper-coord shapes.
  //
  // Shapes also fade on legend hover: if the user hovers a real trace (TL /
  // VL / etc.), all event vlines dim to alpha 0.18 so the trace stands out.
  // If the user hovers an event LI ("trainer_started", "sigterm", "cluster
  // preempt", "death: preempt/cascade/failed", "annotations"), shapes
  // matching that event's color stay full opacity; the rest dim. With no
  // hover, all shapes render at full opacity.
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
    if (activeName.startsWith('annotations')) return ANNOTATION_COLOR
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

  // Hand-curated per-run annotations (LR / BS / data changes etc) from
  // `annotations.ts`. Rendered as dashed vlines spanning all panels, with
  // small info-icons floating just above the top of the plot — hover an
  // icon to see the annotation's label + step in a floating tooltip.
  // Toggle on/off via the "annotations" legend item (events group). Inline
  // text labels were dropped because closely-spaced annotations collided.
  const annotations = annotationsFor(runId)
  // Per-annotation x in CURRENT-axis units (already resolved through
  // `xOfStep`), used both to draw the vline shape and to compute the
  // info-icon's pixel position in the HTML overlay below.
  type AnnotationPos = { ann: RunAnnotation; x: string | number }
  const annotationPositions: AnnotationPos[] = []
  for (const ann of annotations) {
    const x = xOfStep(ann.step)
    if (x === null || (typeof x === 'number' && Number.isNaN(x))) continue
    annotationPositions.push({ ann, x })
  }
  // Whether the "annotations" LI is toggled on. Drives both the dashed
  // vlines and the HTML info-icons rendered as an overlay.
  const [annotationsEnabled, setAnnotationsEnabled] = useState(true)
  if (annotationsEnabled) {
    for (const { ann, x } of annotationPositions) {
      eventShapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color: shapeColor(ann.color ?? ANNOTATION_COLOR), width: 0.8, dash: 'dash' },
      })
    }
  }

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
  // (real lines are `shapes`, which don't legend-ify). All event LIs share
  // the `events` legendgroup so they cluster together under the "events"
  // header below the metric groups.
  const legendOnly = (name: string, color: string, dash: 'dash' | 'dot' | 'solid') => ({
    x: [null], y: [null],
    mode: 'lines' as const,
    type: 'scatter' as const,
    line: { color, width: 1.5, dash },
    name,
    showlegend: true,
    legendgroup: 'events',
    legendgrouptitle: { text: 'events' },
    hoverinfo: 'skip' as const,
  })

  // Annotation-LI name (the only one we need to intercept legend-clicks for).
  // The trailing `(N)` mirrors the lifecycle LIs' count suffix and gives the
  // user a quick "how many" signal in the legend itself.
  const annotationsLegendName = `annotations (${annotationPositions.length})`
  // Toggle visibility of annotation vlines + icons on legend-click.
  //
  // We attach a DOM listener directly to `.js-plotly-plot` (rather than the
  // `onLegendClick` Plot prop) because pltly's internal pinned-legend / solo-
  // trace machinery consumes `plotly_legendclick` and never delegates to the
  // user prop when `onActiveTraceChange` is wired (which we DO need for the
  // shape-fade behavior). Our listener fires alongside pltly's; the pin
  // animation still runs on the LI, which is fine — clicking it again unpins
  // AND re-toggles our state, the obvious mental model.
  //
  // Same gotcha as the box-zoom `onRelayout` thread upstream: a useEffect at
  // mount time runs BEFORE pltly's async first render creates the
  // `.js-plotly-plot` element. The `attachedRef` + `onAfterPlot`-driven
  // recompute hook works around the race by attempting attachment every time
  // the plot redraws, and bailing once successful.
  const legendListenerRef = useRef<{ el: HTMLElement; fn: (ev: unknown) => void } | null>(null)
  const ensureLegendListener = useCallback(() => {
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as (HTMLElement & {
      on?: (evt: string, fn: (ev: unknown) => void) => void
      removeListener?: (evt: string, fn: (ev: unknown) => void) => void
    }) | null
    if (!plotDiv?.on) return
    if (legendListenerRef.current?.el === plotDiv) return
    // Detach any prior (stale element) listener before re-attaching.
    if (legendListenerRef.current) {
      const { el, fn } = legendListenerRef.current
      ;(el as unknown as { removeListener?: (evt: string, fn: (ev: unknown) => void) => void })
        .removeListener?.('plotly_legendclick', fn)
    }
    const fn = (ev: unknown) => {
      const e = ev as { curveNumber?: number; data?: Array<{ name?: string }> }
      const idx = e.curveNumber
      if (typeof idx === 'number' && e.data?.[idx]?.name === annotationsLegendName) {
        setAnnotationsEnabled((v) => !v)
      }
    }
    plotDiv.on('plotly_legendclick', fn)
    legendListenerRef.current = { el: plotDiv, fn }
  }, [annotationsLegendName])
  useEffect(() => {
    ensureLegendListener()
    return () => {
      if (legendListenerRef.current) {
        const { el, fn } = legendListenerRef.current
        ;(el as unknown as { removeListener?: (evt: string, fn: (ev: unknown) => void) => void })
          .removeListener?.('plotly_legendclick', fn)
        legendListenerRef.current = null
      }
    }
  }, [ensureLegendListener])

  // Pixel positions of the info-icons, recomputed on every Plotly redraw
  // (axis rescale, zoom, x-mode swap…). Each entry has the icon's CSS
  // (left, top) inside `plotWrapperRef` plus the annotation it represents.
  type IconPos = { ann: RunAnnotation; left: number; top: number }
  const [iconPositions, setIconPositions] = useState<IconPos[]>([])
  // Annotation positions captured via ref so `recomputeIconPositions`
  // (memoized) always reads the LATEST x-values — `annotationPositions`
  // recomputes on every `xMode` swap (xOfStep returns different things),
  // but the callback's dep list intentionally doesn't include it (otherwise
  // we'd recreate the callback on every render).
  const annotationPositionsRef = useRef(annotationPositions)
  annotationPositionsRef.current = annotationPositions
  const recomputeIconPositions = useCallback(() => {
    // Piggyback the post-render callback to (re)attach the legend-click
    // listener if it isn't already — fixes the mount-time race where the
    // `.js-plotly-plot` element doesn't exist when the listener-registration
    // useEffect first runs.
    ensureLegendListener()
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as (HTMLElement & {
      _fullLayout?: {
        xaxis?: { d2p?: (v: number) => number; _offset?: number; r2c?: (v: string | number) => number }
      }
    }) | null
    const xax = plotDiv?._fullLayout?.xaxis
    if (!plotDiv || !xax?.d2p) {
      setIconPositions((prev) => (prev.length === 0 ? prev : []))
      return
    }
    const wrapperRect = root.getBoundingClientRect()
    const plotRect = plotDiv.getBoundingClientRect()
    // Anchor icons in the top margin between the plot title and the data
    // area. `_fullLayout.margin.t` is the top-margin pixel height; the title
    // sits at the top of that margin, so we shift to land just above the
    // plot's drawing area (below the title row).
    const yAxObj = (plotDiv as unknown as {
      _fullLayout?: { margin?: { t?: number }; yaxis?: { _offset?: number } }
    })._fullLayout
    const yTopOffset = yAxObj?.yaxis?._offset ?? (yAxObj?.margin?.t ?? 50)
    const topPx = plotRect.top - wrapperRect.top + Math.max(0, yTopOffset - 16)
    const next: IconPos[] = []
    for (const { ann, x } of annotationPositionsRef.current) {
      // r2c handles both date strings (wallclock mode) and numbers; d2p then
      // maps the canonical x-coordinate to a pixel offset within the axis.
      const xc = typeof x === 'number' ? x : (xax.r2c ? xax.r2c(x) : NaN)
      if (!Number.isFinite(xc)) continue
      const xPx = xax.d2p(xc)
      if (!Number.isFinite(xPx)) continue
      const xOffset = xax._offset ?? 0
      // `plotRect` is the `.js-plotly-plot` element; axis `_offset` is
      // relative to that. Convert through wrapper-relative coords so the
      // overlay (which lives inside the wrapper, not the plot div) lands
      // on top of the right column.
      const left = (plotRect.left - wrapperRect.left) + xOffset + xPx
      next.push({ ann, left, top: topPx })
    }
    setIconPositions((prev) => {
      if (prev.length === next.length
          && prev.every((p, i) => p.left === next[i].left
                              && p.top === next[i].top
                              && p.ann.step === next[i].ann.step)) {
        return prev
      }
      return next
    })
  // annotationPositionsRef provides the latest x-values per render; deps
  // intentionally don't include it (it's a ref).
  }, [ensureLegendListener])
  // Recompute on window resize too (a wrapper width change moves icons
  // horizontally without firing `plotly_afterplot`).
  useEffect(() => {
    if (typeof window === 'undefined') return
    const onResize = () => recomputeIconPositions()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [recomputeIconPositions])
  // Re-run when the set of annotation x-values changes (xMode swap, lineage
  // glue re-resolving steps, etc). The ref captures the latest values; this
  // effect is what schedules the actual recompute *after* the new x-values
  // commit + Plotly.react finishes redrawing.
  const annotationXKey = annotationPositions
    .map((p) => `${p.ann.step}:${typeof p.x === 'number' ? p.x : p.x.toString()}`)
    .join('|')
  useEffect(() => {
    recomputeIconPositions()
  }, [annotationXKey, recomputeIconPositions])

  // The top-panel (running-max global_step) is degenerate in both `step` and
  // `epoch` modes (epoch is a pure rescaling of step → also y = x there).
  const showTopPanel = xMode !== 'step' && xMode !== 'epoch'
  const showEvalPanel = hasEvalData

  const logType: 'log' = 'log'
  // Domains: stack panels top → bottom with small inter-panel gaps. Each of
  // the 4 (showTopPanel × showEvalPanel) combinations gets its own layout so
  // each panel uses the full available space.
  const [stepDomain, lossDomain, evalDomain]: [
    [number, number] | null, [number, number], [number, number] | null,
  ] = (() => {
    if (showTopPanel && showEvalPanel) {
      // 3 panels: step ~17%, TL/VL ~45%, MT/MV ~35%, with ~1.5% gaps between.
      return [[0.83, 1.0], [0.36, 0.815], [0.0, 0.345]]
    }
    if (showTopPanel && !showEvalPanel) {
      // 2 panels: step ~28% (legacy), TL/VL ~70%.
      return [[0.72, 1.0], [0.0, 0.685], null]
    }
    if (!showTopPanel && showEvalPanel) {
      // 2 panels: TL/VL ~55%, MT/MV ~40%, no step.
      return [null, [0.45, 1.0], [0.0, 0.42]]
    }
    // 1 panel: TL/VL only.
    return [null, [0.0, 1.0], null]
  })()

  const xTitle = xMode === 'time' ? TZ_LABEL
    : xMode === 'elapsed' ? 'elapsed (h)'
    : xMode === 'epoch' ? 'epoch (fractional)'
    : xMode === 'flop' ? `FLOP (${flopUnit})`
    : 'global_step'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#888', alignSelf: 'center' }}>x-axis:</span>
        {(['time', 'elapsed', 'step', 'epoch', 'flop'] as XMode[])
          // Hide `epoch` when the manifest can't compute it (data label not
          // in EPOCH_SEQUENCES, or `train_batch_size` missing) — a useless
          // button would just confuse. Same for `flop` when the parquet
          // doesn't carry `throughput/total_gflops` (added 2026-06-07).
          .filter((m) => m !== 'epoch' || epochAvailable)
          .filter((m) => m !== 'flop' || flopColAvailable)
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
            {m === 'time' ? 'wallclock' : m === 'flop' ? 'FLOP' : m}
          </button>
        ))}
        <SmoothingChips
          mode={smooth} setMode={setSmooth}
          bandsOn={bandsOn} setBandsOn={setBandsOn}
          isDark={isDark}
          fg={isDark ? '#bbb' : '#444'}
          muted={isDark ? '#888' : '#666'}
        />
        <FlopUnitChips
          unit={flopUnit} setUnit={setFlopUnit}
          isDark={isDark}
          fg={isDark ? '#bbb' : '#444'}
          muted={isDark ? '#888' : '#666'}
        />
      </div>
      <div ref={plotWrapperRef} style={{ position: 'relative' }}>
      <Plot
        onActiveTraceChange={setActiveTraceName}
        onRelayout={onRelayout as (ev: unknown) => void}
        onAfterPlot={recomputeIconPositions}
        data={[
          // 1. step (top panel) — only when not in step mode
          ...(showTopPanel ? [{
            x: stepTrace.xs, y: stepTrace.ys, name: 'step',
            type: 'scatter' as const, mode: 'lines' as const,
            line: { color: COLORS.step, width: 2, shape: 'hv' as const },
            yaxis: 'y',
            legendgroup: 'step',
            legendgrouptitle: { text: 'progress' },
            hovertemplate: 'step %{y}<extra></extra>',
          }] : []),
          // 2. losses (shared log y2) — smoothing-aware. Bands (when on) share
          //    each line's `legendgroup` so `applyBandFade` desats them with
          //    their parent on hover.
          // TL/VL share the `'losses'` legendgroup so their LIs sit flush
          // (a single shared "losses (log)" group title above; no inter-group
          // gap between them) — sweeping the cursor from TL to VL no longer
          // crosses a dead zone that thrashes the hover state. Per-trace fade
          // continues to work because pltly fades by `trace.name` and our
          // `applyBandFade` matches band edges to their parent by name.
          ...smoothedSeriesTraces(TL, 'TL (train/loss)', COLORS.TL, 1.2, 'losses'),
          ...smoothedSeriesTraces(VL, 'VL (eval/loss)', COLORS.VL, 1.4, 'losses'),
          // 3. mat-NMAE + mat-NEMD (from eval.json) on yaxis: 'y3' — only when
          //    the run has eval data; otherwise the panel + traces are omitted
          //    and we fall back to the legacy 2-panel layout.
          ...(showEvalPanel ? evalTraces : []),
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
          // Annotation LI — only when the run has any annotations. Click
          // toggles `annotationsEnabled`, which gates the vlines (above) +
          // the HTML info-icon overlay (rendered below the Plot).
          ...(annotationPositions.length > 0 ? [legendOnly(annotationsLegendName, ANNOTATION_COLOR, 'dash')] : []),
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
            // FLOP mode: scientific notation for the axis ticks
            // (`tickformat: '~e'` → 2e20, not "240 EF") so the axis itself
            // stays compact. The unit-formatted display lives in the title +
            // tooltip (`hovertemplate` below). FLOP values span 6+ OOM across
            // runs so SI suffixes (`~s` → "240E") would be ambiguous.
            ...(xMode === 'flop' ? { tickformat: '.2e' } : {}),
            gridcolor, zerolinecolor, linecolor: gridcolor,
            // Anchor to the bottom-most y-axis so tick labels render BELOW
            // the bottom panel (MT/MV when present, TL/VL when not), instead
            // of below the topmost panel (the step counter) which is plotly's
            // default for a shared xaxis. Without this, the bottom panel has
            // no visible x reference.
            anchor: showEvalPanel ? 'y3' : 'y2',
            // Restore user-picked range across re-renders triggered by
            // smoothing / bands toggles. `null` → omit → plotly autoranges.
            ...(userXRange ? { range: userXRange, autorange: false } : {}),
          },
          yaxis: {
            title: { text: 'step', font: { color: COLORS.step } },
            tickfont: { color: COLORS.step },
            // When step panel is hidden, the domain is unused, but plotly
            // requires SOMETHING valid here; pick a tiny offscreen slice.
            domain: stepDomain ?? [0.99, 1.0],
            gridcolor, zerolinecolor, linecolor: gridcolor,
            visible: showTopPanel,
            // Lock y-zoom: click-drag should zoom x-only (constraint imposed
            // by `fixedrange: true` on every y-axis; plotly's box-zoom then
            // spans the full y range automatically).
            fixedrange: true,
          },
          yaxis2: {
            title: { text: 'loss (log)' },
            type: logType,
            domain: lossDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
            fixedrange: true,
          },
          yaxis3: {
            // Linear scale: MT/MV values are typically 100-600% (< 6× range),
            // not orders of magnitude — log here compresses the range without
            // adding information.
            title: { text: 'mat-NMAE / NEMD %' },
            type: 'linear',
            domain: evalDomain ?? [0.0, 0.01],
            gridcolor, zerolinecolor, linecolor: gridcolor,
            visible: showEvalPanel,
            fixedrange: true,
          },
          shapes: eventShapes,
          margin: { t: 50, l: 70, r: 210, b: 50 },
          hovermode: 'x unified',
          // Anchor the unified TT to a fixed paper-Y position so it doesn't
          // chase the (noisy) trace values as the cursor scans horizontally.
          // User reported jitter when TT was bouncing up/down with TL.
          hoverlabel: { ...themedHoverlabel(isDark), align: 'left' },
          // `hovermode: x unified` follows the cursor X but the box y-pos is
          // computed to avoid the highest trace; using `hoverdistance: -1`
          // disables nearest-point Y snapping → cursor-following Y.
          hoverdistance: -1,
          // Single legend on the right with grouped headers per panel.
          // `tracegroupgap` adds vertical breathing room between groups so
          // the per-group `legendgrouptitle` rows visually separate the
          // panels' entries (step / losses / MT·MV / events).
          legend: {
            x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)',
            tracegroupgap: 10,
          },
        }}
      />
      {/* Annotation info-icon overlay. Pixel-positioned by
          `recomputeIconPositions`; each icon wraps in a `<Tooltip>` that
          shows the annotation's label + step on hover (floating-ui-driven,
          escapes the plot container so it doesn't clip). */}
      {annotationsEnabled && iconPositions.map(({ ann, left, top }, i) => (
        <div
          key={`${ann.step}-${i}`}
          style={{
            position: 'absolute',
            left,
            top,
            transform: 'translateX(-50%)',
            zIndex: 5,
            pointerEvents: 'auto',
          }}
        >
          <Tooltip
            content={(
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                  step {ann.step.toLocaleString()}
                </div>
                <div style={{ whiteSpace: 'pre-wrap' }}>{ann.label}</div>
              </div>
            )}
          >
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 14,
                height: 14,
                borderRadius: '50%',
                border: `1px solid ${ann.color ?? ANNOTATION_COLOR}`,
                color: ann.color ?? ANNOTATION_COLOR,
                background: isDark ? 'rgba(20,20,28,0.85)' : 'rgba(255,255,255,0.92)',
                fontSize: 10,
                fontFamily: 'serif',
                fontStyle: 'italic',
                fontWeight: 700,
                lineHeight: 1,
                cursor: 'help',
                userSelect: 'none',
              }}
              aria-label={`annotation at step ${ann.step}`}
            >
              i
            </span>
          </Tooltip>
        </div>
      ))}
      </div>
    </div>
  )
}
