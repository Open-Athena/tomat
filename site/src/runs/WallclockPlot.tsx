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
import { FlopUnitChips, flopTickformat, flopUnitScale, formatFlops, useFlopUnit } from './flops'
import { ancestorRelation } from './lineage'
import { formatStepDetail } from '../lib/runNames'

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
  /** Which mat-eval metric to plot in the MT/MV panel. Shared with the
   *  MEvalTable above via lifted state in RunsPage so the user's NMAE/NEMD
   *  toggle drives both views at once. Defaults to NMAE. */
  mevalMetric?: 'nmae' | 'nemd'
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
  TL: '#ef5350',     // train/loss — red (also used for MT below; train ↔ red)
  VL: '#ffa726',     // eval/loss  — orange (also used for MV below; val ↔ orange)
  // MT (mat_nmae/nemd on train_200) — red, matching TL so "train" reads the
  // same color throughout the plot. MV (val_200) — orange, matching VL.
  // K=1 (oneshot, the trainer's bare `val_200`/`train_200` setKey) renders
  // DASHED; K=12 (full MaskGIT iterative decode, the `…-maskgit` setKey)
  // renders SOLID. The cost/quality intuition: K=12 is the expensive, more
  // honest decode, so it gets the visually heavier solid line.
  MT: '#ef5350',     // mat-NMAE/NEMD · train_200 — red (matches TL)
  MV: '#ffa726',     // mat-NMAE/NEMD · val_200   — orange (matches VL)
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

export function WallclockPlot({ history, evalSeries, runId, defaultXMode = 'step', attempts, manifest = null, lineageInfo = null, mevalMetric = 'nmae' }: Props) {
  const { isDark } = useTheme()
  // Hoisted early so `applyShapeFade` (declared next) can reference it; the
  // applyShapeFade callback runs in a `useEffect` and Plotly's afterplot
  // listener, both of which read this color when computing the active-trace
  // tint for annotation vlines.
  const ANNOTATION_COLOR = isDark ? 'rgba(190,190,210,0.6)' : 'rgba(80,80,100,0.55)'
  // Wrapper around <Plot> so we can DOM-walk to the `.js-plotly-plot` element
  // and call `Plotly.restyle` on the band traces directly. Bands have
  // `showlegend: false`; pltly's built-in `applyFadeSolo` skips those (since
  // they don't appear in the legend) — so without this fix the teal NEMD bands
  // stay full opacity when MV NMAE is hovered.
  const plotWrapperRef = useRef<HTMLDivElement | null>(null)
  // Two independent inputs feed the "which trace is highlighted" state:
  //   - `hoveredTraceName`: pltly's `onActiveTraceChange` (legend MOUSEOVER)
  //   - `pinnedTraceName`:  our `plotly_legendclick` handler (legend CLICK)
  // Pin wins over hover so the user can click-pin TL, mouseover something
  // else briefly, mouseleave, and find TL still pinned. Click the SAME pinned
  // LI to unpin. The downstream fade machinery (`applyBandFade` /
  // `applyShapeFade`) reads `activeTraceName` only, so it doesn't care which
  // input is driving — visual treatment is identical.
  const [hoveredTraceName, setHoveredTraceName] = useState<string | null>(null)
  const [pinnedTraceName, setPinnedTraceName] = useState<string | null>(null)
  const activeTraceName = pinnedTraceName ?? hoveredTraceName
  type PlotlyDiv = HTMLElement & {
    data?: Array<Record<string, unknown>>
    _Plotly?: { restyle: (el: HTMLElement, attrs: Record<string, unknown>, indices?: number[]) => Promise<void> }
  }
  // Fade bands by trace `name` whenever the active trace changes. Each band
  // edge trace shares its parent line's `name` (e.g. `'TL (train loss)'`), so
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

  // Shape-fade side-effect (Bug 3 fix). Event vlines' active-color tint is
  // applied here via `Plotly.relayout({ shapes })` instead of baked into the
  // layout prop — that way the inline layout object's `shapes:` stays
  // referentially stable across `activeTraceName` flips, so `Plotly.react`
  // doesn't fire for every legend mouseover. The ref pattern matches
  // `applyBandFade` above: the latest `baseEventShapes` is stashed during
  // render and read by this effect when `activeTraceName` changes. Idempotent
  // — bails when the freshly-computed shape colors match what's already on
  // `_fullLayout.shapes`.
  type ShapeColorTuple = string
  const baseEventShapesRef = useRef<Array<{
    type: 'line'; xref: 'x'; yref: 'paper'
    x0: string | number; x1: string | number; y0: number; y1: number
    line: { color: string; width: number; dash: 'dash' | 'dot' | 'solid' }
    _baseColor: string
  }>>([])
  const applyShapeFade = useCallback(() => {
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as (HTMLElement & {
      _Plotly?: { relayout: (el: HTMLElement, attrs: Record<string, unknown>) => Promise<void> }
      _fullLayout?: { shapes?: Array<{ line?: { color?: string } }> }
    }) | null
    const P = plotDiv?._Plotly
    if (!plotDiv || !P) return
    const shapes = baseEventShapesRef.current
    if (shapes.length === 0) return
    // Recompute active color → per-shape tint mapping. Anything not in this
    // map (TL, VL, NMAE, NEMD, step, etc.) means "user hovered a non-event
    // trace" → fade ALL shapes.
    let activeColor: string | null
    const a = activeTraceName
    if (a === null) activeColor = null
    else if (a.startsWith('trainer_started')) activeColor = COLORS.start
    else if (a.startsWith('sigterm')) activeColor = COLORS.sigterm
    else if (a.startsWith('cluster preempt')) activeColor = COLORS.preempt
    else if (a.startsWith('death: preempt')) activeColor = DEATH_COLORS.preempt
    else if (a.startsWith('death: cascade')) activeColor = DEATH_COLORS.cascade
    else if (a.startsWith('death: failed')) activeColor = DEATH_COLORS.failed
    else if (a.startsWith('annotations')) activeColor = ANNOTATION_COLOR
    else activeColor = ''
    const tint = (base: string): ShapeColorTuple => {
      if (activeColor === null || activeColor === base) return base
      const rgb = hexToRgbTuple(base)
      // 0.08 alpha (was 0.18): runs with many event vlines (e.g. 72 starts
      // on `train-mg-kl-bin5-fs-tpu`) cluster densely on the x-axis. At 0.18,
      // overlapping faded dashes alpha-blend back to near-full brightness
      // and the "events stay full opacity" bug 1 manifests visually even
      // though each shape is technically faded. 0.08 keeps individual
      // shapes legible-but-clearly-dim and survives the dense-cluster
      // compositing without re-becoming bright.
      return `rgba(${rgb}, 0.08)`
    }
    // Build the new shapes array with retinted colors. Idempotency check:
    // bail when the new colors exactly match the live `_fullLayout.shapes`
    // colors (Plotly normalizes through r/g/b → string equality is the only
    // contract we can trust here, but matches in practice).
    const liveShapes = plotDiv._fullLayout?.shapes ?? []
    const tintedRaw = shapes.map((s) => ({
      type: s.type, xref: s.xref, yref: s.yref,
      x0: s.x0, x1: s.x1, y0: s.y0, y1: s.y1,
      line: { ...s.line, color: tint(s._baseColor) },
    }))
    // Cluster-dedup faded shapes: when N event vlines pile up near the same
    // x (e.g. 36 trainer_starts in a retry-storm), α=0.08 alpha-blends back
    // to ~95% effective opacity, defeating the fade. Per (faded color, x-
    // bucket), keep only one. The active-color shapes (not faded) keep their
    // full density — only the de-emphasized ones collapse. Bucket is 1/300th
    // of the visible x-range so the dedup tracks the rendered pixel
    // resolution rather than absolute step / wallclock units.
    type Range = [number, number]
    const xRange = ((plotDiv._fullLayout as { xaxis?: { range?: Range } } | undefined)?.xaxis?.range) ?? null
    const bucketWidth = xRange ? Math.max(1, (xRange[1] - xRange[0]) / 300) : null
    const tinted: typeof tintedRaw = []
    const seenBucketKey = new Set<string>()
    for (const s of tintedRaw) {
      const isFaded = typeof s.line.color === 'string' && s.line.color.startsWith('rgba(')
      if (!isFaded || bucketWidth == null) {
        tinted.push(s)
        continue
      }
      const x = typeof s.x0 === 'number' ? s.x0 : Number(s.x0)
      const bucket = Number.isFinite(x) ? Math.floor(x / bucketWidth) : null
      const key = `${bucket}|${s.line.color}`
      if (seenBucketKey.has(key)) continue
      seenBucketKey.add(key)
      tinted.push(s)
    }
    let changed = liveShapes.length !== tinted.length
    if (!changed) {
      for (let i = 0; i < tinted.length; i++) {
        if (liveShapes[i]?.line?.color !== tinted[i].line.color) { changed = true; break }
      }
    }
    if (!changed) return
    P.relayout(plotDiv, { shapes: tinted })
  }, [activeTraceName, ANNOTATION_COLOR])
  useEffect(applyShapeFade, [applyShapeFade])
  // Re-apply on every Plotly redraw — `Plotly.react` (xMode swap, data
  // refetch, smoothing change) resets `_fullLayout.shapes` to whatever's
  // in the inline layout (which we pin to BASE colors). If an active LI was
  // hovered during the react, we need to re-tint after the redraw lands.
  // Idempotency check inside `applyShapeFade` breaks the otherwise-infinite
  // afterplot → relayout → afterplot loop.
  useEffect(() => {
    const root = plotWrapperRef.current
    if (!root) return
    const plotDiv = root.querySelector('.js-plotly-plot') as (HTMLElement & {
      on?: (evt: string, fn: () => void) => void
      removeListener?: (evt: string, fn: () => void) => void
    }) | null
    if (!plotDiv?.on) return
    plotDiv.on('plotly_afterplot', applyShapeFade)
    return () => plotDiv.removeListener?.('plotly_afterplot', applyShapeFade)
  }, [applyShapeFade])


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

  // FLOP-unit display preference (`?fopu=`). Hoisted to the top of the
  // component so `xOfRow` / `xOfTs` / `xOfStep` / `stepTrace` (declared
  // below) can divide raw FLOPs by `flopXScale` to land x in the user-chosen
  // unit. `formatFlops` callers (hovertemplates) still receive RAW flops
  // because that helper does its own unit-aware scaling.
  const [flopUnit, setFlopUnit] = useFlopUnit()
  const flopXScale = flopUnitScale(flopUnit)

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
  // `formatFlops(x, unit)` lands a value in the unit the user chose. Drives
  // `flopAtTs` for the `flop` x-axis mode. Empty for parquets logged before
  // the column was added to the schema — the `flop` button is hidden in that
  // case.
  //
  // The cumulative IS continuous across trainer-restart boundaries (Levanter
  // re-loads the prior cumulative from the checkpoint after resume), but
  // `_timestamp` is upload-time noise on iris-TPU runs that async-upload
  // metric batches: rows can land in wandb with their ts hours late relative
  // to log time. The flop value paired with a "late" ts is its OLD logical-
  // order cumulative, which is much smaller than the cumulative the run
  // had actually reached by that wallclock — so a naive ts→flop binary
  // search would snap back to a stale value and produce "horizontal
  // slashes" (the running-max step stays high while x snaps low).
  //
  // Walk row-index (= `_step`) order — which preserves the (cumulative,
  // step) pairing — and store the RUNNING MAX cumulative at each ts so
  // `flopAtTs(ts)` always returns "the highest cumulative the run had
  // reached by ts". Then sort by ts so the binary search still works.
  // Same rationale + fix as the `segments` useMemo below (`_step` is
  // monotonic and authoritative; `_timestamp` is upload noise).
  const tsFlop = useMemo(() => {
    const totalGflops = cols.get('throughput/total_gflops') ?? []
    if (totalGflops.length === 0) return []
    const pairs: { ts: number; flop: number }[] = []
    let runningMaxFlop = 0
    for (let i = 0; i < history.rowCount; i++) {
      const g = totalGflops[i]
      if (g == null) continue
      const ts = timestamps[i]
      if (ts == null) continue
      // The parquet IS `_step`-ascending; cumulative is monotonic in `_step`,
      // so this `Math.max` is a no-op for in-order runs and ONLY clamps the
      // pathological async-upload disorder. Kept explicit so a future
      // schema change can't silently regress the contract.
      runningMaxFlop = Math.max(runningMaxFlop, (g as number) * 1e9)
      pairs.push({ ts: ts as number, flop: runningMaxFlop })
    }
    // ts-sort, but also enforce monotone flop within ties / out-of-order ts:
    // after sort, if pair[i].ts < pair[i-1].ts due to wandb upload noise,
    // pair[i].flop (from an EARLIER `_step`) can be smaller than pair[i-1].flop.
    // Running-max-after-sort returns to the same monotone semantics ("highest
    // flop the run had reached by ts").
    pairs.sort((a, b) => a.ts - b.ts)
    let m = 0
    for (const p of pairs) { m = Math.max(m, p.flop); p.flop = m }
    return pairs
  }, [history.rowCount, timestamps, cols])
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
    if (xMode === 'flop') {
      const f = flopAtTs(ts)
      return Number.isFinite(f) ? f / flopXScale : NaN
    }
    if (xMode === 'elapsed') return (ts - t0) / 3600
    return toLocal(ts)
  }

  /** x-coordinate for the parquet row at index `i`. Used by traces that walk
   *  rows in `_step` (row-index) order — `stepTrace`, `series` — so the
   *  cumulative pairs strictly with the row's own logical step. In FLOP mode
   *  this reads the row's own `throughput/total_gflops` directly instead of
   *  routing through `flopAtTs(ts)`: the ts-keyed binary search would land on
   *  the wrong cumulative when wandb's `_timestamp` is upload-time noise
   *  (iris async-uploads metric batches), producing back-and-forth jitter on
   *  what should be a monotonic loss-vs-FLOP trace. For other axes, the row's
   *  ts is the right `xOfTs` input. The FLOP value is divided by
   *  `flopXScale` so plotly's axis ticks read in the user-chosen unit
   *  (EF/PF/TF/sci) — keep `formatFlops` callers feeding RAW flops since
   *  that helper does its own unit scaling. */
  function xOfRow(i: number, ts: number): string | number {
    if (xMode === 'flop') {
      const totalGflops = cols.get('throughput/total_gflops')
      const g = totalGflops?.[i]
      return g == null ? NaN : ((g as number) * 1e9) / flopXScale
    }
    return xOfTs(ts)
  }

  /** x-coordinate for an eval point at checkpoint `step`. */
  function xOfStep(step: number): string | number | null {
    if (xMode === 'step') return step
    if (xMode === 'epoch') return epochOfStep(step, manifest)
    if (xMode === 'flop') {
      const ts = tsAtGstep(step)
      if (ts === null) return null
      const f = flopAtTs(ts)
      return Number.isFinite(f) ? f / flopXScale : null
    }
    const ts = tsAtGstep(step)
    if (ts === null) return null
    return xMode === 'elapsed' ? (ts - t0) / 3600 : toLocal(ts)
  }

  // Per-metric parquet series (TL/VL): xs, ys, gsteps (gstep for the tooltip).
  type Series = { xs: (string | number)[]; ys: number[]; gsteps: (number | null)[] }

  // Restart-segment boundaries: row indices into the parquet (NOT into
  // `ordered`) keyed on `lifecycle/trainer_started` events (each one marks
  // a new trainer process). Splitting at these boundaries lets us render
  // older restart trajectories at low opacity so they don't deface the plot,
  // while the latest segment stays fully visible.
  //
  // ## Why row-index, not `_timestamp`?
  //
  // We previously walked `ordered` (ts-sorted) and placed seams when the
  // running `_timestamp` cursor crossed each trainer_started's ts. That's
  // a footgun on iris-TPU runs: Levanter's `BackgroundIterator` async-
  // uploads metric batches in chunks, so wandb's server-side `_timestamp`
  // (= upload time) drifts from the original client-side `log()` call time
  // by minutes-to-hours. After a `trainer_started` event uploads at
  // wall-time T_new, late-arriving rows from BEFORE the restart (whose
  // logical step is e.g. 267) can still upload at wall-time > T_new and
  // land inside the "post-restart" ts bucket — getting wrongly tagged as
  // segment #2 by the ts-cursor walk. The plotted segment #2 trace then
  // spans the full x range (because some of its members have gstep ≈ 267
  // while others have gstep ≈ 34k), producing three full-x overlapping
  // zigzags instead of three non-overlapping per-restart trajectories.
  //
  // The parquet's `_step` IS monotonic and uniquely identifies each row's
  // logical position; `_timestamp` is upload-time noise. `scripts/runs-
  // sync.py` writes rows in ascending `_step` order, so the natural row
  // index 0..rowCount-1 already IS the correct seam space. Seams go at:
  //   (1) row 0,
  //   (2) every part boundary (lineage glue), and
  //   (3) the row index of each `trainer_started == 1` row in the current
  //       part, EXCLUDING the first one (it coincides with the current
  //       run's start, already a seam from (1)/(2)).
  //
  // The proper upstream fix lives in iris task #219 (Levanter's
  // BackgroundIterator ContextVars fix); this is the FE workaround for
  // its consequences on parquets already in the wild.
  //
  // Each segment tracks which lineage part it belongs to (`partIdx`):
  // 0..ancestors.length-1 → that ancestor, ancestors.length → current run.
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
    // Precompute cumulative part boundaries so we can detect part transitions
    // in a single O(rowCount) pass instead of paying the per-row scan cost.
    const partBoundaries: number[] = []  // partBoundaries[p] = end-exclusive row index of part p
    {
      let acc = 0
      for (let p = 0; p < numParts; p++) {
        acc += partRowCounts?.[p] ?? history.rowCount
        partBoundaries.push(acc)
      }
    }
    const partOfRaw = (rawIdx: number): number => {
      if (!partRowCounts || numParts === 1) return 0
      for (let p = 0; p < numParts; p++) {
        if (rawIdx < partBoundaries[p]) return p
      }
      return numParts - 1
    }
    if (history.rowCount === 0) return [{ start: 0, partIdx: currentPartIdx }]
    const startedCol = cols.get('lifecycle/trainer_started') ?? []
    // Walk row indices in their natural (ascending-`_step`) order. Insert a
    // seam at row 0, at each part boundary, and at each trainer_started row
    // within the current part (skipping the first such row since it
    // coincides with the current run's start).
    const out: SegMeta[] = []
    let currentPartStartsSeen = 0
    let lastPartIdx = -1
    for (let i = 0; i < history.rowCount; i++) {
      const p = partOfRaw(i)
      const partChanged = p !== lastPartIdx
      let restartHere = false
      if (startedCol[i] === 1 && p === currentPartIdx) {
        // Skip the very first trainer_started in the current run — it
        // coincides with the run's start (already a seam via partChanged /
        // i===0). Only 2nd+ starts indicate intra-run restarts.
        if (currentPartStartsSeen > 0) restartHere = true
        currentPartStartsSeen++
      }
      if (i === 0 || partChanged || restartHere) {
        if (out.length === 0 || out[out.length - 1].start !== i) {
          out.push({ start: i, partIdx: p })
        }
      }
      lastPartIdx = p
    }
    return out
  }, [history.rowCount, history.partRowCounts, cols])
  const numSegments = segments.length
  const segmentStarts = useMemo(() => segments.map((s) => s.start), [segments])

  // Segment index for a given RAW ROW INDEX (NOT `ordered`-index). Binary
  // search over `segmentStarts` (monotonic in row-index space).
  function segmentOf(rowIdx: number): number {
    let lo = 0, hi = segmentStarts.length - 1, best = 0
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (segmentStarts[mid] <= rowIdx) { best = mid; lo = mid + 1 }
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
  //
  // Walks raw row indices in natural (parquet / `_step`-ascending) order
  // rather than `ordered` (ts-sorted): segment membership is in row-index
  // space (see the `segments` useMemo for the rationale — `_timestamp` is
  // upload-time noise on iris-TPU runs that async-upload metric batches).
  // Within a segment, x=step / epoch / flop picks up monotonically in row
  // order naturally (gstep / epoch-of-gstep / cumulative-flop are all
  // monotone in `_step`); x=wallclock/elapsed instead carry tiny ts
  // back-jumps where async-upload reordered rows — we post-sort each
  // segment's xs/ys/gsteps in those modes so downstream gap detection
  // (`findGapEndIndices`) doesn't trip on the jitter and emit hundreds of
  // phantom dotted bridge segments.
  function series(key: keyof RunHistoryRow): Series[] {
    const col = cols.get(key) ?? []
    const out: Series[] = Array.from({ length: numSegments },
      () => ({ xs: [], ys: [], gsteps: [] }))
    for (let i = 0; i < history.rowCount; i++) {
      const ts = timestamps[i]
      if (ts === null) continue
      const v = col[i]
      if (v === null || v === undefined) continue
      const x = xOfRow(i, ts as number)
      // In FLOP mode the row may not carry the cumulative column (older
      // pre-warm rows, ancestor parquets that predate the schema bump);
      // drop those points so the trace doesn't render `NaN` gaps mid-line.
      if (typeof x === 'number' && !Number.isFinite(x)) continue
      const seg = out[segmentOf(i)]
      seg.gsteps.push(gstepAtTs(ts))
      seg.xs.push(x)
      seg.ys.push(v as number)
    }
    // Sort wallclock/elapsed segments by x. The row-index walk above doesn't
    // guarantee ts-monotonicity (Levanter `BackgroundIterator` async-uploads
    // metric batches → wandb `_timestamp` can drift minutes-to-hours from
    // log time, putting an "earlier" row's ts after a "later" row's). The
    // resulting non-monotone xs trip both rendering (zigzag) and gap
    // detection (`findGapEndIndices`'s median-delta threshold flags every
    // tiny back-jump as a gap, producing hundreds of phantom dotted bridge
    // segments — `train-mg-kl-bin5-fs-tpu` exhibited ~700 false gaps in
    // seg #1 alone). step/epoch/flop xs are monotone in row order already;
    // skip the sort there.
    if (xMode === 'time' || xMode === 'elapsed') {
      for (const seg of out) {
        if (seg.xs.length < 2) continue
        const indices = seg.xs.map((_, k) => k)
        indices.sort((a, b) => {
          const xa = seg.xs[a], xb = seg.xs[b]
          if (typeof xa === 'number' && typeof xb === 'number') return xa - xb
          // wallclock: xs are `YYYY-MM-DD HH:MM:SS` strings, where lex order
          // matches chronological order. localeCompare is overkill but safe.
          return String(xa).localeCompare(String(xb))
        })
        seg.xs = indices.map((k) => seg.xs[k])
        seg.ys = indices.map((k) => seg.ys[k])
        seg.gsteps = indices.map((k) => seg.gsteps[k])
      }
    }
    return out
  }

  // Step (top-panel) trace: running-max of global_step.
  //
  // In FLOP mode, walk row-index order (`_step`-ascending in the parquet) so
  // x = cumulative_at_row_i pairs strictly with the row's own logical step
  // — `xOfRow` reads `throughput/total_gflops[i]` directly. Other modes still
  // walk ts-sorted (`ordered`) because the running-max-step + x-of-ts pairing
  // is correct there and the ts-sort gives natural ascending x on the time
  // axes.
  const stepTrace = useMemo(() => {
    const globalStep = cols.get('global_step') ?? []
    const xs: (string | number)[] = []
    const ys: number[] = []
    let runningMax = -Infinity
    if (xMode === 'flop') {
      const totalGflops = cols.get('throughput/total_gflops') ?? []
      for (let i = 0; i < history.rowCount; i++) {
        const ts = timestamps[i]
        if (ts === null) continue
        const s = globalStep[i]
        if (s === null) continue
        const g = totalGflops[i]
        if (g == null) continue
        runningMax = Math.max(runningMax, s)
        xs.push(((g as number) * 1e9) / flopXScale)
        ys.push(runningMax)
      }
    } else {
      for (const { ts, i } of ordered) {
        const s = globalStep[i]
        if (s === null) continue
        runningMax = Math.max(runningMax, s)
        xs.push(xOfTs(ts as number))
        ys.push(runningMax)
      }
    }
    return { xs, ys }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ordered, cols, xMode, history.rowCount, timestamps, flopXScale])

  const customGsteps = (s: Series) => s.gsteps.map((g) => (g === null ? '?' : g))

  // Memoize TL / VL so the Plot's `data` prop stays referentially stable across
  // re-renders that don't change the underlying parquet / xMode / segments
  // (e.g. legend-hover state changes flowing through `activeTraceName` would
  // otherwise rebuild these per render → Plotly.react fires for every hover).
  // Same dep set as `stepTrace` above plus `segments` (whose change forces
  // `series()` to re-bucket rows). Bug 3 (flicker) regressed when these were
  // computed inline.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const TL = useMemo(() => series('train/loss'), [ordered, cols, xMode, history.rowCount, timestamps, flopXScale, segments])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const VL = useMemo(() => series('eval/loss'), [ordered, cols, xMode, history.rowCount, timestamps, flopXScale, segments])

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
  // (FLOP-unit hook hoisted to the top of the component so the x-coordinate
  // helpers — `xOfRow`, `xOfTs`, `xOfStep`, `stepTrace` — can scale raw
  // flops by `flopXScale` to land x in the user-chosen unit.)

  // Gap detection in wallclock / elapsed modes: when a run is paused
  // (e.g. a Modal-side respawn between two real data points without an
  // intervening `lifecycle/trainer_started`), the parquet rows that bookend
  // the pause would otherwise be connected by a solid line indistinguishable
  // from a normal training step — especially misleading once rolling
  // smoothing is on. Detect "gaps" as consecutive-x deltas ≥ both
  // `GAP_THRESHOLD_RATIO` × the segment's median delta AND `GAP_MIN_SECONDS`
  // absolute. Both gates have to fire so we don't flag (a) wandb async-upload
  // jitter at the ~minute scale (~100× median but only ~1 min absolute —
  // training resumes fine, no real pause) or (b) a long-cadence eval column
  // whose median is already minutes (a 10-min gap there is normal). Real
  // Modal-respawn pauses are 5+ min AND many-× the median; the AND-gate
  // catches them and nothing else. Step-mode is exempt: restart-segment
  // splitting already breaks at step regressions and intra-segment step
  // deltas are unit-1 monotone.
  const GAP_THRESHOLD_RATIO = 10
  const GAP_MIN_SECONDS = 300  // 5 minutes
  // Numeric distance between consecutive xs, in SECONDS. `NaN` if either is
  // null/mixed. Wallclock xs are `YYYY-MM-DD HH:MM:SS` strings (parse via
  // `Date.parse` → ms → / 1000); `elapsed` xs are hours (× 3600).
  function xDeltaSec(prev: string | number, curr: string | number): number {
    if (typeof prev === 'number' && typeof curr === 'number') {
      // elapsed: hours
      return (curr - prev) * 3600
    }
    if (typeof prev === 'string' && typeof curr === 'string') {
      // wallclock: date strings → ms → s
      return (new Date(curr).getTime() - new Date(prev).getTime()) / 1000
    }
    return NaN
  }
  // Indices `i` such that `xs[i-1] → xs[i]` is a gap. Empty in step-mode
  // (caller short-circuits) and for short series.
  function findGapEndIndices(xs: (string | number)[]): number[] {
    if (xMode === 'step' || xs.length < 3) return []
    const deltas: number[] = []
    for (let i = 1; i < xs.length; i++) {
      const d = xDeltaSec(xs[i - 1], xs[i])
      if (Number.isFinite(d) && d > 0) deltas.push(d)
    }
    if (deltas.length === 0) return []
    const sorted = [...deltas].sort((a, b) => a - b)
    const median = sorted[Math.floor(sorted.length / 2)]
    if (!(median > 0)) return []
    const cutoff = Math.max(median * GAP_THRESHOLD_RATIO, GAP_MIN_SECONDS)
    const out: number[] = []
    for (let i = 1; i < xs.length; i++) {
      const d = xDeltaSec(xs[i - 1], xs[i])
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
    void numCurrentSegs
    void lastNonEmptyCurrent
    // Group segments into buckets: one "current" bucket + one per ancestor.
    // Each bucket emits a SINGLE concatenated trace so the x-unified tooltip
    // shows ONE TL + ONE VL entry per group (current, parent, grandparent),
    // not one per restart segment. Bucketing in insertion order keeps the
    // legend order stable (current first, then ancestors as they appear).
    type Bucket = {
      key: string
      segIndices: number[]
      isCurrent: boolean
      color: string
      legendGroup: string
      groupTitle: string
      hoverName: string
      legendName: string
      showLegend: boolean
    }
    const buckets = new Map<string, Bucket>()
    for (let i = 0; i < N; i++) {
      const sm = segments[i]
      const ancestor = ancestorOf(sm.partIdx)
      if (ancestor) {
        const bk = `ancestor:${ancestor.name}`
        if (!buckets.has(bk)) {
          const rel = ancestorRelation(sm.partIdx, lineageInfo!.length)
          buckets.set(bk, {
            key: bk,
            segIndices: [],
            isCurrent: false,
            color: ancestor.color,
            // Share the current run's legendgroup so all TL/VL traces — the
            // current run's plus every ancestor's — collapse to ONE legend
            // row per metric, toggle together. Each ancestor still draws in
            // its own color (line color is per-trace), but the legend reads
            // as a single virtual TL + VL pair.
            legendGroup: lg,
            groupTitle: rel,
            hoverName: `${name} (${rel})`,
            legendName: name,
            showLegend: false,
          })
        }
        buckets.get(bk)!.segIndices.push(i)
      } else {
        const bk = 'current'
        if (!buckets.has(bk)) {
          buckets.set(bk, {
            key: bk,
            segIndices: [],
            isCurrent: true,
            color,
            legendGroup: lg,
            groupTitle: 'losses (log)',
            hoverName: name,
            legendName: name,
            showLegend: true,
          })
        }
        buckets.get(bk)!.segIndices.push(i)
      }
    }
    const SPARSE_THRESHOLD = 30
    const out: SmoothedTrace[] = []
    for (const bucket of buckets.values()) {
      // Concatenate this bucket's segments. Insert a `null` y between
      // consecutive segments so plotly breaks the line (the segment boundary
      // is real — don't draw across restart gaps). Smoothing runs PER segment
      // so the kernel doesn't reach back across a restart.
      const cx: (string | number)[] = []
      const cy: (number | null)[] = []
      const cg: (number | string | null)[] = []
      const cyStd: (number | null)[] = []
      let lastSegFirstIdx = -1
      let lastSegLastIdx = -1
      let pointCount = 0
      bucket.segIndices.forEach((segIdx, pos) => {
        const s = seriesPerSeg[segIdx]
        if (s.xs.length === 0) return
        const nonNullYs = s.ys.reduce((acc, y) => acc + (y == null ? 0 : 1), 0)
        const isSparseSeg = nonNullYs < SPARSE_THRESHOLD
        const { mean: ySm, std: ySt } = isSparseSeg
          ? { mean: s.ys, std: null as (number | null)[] | null }
          : computeSmoothedSeries(s.ys, smooth)
        const segG = customGsteps(s)
        const isLastBucketSeg = pos === bucket.segIndices.length - 1
        if (cx.length > 0) {
          // Inter-segment null break — share the gap-start x so the break
          // marker doesn't render anywhere visible.
          cx.push(s.xs[0]); cy.push(null); cg.push(null); cyStd.push(null)
        }
        if (isLastBucketSeg) lastSegFirstIdx = cx.length
        const gapEnds = findGapEndIndices(s.xs)
        const gapSet = new Set(gapEnds)
        for (let i = 0; i < s.xs.length; i++) {
          if (gapSet.has(i) && i > 0) {
            cx.push(s.xs[i - 1]); cy.push(null); cg.push(null); cyStd.push(null)
          }
          cx.push(s.xs[i])
          cy.push(ySm[i])
          cg.push(segG[i])
          cyStd.push(ySt ? ySt[i] : null)
          if (ySm[i] != null) pointCount += 1
        }
        if (isLastBucketSeg) lastSegLastIdx = cx.length - 1
      })
      if (pointCount === 0) continue
      // Sparse-overall buckets get visible markers (each datapoint stands
      // out — VL on long resume chains has a handful of points). Dense
      // buckets stay lines-only (TL would be a wall of red dots otherwise).
      const denseOverall = pointCount >= SPARSE_THRESHOLD
      const mainTrace: SmoothedTrace = {
        x: cx, y: cy, name: bucket.legendName,
        type: 'scatter',
        mode: denseOverall ? 'lines' : 'lines+markers',
        line: { color: bucket.color, width: lineWidth },
        ...(denseOverall ? {} : { marker: { color: bucket.color, size: 5 } }),
        opacity: 1,
        yaxis: 'y2',
        legendgroup: bucket.legendGroup,
        // Only the current bucket sets the legendgrouptitle. Ancestors share
        // its legendgroup, so plotly nests their (showlegend:false) traces
        // under the same title — one legend section, one TL + one VL row,
        // and one "losses (log)" header in the unified tooltip.
        ...(bucket.isCurrent ? { legendgrouptitle: { text: bucket.groupTitle } } : {}),
        showlegend: bucket.showLegend,
        customdata: cg,
        hovertemplate: `${bucket.hoverName} %{y:.3f}<br>gstep %{customdata}<extra></extra>`,
      }
      // ±σ bands only on the current bucket's last segment (avoids smearing
      // bands across the full lineage).
      if (bandsOn && bucket.isCurrent && lastSegFirstIdx >= 0 && lastSegLastIdx >= 0) {
        const yLower: (number | null)[] = new Array(cx.length).fill(null)
        const yUpper: (number | null)[] = new Array(cx.length).fill(null)
        let hasBand = false
        for (let i = lastSegFirstIdx; i <= lastSegLastIdx; i++) {
          const m = cy[i]
          const sd = cyStd[i]
          if (m == null || sd == null) continue
          yLower[i] = Math.max(1e-6, m - sd)
          yUpper[i] = m + sd
          hasBand = true
        }
        if (hasBand) {
          const edge = (y: (number | null)[], fillKey: string | null): SmoothedTrace => ({
            x: cx, y, name: bucket.legendName,
            type: 'scatter', mode: 'lines',
            line: { width: 0, color: 'rgba(0,0,0,0)' },
            yaxis: 'y2', legendgroup: bucket.legendGroup,
            showlegend: false,
            hoverinfo: 'skip',
            ...(fillKey ? { fill: 'tonexty', fillcolor: fillKey } : {}),
          })
          const rgb = hexToRgbTuple(bucket.color)
          out.push(edge(yLower, null), edge(yUpper, `rgba(${rgb}, 0.18)`))
        }
      }
      out.push(mainTrace)
    }
    return out
  }

  // ── eval points from eval.json ──
  // Per setKey: per-step median rendered as a `lines+markers` trace on the
  // dedicated MT/MV panel (yaxis: 'y3'). The active metric (NMAE or NEMD)
  // is driven by `mevalMetric` (URL-shared with MEvalTable above).
  //
  // Visual encoding (matches MEvalTable's column labels):
  //   • MT (train_200)  → red    (matches TL)
  //   • MV (val_200)    → orange (matches VL)
  //   • K=1  (oneshot, bare setKey)        → DASHED
  //   • K=12 (full MaskGIT, `-maskgit`)    → SOLID
  // Other modes (`-free`, `…`) keep dashed-K=1 styling for now (no K-shaped
  // setKey suffix yet) but render in MT/MV's set color — spec 25 follow-up
  // is the eventual per-mode color-coding.
  //
  // Markers are size 6 so the typically-handful (~4) of timepoints are clearly
  // visible. ±p25-p75 / p1-p99 bands were dropped — see header comment.
  const evalMeanTrace = (
    setKey: string, mvmt: 'MT' | 'MV', kLabel: string,
    dash: 'solid' | 'dash',
    metric: 'nmae' | 'nemd',
  ) => {
    // Use the MEAN of nmae_filled_* / nemd_filled_* to match the MEvalTable
    // (table reads `${metric}_filled_mean`). Plot was previously using
    // `_median`, which produced different numbers per step (e.g. bin5 step
    // 100k: median 5.35% vs mean 7.46%). Median is more outlier-robust but
    // diverging from the table broke user trust in the dashboard's coherence.
    const pts = evalSeries?.sets[setKey] ?? []
    const xs: (string | number)[] = []
    const ys: (number | null)[] = []
    const customdata: [number, number | string][] = []
    for (const pt of pts) {
      const x = xOfStep(pt.step)
      if (x === null) continue
      const rec = pt as unknown as Record<string, number | null>
      const v = rec[`${metric}_filled_mean`] ?? rec[`${metric}_mean`]
      const nMats = (pt as unknown as { n_mats?: number }).n_mats
      xs.push(x)
      customdata.push([pt.step, nMats ?? '?'])
      ys.push(typeof v === 'number' ? v * 100 : null)  // fraction → %
    }
    const color = COLORS[mvmt]  // MT → red, MV → orange (matches TL/VL).
    const name = `${mvmt} · ${kLabel}`
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
      legendgrouptitle: { text: `MT/MV (mat-${metric.toUpperCase()} %)` },
      customdata,
      hovertemplate: `${name} %{y:.3f}%%<br>step %{customdata[0]} · n_mats %{customdata[1]}<extra></extra>`,
    }
  }
  const evalTraces = useMemo(() => {
    const out: Record<string, unknown>[] = []
    // setKey conventions on eval.json:
    //   val_200             → MV · K=1   (oneshot / teacher-mode bare key)
    //   train_200           → MT · K=1
    //   val_200-maskgit     → MV · K=12  (bare maskgit = legacy K=12 iterative)
    //   train_200-maskgit   → MT · K=12
    //   val_200-maskgit-K1  → MV · K=1   (`--output-suffix K1` ablation —
    //                          honest single-MG-step decode; the dashboard's
    //                          primary number post-reforward-removal)
    //   …-<other-mode>      → MV/MT · <mode>  (e.g. `-free`; falls through
    //                          as a dashed trace to keep the legend coherent)
    //
    // `-K<N>` variants get K=N labelling and override the mode's default K
    // (so e.g. `-maskgit-K1` renders dashed K=1, not solid K=12).
    const setKeys = Object.keys(evalSeries?.sets ?? {})
    const labelFor = (setKey: string): {
      mvmt: 'MT' | 'MV'; kLabel: string; dash: 'solid' | 'dash'
    } => {
      const parts = setKey.split('-')
      const mvmt: 'MT' | 'MV' = parts[0] === 'val_200' ? 'MV' : 'MT'
      const mode: string | undefined = parts[1]
      const variant: string = parts.slice(2).join('-')
      const kMatch = variant.match(/^K(\d+)/)
      let kLabel: string
      let dash: 'solid' | 'dash'
      if (kMatch) {
        // Explicit K=N variant from `--output-suffix`; honour it.
        const k = kMatch[1]
        kLabel = `K=${k}`
        dash = k === '1' ? 'dash' : 'solid'
      } else if (mode === 'maskgit') {
        // Bare `-maskgit` is the legacy K=12 iterative-decode bucket.
        kLabel = 'K=12'
        dash = 'solid'
      } else {
        // Bare matSet or any other mode (e.g. `-free`) is K=1-style dashed.
        kLabel = mode == null ? 'K=1' : mode
        dash = 'dash'
      }
      return { mvmt, kLabel, dash }
    }
    for (const setKey of setKeys) {
      const { mvmt, kLabel, dash } = labelFor(setKey)
      out.push(evalMeanTrace(setKey, mvmt, kLabel, dash, mevalMetric))
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalSeries, xMode, mevalMetric])
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

  // Lifecycle event timestamps. Memoized so that `plotDataMemo` and
  // `plotLayoutMemo` can include them in deps without re-invalidating on
  // every render (these returned fresh array refs each time before, which
  // defeated all of the layout/data memoization downstream).
  const startTs = useMemo<number[]>(() => {
    const ts: number[] = []
    const col = cols.get('lifecycle/trainer_started') ?? []
    for (let i = 0; i < col.length; i++) {
      if (col[i] === 1 && timestamps[i] !== null) ts.push(timestamps[i] as number)
    }
    return ts.sort((a, b) => a - b)
  }, [cols, timestamps])
  const sigtermTs = useMemo<number[]>(() => {
    const ts: number[] = []
    const col = cols.get('lifecycle/sigterm_received') ?? []
    for (let i = 0; i < col.length; i++) {
      if (col[i] === 1 && timestamps[i] !== null) ts.push(timestamps[i] as number)
    }
    return ts.sort((a, b) => a - b)
  }, [cols, timestamps])
  const preemptTs = useMemo<number[]>(() => {
    const preemptCol = cols.get('cluster/preemptions') ?? []
    const out: number[] = []
    let prev: number | null = null
    for (const { ts, i } of ordered) {
      const v = preemptCol[i]
      if (v === null) continue
      if (prev !== null && v > prev) out.push(ts as number)
      prev = v
    }
    return out
  }, [cols, ordered])

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'

  // (`ANNOTATION_COLOR` is hoisted to the top of the component so
  // `applyShapeFade` can reference it. Originally declared here.)

  // Event vlines spanning both panels via paper-coord shapes.
  //
  // Shapes also fade on legend hover: if the user hovers a real trace (TL /
  // VL / etc.), all event vlines dim to alpha 0.08 so the trace stands out
  // (was 0.18; bumped down because dense overlapping dashed vlines on a
  // 72-restart run alpha-blended back to near-baseline brightness).
  // If the user hovers an event LI ("trainer_started", "sigterm", "cluster
  // preempt", "death: preempt/cascade/failed", "annotations"), shapes
  // matching that event's color stay full opacity; the rest dim. With no
  // hover, all shapes render at full opacity.
  //
  // The active-color application is split into two phases so legend-hover
  // doesn't churn the `layout` prop on every mouse move (Bug 3 fix —
  // previously each hover rebuilt every shape's `line.color` with the
  // current `activeTraceName`, which made `layout` change reference and
  // triggered `Plotly.react` for every hover step). Now the shapes baked
  // into `layout` carry BASE colors (stable across `activeTraceName`); a
  // dedicated `useEffect` on `activeTraceName` recomputes the tinted
  // versions and pushes them via `Plotly.relayout({ shapes })`.
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
  // Each "base" shape pairs the canonical (un-tinted) color with the
  // currently-drawn color. `applyShapeFade` (below) walks `baseEventShapes`
  // to recompute `line.color` per-hover without re-touching layout.
  type BaseShape = Shape & { _baseColor: string }
  // The active-trace → tinted-color logic lives inline in `applyShapeFade`
  // (see the dedicated `useEffect` above). Old `activeShapeColor` /
  // `shapeColor` helpers were removed: that path was per-render and
  // dirtied the layout reference on every hover (Bug 3 root cause).
  const baseEventShapes: BaseShape[] = []
  const addShapes = (ts: number[], color: string, dash: 'dash' | 'dot' | 'solid') => {
    for (const t of ts) {
      const x = xOfTs(t)
      if (typeof x === 'number' && Number.isNaN(x)) continue
      baseEventShapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color, width: dash === 'solid' ? 1 : 1.2, dash },
        _baseColor: color,
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
  // Memoized so that data/layout/applyShapeFade deps see a stable ref when
  // the underlying inputs (annotations list, xMode, tsGstep mappings)
  // haven't actually changed. Without this, `xOfStep` returns the same
  // value but `annotationPositions` is a new array every render → memos
  // downstream invalidate every hover. eslint-disable-next-line because
  // `xOfStep` closes over many things that aren't in scope here; the
  // explicit deps below capture what really matters.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const annotationPositions = useMemo<AnnotationPos[]>(() => {
    const out: AnnotationPos[] = []
    for (const ann of annotations) {
      const x = xOfStep(ann.step)
      if (x === null || (typeof x === 'number' && Number.isNaN(x))) continue
      out.push({ ann, x })
    }
    return out
  }, [annotations, xMode, manifest, tsGstep, flopXScale])
  // Whether the "annotations" LI is toggled on. Drives both the dashed
  // vlines and the HTML info-icons rendered as an overlay.
  const [annotationsEnabled, setAnnotationsEnabled] = useState(true)
  if (annotationsEnabled) {
    for (const { ann, x } of annotationPositions) {
      const color = ann.color ?? ANNOTATION_COLOR
      baseEventShapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color, width: 0.8, dash: 'dash' },
        _baseColor: color,
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
  // Stash base shapes into the ref so `applyShapeFade` (defined above) can
  // recompute tinted colors on `activeTraceName` flips without re-running
  // the `addShapes` chain. Done during render so the ref always points at
  // the current shape set even when activeTraceName hasn't changed (other
  // re-renders, e.g. xMode swap, refresh the shape geometry).
  baseEventShapesRef.current = baseEventShapes

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
      if (typeof idx !== 'number') return undefined
      const name = e.data?.[idx]?.name
      if (!name) return undefined
      // Annotations LI gets its dedicated toggle — fall through to pltly's
      // default after flipping our flag so the dashed vlines actually
      // appear/disappear via the layout `shapes:` update.
      if (name === annotationsLegendName) {
        setAnnotationsEnabled((v) => !v)
        return undefined
      }
      // Every other LI (TL/VL/MT/MV/event traces) → toggle pin. Same-name
      // re-click unpins; different name switches pin. Return false to
      // suppress pltly's default click handler (which toggles trace
      // visibility + shows the "Double-click to isolate" toast — we're
      // commandeering legend-click semantics for pin-style highlighting).
      setPinnedTraceName((prev) => (prev === name ? null : name))
      return false
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
  // each panel uses the full available space. Memoized so that
  // `plotLayoutMemo` deps see stable array refs (the inline array literals
  // changed identity every render and defeated layout memoization).
  const [stepDomain, lossDomain, evalDomain] = useMemo<[
    [number, number] | null, [number, number], [number, number] | null,
  ]>(() => {
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
  }, [showTopPanel, showEvalPanel])

  const xTitle = xMode === 'time' ? TZ_LABEL
    : xMode === 'elapsed' ? 'elapsed (h)'
    : xMode === 'epoch' ? 'epoch'
    : xMode === 'flop' ? `FLOP (${flopUnit})`
    : 'global_step'

  // Memoize `data` so the reference stays stable across `activeTraceName`
  // flips that don't actually change trace content (Bug 3 fix — previously,
  // every hover step rebuilt the inline data array and pltly's chain
  // `styledData` → `finalData` → `dataWithSoloVisibility` → `plotData` saw
  // new refs at every level, triggering a `Plotly.react` per hover). The dep
  // list captures everything that ACTUALLY changes trace content; hover state
  // is intentionally absent.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const plotDataMemo = useMemo(() => [
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
    ...smoothedSeriesTraces(TL, 'TL (train loss)', COLORS.TL, 1.2, 'losses'),
    ...smoothedSeriesTraces(VL, 'VL (eval loss)', COLORS.VL, 1.4, 'losses'),
    ...(showEvalPanel ? evalTraces : []),
    ...(startTs.length > 0 ? [legendOnly(`trainer_started (${startTs.length})`, COLORS.start, 'dash')] : []),
    ...(sigtermTs.length > 0 ? [legendOnly(`sigterm (${sigtermTs.length})`, COLORS.sigterm, 'dot')] : []),
    ...(preemptTs.length > 0 ? [legendOnly(`cluster preempt (${preemptTs.length})`, COLORS.preempt, 'solid')] : []),
    ...(attempts ? (['preempt', 'cascade', 'failed'] as const)
      .filter((c) => deathBuckets[c].length > 0)
      .map((c) => legendOnly(
        `death: ${c} (${deathBuckets[c].length})`,
        DEATH_COLORS[c], 'solid',
      )) : []),
    ...(annotationPositions.length > 0 ? [legendOnly(annotationsLegendName, ANNOTATION_COLOR, 'dash')] : []),
  ], [
    showTopPanel, stepTrace, TL, VL, smooth, bandsOn, segments, lineageInfo,
    showEvalPanel, evalTraces,
    startTs, sigtermTs, preemptTs, attempts,
    annotationPositions, annotationsLegendName, ANNOTATION_COLOR, isDark,
  ])

  // Memoize `layout` similarly. `shapes:` carries BASE colors that don't
  // change with `activeTraceName` (the per-hover tint is applied in
  // `applyShapeFade` via `Plotly.relayout`, not by rebuilding the layout
  // prop). Without this, every hover dirtied the layout ref → `Plotly.react`
  // fired per hover step instead of per actual content change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const plotLayoutMemo = useMemo(() => ({
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
      type: (xMode === 'time' ? 'date' : 'linear') as 'date' | 'linear',
      ...(xMode === 'time' ? { tickformat: '%-m/%-d %H:%M' } : {}),
      ...(xMode === 'flop' ? { tickformat: flopTickformat(flopUnit) } : {}),
      gridcolor, zerolinecolor, linecolor: gridcolor,
      anchor: (showEvalPanel ? 'y3' : 'y2') as 'y3' | 'y2',
      ...(userXRange ? { range: userXRange, autorange: false } : {}),
    },
    yaxis: {
      title: { text: 'step', font: { color: COLORS.step } },
      tickfont: { color: COLORS.step },
      domain: stepDomain ?? [0.99, 1.0],
      gridcolor, zerolinecolor, linecolor: gridcolor,
      visible: showTopPanel,
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
      title: { text: `mat-${mevalMetric.toUpperCase()} %` },
      type: 'linear' as const,
      domain: evalDomain ?? [0.0, 0.01],
      gridcolor, zerolinecolor, linecolor: gridcolor,
      visible: showEvalPanel,
      fixedrange: true,
    },
    // shapes intentionally OMITTED here — `applyShapeFade` pushes them
    // via `Plotly.relayout({ shapes })` on render and on every active-
    // trace flip, decoupling shape churn from the layout-prop reference.
    // See the Bug 3 fix comment above.
    margin: { t: 50, l: 70, r: 210, b: 50 },
    hovermode: 'x unified' as const,
    // `hoversubplots: 'axis'` makes the x-unified tooltip span ALL subplots
    // sharing the x axis — TL/VL panel + MT/MV panel — so a single hover
    // box surfaces every trace the spikeline intersects. plotly default is
    // 'single' which restricts hover to the cursor's subplot only.
    hoversubplots: 'axis' as const,
    // `hoverdistance: 1` so only traces whose nearest data point is
    // essentially on the spikeline contribute to the unified tooltip.
    // The user explicitly asked for "exactly the traces the x-sparkline
    // intersects, none more none less" — a wider value re-introduces an
    // "overlap buffer" near restart cutovers where the parent's terminal
    // point (last x = N) leaks into the tooltip for cursors at N+ε.
    hoverdistance: 1,
    hoverlabel: { ...themedHoverlabel(isDark), align: 'left' as const },
    legend: {
      x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)',
      tracegroupgap: 10,
    },
  }), [
    runId, numSegments, startTs.length, sigtermTs.length, preemptTs.length,
    attempts, xTitle, xMode, flopUnit, gridcolor, zerolinecolor,
    showEvalPanel, userXRange, stepDomain, showTopPanel, lossDomain,
    mevalMetric, evalDomain, isDark,
  ])

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
      {/* CSS-enforced minHeight so the wrapper holds its layout space even if
          Plotly's auto-resize misfires (e.g. tab background → foreground:
          Chrome throttles rAF on hidden tabs; Plotly's ResizeObserver
          callback may not fire on return, collapsing the inner div to 0
          height. Without minHeight here, RecentEvents below reflows UP into
          the collapsed plot's space. Matches the plot's `height: 640` in
          plotLayoutMemo so the visual footprint is unchanged. */}
      <div ref={plotWrapperRef} style={{ position: 'relative', minHeight: 640 }}>
      <Plot
        onActiveTraceChange={setHoveredTraceName}
        onRelayout={onRelayout as (ev: unknown) => void}
        onAfterPlot={recomputeIconPositions}
        data={plotDataMemo}
        layout={plotLayoutMemo}
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
            content={(() => {
              // Inline the snap explanation when the annotation step was
              // snapped — we're already inside a Tooltip, so nesting another
              // would be janky.
              const sd = formatStepDetail(ann.step)
              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                    step {sd.display}
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>{ann.label}</div>
                  {sd.isLegacy && (
                    <div style={{ fontSize: '0.65rem', opacity: 0.55,
                                  borderTop: '1px solid rgba(255,255,255,0.1)',
                                  paddingTop: 3, marginTop: 2 }}>
                      {sd.tooltip}
                    </div>
                  )}
                </div>
              )
            })()}
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
