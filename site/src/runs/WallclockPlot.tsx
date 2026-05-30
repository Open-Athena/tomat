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
import { enumParam, useUrlState } from 'use-prms'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'
import type { RunEval } from './api'
import { SmoothingChips, useBandsToggle, useSmoothMode } from './RunsTimelinePlot'
import { applySmoothing, smoothingStd } from './smoothing'

interface Props {
  history: RunHistory
  evalSeries: RunEval | null
  runId: string
  /** Initial x-axis mode when the URL has no `?x=…`. Defaults to `'step'`
   *  for the run-detail page (training-progress is the obvious x for a
   *  single-run view); callers using this on a multi-run context can
   *  override to `'wallclock'` or `'elapsed'`. */
  defaultXMode?: UrlXMode
}

type XMode = 'time' | 'elapsed' | 'step'

// URL-facing x-axis mode names (`?x=wallclock|elapsed|step`). The internal
// XMode uses `'time'` for the wallclock axis; `wallclock` reads better in
// shared links.
type UrlXMode = 'wallclock' | 'elapsed' | 'step'
const URL_X_MODES = ['wallclock', 'elapsed', 'step'] as const
const X_TO_URL: Record<XMode, UrlXMode> = { time: 'wallclock', elapsed: 'elapsed', step: 'step' }
const URL_TO_X: Record<UrlXMode, XMode> = { wallclock: 'time', elapsed: 'elapsed', step: 'step' }

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

export function WallclockPlot({ history, evalSeries, runId, defaultXMode = 'step' }: Props) {
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
      if (t.showlegend !== false) continue
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
  // `?x=wallclock|elapsed|step` — URL-persisted so deep-links carry the view
  // choice. The run-detail page defaults to `'step'` (training progress is
  // the obvious x for a single run); callers can override.
  const [urlXMode, setUrlXMode] = useUrlState('x', enumParam<UrlXMode>(defaultXMode, URL_X_MODES))
  const xMode: XMode = URL_TO_X[urlXMode]
  const setXMode = (m: XMode) => setUrlXMode(X_TO_URL[m])
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
    if (xMode === 'elapsed') return (ts - t0) / 3600
    return toLocal(ts)
  }

  /** x-coordinate for an eval point at checkpoint `step`. */
  function xOfStep(step: number): string | number | null {
    if (xMode === 'step') return step
    const ts = tsAtGstep(step)
    if (ts === null) return null
    return xMode === 'elapsed' ? (ts - t0) / 3600 : toLocal(ts)
  }

  // Per-metric parquet series (TL/VL): xs, ys, gsteps (gstep for the tooltip).
  type Series = { xs: (string | number)[]; ys: number[]; gsteps: (number | null)[] }
  function series(key: string): Series {
    const col = cols.get(key) ?? []
    const xs: (string | number)[] = []
    const ys: number[] = []
    const gsteps: (number | null)[] = []
    for (const { ts, i } of ordered) {
      const v = col[i]
      if (v === null || v === undefined) continue
      gsteps.push(gstepAtTs(ts as number))
      xs.push(xOfTs(ts as number))
      ys.push(v as number)
    }
    return { xs, ys, gsteps }
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
  const [smooth, setSmooth] = useSmoothMode()
  const [bandsOn, setBandsOn] = useBandsToggle()
  const wantSmooth = smooth.kind !== 'raw'

  // Smooth a parquet-derived Series in place + (optionally) build paired
  // ±σ band traces. Bands inherit the line's `name` (so the closest-trace
  // matching logic upstream remains correct) and a shared `legendgroup` so
  // pltly's solo/fade + our `applyBandFade` desaturate them together.
  type SmoothedTrace = Record<string, unknown>
  function smoothedSeriesTraces(
    s: Series, name: string, color: string, lineWidth: number, lg: string,
  ): SmoothedTrace[] {
    const ySmoothed = applySmoothing(s.ys, smooth)
    const lineTrace: SmoothedTrace = {
      x: s.xs, y: ySmoothed, name,
      type: 'scatter', mode: 'lines',
      line: { color, width: lineWidth },
      yaxis: 'y2',
      legendgroup: lg,
      customdata: customGsteps(s),
      hovertemplate: `${name} %{y:.3f}<br>gstep %{customdata}<extra></extra>`,
    }
    if (!wantSmooth || !bandsOn) return [lineTrace]
    const yStd = smoothingStd(s.ys, smooth)
    if (!yStd) return [lineTrace]
    const yLower: (number | null)[] = []
    const yUpper: (number | null)[] = []
    for (let i = 0; i < ySmoothed.length; i++) {
      const m = ySmoothed[i]
      const sd = yStd[i]
      if (m == null || sd == null) { yLower.push(null); yUpper.push(null); continue }
      // y2 is log-scaled — keep band edges strictly positive so plotly doesn't
      // drop them (`log(<=0) = NaN`). 1e-6 is well below any real loss value.
      yLower.push(Math.max(1e-6, m - sd))
      yUpper.push(m + sd)
    }
    const edge = (y: (number | null)[], fillKey: string | null): SmoothedTrace => ({
      x: s.xs, y, name,
      type: 'scatter', mode: 'lines',
      line: { width: 0, color: 'rgba(0,0,0,0)' },
      yaxis: 'y2', legendgroup: lg,
      showlegend: false,
      hoverinfo: 'skip',
      ...(fillKey ? { fill: 'tonexty', fillcolor: fillKey } : {}),
    })
    // Match the eval-band convention (alpha 0.20 for the inner ribbon).
    const rgb = hexToRgbTuple(color)
    const fillcolor = `rgba(${rgb}, 0.18)`
    return [edge(yLower, null), edge(yUpper, fillcolor), lineTrace]
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
    const medianSmoothed = applySmoothing(pct.median, smooth)
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
  const eventShapes: Shape[] = []
  const addShapes = (ts: number[], color: string, dash: 'dash' | 'dot' | 'solid') => {
    for (const t of ts) {
      const x = xOfTs(t)
      if (typeof x === 'number' && Number.isNaN(x)) continue
      eventShapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color, width: dash === 'solid' ? 1 : 1.2, dash },
      })
    }
  }
  addShapes(startTs, COLORS.start, 'dash')
  addShapes(sigtermTs, COLORS.sigterm, 'dot')
  addShapes(preemptTs, COLORS.preempt, 'solid')

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

  const showTopPanel = xMode !== 'step'

  const logType: 'log' = 'log'
  const lossDomain: [number, number] = showTopPanel ? [0.0, 0.66] : [0.0, 1.0]
  const stepDomain: [number, number] = [0.72, 1.0]

  const xTitle = xMode === 'time' ? TZ_LABEL
    : xMode === 'elapsed' ? 'elapsed (h)' : 'global_step'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#888', alignSelf: 'center' }}>x-axis:</span>
        {(['time', 'elapsed', 'step'] as XMode[]).map((m) => (
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
          legendOnly(`trainer_started (${startTs.length})`, COLORS.start, 'dash'),
          legendOnly(`sigterm (${sigtermTs.length})`, COLORS.sigterm, 'dot'),
          legendOnly(`cluster preempt (${preemptTs.length})`, COLORS.preempt, 'solid'),
        ]}
        layout={{
          title: {
            text: `${runId}  ·  ${startTs.length} starts, ${sigtermTs.length} sigterms, ${preemptTs.length} preempts`,
            font: { size: 14 },
          },
          autosize: true,
          height: 640,
          xaxis: {
            title: { text: xTitle },
            type: xMode === 'time' ? 'date' : 'linear',
            ...(xMode === 'time' ? { tickformat: '%-m/%-d %H:%M' } : {}),
            gridcolor, zerolinecolor, linecolor: gridcolor,
          },
          yaxis: {
            title: { text: 'step', font: { color: COLORS.step } },
            tickfont: { color: COLORS.step },
            domain: stepDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
            visible: showTopPanel,
          },
          yaxis2: {
            title: { text: 'loss / NMAE·NEMD % (log)' },
            type: logType,
            domain: lossDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
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
