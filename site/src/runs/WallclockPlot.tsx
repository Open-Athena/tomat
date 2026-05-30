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

import { useMemo } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { enumParam, useUrlState } from 'use-prms'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'
import type { RunEval } from './api'

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

  const TL = series('train/loss')
  const VL = series('eval/loss')

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
    return [
      edge('p1', null),     // lower edge of the p1–p99 band
      edge('p99', 0.09),    // upper edge → fills down to p1
      edge('p25', null),    // lower edge of the IQR band
      edge('p75', 0.20),    // upper edge → fills down to p25
      {                     // median — the legend-bearing center line
        x: xs, y: pct.median, name,
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
  }, [evalSeries, xMode])

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
  const customGsteps = (s: Series) => s.gsteps.map((g) => (g === null ? '?' : g))

  const logType: 'log' = 'log'
  const lossDomain: [number, number] = showTopPanel ? [0.0, 0.66] : [0.0, 1.0]
  const stepDomain: [number, number] = [0.72, 1.0]

  const xTitle = xMode === 'time' ? TZ_LABEL
    : xMode === 'elapsed' ? 'elapsed (h)' : 'global_step'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.4rem', marginBottom: '0.3rem' }}>
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
      </div>
      <Plot
        data={[
          // 1. step (top panel) — only when not in step mode
          ...(showTopPanel ? [{
            x: stepTrace.xs, y: stepTrace.ys, name: 'step',
            type: 'scatter' as const, mode: 'lines' as const,
            line: { color: COLORS.step, width: 2, shape: 'hv' as const },
            yaxis: 'y',
            hovertemplate: 'step %{y}<extra></extra>',
          }] : []),
          // 2. losses (shared log y2)
          {
            x: TL.xs, y: TL.ys, name: 'TL (train/loss)',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.TL, width: 1.2 },
            yaxis: 'y2',
            customdata: customGsteps(TL),
            hovertemplate: 'TL %{y:.3f}<br>gstep %{customdata}<extra></extra>',
          },
          {
            x: VL.xs, y: VL.ys, name: 'VL (eval/loss)',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.VL, width: 1.4 },
            yaxis: 'y2',
            customdata: customGsteps(VL),
            hovertemplate: 'VL %{y:.3f}<br>gstep %{customdata}<extra></extra>',
          },
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
  )
}
