// Per-run plot — 2 stacked subplots over a shared x-axis (wallclock UTC by
// default; toggleable to gstep):
//   1. step (Levanter's `global_step`, running max — top, 28%). Hidden in
//      step mode (would degenerate to y=x).
//   2. losses + NMAE on a single log y-axis (bottom):
//        TL, VL, and MV/MT × {mean, median, p99}.
// Lifecycle events (trainer_started / sigterm / cluster preempt) render as
// vertical lines via `shapes` (yref='paper') so they span both panels.
//
// MT (train_200) traces are dashed; MV (val_200) traces solid. NMAE stat is
// encoded by shade: mean (lime) → median (mid green) → p99 (dark green).
// NMAE traces include markers because eval logging is sparse (~17 entries
// per run); a pure line trace would be invisible between widely-spaced rows.
//
// Eval rows (MV/MT/VL) lack `global_step` in wandb, so in step mode we
// back-fill it from the nearest preceding TL row by `_timestamp`.

import { useMemo, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'

interface Props {
  history: RunHistory
  runId: string
}

type XMode = 'time' | 'step'

const COLORS = {
  step: '#2196f3',
  TL: '#ef5350',     // train/loss
  VL: '#ffa726',     // eval/loss
  // NMAE shades, light → dark = mean → median → p99
  mean: '#9ccc65',
  median: '#43a047',
  p99: '#1b5e20',
  start: '#ffa726',
  sigterm: '#bdbdbd',
  preempt: '#ba68c8',
} as const

export function WallclockPlot({ history, runId }: Props) {
  const { isDark } = useTheme()
  const [xMode, setXMode] = useState<XMode>('time')
  const { timestamps, cols } = history
  const toIso = (ts: number) => new Date(ts * 1000).toISOString()

  const ordered = useMemo(
    () => timestamps
      .map((ts, i) => ({ ts, i }))
      .filter((r) => r.ts !== null)
      .sort((a, b) => (a.ts as number) - (b.ts as number)),
    [timestamps],
  )

  // (ts → gstep) map: derived from any row that has a `global_step`. Used to
  // back-fill the gstep for eval/lifecycle rows that wandb didn't tag.
  const tsToGstep = useMemo(() => {
    const globalStep = cols.get('global_step') ?? []
    const pairs: { ts: number; gstep: number }[] = []
    for (const { ts, i } of ordered) {
      const g = globalStep[i]
      if (g === null || g === undefined) continue
      pairs.push({ ts: ts as number, gstep: g as number })
    }
    return pairs
  }, [ordered, cols])

  function gstepAtTs(ts: number): number | null {
    if (tsToGstep.length === 0) return null
    // Binary search for largest ts <= target.
    let lo = 0, hi = tsToGstep.length - 1, best = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (tsToGstep[mid].ts <= ts) { best = mid; lo = mid + 1 }
      else hi = mid - 1
    }
    if (best < 0) return tsToGstep[0]?.gstep ?? null
    return tsToGstep[best].gstep
  }

  // Per-metric series: xs (date strings or gstep numbers depending on mode),
  // ys, and gsteps (used for the hover tooltip in either mode).
  type Series = { xs: (string | number)[]; ys: number[]; gsteps: (number | null)[] }
  function series(key: string): Series {
    const col = cols.get(key) ?? []
    const xs: (string | number)[] = []
    const ys: number[] = []
    const gsteps: (number | null)[] = []
    for (const { ts, i } of ordered) {
      const v = col[i]
      if (v === null || v === undefined) continue
      const g = gstepAtTs(ts as number)
      gsteps.push(g)
      xs.push(xMode === 'time' ? toIso(ts as number) : (g ?? NaN))
      ys.push(v as number)
    }
    return { xs, ys, gsteps }
  }

  // Step (top-panel) trace: running-max of global_step.
  const stepTrace = useMemo(() => {
    const globalStep = cols.get('global_step') ?? []
    const xs: string[] = []
    const ys: number[] = []
    let runningMax = -Infinity
    for (const { ts, i } of ordered) {
      const s = globalStep[i]
      if (s === null) continue
      runningMax = Math.max(runningMax, s)
      xs.push(toIso(ts as number))
      ys.push(runningMax)
    }
    return { xs, ys }
  }, [ordered, cols])

  const TL = series('train/loss')
  const VL = series('eval/loss')
  const MVmean = series('eval/mat_nmae/val_200/mean')
  const MVmed = series('eval/mat_nmae/val_200/median')
  const MVp99 = series('eval/mat_nmae/val_200/p99')
  const MTmean = series('eval/mat_nmae/train_200/mean')
  const MTmed = series('eval/mat_nmae/train_200/median')
  const MTp99 = series('eval/mat_nmae/train_200/p99')

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
      const x: string | number = xMode === 'time' ? toIso(t) : (gstepAtTs(t) ?? NaN)
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

  const showTopPanel = xMode === 'time'
  // gstep gets shown as customdata for the tooltip in both x-modes.
  const customGsteps = (s: Series) => s.gsteps.map((g) => (g === null ? '?' : g))

  const tlHover = xMode === 'time'
    ? 'TL %{y:.3f}<br>gstep %{customdata}<extra></extra>'
    : 'TL %{y:.3f}<br>gstep %{x}<extra></extra>'
  const vlHover = xMode === 'time'
    ? 'VL %{y:.3f}<br>gstep %{customdata}<extra></extra>'
    : 'VL %{y:.3f}<br>gstep %{x}<extra></extra>'
  const nmaeHoverTmpl = (name: string) =>
    xMode === 'time'
      ? `${name} %{y:.2f}%%<br>gstep %{customdata}<extra></extra>`
      : `${name} %{y:.2f}%%<br>gstep %{x}<extra></extra>`

  const nmaeTrace = (name: string, s: Series, color: string, dash: 'solid' | 'dash') => ({
    x: s.xs, y: s.ys, name,
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    line: { color, width: 1.5, dash },
    marker: { color, size: 5 },
    yaxis: 'y2',
    customdata: customGsteps(s),
    hovertemplate: nmaeHoverTmpl(name),
  })

  const logType: 'log' = 'log'

  // Top-panel domain depends on whether we render it.
  const lossDomain: [number, number] = showTopPanel ? [0.0, 0.66] : [0.0, 1.0]
  const stepDomain: [number, number] = [0.72, 1.0]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.4rem', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#888', alignSelf: 'center' }}>x-axis:</span>
        {(['time', 'step'] as XMode[]).map((m) => (
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
            {m === 'time' ? 'wallclock' : 'step'}
          </button>
        ))}
      </div>
      <Plot
        data={[
          // 1. step (top panel) — only when in time mode
          ...(showTopPanel ? [{
            x: stepTrace.xs, y: stepTrace.ys, name: 'step',
            type: 'scatter' as const, mode: 'lines' as const,
            line: { color: COLORS.step, width: 2, shape: 'hv' as const },
            yaxis: 'y',
            hovertemplate: '%{x|%Y-%m-%d %H:%M:%S}<br>step %{y}<extra></extra>',
          }] : []),
          // 2. losses + NMAE (shared log y2)
          {
            x: TL.xs, y: TL.ys, name: 'TL (train/loss)',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.TL, width: 1.2 },
            yaxis: 'y2',
            customdata: customGsteps(TL),
            hovertemplate: tlHover,
          },
          {
            x: VL.xs, y: VL.ys, name: 'VL (eval/loss)',
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.VL, width: 1.4 },
            yaxis: 'y2',
            customdata: customGsteps(VL),
            hovertemplate: vlHover,
          },
          nmaeTrace('MV mean',   MVmean, COLORS.mean,   'solid'),
          nmaeTrace('MV median', MVmed,  COLORS.median, 'solid'),
          nmaeTrace('MV p99',    MVp99,  COLORS.p99,    'solid'),
          nmaeTrace('MT mean',   MTmean, COLORS.mean,   'dash'),
          nmaeTrace('MT median', MTmed,  COLORS.median, 'dash'),
          nmaeTrace('MT p99',    MTp99,  COLORS.p99,    'dash'),
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
            title: { text: xMode === 'time' ? 'UTC' : 'global_step' },
            type: xMode === 'time' ? 'date' : 'linear',
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
            title: { text: 'loss / NMAE % (log)' },
            type: logType,
            domain: lossDomain,
            gridcolor, zerolinecolor, linecolor: gridcolor,
          },
          shapes: eventShapes,
          margin: { t: 50, l: 70, r: 160, b: 50 },
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          legend: { x: 1.02, y: 1, bgcolor: 'rgba(0,0,0,0)' },
        }}
      />
    </div>
  )
}
