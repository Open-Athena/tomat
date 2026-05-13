// Wallclock plot — step + loss over UTC time, with vertical markers (or
// cumulative-count overlays) at every lifecycle / cluster event.
// Port of the matplotlib `tomat runs wallclock-plot`.

import { useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'

interface Props {
  history: RunHistory
  runId: string
}

const COLORS = {
  step: '#2196f3',
  loss: '#ef5350',
  start: '#ffa726',
  sigterm: '#bdbdbd',
  preempt: '#ba68c8',
} as const

type EventMode = 'vlines' | 'cumsum'

export function WallclockPlot({ history, runId }: Props) {
  const { isDark } = useTheme()
  const [eventMode, setEventMode] = useState<EventMode>('vlines')

  const { timestamps, cols } = history
  const toIso = (ts: number) => new Date(ts * 1000).toISOString()

  // Build a time-ordered index used by every series — wandb history rows
  // can be timestamp-out-of-order because the lifecycle daemon-thread logs
  // interleave with the trainer's main thread.
  const ordered = timestamps
    .map((ts, i) => ({ ts, i }))
    .filter((r) => r.ts !== null)
    .sort((a, b) => (a.ts as number) - (b.ts as number))

  // Step curve: running max of `global_step` (Levanter's actual training
  // step) along ascending `_timestamp`. Was previously `_step` (wandb's
  // log-call counter) which leaps non-physically during restart-burst
  // log-sync. global_step may be null on rows that aren't from the
  // trainer's main loop (e.g. lifecycle/sigterm logs) — skip those.
  const globalStep = cols.get('global_step') ?? []
  const stepXs: string[] = []
  const stepYs: number[] = []
  let runningMax = -Infinity
  for (const { ts, i } of ordered) {
    const s = globalStep[i]
    if (s === null) continue
    runningMax = Math.max(runningMax, s)
    stepXs.push(toIso(ts as number))
    stepYs.push(runningMax)
  }

  // train/loss — time-ordered (was previously in _step order, producing
  // a back-in-time connector segment at the right edge).
  const trainLoss = cols.get('train/loss') ?? []
  const lossXs: string[] = []
  const lossYs: number[] = []
  for (const { ts, i } of ordered) {
    const v = trainLoss[i]
    if (v === null) continue
    lossXs.push(toIso(ts as number))
    lossYs.push(v)
  }

  // Lifecycle event timestamps (sorted ascending).
  const eventTimes = (key: 'lifecycle/trainer_started' | 'lifecycle/sigterm_received'): number[] => {
    const ts: number[] = []
    const col = cols.get(key) ?? []
    for (let i = 0; i < col.length; i++) {
      if (col[i] === 1 && timestamps[i] !== null) ts.push(timestamps[i] as number)
    }
    return ts.sort((a, b) => a - b)
  }
  const startTs = eventTimes('lifecycle/trainer_started')
  const sigtermTs = eventTimes('lifecycle/sigterm_received')

  // cluster/preemptions jumps in time-order (sparse).
  const preemptCol = cols.get('cluster/preemptions') ?? []
  const preemptTs: number[] = []
  let prev: number | null = null
  for (const { ts, i } of ordered) {
    const v = preemptCol[i]
    if (v === null) continue
    if (prev !== null && v > prev) preemptTs.push(ts as number)
    prev = v
  }

  // Build per-mode traces for the 3 event series.
  const eventTraces: Partial<Plotly.PlotData>[] = []
  const addEventSeries = (
    ts: number[],
    name: string,
    color: string,
    dash: 'dash' | 'dot' | 'solid',
  ) => {
    if (eventMode === 'vlines') {
      // null-separated [x,x] segments on hidden yaxis3 (paper coords 0→1).
      const xArr: (string | null)[] = []
      const yArr: (number | null)[] = []
      for (const t of ts) {
        const x = toIso(t)
        xArr.push(x, x, null)
        yArr.push(0, 1, null)
      }
      eventTraces.push({
        x: xArr as string[],
        y: yArr as number[],
        mode: 'lines',
        type: 'scatter',
        name: `${name} (${ts.length})`,
        line: { color, width: dash === 'solid' ? 1 : 1.2, dash },
        yaxis: 'y3',
        hoverinfo: 'skip',
        showlegend: true,
        opacity: 0.85,
      })
    } else {
      // cumsum: staircase line, y = cumulative count on yaxis3 (auto-scaled).
      const xs = ts.map((t) => toIso(t))
      const ys = ts.map((_, i) => i + 1)
      eventTraces.push({
        x: xs,
        y: ys,
        mode: 'lines',
        type: 'scatter',
        name: `${name} (${ts.length})`,
        line: { color, width: 2, dash, shape: 'hv' },
        yaxis: 'y3',
        hovertemplate: `${name} #%{y}<br>%{x|%H:%M:%S}<extra></extra>`,
        showlegend: true,
        opacity: 0.95,
      })
    }
  }
  addEventSeries(startTs, 'trainer_started', COLORS.start, 'dash')
  addEventSeries(sigtermTs, 'sigterm', COLORS.sigterm, 'dot')
  addEventSeries(preemptTs, 'cluster preempt', COLORS.preempt, 'solid')

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'

  // yaxis3: hidden in vlines mode (just used to anchor 0→1), visible+
  // labeled in cumsum mode so the event counts are readable.
  const y3CumMax = Math.max(startTs.length, sigtermTs.length, preemptTs.length, 1)
  const yaxis3 =
    eventMode === 'vlines'
      ? { overlaying: 'y', range: [0, 1], showgrid: false, showticklabels: false, zeroline: false, visible: false }
      : {
          overlaying: 'y',
          side: 'right',
          range: [0, y3CumMax * 1.05],
          showgrid: false,
          zeroline: false,
          tickfont: { color: '#9e9e9e' },
          title: { text: 'event cumsum', font: { color: '#9e9e9e' } },
          // shifted further out so it doesn't overlap yaxis2.
          anchor: 'free',
          position: 0.97,
        }

  return (
    <>
      <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center',
                    fontSize: '0.85rem', color: '#888', marginBottom: '0.3rem' }}>
        <span>events:</span>
        <label style={{ cursor: 'pointer' }}>
          <input type="radio" name="eventMode" value="vlines"
            checked={eventMode === 'vlines'}
            onChange={() => setEventMode('vlines')} /> vlines
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input type="radio" name="eventMode" value="cumsum"
            checked={eventMode === 'cumsum'}
            onChange={() => setEventMode('cumsum')} /> cumsum
        </label>
      </div>
      <Plot
        data={[
          {
            x: stepXs,
            y: stepYs,
            name: 'step',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.step, width: 2.2, shape: 'hv' },
            yaxis: 'y',
            hovertemplate: '%{x|%Y-%m-%d %H:%M:%S}<br>step %{y}<extra></extra>',
          },
          {
            x: lossXs,
            y: lossYs,
            name: 'train/loss',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.loss, width: 1.2 },
            opacity: 0.9,
            yaxis: 'y2',
            hovertemplate: 'loss %{y:.3f}<extra></extra>',
          },
          ...eventTraces,
        ]}
        layout={{
          title: {
            text: `${runId}  ·  ${startTs.length} starts, ${sigtermTs.length} sigterms, ${preemptTs.length} preempts`,
            font: { size: 14 },
          },
          autosize: true,
          height: 460,
          xaxis: {
            title: { text: 'UTC' }, type: 'date',
            gridcolor, zerolinecolor, linecolor: gridcolor,
            layer: 'below traces',
          },
          yaxis: {
            title: { text: 'step', font: { color: COLORS.step } },
            tickfont: { color: COLORS.step },
            gridcolor, zerolinecolor, linecolor: gridcolor,
            layer: 'below traces',
          },
          yaxis2: {
            title: { text: 'train/loss', font: { color: COLORS.loss } },
            tickfont: { color: COLORS.loss },
            overlaying: 'y', side: 'right', type: 'log',
            gridcolor: 'rgba(0,0,0,0)', zerolinecolor: 'rgba(0,0,0,0)',
            layer: 'below traces',
          },
          yaxis3,
          margin: { t: 50, l: 60, r: eventMode === 'cumsum' ? 110 : 60, b: 50 },
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          legend: { x: 1.08, y: 1, bgcolor: 'rgba(0,0,0,0)' },
        }}
      />
    </>
  )
}
