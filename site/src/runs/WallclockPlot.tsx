// Wallclock plot — step + loss over UTC time, with vertical markers at
// every lifecycle event. Port of the matplotlib `tomat runs wallclock-plot`.

import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'

interface Props {
  history: RunHistory
  runId: string
}

const COLORS = {
  step: '#2196f3',           // brighter blue
  loss: '#ef5350',           // brighter red
  start: '#ffa726',          // brighter orange
  sigterm: '#bdbdbd',        // lighter gray (more visible than 7f7f7f on dark)
  preempt: '#ba68c8',        // brighter purple
} as const

export function WallclockPlot({ history, runId }: Props) {
  const { isDark } = useTheme()
  const { timestamps, steps, cols } = history
  const toIso = (ts: number) => new Date(ts * 1000).toISOString()

  // Step curve: running-max of _step over _timestamp.
  // Why: wandb's `_step` is monotone in log-order, but events from our
  // lifecycle daemon-thread interleave out-of-order on the wall-time axis.
  // Plot the honest "how much training has been completed at time T".
  const indices = timestamps
    .map((ts, i) => ({ ts, s: steps[i], i }))
    .filter((r) => r.ts !== null && r.s !== null)
    .sort((a, b) => (a.ts as number) - (b.ts as number))
  const stepXs: string[] = []
  const stepYs: number[] = []
  let runningMax = -Infinity
  for (const { ts, s } of indices) {
    runningMax = Math.max(runningMax, s as number)
    stepXs.push(toIso(ts as number))
    stepYs.push(runningMax)
  }

  // train/loss (sparse).
  const trainLoss = cols.get('train/loss') ?? []
  const lossXs: string[] = []
  const lossYs: number[] = []
  for (let i = 0; i < trainLoss.length; i++) {
    const v = trainLoss[i]
    const ts = timestamps[i]
    if (v === null || ts === null) continue
    lossXs.push(toIso(ts))
    lossYs.push(v)
  }

  // Lifecycle events → as scatter "fake-vline" traces so they get legend
  // entries (shapes don't). Each event becomes a small null-separated
  // [x,x,null] segment with y in paper-coords via a dedicated invisible axis.
  const eventXs = (
    key: 'lifecycle/trainer_started' | 'lifecycle/sigterm_received',
  ): string[] => {
    const xs: string[] = []
    const col = cols.get(key) ?? []
    for (let i = 0; i < col.length; i++) {
      if (col[i] === 1) {
        const ts = timestamps[i]
        if (ts !== null) xs.push(toIso(ts))
      }
    }
    return xs
  }
  const startXs = eventXs('lifecycle/trainer_started')
  const sigtermXs = eventXs('lifecycle/sigterm_received')

  // cluster/preemptions jumps (sparse).
  const preemptCol = cols.get('cluster/preemptions') ?? []
  const preemptXs: string[] = []
  let prev: number | null = null
  for (let i = 0; i < preemptCol.length; i++) {
    const v = preemptCol[i]
    if (v === null) continue
    if (prev !== null && v > prev) {
      const ts = timestamps[i]
      if (ts !== null) preemptXs.push(toIso(ts))
    }
    prev = v
  }

  // Helper: build a single scatter trace that draws N vertical lines on
  // the hidden `yaxis3` (paper-anchored 0→1) so they don't fight with
  // the step/loss y-scales.
  const vlinesTrace = (xs: string[], name: string, color: string, dash: 'dash' | 'dot' | 'solid'):
    Partial<Plotly.PlotData> => {
    const xArr: (string | null)[] = []
    const yArr: (number | null)[] = []
    for (const x of xs) {
      xArr.push(x, x, null)
      yArr.push(0, 1, null)
    }
    return {
      x: xArr as string[],
      y: yArr as number[],
      mode: 'lines',
      type: 'scatter',
      name: `${name} (${xs.length})`,
      line: { color, width: dash === 'solid' ? 1 : 1.2, dash },
      yaxis: 'y3',
      hoverinfo: 'skip',
      showlegend: true,
      opacity: 0.85,
    }
  }

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'

  return (
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
        vlinesTrace(startXs, 'trainer_started', COLORS.start, 'dash'),
        vlinesTrace(sigtermXs, 'sigterm', COLORS.sigterm, 'dot'),
        vlinesTrace(preemptXs, 'cluster preempt', COLORS.preempt, 'solid'),
      ]}
      layout={{
        title: {
          text: `${runId}  ·  ${startXs.length} starts, ${sigtermXs.length} sigterms, ${preemptXs.length} preempts`,
          font: { size: 14 },
        },
        autosize: true,
        height: 460,
        // Gridlines: render BEHIND traces + dim color.
        // Plotly draws gridlines in the order axes are declared; setting
        // layer='below traces' keeps them out of the way.
        xaxis: {
          title: { text: 'UTC' },
          type: 'date',
          gridcolor,
          zerolinecolor,
          linecolor: gridcolor,
          layer: 'below traces',
        },
        yaxis: {
          title: { text: 'step', font: { color: COLORS.step } },
          tickfont: { color: COLORS.step },
          gridcolor,
          zerolinecolor,
          linecolor: gridcolor,
          layer: 'below traces',
        },
        yaxis2: {
          title: { text: 'train/loss', font: { color: COLORS.loss } },
          tickfont: { color: COLORS.loss },
          overlaying: 'y',
          side: 'right',
          type: 'log',
          gridcolor: 'rgba(0,0,0,0)',     // y2 doesn't draw its own grid (avoid double lines)
          zerolinecolor: 'rgba(0,0,0,0)',
          layer: 'below traces',
        },
        // Hidden axis for event vlines so they span 0→1 paper-coords
        // without affecting the visible y-axes.
        yaxis3: {
          overlaying: 'y',
          range: [0, 1],
          showgrid: false,
          showticklabels: false,
          zeroline: false,
          visible: false,
        },
        margin: { t: 50, l: 60, r: 60, b: 50 },
        hovermode: 'x unified',
        hoverlabel: themedHoverlabel(isDark),
        legend: { x: 1.04, y: 1, bgcolor: 'rgba(0,0,0,0)' },
      }}
    />
  )
}
