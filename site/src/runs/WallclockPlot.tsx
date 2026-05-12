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
  step: '#1f77b4',
  loss: '#d62728',
  start: '#ff7f0e',
  sigterm: '#7f7f7f',
  preempt: '#9467bd',
} as const

export function WallclockPlot({ history, runId }: Props) {
  const { isDark } = useTheme()
  const { timestamps, steps, cols } = history
  const toIso = (ts: number) => new Date(ts * 1000).toISOString()

  // Step curve (every row that has a step + a timestamp).
  const stepXs: string[] = []
  const stepYs: number[] = []
  for (let i = 0; i < timestamps.length; i++) {
    const ts = timestamps[i]
    const s = steps[i]
    if (ts === null || s === null) continue
    stepXs.push(toIso(ts))
    stepYs.push(s)
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

  // Lifecycle events → vertical shapes.
  const eventXs = (key: keyof Map<string, unknown> | string): string[] => {
    const xs: string[] = []
    const col = cols.get(key as never) ?? []
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

  // cluster/preemptions jumps.
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

  const eventShapes = [
    ...startXs.map((x) => vshape(x, COLORS.start, 'dash')),
    ...sigtermXs.map((x) => vshape(x, COLORS.sigterm, 'dot')),
    ...preemptXs.map((x) => vshape(x, COLORS.preempt, 'solid')),
  ]

  return (
    <Plot
      data={[
        {
          x: stepXs,
          y: stepYs,
          name: 'step',
          type: 'scatter',
          mode: 'lines',
          line: { color: COLORS.step, width: 1.5 },
          yaxis: 'y',
          hovertemplate: '%{x}<br>step %{y}<extra></extra>',
        },
        {
          x: lossXs,
          y: lossYs,
          name: 'train/loss',
          type: 'scatter',
          mode: 'lines',
          opacity: 0.4,
          line: { color: COLORS.loss, width: 0.6 },
          yaxis: 'y2',
          hovertemplate: 'loss %{y:.3f}<extra></extra>',
        },
      ]}
      layout={{
        title: {
          text: `${runId}  ·  ${startXs.length} starts, ${sigtermXs.length} sigterms, ${preemptXs.length} preempts`,
          font: { size: 14 },
        },
        autosize: true,
        height: 460,
        xaxis: { title: { text: 'UTC' }, type: 'date' },
        yaxis: {
          title: { text: 'step', font: { color: COLORS.step } },
          tickfont: { color: COLORS.step },
        },
        yaxis2: {
          title: { text: 'train/loss', font: { color: COLORS.loss } },
          tickfont: { color: COLORS.loss },
          overlaying: 'y',
          side: 'right',
          type: 'log',
        },
        shapes: eventShapes,
        margin: { t: 50, l: 60, r: 60, b: 40 },
        hovermode: 'x unified',
        hoverlabel: themedHoverlabel(isDark),
        legend: { x: 1.05, y: 1 },
      }}
    />
  )
}

function vshape(x: string, color: string, dash: 'solid' | 'dash' | 'dot') {
  return {
    type: 'line' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    line: { color, width: dash === 'solid' ? 0.6 : 0.9, dash },
    opacity: dash === 'dot' ? 0.5 : 0.7,
  }
}
