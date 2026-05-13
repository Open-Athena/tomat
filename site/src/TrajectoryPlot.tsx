// Interactive port of `scripts/plot_nmae_nemd_trajectory.py`: NMAE (top) +
// NEMD (bottom) over training step, one color per run, solid=val/dashed=train,
// star markers at each run's best val ckpt, vertical reference lines at the
// Chinchilla-optimal step + 1 epoch.
//
// Data shape produced by the script's `--json-out`:
//   { schema_version, chinchilla_step, epoch_step,
//     runs: [{ label, full, splits: { val: { steps, nmae, nemd, nmae_p99 },
//                                      train: { …same shape… } } }, …] }

import { useEffect, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from './theme'

interface SplitSeries {
  steps: number[]
  nmae: (number | null)[]
  nemd: (number | null)[]
  nmae_p99: (number | null)[]
}

interface Run {
  label: string
  full: string
  splits: { val: SplitSeries; train: SplitSeries }
}

interface TrajectoryData {
  schema_version: number
  chinchilla_step: number
  epoch_step: number
  runs: Run[]
}

// Matplotlib tab10 (the script's default cycle), matched so colors are
// stable when the static PNG and the live plot coexist on the page.
const TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

function bestStep(s: SplitSeries, key: 'nmae' | 'nemd'): { step: number; value: number } | null {
  let best: { step: number; value: number } | null = null
  for (let i = 0; i < s.steps.length; i++) {
    const v = s[key][i]
    if (v == null) continue
    const pct = v * 100
    if (best === null || pct < best.value) best = { step: s.steps[i], value: pct }
  }
  return best
}

interface Props {
  url: string
}

export function TrajectoryPlot({ url }: Props) {
  const { isDark } = useTheme()
  const [data, setData] = useState<TrajectoryData | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setData)
      .catch((e) => setErr(String(e)))
  }, [url])

  if (err) return <p style={{ color: 'crimson' }}>trajectories: {err}</p>
  if (!data) return <p style={{ color: '#888' }}>loading trajectories…</p>

  // Per-(run × split × metric) trace + a best-val star per run per metric.
  const traces: Partial<Plotly.PlotData>[] = []
  for (let i = 0; i < data.runs.length; i++) {
    const r = data.runs[i]
    const color = TAB10[i % TAB10.length]
    for (const [split, dash, opacity] of [['val', 'solid', 1.0], ['train', 'dash', 0.7]] as const) {
      const s = r.splits[split]
      for (const [metric, axis] of [['nmae', 'y'], ['nemd', 'y2']] as const) {
        const ys = s[metric].map((v) => (v == null ? null : v * 100))
        if (ys.every((v) => v == null)) continue
        traces.push({
          x: s.steps,
          y: ys as number[],
          name: `${r.label} ${split}`,
          legendgroup: `${r.label}-${split}`,
          showlegend: metric === 'nmae',     // legend item once per run+split (the NMAE panel)
          mode: 'lines+markers',
          type: 'scatter',
          line: { color, dash, width: 2 },
          marker: { color, size: split === 'val' ? 6 : 4 },
          opacity,
          yaxis: axis,
          xaxis: metric === 'nmae' ? 'x' : 'x2',
          hovertemplate: `${r.label} ${split} step %{x}<br>${metric.toUpperCase()} %{y:.2f}%<extra></extra>`,
        })
      }
    }
    // Best-val markers on both panels.
    for (const [metric, axis] of [['nmae', 'y'], ['nemd', 'y2']] as const) {
      const best = bestStep(r.splits.val, metric)
      if (!best) continue
      traces.push({
        x: [best.step],
        y: [best.value],
        name: `best ${r.label}`,
        showlegend: false,
        mode: 'markers+text',
        type: 'scatter',
        marker: { symbol: 'star', size: 16, color, line: { color: '#000', width: 0.7 } },
        text: [`step-${best.step}<br>${best.value.toFixed(2)}%`],
        textposition: 'top right',
        textfont: { color, size: 10 },
        yaxis: axis,
        xaxis: metric === 'nmae' ? 'x' : 'x2',
        hoverinfo: 'skip',
      })
    }
  }

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'

  // Two stacked panels: top (NMAE) shares x with bottom (NEMD).
  const layout: Partial<Plotly.Layout> = {
    autosize: true,
    height: 560,
    grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    margin: { t: 30, l: 56, r: 20, b: 50 },
    showlegend: true,
    legend: { orientation: 'h', y: -0.12, x: 0, xanchor: 'left', yanchor: 'top', font: { size: 11 } },
    hovermode: 'closest',
    hoverlabel: themedHoverlabel(isDark),
    xaxis: { matches: 'x2', showticklabels: false, gridcolor, zerolinecolor, layer: 'below traces' },
    xaxis2: { title: { text: 'training step' }, gridcolor, zerolinecolor, layer: 'below traces' },
    yaxis: { title: { text: 'mat-NMAE mean (%)' }, gridcolor, zerolinecolor, layer: 'below traces' },
    yaxis2: { title: { text: 'mat-NEMD mean (%)' }, gridcolor, zerolinecolor, layer: 'below traces' },
    shapes: [
      // Chinchilla-optimal + 1-epoch vertical lines on both panels.
      ...(['y', 'y2'] as const).flatMap((yref) => [
        { type: 'line' as const, x0: data.chinchilla_step, x1: data.chinchilla_step, yref, y0: 0, y1: 1, xref: yref === 'y' ? 'x' : 'x2', line: { color: '#888', width: 1, dash: 'dot' as const } },
        { type: 'line' as const, x0: data.epoch_step,      x1: data.epoch_step,      yref, y0: 0, y1: 1, xref: yref === 'y' ? 'x' : 'x2', line: { color: '#888', width: 1, dash: 'dot' as const } },
      ]),
    ],
    annotations: [
      { x: data.chinchilla_step, y: 1.02, xref: 'x2', yref: 'paper', text: 'Chinchilla-opt', showarrow: false, font: { size: 10, color: '#888' }, xanchor: 'left' },
      { x: data.epoch_step,      y: 1.02, xref: 'x2', yref: 'paper', text: '1 epoch',        showarrow: false, font: { size: 10, color: '#888' }, xanchor: 'right' },
    ],
  }

  return <Plot data={traces} layout={layout} />
}
