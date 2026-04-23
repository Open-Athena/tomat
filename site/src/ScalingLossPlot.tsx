import { useEffect, useState } from 'react'
import { csvParse } from 'd3-dsv'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from './theme'

interface LossRow { step: number; loss: number }
interface RunSpec {
  file: string
  name: string
  color: string
}

const RUNS: RunSpec[] = [
  { file: 'val-full-5k-bs32-bs32-seed42.csv',        name: 'A100:1 bs=32 (val-full)',     color: '#1f77b4' },
  { file: 'val-full-5k-bs32-2gpu-bs32-seed42.csv',   name: 'A100:2 bs=32 (val-full)',     color: '#2ca02c' },
  { file: 'val-full-5k-bs64-4gpu-bs64-seed42.csv',   name: 'A100:4 bs=64 (val-full)',     color: '#ff7f0e' },
  { file: 'val-full-5k-bs128-8gpu-bs128-seed42.csv', name: 'A100:8 bs=128 (val-full)',    color: '#d62728' },
  { file: 'val-full-tpu-bs128-seed42.csv',           name: 'TPU v6e-4 bs=128 (val-full)', color: '#9467bd' },
  { file: 'train-full-tpu8-bs256-seed42.csv',        name: '30M: TPU v6e-8 bs=256 (train-full)', color: '#8c564b' },
  { file: 'train-full-tpu16-30M-bs512-seed42.csv',   name: '30M: TPU v6e-16 bs=512 (train-full)', color: '#17becf' },
  { file: 'train-full-tpu8-200M-bs128-val-bf16-seed42.csv', name: '200M: TPU v6e-8 bs=128 (train-full)', color: '#000000' },
]

export function ScalingLossPlot({ baseUrl }: { baseUrl: string }) {
  const [data, setData] = useState<Record<string, LossRow[]>>({})
  const [error, setError] = useState<Error | null>(null)
  const { isDark } = useTheme()

  useEffect(() => {
    Promise.all(RUNS.map(run =>
      fetch(`${baseUrl}/${run.file}`)
        .then(r => r.ok ? r.text() : Promise.reject(new Error(`fetch ${run.file}: ${r.status}`)))
        .then(text => {
          const parsed = csvParse(text, d => ({
            step: Number(d.step),
            loss: Number(d.train_loss),
          }))
          return [run.file, parsed as LossRow[]] as const
        })
    ))
      .then(entries => setData(Object.fromEntries(entries)))
      .catch(setError)
  }, [baseUrl])

  if (error) return <p className="note">Error loading run histories: {error.message}</p>
  if (Object.keys(data).length < RUNS.length) return <p className="note">Loading scaling loss curves…</p>

  const baseline = Math.log(6792)

  return (
    <>
      <Plot
        data={[
          ...RUNS.map(run => {
            const rows = data[run.file] ?? []
            return {
              x: rows.map(r => r.step),
              y: rows.map(r => r.loss),
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: run.name,
              line: { color: run.color, width: 2 },
              hovertemplate: `${run.name}<br>step %{x}<br>loss %{y:.3f}<extra></extra>`,
            }
          }),
          {
            x: [0, 5000],
            y: [baseline, baseline],
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'uniform baseline (ln 6792)',
            line: { color: '#888', width: 1, dash: 'dash' },
            hovertemplate: 'uniform baseline: %{y:.3f}<extra></extra>',
          },
        ]}
        layout={{
          autosize: true,
          height: 420,
          margin: { t: 10, r: 40, b: 50, l: 60 },
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          xaxis: { title: { text: 'step' } },
          yaxis: { title: { text: 'train loss (nats/token)' }, rangemode: 'tozero' },
          legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2, yanchor: 'top' },
        }}
        style={{ width: '100%' }}
      />
      <p className="note">
        Same 30 M Qwen3 on val-full (4,305 mats × 32 patches × 8k context, seed 42).
        A100:1 (blue) OOMed at step 2,560 on bs=32 per-device=32; the per-device bs=16
        sweep (green/orange/red) scales nearly linearly through bs=128. TPU (purple) ran
        1,000 steps of the same config for the 10×-tok/s datapoint; different step budget,
        so its curve ends earlier on the x-axis.
      </p>
    </>
  )
}
