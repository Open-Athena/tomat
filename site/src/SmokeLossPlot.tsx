import { useEffect, useState } from 'react'
import { csvParse } from 'd3-dsv'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from './theme'

interface LossRow { step: number; loss: number }

export function SmokeLossPlot({ url }: { url: string }) {
  const [rows, setRows] = useState<LossRow[] | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const { isDark } = useTheme()

  useEffect(() => {
    fetch(url)
      .then(r => r.ok ? r.text() : Promise.reject(new Error(`fetch ${url}: ${r.status}`)))
      .then(text => {
        const parsed = csvParse(text, d => ({
          step: Number(d.step),
          loss: Number(d.train_loss),
        }))
        setRows(parsed as LossRow[])
      })
      .catch(setError)
  }, [url])

  if (error) return <p className="note">Error loading smoke loss CSV: {error.message}</p>
  if (!rows) return <p className="note">Loading smoke loss curve…</p>

  const baseline = Math.log(6792)  // uniform-over-vocab; matches step-0 loss
  const final = rows.length > 0 ? rows[rows.length - 1].loss : NaN
  return (
    <>
      <Plot
        data={[
          {
            x: rows.map(r => r.step),
            y: rows.map(r => r.loss),
            type: 'scatter',
            mode: 'lines',
            name: 'train/loss',
            line: { color: '#1f77b4', width: 2 },
            hovertemplate: 'step %{x}<br>loss %{y:.3f}<extra></extra>',
          },
          {
            x: [0, rows[rows.length - 1]?.step ?? 99],
            y: [baseline, baseline],
            type: 'scatter',
            mode: 'lines',
            name: `uniform baseline (ln 6792)`,
            line: { color: '#888', width: 1, dash: 'dash' },
            hovertemplate: `uniform baseline: %{y:.3f}<extra></extra>`,
          },
        ]}
        layout={{
          autosize: true,
          height: 360,
          margin: { t: 10, r: 40, b: 50, l: 60 },
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          xaxis: { title: { text: 'step' }, fixedrange: false },
          yaxis: { title: { text: 'train loss (nats/token)' }, fixedrange: false },
          legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2, yanchor: 'top' },
        }}
        style={{ width: '100%' }}
      />
      <p className="note">
        100-step Qwen3 smoke on Modal A100 (28.6 M params, 8k context, batch 8).
        Loss went <strong>{rows[0].loss.toFixed(3)} → {final.toFixed(3)}</strong> in 100 steps;
        step 0 sits on the uniform-over-vocab baseline as expected at init.
      </p>
    </>
  )
}
