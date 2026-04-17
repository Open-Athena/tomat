import { useEffect, useMemo, useState } from 'react'
import { Plot, PlotlyProvider } from 'pltly/react'

const plotlyLoader = () =>
  import('plotly.js-basic-dist') as unknown as Promise<typeof import('plotly.js')>
import { METRIC_LABEL, type Metric, parseConfig, type SweepRow } from './types'
import { useSweep } from './useSweep'

const SCHEME_COLORS: Record<string, string> = {
  cutoff: '#c44',
  fourier: '#4480e0',
  'delta-fourier': '#2a8c2a',
}

const SCHEME_LABEL: Record<string, string> = {
  cutoff: 'cutoff-top',
  fourier: 'fourier-lowg',
  'delta-fourier': 'Δρ-fourier-lowg',
}

const SCHEME_DESC: Record<string, string> = {
  cutoff: 'voxel value',
  fourier: 'low |G|',
  'delta-fourier': 'PADS subtracted, low |G|',
}

const SCHEMES_WITH_FRACTION = ['cutoff-top', 'fourier-lowg', 'delta-fourier-lowg']

function schemeKey(scheme: string): string {
  if (scheme === 'cutoff-top') return 'cutoff'
  if (scheme === 'fourier-lowg') return 'fourier'
  if (scheme === 'delta-fourier-lowg') return 'delta-fourier'
  return scheme
}

type ThemeMode = 'light' | 'dark' | 'auto'

function useThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>(
    () => (localStorage.getItem('theme') as ThemeMode | null) ?? 'auto',
  )
  useEffect(() => {
    const html = document.documentElement
    if (mode === 'auto') {
      html.removeAttribute('data-theme')
    } else {
      html.setAttribute('data-theme', mode)
    }
    localStorage.setItem('theme', mode)
  }, [mode])
  return [mode, setMode] as const
}

export function App() {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
  const { rows, error } = useSweep(`${base}/sweep-n50.csv`)

  const [metric, setMetric] = useState<Metric>('nmae')
  const [logY, setLogY] = useState(true)
  const [showMinMax, setShowMinMax] = useState(true)
  const [enabledSchemes, setEnabledSchemes] = useState(new Set(Object.keys(SCHEME_COLORS)))
  const [theme, setTheme] = useThemeToggle()
  const toggleScheme = (s: string) => {
    const next = new Set(enabledSchemes)
    next.has(s) ? next.delete(s) : next.add(s)
    setEnabledSchemes(next)
  }

  const fractionCurves = useMemo(() => buildFractionCurves(rows ?? [], metric), [rows, metric])
  const byCategory = useMemo(() => buildByCategory(rows ?? [], metric), [rows, metric])

  if (error) return <pre>Error loading sweep CSV: {error.message}</pre>
  if (!rows) return <p>Loading sweep CSV…</p>

  const samples = new Set(rows.map(r => r.mp_id)).size
  const metricLabel = METRIC_LABEL[metric]

  return (
    <PlotlyProvider loader={plotlyLoader}>
      <header>
        <h1>tomat 🍅 — tokenizer fidelity</h1>
        <button
          className="theme-toggle"
          onClick={() => setTheme(theme === 'dark' ? 'light' : theme === 'light' ? 'auto' : 'dark')}
          aria-label="theme"
          title={`theme: ${theme}`}
        >
          {theme === 'dark' ? '🌙' : theme === 'light' ? '☀️' : '🖥️'}
        </button>
      </header>

      <p className="meta">
        Reconstruction-error floor per tokenization scheme on n={samples} Materials
        Project CHGCARs (128³ grid). Backed by live
        {' '}<a href="https://github.com/Open-Athena/tomat/blob/main/results/sweep-n50.csv">sweep CSV</a>{' '}
        · <a href="https://github.com/Open-Athena/tomat">Open-Athena/tomat</a>
      </p>

      <div className="controls">
        <label>
          Metric:
          <select value={metric} onChange={e => setMetric(e.target.value as Metric)}>
            {(Object.keys(METRIC_LABEL) as Metric[]).map(m => (
              <option key={m} value={m}>{METRIC_LABEL[m]}</option>
            ))}
          </select>
        </label>
        <label>
          <input type="checkbox" checked={logY} onChange={e => setLogY(e.target.checked)} />
          log y
        </label>
        <label>
          <input type="checkbox" checked={showMinMax} onChange={e => setShowMinMax(e.target.checked)} />
          min/max band
        </label>
        {Object.keys(SCHEME_COLORS).map(s => (
          <label key={s} title={SCHEME_DESC[s]}>
            <input
              type="checkbox"
              checked={enabledSchemes.has(s)}
              onChange={() => toggleScheme(s)}
            />
            <span style={{ color: SCHEME_COLORS[s] }}>{SCHEME_LABEL[s]}</span>
          </label>
        ))}
      </div>

      <h2>{metricLabel} vs fraction of representation kept</h2>
      <div className="plot-card">
        <Plot
          data={fractionCurves.flatMap(c =>
            enabledSchemes.has(c.key)
              ? [
                  ...(showMinMax
                    ? [{
                        x: [...c.fractions, ...[...c.fractions].reverse()],
                        y: [...c.max, ...[...c.min].reverse()],
                        fill: 'toself' as const,
                        fillcolor: hexToRgba(SCHEME_COLORS[c.key], 0.12),
                        line: { color: 'transparent' },
                        hoverinfo: 'skip' as const,
                        name: `${c.key} (min-max)`,
                        showlegend: false,
                        type: 'scatter' as const,
                      }]
                    : []),
                  {
                    x: c.fractions,
                    y: c.median,
                    type: 'scatter' as const,
                    mode: 'lines+markers' as const,
                    name: SCHEME_LABEL[c.key],
                    line: { color: SCHEME_COLORS[c.key], width: 2 },
                    marker: { size: 7 },
                  },
                ]
              : [],
          )}
          layout={{
            autosize: true,
            height: 500,
            margin: { r: 20 },
            xaxis: { type: 'log', title: { text: 'Fraction of representation kept' } },
            yaxis: { type: logY ? 'log' : 'linear', title: { text: `Reconstruction ${metricLabel}` }, fixedrange: false },
            legend: { orientation: 'h', y: -0.2, yanchor: 'top' },
            shapes: [{
              type: 'line',
              x0: 0.0001, x1: 1, y0: 0.026, y1: 0.026,
              line: { color: 'gray', dash: 'dash', width: 1 },
            }],
            annotations: [{
              x: Math.log10(0.01), y: Math.log10(0.026),
              xref: 'x', yref: 'y',
              text: 'electrAI best 2.6% NMAE',
              showarrow: false,
              xanchor: 'left',
              yshift: 8,
              font: { color: 'gray', size: 11 },
            }],
          }}
          style={{ width: '100%' }}
        />
      </div>

      <h2>{metricLabel} by material category (mean, 5% fraction)</h2>
      <div className="plot-card">
        <Plot
          data={byCategory.schemes
            .filter(s => enabledSchemes.has(s.key))
            .map(s => ({
              x: byCategory.categories.map(c => `${c} (n=${byCategory.counts[c]})`),
              y: s.values,
              type: 'bar' as const,
              name: SCHEME_LABEL[s.key],
              marker: { color: SCHEME_COLORS[s.key] },
            }))}
          layout={{
            autosize: true,
            height: 460,
            margin: { r: 20 },
            barmode: 'group',
            yaxis: { type: logY ? 'log' : 'linear', title: { text: `Mean ${metricLabel}` }, fixedrange: false },
            legend: { orientation: 'h', y: -0.2, yanchor: 'top' },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="footer">
        Data regenerated via <code>uv run scripts/fidelity_sweep.py -n 50</code>; provenance
        tracked in <code>results/sweep-n50.csv.dvc</code> (<code>dvx status</code>).
        Plots built from the committed CSV — no server needed.
      </div>
    </PlotlyProvider>
  )
}

interface FractionCurve {
  key: string
  fractions: number[]
  median: number[]
  min: number[]
  max: number[]
}

function buildFractionCurves(rows: SweepRow[], metric: Metric): FractionCurve[] {
  const bySchemeFrac = new Map<string, Map<number, number[]>>()
  for (const r of rows) {
    const parsed = parseConfig(r.config)
    if (!parsed) continue
    if (!SCHEMES_WITH_FRACTION.includes(parsed.scheme)) continue
    const key = schemeKey(parsed.scheme)
    if (!bySchemeFrac.has(key)) bySchemeFrac.set(key, new Map())
    const inner = bySchemeFrac.get(key)!
    if (!inner.has(parsed.fraction)) inner.set(parsed.fraction, [])
    const v = r[metric]
    if (Number.isFinite(v)) inner.get(parsed.fraction)!.push(v as number)
  }

  return [...bySchemeFrac.entries()].map(([key, inner]) => {
    const fractions = [...inner.keys()].sort((a, b) => a - b)
    return {
      key,
      fractions,
      median: fractions.map(f => median(inner.get(f)!)),
      min: fractions.map(f => Math.min(...inner.get(f)!)),
      max: fractions.map(f => Math.max(...inner.get(f)!)),
    }
  })
}

interface ByCategory {
  categories: string[]
  counts: Record<string, number>
  schemes: { key: string; values: number[] }[]
}

function buildByCategory(rows: SweepRow[], metric: Metric, targetFraction = 0.05): ByCategory {
  const byCatConfig = new Map<string, Map<string, number[]>>()
  const catCounts: Record<string, Set<string>> = {}
  for (const r of rows) {
    const parsed = parseConfig(r.config)
    if (!parsed || parsed.fraction !== targetFraction) continue
    if (!SCHEMES_WITH_FRACTION.includes(parsed.scheme)) continue
    const key = schemeKey(parsed.scheme)
    if (!byCatConfig.has(r.category)) byCatConfig.set(r.category, new Map())
    const inner = byCatConfig.get(r.category)!
    if (!inner.has(key)) inner.set(key, [])
    const v = r[metric]
    if (Number.isFinite(v)) inner.get(key)!.push(v as number)
    if (!catCounts[r.category]) catCounts[r.category] = new Set()
    catCounts[r.category].add(r.mp_id)
  }
  const categories = [...byCatConfig.keys()].sort(
    (a, b) => catCounts[b].size - catCounts[a].size,
  )
  const counts = Object.fromEntries(
    Object.entries(catCounts).map(([c, s]) => [c, s.size]),
  )
  const schemeKeys = [...new Set(categories.flatMap(c => [...byCatConfig.get(c)!.keys()]))]
  const schemes = schemeKeys.map(key => ({
    key,
    values: categories.map(cat => {
      const arr = byCatConfig.get(cat)?.get(key) ?? []
      return arr.length ? mean(arr) : NaN
    }),
  }))
  return { categories, counts, schemes }
}

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b)
  const n = sorted.length
  return n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
}

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}
