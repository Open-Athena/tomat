import { useMemo, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { METRIC_LABEL, type Metric, parseConfig, type SweepRow } from './types'
import { useSweep } from './useSweep'
import { FractionPlot, SCHEME_COLORS, SCHEME_LABEL } from './FractionPlot'
import { ParetoPlot } from './ParetoPlot'
import { ThemeToggle, themedHoverlabel } from './theme'

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
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

export function HomePage() {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
  const { rows, error } = useSweep(`${base}/sweep-n50.csv`)

  const [metric, setMetric] = useState<Metric>('nmae')
  const [logY, setLogY] = useState(true)
  const [showMinMax, setShowMinMax] = useState(true)
  const [enabledSchemes, setEnabledSchemes] = useState(new Set(Object.keys(SCHEME_COLORS)))
  const { isDark } = useTheme()
  const hoverlabel = themedHoverlabel(isDark)
  const toggleScheme = (s: string) => {
    const next = new Set(enabledSchemes)
    next.has(s) ? next.delete(s) : next.add(s)
    setEnabledSchemes(next)
  }

  const byCategory = useMemo(() => buildByCategory(rows ?? [], metric), [rows, metric])

  if (error) return <pre>Error loading sweep CSV: {error.message}</pre>
  if (!rows) return <p>Loading sweep CSV…</p>

  const samples = new Set(rows.map(r => r.mp_id)).size
  const metricLabel = METRIC_LABEL[metric]

  return (
    <>
      <header>
        <h1>tomat 🍅 — tokenizer fidelity</h1>
        <a className="deck-link" href="#/deck" title="Open slide deck">slides →</a>
        <ThemeToggle />
      </header>

      <p className="meta">
        Reconstruction-error floor per tokenization scheme on n={samples} Materials
        Project CHGCARs (128³ grid). Backed by live
        {' '}<ExtLink href="https://github.com/Open-Athena/tomat/blob/main/results/sweep-n50.csv">sweep CSV</ExtLink>{' '}
        · <ExtLink href="https://github.com/Open-Athena/tomat">Open-Athena/tomat</ExtLink>
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
        <FractionPlot
          rows={rows}
          metric={metric}
          logY={logY}
          showMinMax={showMinMax}
          enabledSchemes={enabledSchemes}
        />
      </div>

      <h2>{metricLabel} vs tokens per structure (Pareto)</h2>
      <p className="meta">
        Coded variants assume FP16 codec fidelity end-to-end (3 tokens per real value,
        6 per complex). Vertical lines mark typical transformer context lengths.
      </p>
      <div className="plot-card">
        <ParetoPlot
          rows={rows}
          metric={metric}
          showMinMax={showMinMax}
          enabledSchemes={enabledSchemes}
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
              hovertemplate: '%{y:.3~%}<extra></extra>',
            }))}
          layout={{
            autosize: true,
            height: 500,
            margin: { t: 10, r: 40, b: 50, l: 60 },
            hovermode: 'x unified',
            hoverlabel,
            barmode: 'group',
            yaxis: {
              type: logY ? 'log' : 'linear',
              title: { text: `Mean ${metricLabel}` },
              fixedrange: false,
              tickformat: '.2~%',
              hoverformat: '.3~%',
            },
            legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.15, yanchor: 'top' },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="footer">
        Data regenerated via{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/scripts/fidelity_sweep.py">
          <code>uv run scripts/fidelity_sweep.py -n 50</code>
        </ExtLink>
        ; provenance tracked in{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/results/sweep-n50.csv.dvc">
          <code>results/sweep-n50.csv.dvc</code>
        </ExtLink>{' '}
        (<code>dvx status</code>). Plots built from the committed CSV — no server needed.
      </div>
    </>
  )
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

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length
}
