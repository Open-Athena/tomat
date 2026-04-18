import { useMemo } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { METRIC_LABEL, type Metric, parseConfig, type SweepRow } from './types'
import { SCHEME_COLORS, SCHEME_LABEL, hexToRgba } from './FractionPlot'
import { themedHoverlabel } from './theme'

/** Context-length reference lines to overlay on the token axis. */
const CONTEXT_LINES: Array<{ x: number; label: string }> = [
  { x: 4_096, label: '4k' },
  { x: 16_384, label: '16k' },
  { x: 65_536, label: '64k' },
  { x: 262_144, label: '256k' },
  { x: 1_048_576, label: '1M' },
]

const SCHEMES_ON_PARETO = ['cutoff-top', 'fourier-coded-lowg', 'delta-fourier-coded-lowg']

function schemeKey(scheme: string): string {
  if (scheme === 'cutoff-top') return 'cutoff'
  if (scheme === 'fourier-coded-lowg') return 'fourier'
  if (scheme === 'delta-fourier-coded-lowg') return 'delta-fourier'
  return scheme
}

interface ParetoCurve {
  key: string
  tokens: number[]
  nmae: number[]
  min: number[]
  max: number[]
}

function buildParetoCurves(rows: SweepRow[], metric: Metric): ParetoCurve[] {
  // Group by (scheme, fraction), average tokens and aggregate metric across samples.
  const bySchemeFrac = new Map<string, Map<number, { tokens: number[]; metric: number[] }>>()
  for (const r of rows) {
    const parsed = parseConfig(r.config)
    if (!parsed) continue
    if (!SCHEMES_ON_PARETO.includes(parsed.scheme)) continue
    const key = schemeKey(parsed.scheme)
    if (!bySchemeFrac.has(key)) bySchemeFrac.set(key, new Map())
    const inner = bySchemeFrac.get(key)!
    if (!inner.has(parsed.fraction)) inner.set(parsed.fraction, { tokens: [], metric: [] })
    const v = r[metric]
    if (Number.isFinite(v) && Number.isFinite(r.tokens)) {
      inner.get(parsed.fraction)!.tokens.push(r.tokens)
      inner.get(parsed.fraction)!.metric.push(v as number)
    }
  }
  return [...bySchemeFrac.entries()].map(([key, inner]) => {
    const fractions = [...inner.keys()].sort((a, b) => a - b)
    const tokens = fractions.map(f => mean(inner.get(f)!.tokens))
    const nmae = fractions.map(f => median(inner.get(f)!.metric))
    const min = fractions.map(f => Math.min(...inner.get(f)!.metric))
    const max = fractions.map(f => Math.max(...inner.get(f)!.metric))
    return { key, tokens, nmae, min, max }
  })
}

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b)
  const n = sorted.length
  return n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
}

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length
}

export function ParetoPlot({
  rows,
  metric = 'nmae',
  height = 560,
  showMinMax = true,
  enabledSchemes,
}: {
  rows: SweepRow[]
  metric?: Metric
  height?: number
  showMinMax?: boolean
  enabledSchemes?: Set<string>
}) {
  const { isDark } = useTheme()
  const hoverlabel = themedHoverlabel(isDark)
  const curves = useMemo(() => buildParetoCurves(rows, metric), [rows, metric])
  const enabled = enabledSchemes ?? new Set(Object.keys(SCHEME_COLORS))
  const metricLabel = METRIC_LABEL[metric]

  return (
    <Plot
      data={curves.flatMap(c =>
        enabled.has(c.key)
          ? [
              ...(showMinMax
                ? [{
                    x: [...c.tokens, ...[...c.tokens].reverse()],
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
                x: c.tokens,
                y: c.nmae,
                type: 'scatter' as const,
                mode: 'lines+markers' as const,
                name: SCHEME_LABEL[c.key],
                line: { color: SCHEME_COLORS[c.key], width: 2 },
                marker: { size: 7 },
                hovertemplate: '%{x:,d} tokens<br>%{y:.3~%}<extra></extra>',
              },
            ]
          : [],
      )}
      layout={{
        autosize: true,
        height,
        margin: { t: 10, r: 40, b: 50, l: 60 },
        hovermode: 'closest',
        hoverlabel,
        xaxis: {
          type: 'log',
          title: { text: 'Tokens per structure' },
          tickformat: '.2s',  // 1k, 10k, 1M, ...
          hoverformat: ',d',
        },
        yaxis: {
          type: 'log',
          title: { text: `Reconstruction ${metricLabel}` },
          fixedrange: false,
          tickformat: '.1~e',
          hoverformat: '.3~%',
        },
        legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.12, yanchor: 'top' },
        shapes: [
          // electrAI 2.6% horizontal reference
          {
            type: 'line',
            xref: 'paper', x0: 0, x1: 1, y0: 0.026, y1: 0.026,
            line: { color: 'gray', dash: 'dash', width: 1 },
          },
          // Context-length verticals
          ...CONTEXT_LINES.map(c => ({
            type: 'line' as const,
            yref: 'paper' as const,
            x0: c.x, x1: c.x, y0: 0, y1: 1,
            line: { color: 'gray', dash: 'dot' as const, width: 1 },
          })),
        ],
        annotations: [
          {
            x: Math.log10(3e3), y: Math.log10(0.026),
            xref: 'x', yref: 'y',
            text: 'electrAI 2.6%',
            showarrow: false, xanchor: 'left', yshift: 8,
            font: { color: 'gray', size: 11 },
          },
          ...CONTEXT_LINES.map(c => ({
            x: Math.log10(c.x), y: 1.02,
            xref: 'x' as const, yref: 'paper' as const,
            text: c.label,
            showarrow: false, xanchor: 'center' as const,
            font: { color: 'gray', size: 11 },
          })),
        ],
      }}
      style={{ width: '100%' }}
    />
  )
}
