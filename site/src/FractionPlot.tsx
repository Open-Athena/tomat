import { useMemo } from 'react'
import { Plot, useTheme } from 'pltly/react'
import { METRIC_LABEL, type Metric, parseConfig, type SweepRow } from './types'
import { themedHoverlabel } from './theme'

export const SCHEME_COLORS: Record<string, string> = {
  cutoff: '#cc4444',
  fourier: '#4480e0',
  'delta-fourier': '#2a8c2a',
}

export const SCHEME_LABEL: Record<string, string> = {
  cutoff: 'cutoff-top',
  fourier: 'fourier-lowg',
  'delta-fourier': 'Δρ-fourier-lowg',
}

const SCHEMES_WITH_FRACTION = ['cutoff-top', 'fourier-lowg', 'delta-fourier-lowg']

function schemeKey(scheme: string): string {
  if (scheme === 'cutoff-top') return 'cutoff'
  if (scheme === 'fourier-lowg') return 'fourier'
  if (scheme === 'delta-fourier-lowg') return 'delta-fourier'
  return scheme
}

export function hexToRgba(hex: string, alpha: number): string {
  let h = hex.replace('#', '')
  if (h.length === 3) h = h.split('').map(c => c + c).join('')
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

interface FractionCurve {
  key: string
  fractions: number[]
  median: number[]
  min: number[]
  max: number[]
}

export function buildFractionCurves(rows: SweepRow[], metric: Metric): FractionCurve[] {
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

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b)
  const n = sorted.length
  return n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
}

export function FractionPlot({
  rows,
  metric = 'nmae',
  height = 560,
  logY = true,
  showMinMax = true,
  enabledSchemes,
  showElectrAI = true,
}: {
  rows: SweepRow[]
  metric?: Metric
  height?: number
  logY?: boolean
  showMinMax?: boolean
  enabledSchemes?: Set<string>
  showElectrAI?: boolean
}) {
  const { isDark } = useTheme()
  const hoverlabel = themedHoverlabel(isDark)
  const curves = useMemo(() => buildFractionCurves(rows, metric), [rows, metric])
  const enabled = enabledSchemes ?? new Set(Object.keys(SCHEME_COLORS))
  const metricLabel = METRIC_LABEL[metric]

  return (
    <Plot
      data={curves.flatMap(c =>
        enabled.has(c.key)
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
                hovertemplate: '%{y:.3~%}<extra></extra>',
              },
            ]
          : [],
      )}
      layout={{
        autosize: true,
        height,
        margin: { t: 10, r: 40, b: 50, l: 60 },
        hovermode: 'x unified',
        hoverlabel,
        xaxis: {
          type: 'log',
          title: { text: 'Fraction of representation kept' },
          range: [Math.log10(0.002), Math.log10(1.3)],
          dtick: 1,
          tickformat: '.2~%',
          hoverformat: '.3~%',
        },
        yaxis: {
          type: logY ? 'log' : 'linear',
          title: { text: `Reconstruction ${metricLabel}` },
          fixedrange: false,
          tickformat: '.1~e',
          hoverformat: '.3~%',
        },
        legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.12, yanchor: 'top' },
        shapes: showElectrAI ? [{
          type: 'line',
          x0: 0.001, x1: 1.5, y0: 0.026, y1: 0.026,
          line: { color: 'gray', dash: 'dash', width: 1 },
        }] : [],
        annotations: showElectrAI ? [{
          x: Math.log10(0.005), y: Math.log10(0.026),
          xref: 'x', yref: 'y',
          text: 'electrAI best 2.6% NMAE',
          showarrow: false,
          xanchor: 'left',
          yshift: 8,
          font: { color: 'gray', size: 11 },
        }] : [],
      }}
      style={{ width: '100%' }}
    />
  )
}
