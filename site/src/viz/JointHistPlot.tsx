// Plotly view of one (x-source, y-source) joint histogram. Main
// heatmap + top/right marginal PDFs in a 3×3 layout via Plotly subplot
// domains. Diagonal y=x line on the main trace. Drag-to-zoom is the
// out-of-box Plotly behavior; the page wrapper picks it up via
// `onRelayout` to sync to URL.

import { useMemo } from 'react'
import type { Layout } from 'plotly.js'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { JointHistResponse } from './jointHistWorker'

export interface JointHistRange {
  xRange: [number, number] | null
  yRange: [number, number] | null
}

export interface JointHistPlotProps {
  /** Histogram result from the worker. */
  hist: JointHistResponse
  /** Title-bar text — the page wrapper fills in mp + dims + spec labels. */
  title: string
  /** Short labels for the two axes (used by hover + axis titles). */
  xLabel: string
  yLabel: string
  /** Log or linear coloring of `H` (axes always render in `scale` units). */
  scale: 'log' | 'linear'
  /** Show top + right 1-D marginal PDFs. When false, the heatmap fills the
   *  whole plot area (useful when the marginal histograms duplicate info
   *  already shown elsewhere in the page). */
  showMarginals: boolean
  /** Optional drag-zoom range to apply on render. */
  xRange: [number, number] | null
  yRange: [number, number] | null
  /** Fired whenever Plotly's `onRelayout` reports a zoom/pan; page wrapper
   *  syncs these to URL params (debounced). */
  onRangeChange: (r: JointHistRange) => void
}

export function JointHistPlot({
  hist, title, xLabel, yLabel, scale, showMarginals,
  xRange, yRange, onRangeChange,
}: JointHistPlotProps) {
  const { isDark } = useTheme()
  const { traces, layout } = useMemo(
    () => buildFigure({ hist, title, xLabel, yLabel, scale, showMarginals, xRange, yRange, isDark }),
    [hist, title, xLabel, yLabel, scale, showMarginals, xRange, yRange, isDark],
  )

  // Plotly fires `plotly_relayout` with the new axis ranges after every
  // drag-zoom / pan / autoscale. We round-trip `[xaxis.range[0], xaxis.range[1]]`
  // (heatmap) into the page wrapper. Reset (`xaxis.autorange: true`) maps to
  // `null` so the URL params drop, restoring the data extent.
  const onRelayout = (e: Readonly<Record<string, unknown>>) => {
    const next: JointHistRange = { xRange: null, yRange: null }
    const xLo = e['xaxis.range[0]']
    const xHi = e['xaxis.range[1]']
    if (typeof xLo === 'number' && typeof xHi === 'number') {
      next.xRange = [xLo, xHi]
    } else if (e['xaxis.autorange'] === true) {
      next.xRange = null
    } else {
      next.xRange = xRange
    }
    const yLo = e['yaxis.range[0]']
    const yHi = e['yaxis.range[1]']
    if (typeof yLo === 'number' && typeof yHi === 'number') {
      next.yRange = [yLo, yHi]
    } else if (e['yaxis.autorange'] === true) {
      next.yRange = null
    } else {
      next.yRange = yRange
    }
    if (next.xRange === xRange && next.yRange === yRange) return
    onRangeChange(next)
  }

  // Title + subtitle rendered as selectable HTML, OUTSIDE plotly. Plotly's
  // SVG-rendered titles can't be copy/pasted, and as a bonus we get more
  // typographic control AND we free plotly's top margin (it was 130px to
  // make room for 4 lines of `<sub>` text, which crowded the top
  // marginal — see Yael's CIC).
  const { total, xStats, yStats, onDiagFrac } = hist
  const fmt = (s: { mean: number; p50: number; p99: number; max: number }) =>
    `μ=${s.mean.toFixed(1)}, p50=${s.p50.toFixed(1)}, p99=${s.p99.toFixed(1)}, max=${s.max.toFixed(1)}`
  return (
    <div className="plot-card">
      <div style={{
        textAlign: 'center', marginBottom: '0.4rem', userSelect: 'text',
      }}>
        <div style={{ fontSize: '1.05rem', fontWeight: 600 }}>{title}</div>
        <div style={{ fontSize: '0.8rem', opacity: 0.85, marginTop: 2 }}>
          {xLabel}: {fmt(xStats)}
        </div>
        <div style={{ fontSize: '0.8rem', opacity: 0.85 }}>
          {yLabel}: {fmt(yStats)}
        </div>
        <div style={{ fontSize: '0.8rem', opacity: 0.85 }}>
          on-diag(|y/x−1|≤10%): {(onDiagFrac * 100).toFixed(2)}% · n={total.toLocaleString()} voxels
        </div>
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: '100%', height: showMarginals ? 720 : 600 }}
        onRelayout={onRelayout as never}
      />
    </div>
  )
}

interface BuildArgs {
  hist: JointHistResponse
  title: string
  xLabel: string
  yLabel: string
  scale: 'log' | 'linear'
  showMarginals: boolean
  xRange: [number, number] | null
  yRange: [number, number] | null
  isDark: boolean
}

interface BuiltFigure {
  // Plotly's data array is heterogeneous — heatmap traces + bar traces — so
  // the union type would balloon and add no safety here. The Plot wrapper
  // expects `Partial<Plotly.Data>[]` anyway.
  traces: Plotly.Data[]
  layout: Partial<Layout>
}

function buildFigure(args: BuildArgs): BuiltFigure {
  // `title` + the stats summary (`xStats`/`yStats`/`onDiagFrac`) are
  // rendered as selectable HTML in the wrapper component, NOT here.
  const { hist, xLabel, yLabel, scale, showMarginals, xRange, yRange, isDark } = args
  const { H, xEdges, yEdges, total } = hist
  const bins = xEdges.length - 1

  // Plotly's `heatmap` trace wants 2-D `z` and 1-D `x`/`y` arrays. We
  // reshape the flat row-major H into a JS array-of-arrays (the perf cost
  // is dwarfed by Plotly's own typed→untyped marshaling). Each row is a
  // y-bin, each column is an x-bin — matches the row-major layout the
  // worker emits (`H[by * bins + bx]`).
  const z: number[][] = []
  const xCenters = bincenters(xEdges)
  const yCenters = bincenters(yEdges)
  for (let yi = 0; yi < bins; yi++) {
    const row: number[] = []
    for (let xi = 0; xi < bins; xi++) {
      const v = H[yi * bins + xi]
      // Log-color: render zero cells as null so they're left blank, avoiding
      // a giant "log(0)" hole that obscures the actual signal. In linear
      // mode we keep zeros visible.
      if (scale === 'log' && v === 0) row.push(NaN)
      else row.push(v)
    }
    z.push(row)
  }

  // X-marginal: sum H over y. Y-marginal: sum over x.
  const xMarg = new Float64Array(bins)
  const yMarg = new Float64Array(bins)
  for (let yi = 0; yi < bins; yi++) {
    for (let xi = 0; xi < bins; xi++) {
      const v = H[yi * bins + xi]
      xMarg[xi] += v
      yMarg[yi] += v
    }
  }
  // Marginals share the same total mass (n voxels) by construction. We
  // pin both count-axes to the same range so visually-equal-area bars
  // represent equal counts. Without this, Plotly auto-fits each marginal
  // independently, and a tall-narrow PDF (e.g. low-density GT mode at
  // ρ≈30) normalizes against its own peak while the partner's broader
  // PDF (e.g. bin5's mode-collapsed prediction) auto-fits to ITS lower
  // per-bin peak — making it look like the broader marginal carries
  // MORE total mass, which is impossible by construction. Yael spotted
  // this on mp-1797712 (GT mode 37 e/Å³ vs bin5 mode 28 e/Å³ rendered
  // misleadingly).
  let margCountMax = 0
  for (let i = 0; i < bins; i++) {
    if (xMarg[i] > margCountMax) margCountMax = xMarg[i]
    if (yMarg[i] > margCountMax) margCountMax = yMarg[i]
  }
  // Padding so the tallest bar doesn't kiss the subplot edge.
  margCountMax = margCountMax * 1.05

  const xMin = xEdges[0]
  const xMax = xEdges[bins]
  const yMin = yEdges[0]
  const yMax = yEdges[bins]
  // Diagonal y = x line — clip to the data extent so it doesn't extend off
  // the plot.
  const diagLo = Math.max(xMin, yMin)
  const diagHi = Math.min(xMax, yMax)

  const fg = isDark ? '#e4e4e4' : '#222'
  const grid = isDark ? '#2f343a' : '#e0e0e0'
  const heatmapAlpha = isDark ? 1.0 : 1.0
  // Custom colorscale: in dark mode the default Viridis lower end
  // (`#440154` deep purple) reads as background, hiding 1-count cells.
  // We compress the ramp so even the smallest log10(1+1)=0.3 stop lands
  // at a clearly-saturated blue, then progress green → yellow → red. The
  // visual effect: every non-zero cell is visible; ratios of counts read
  // off the ramp.
  const colorscale: Plotly.ColorScale = isDark
    ? [
        [0.0, '#0a2540'],   // floor — for `log10(count+1)=0` (empty/NaN)
        [0.001, '#3a78b7'], // bright blue at the first non-empty stop
        [0.25, '#4caf50'],  // green
        [0.55, '#ffd54f'],  // yellow
        [1.0, '#ff5252'],   // red — peak diagonal cells
      ]
    : [
        [0.0, '#ffffff'],
        [0.001, '#bbdefb'],
        [0.25, '#42a5f5'],
        [0.55, '#ffa726'],
        [1.0, '#d32f2f'],
      ]

  const heat: Plotly.Data = {
    type: 'heatmap',
    x: xCenters,
    y: yCenters,
    z,
    colorscale,
    showscale: true,
    zsmooth: false,
    opacity: heatmapAlpha,
    hovertemplate:
      `${xLabel}=%{x:.3g}<br>${yLabel}=%{y:.3g}<br>count=%{z:,.0f}` +
      `<br>%{customdata:.3%} of total<extra></extra>`,
    customdata: total > 0
      ? z.map(row => row.map(v => (Number.isFinite(v) ? v / total : NaN)))
      : z,
    colorbar: {
      // `z` carries `log10(count+1)` under `scale: 'log'` (see transform
      // below) but the user thinks in raw counts. Tick labels are pinned
      // to round powers of 10 in count-space; the underlying tick
      // POSITIONS are the same numbers transformed (1 → log10(2), 10 →
      // log10(11), 100 → log10(101), …). Computed lazily after we know
      // `maxCount`; see the `scale === 'log'` block below.
      title: { text: 'count', font: { color: fg } },
      len: showMarginals ? 0.7 : 0.9,
      x: showMarginals ? 1.12 : 1.02,
      y: showMarginals ? 0.39 : 0.5,
      tickfont: { color: fg },
      thickness: 12,
    },
    xaxis: 'x',
    yaxis: 'y',
  }
  // Plotly's heatmap has no `colorscale='log'` shortcut, so we transform
  // the data: feed it `log10(count + 1)`. Zero cells stay at 0 (mapped
  // to the colorscale's floor stop); 1-count cells jump to log10(2)=0.301
  // (mapped to bright blue); peak cells land at log10(maxCount+1) (red).
  if (scale === 'log') {
    let maxCount = 0
    for (let i = 0; i < H.length; i++) if (H[i] > maxCount) maxCount = H[i]
    const zLog: number[][] = []
    for (let yi = 0; yi < bins; yi++) {
      const row: number[] = []
      for (let xi = 0; xi < bins; xi++) {
        const v = H[yi * bins + xi]
        row.push(v > 0 ? Math.log10(v + 1) : 0)
      }
      zLog.push(row)
    }
    ;(heat as { z: number[][] }).z = zLog
    // Pin the colorscale range explicitly so Plotly doesn't auto-fit it
    // away. zmin=0 anchors empty cells to the floor color; zmax=log10
    // (maxCount + 1) puts the brightest diagonal cells at the top stop.
    ;(heat as { zauto: boolean; zmin: number; zmax: number }).zauto = false
    ;(heat as { zauto: boolean; zmin: number; zmax: number }).zmin = 0
    ;(heat as { zauto: boolean; zmin: number; zmax: number }).zmax =
      maxCount > 0 ? Math.log10(maxCount + 1) : 1
    // Update the hovertemplate so the user sees the raw count, not log10(cnt).
    const rawCustom: number[][] = []
    for (let yi = 0; yi < bins; yi++) {
      const row: number[] = []
      for (let xi = 0; xi < bins; xi++) {
        const v = H[yi * bins + xi]
        row.push(v)
      }
      rawCustom.push(row)
    }
    ;(heat as { customdata: number[][] }).customdata = rawCustom
    ;(heat as { hovertemplate: string }).hovertemplate =
      `${xLabel}=%{x:.3g}<br>${yLabel}=%{y:.3g}<br>count=%{customdata:,.0f}<extra></extra>`
    // Colorbar ticks: pin to powers of 10 in count-space (1, 10, 100,
    // 1k, …). z-values are `log10(count+1)`, so tickvals are
    // `log10(N+1)` for each chosen `N`; ticktext is the human-readable
    // count. Always include 0 (the empty-bin floor) and a 1-count tick
    // (smallest meaningful signal). Above 1 we step by decades to the
    // largest decade ≤ maxCount.
    const tickCounts: number[] = [0, 1]
    for (let p = 1; ; p++) {
      const v = 10 ** p
      if (v > maxCount) break
      tickCounts.push(v)
    }
    const fmtCount = (n: number): string => {
      if (n === 0) return '0'
      if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(0)}M`
      if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`
      return String(n)
    }
    ;(heat as { colorbar: Record<string, unknown> }).colorbar = {
      ...((heat as { colorbar: Record<string, unknown> }).colorbar ?? {}),
      tickvals: tickCounts.map(n => Math.log10(n + 1)),
      ticktext: tickCounts.map(fmtCount),
    }
  }

  // 1-D marginal histograms as bar traces, sharing the main trace's x/y
  // axes via `xaxis: 'x2'` / `yaxis: 'y2'` Plotly conventions.
  const xMargTrace: Plotly.Data = {
    type: 'bar',
    x: xCenters,
    y: Array.from(xMarg),
    marker: { color: isDark ? '#90caf9' : '#1976d2' },
    showlegend: false,
    hovertemplate: `${xLabel}=%{x:.3g}<br>count=%{y:,.0f}<extra></extra>`,
    xaxis: 'x',
    yaxis: 'y2',
  }
  const yMargTrace: Plotly.Data = {
    type: 'bar',
    x: Array.from(yMarg),
    y: yCenters,
    orientation: 'h',
    marker: { color: isDark ? '#a5d6a7' : '#388e3c' },
    showlegend: false,
    hovertemplate: `${yLabel}=%{y:.3g}<br>count=%{x:,.0f}<extra></extra>`,
    xaxis: 'x2',
    yaxis: 'y',
  }

  const traces: Plotly.Data[] = showMarginals
    ? [heat, xMargTrace, yMargTrace]
    : [heat]

  // Subplot domains: main heat = bottom-left 75%×75%, x-marg = top strip,
  // y-marg = right strip. When marginals are off the main fills 100%.
  const mainXDomain: [number, number] = showMarginals ? [0, 0.78] : [0, 1]
  const mainYDomain: [number, number] = showMarginals ? [0, 0.78] : [0, 1]
  const margXDomain: [number, number] = [0.82, 1]
  const margYDomain: [number, number] = [0.82, 1]

  const axType = scale === 'log' ? 'log' : 'linear'

  // Heatmap axis ranges — if scale=log, Plotly wants log-space ranges
  // (log10). For zoom-from-URL we accept raw data-space and convert.
  const toAxRange = (r: [number, number] | null, defLo: number, defHi: number): [number, number] => {
    const [lo, hi] = r ?? [defLo, defHi]
    return axType === 'log' ? [Math.log10(lo), Math.log10(hi)] : [lo, hi]
  }

  const xAxRange = toAxRange(xRange, xMin, xMax)
  const yAxRange = toAxRange(yRange, yMin, yMax)

  // `title` + the per-axis stats line + on-diag stat are rendered as
  // SELECTABLE HTML above the Plot (see JSX in the wrapper component).
  // Leaving Plotly's `title:` empty means margin.t only needs to clear
  // the top marginal's count-axis tick labels (~30px), not 4 stacked
  // lines of `<sub>` SVG (was 130px and crowded the GT marginal —
  // Yael's CIC).
  const layout: Partial<Layout> = {
    autosize: true,
    margin: { t: 30, r: 80, b: 60, l: 80 },
    showlegend: false,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: isDark ? '#1d2125' : '#ffffff',
    font: { color: fg },
    hoverlabel: themedHoverlabel(isDark),
    xaxis: {
      type: axType,
      domain: mainXDomain,
      range: xAxRange,
      title: { text: `${xLabel} (e/Å³)` },
      gridcolor: grid,
      zeroline: false,
    },
    yaxis: {
      type: axType,
      domain: mainYDomain,
      range: yAxRange,
      title: { text: `${yLabel} (e/Å³)` },
      gridcolor: grid,
      zeroline: false,
      // Lock the heatmap to a 1:1 data-aspect ONLY when marginals are
      // showing — the marginal subplots are positioned alongside fixed
      // (0..0.78) heatmap domains and we want the heatmap to remain
      // square-in-data-space so y=x reads at 45°. With marginals off the
      // heatmap fills the full plot area; constraining scaleratio there
      // strands the heatmap as a small centered square with empty bands on
      // both sides (the bug we're fixing).
      ...(showMarginals ? { scaleanchor: 'x' as const, scaleratio: 1 } : {}),
    },
    // Count-axis treatment for both marginals — small, faded ticks +
    // gridlines so the reader can put a magnitude on the peak heights
    // ("is that 1k or 10k?") without the axis dominating the layout.
    // Shared `margCountMax` range (see above) means the two count axes
    // tick at the same values — visually-equal bars represent equal
    // counts.
    xaxis2: {
      domain: margXDomain,
      type: 'linear',
      showticklabels: true,
      showgrid: true,
      gridcolor: grid,
      zeroline: false,
      nticks: 3,
      tickfont: { size: 9, color: fg },
      tickformat: '~s',  // SI suffix: 1.2k / 3.4k / 12k
      range: [0, margCountMax],
    },
    yaxis2: {
      domain: margYDomain,
      type: 'linear',
      showticklabels: true,
      showgrid: true,
      gridcolor: grid,
      zeroline: false,
      nticks: 3,
      tickfont: { size: 9, color: fg },
      tickformat: '~s',
      range: [0, margCountMax],
    },
    shapes: [
      // y = x diagonal across the heatmap's data extent. In log mode Plotly
      // computes positions in data space natively, so we don't need to
      // pre-log the endpoints.
      {
        type: 'line',
        xref: 'x',
        yref: 'y',
        x0: diagLo,
        x1: diagHi,
        y0: diagLo,
        y1: diagHi,
        line: { color: isDark ? '#ff5252' : '#d32f2f', width: 1.2, dash: 'dash' },
      },
    ],
    annotations: [],
  }

  return { traces, layout }
}

function bincenters(edges: Float64Array): number[] {
  const n = edges.length - 1
  const out: number[] = new Array(n)
  for (let i = 0; i < n; i++) out[i] = 0.5 * (edges[i] + edges[i + 1])
  return out
}

