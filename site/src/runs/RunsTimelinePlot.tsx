// Cross-run timeline: one line per run, x-axis mode selectable.
//
// - clock:   y = running-max global_step vs absolute wallclock (when runs ran)
// - elapsed: same, but x = hours since each run's own start (aligns at t=0)
// - loss:    y = train/loss vs global_step (overlaid training curves)
//
// Intended as the at-a-glance "what's happening across all my training jobs?"
// visual at the top of the /runs index.

import { useState } from 'react'
import { LegendItem, Plot, useTheme } from 'pltly/react'
import type { UseTraceHighlightReturn } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'

export interface RunTimelineSeries {
  id: string
  history: RunHistory
  /** Short label for legend/hover (run name without the train-full-v3 prefix). */
  label: string
  /** Hex color string. */
  color: string
}

interface Props {
  runs: RunTimelineSeries[]
  /** Optional time-window cutoff in hours (default: full history). */
  hoursBack?: number
  /** Shared trace-highlight state machine (from `useTraceHighlight` in parent).
   *  Drives fade/solo + lets cards + the custom legend brush the plot. */
  highlight?: UseTraceHighlightReturn
}

/** Local-TZ datetime string for a Plotly date axis. Plotly treats a string
 *  with no `Z`/offset as timezone-naive and renders it verbatim — so by
 *  formatting with the browser's local getters we get wallclock in the
 *  viewer's timezone rather than UTC. */
function localDateStr(ms: number): string {
  const d = new Date(ms)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} `
    + `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

type XMode = 'clock' | 'rel' | 'loss'
const X_MODES: { id: XMode; label: string; help: string }[] = [
  { id: 'clock', label: 'clock', help: 'absolute wallclock — when each run was active' },
  { id: 'rel', label: 'elapsed', help: 'hours since each run’s own start — aligns runs at t=0' },
  { id: 'loss', label: 'loss vs step', help: 'training-loss curves against step' },
]
const X_MODE_KEY = 'tomat:runs-xmode'
const LEGEND_COLLAPSED_KEY = 'tomat:runs-legend-collapsed'

/** One run's (x, y) series for the given x-axis mode.
 *  clock/rel: y = running-max `global_step` (flats = idle/preempt).
 *  loss:      y = `train/loss` vs `global_step`. */
function traceFor(history: RunHistory, mode: XMode, cutoffSec: number | null): {
  x: (string | number)[]; y: number[]
} {
  const { timestamps, cols, rowCount } = history
  const globalStep = cols.get('global_step') ?? []

  if (mode === 'loss') {
    const loss = cols.get('train/loss') ?? []
    const pts: { s: number; l: number }[] = []
    for (let i = 0; i < rowCount; i++) {
      const s = globalStep[i], l = loss[i]
      if (s != null && l != null) pts.push({ s, l })
    }
    pts.sort((a, b) => a.s - b.s)
    return { x: pts.map((p) => p.s), y: pts.map((p) => p.l) }
  }

  // clock / rel: running max of global_step along ascending _timestamp.
  const ordered = timestamps
    .map((ts, i) => ({ ts, i }))
    .filter((r) => r.ts !== null && (cutoffSec == null || (r.ts as number) >= cutoffSec))
    .sort((a, b) => (a.ts as number) - (b.ts as number))
  const t0 = ordered.length ? (ordered[0].ts as number) : 0
  const x: (string | number)[] = []
  const y: number[] = []
  let runningMax = -Infinity
  for (const { ts, i } of ordered) {
    const s = globalStep[i]
    if (s == null) continue
    runningMax = Math.max(runningMax, s)
    const tsec = ts as number
    x.push(mode === 'rel' ? (tsec - t0) / 3600 : localDateStr(tsec * 1000))
    y.push(runningMax)
  }
  return { x, y }
}

export function RunsTimelinePlot({ runs, hoursBack, highlight }: Props) {
  const { isDark } = useTheme()

  // Legend collapse persists in localStorage — it's long, some users tuck it
  // away once they've keyed the colours to the cards below.
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(LEGEND_COLLAPSED_KEY) === '1' } catch { return false }
  })
  const toggleCollapsed = () => setCollapsed((c) => {
    const next = !c
    try { localStorage.setItem(LEGEND_COLLAPSED_KEY, next ? '1' : '0') } catch { /* ignore */ }
    return next
  })

  // x-axis mode also persists.
  const [xMode, setXModeRaw] = useState<XMode>(() => {
    try {
      const v = localStorage.getItem(X_MODE_KEY)
      if (v === 'clock' || v === 'rel' || v === 'loss') return v
    } catch { /* ignore */ }
    return 'clock'
  })
  const setXMode = (m: XMode) => {
    setXModeRaw(m)
    try { localStorage.setItem(X_MODE_KEY, m) } catch { /* ignore */ }
  }

  const cutoffSec = hoursBack ? (Date.now() / 1000 - hoursBack * 3600) : null

  const activeTrace = highlight?.activeTrace ?? null
  // When a run is pinned, plot ONLY that run so Plotly autoranges (x + y) to
  // its extent — short/small runs are invisible squished against the shared
  // axis. Pinning is driven by clicking a run card or legend item.
  const pinnedTrace = highlight?.pinnedTrace ?? null
  const plotted = pinnedTrace ? runs.filter((r) => r.label === pinnedTrace) : runs

  // Fade non-highlighted traces to a true neutral grey (pltly's built-in fade
  // only desaturates partway / keeps a tint); the highlighted run keeps full
  // colour + a thicker stroke so it pops.
  const data = plotted
    .map((r) => {
      const { x, y } = traceFor(r.history, xMode, cutoffSec)
      if (x.length === 0) return null
      const isActive = r.label === activeTrace
      const faded = activeTrace != null && !isActive
      return {
        x,
        y,
        name: r.label,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: {
          color: faded ? '#666' : r.color,
          width: isActive ? 3 : 2,
          // step-progress curves are step functions; loss is continuous.
          shape: (xMode === 'loss' ? 'linear' : 'hv') as 'linear' | 'hv',
        },
        hovertemplate: xMode === 'loss'
          ? `loss %{y:.3f}<extra></extra>`
          : `step %{y:,}<extra></extra>`,
      }
    })
    .filter((d): d is NonNullable<typeof d> => d !== null)

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
  const fg = isDark ? '#bbb' : '#444'
  const muted = isDark ? '#888' : '#666'

  const xaxis = xMode === 'clock'
    ? { type: 'date' as const, gridcolor, zerolinecolor, linecolor: gridcolor }
    : {
      type: 'linear' as const,
      title: { text: xMode === 'rel' ? 'elapsed (h)' : 'step' },
      gridcolor, zerolinecolor, linecolor: gridcolor,
    }

  return (
    <div>
      {/* x-axis mode toggle */}
      <div style={{
        display: 'flex', justifyContent: 'flex-end', gap: 4, marginBottom: 4,
      }}>
        {X_MODES.map((m) => {
          const on = m.id === xMode
          return (
            <button
              key={m.id}
              onClick={() => setXMode(m.id)}
              title={m.help}
              style={{
                background: on ? (isDark ? '#2a2a2a' : '#e8e8e8') : 'transparent',
                border: `1px solid ${on ? (isDark ? '#444' : '#bbb') : 'transparent'}`,
                borderRadius: 4, cursor: 'pointer', padding: '2px 8px',
                fontSize: '0.72rem', fontFamily: 'inherit',
                color: on ? fg : muted,
              }}
            >
              {m.label}
            </button>
          )
        })}
      </div>
      <Plot
        data={data}
        // Fade is applied in `data` above (true grey), not via pltly's
        // highlight fade. `disableSoloTrace` stops a stray click on a plotted
        // line from creating a solo state disconnected from our `highlight`.
        disableSoloTrace
        layout={{
          autosize: true,
          height: 320,
          // Extra bottom room when there's an x-axis title (rel / loss modes).
          margin: { t: 30, l: 60, r: 12, b: xMode === 'clock' ? 36 : 48 },
          xaxis,
          yaxis: {
            title: { text: xMode === 'loss' ? 'train loss' : 'step' },
            gridcolor, zerolinecolor, linecolor: gridcolor,
          },
          // 'x unified' shows every trace's value at the hover x — much easier
          // to compare runs at a given moment than 'closest' (per-trace).
          hovermode: 'x unified',
          hoverlabel: themedHoverlabel(isDark),
          // Built-in legend off — we render a custom one below, wired to the
          // shared trace-highlight so legend hover brushes the plot + cards.
          showlegend: false,
        }}
      />
      {/* Custom collapsible legend. Each item hovers→highlight, clicks→pin,
          via the shared `useTraceHighlight` handlers (the pltly idiom). */}
      <div style={{ marginTop: 2, fontSize: '0.75rem' }}>
        <button
          onClick={toggleCollapsed}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: muted, fontSize: '0.75rem', padding: '2px 4px',
            fontFamily: 'inherit',
          }}
        >
          {collapsed ? '▸' : '▾'} legend · {runs.length} run{runs.length === 1 ? '' : 's'}
        </button>
        {!collapsed && (
          <div style={{
            display: 'flex', flexWrap: 'wrap', alignItems: 'center',
            marginTop: 2, color: fg,
          }}>
            {runs.map((r) => (
              <LegendItem
                key={r.id}
                type="line"
                color={r.color}
                label={r.label}
                active={highlight?.activeTrace === r.label}
                faded={!!highlight?.activeTrace && highlight.activeTrace !== r.label}
                pinned={highlight?.pinnedTrace === r.label}
                {...(highlight ? highlight.handlers(r.label) : {})}
                style={{ fontSize: '0.72rem' }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// Stable color palette with good distinguishability on dark + light backgrounds.
const PALETTE = [
  '#22c55e', // green
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ec4899', // pink
  '#a855f7', // purple
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#ef4444', // red
  '#8b5cf6', // violet
]

export function colorForIndex(i: number): string {
  return PALETTE[i % PALETTE.length]
}

/** Truncate run name for legend — drop the train-full-v3 / train-cont prefix. */
export function shortLabel(id: string): string {
  return id
    .replace(/^train-full-v3-/, '')
    .replace(/^train-full-/, '')
    .replace(/^train-cont-/, '')
}
