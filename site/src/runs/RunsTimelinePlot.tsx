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
import { Tooltip } from '../Tooltip'
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

type XMode = 'clock' | 'rel' | 'active' | 'loss'
const X_MODES: { id: XMode; label: string; help: string }[] = [
  { id: 'clock', label: 'clock', help: 'absolute wallclock — when each run was active' },
  { id: 'rel', label: 'elapsed', help: 'hours since each run’s own start — aligns runs at t=0' },
  { id: 'active', label: 'active', help: 'training time only — idle / preempt gaps (the flat segments) removed' },
  { id: 'loss', label: 'loss vs step', help: 'training-loss curves against step' },
]
const X_MODE_KEY = 'tomat:runs-xmode'
const LEGEND_COLLAPSED_KEY = 'tomat:runs-legend-collapsed'

// `active` x-mode: cap on a single inter-sample interval's contribution.
// Runs log every ~minute while training, so anything longer is a gap the
// run wasn't scheduled/running — collapse it to this many seconds rather
// than its full wall duration.
const IDLE_CAP_SEC = 300
const LOGY_KEY = 'tomat:runs-logy'

/** q-th quantile (0–1) of an ascending-sorted array. */
function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))
  return sorted[i]
}

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

  // clock / rel / active: running max of global_step along ascending
  // _timestamp. `active` accumulates only the intervals in which the step
  // advanced — idle / preempt stretches (the flat segments) collapse to zero,
  // so the x-axis measures time the run was actually training.
  const ordered = timestamps
    .map((ts, i) => ({ ts, i }))
    .filter((r) => r.ts !== null && (cutoffSec == null || (r.ts as number) >= cutoffSec))
    .sort((a, b) => (a.ts as number) - (b.ts as number))
  const t0 = ordered.length ? (ordered[0].ts as number) : 0
  const x: (string | number)[] = []
  const y: number[] = []
  let runningMax = -Infinity
  let prevTs: number | null = null
  let activeCum = 0
  for (const { ts, i } of ordered) {
    const s = globalStep[i]
    if (s == null) continue
    const tsec = ts as number
    // `advanced` = this sample pushes a new step high → the interval since
    // the previous sample saw real training. Cap it: a preemption gap ends
    // with a step bump when the run resumes, so the gap interval reads as
    // "advanced" — the cap collapses its multi-hour duration to a sliver.
    // Non-advancing intervals (logged-but-stuck, or post-restore catch-up
    // below the running max) contribute nothing.
    const advanced = s > runningMax
    if (mode === 'active' && prevTs !== null && advanced) {
      activeCum += Math.min(tsec - prevTs, IDLE_CAP_SEC)
    }
    runningMax = Math.max(runningMax, s)
    if (mode === 'clock') x.push(localDateStr(tsec * 1000))
    else if (mode === 'rel') x.push((tsec - t0) / 3600)
    else x.push(activeCum / 3600)
    y.push(runningMax)
    prevTs = tsec
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
      if (v === 'clock' || v === 'rel' || v === 'active' || v === 'loss') return v
    } catch { /* ignore */ }
    return 'clock'
  })
  const setXMode = (m: XMode) => {
    setXModeRaw(m)
    try { localStorage.setItem(X_MODE_KEY, m) } catch { /* ignore */ }
  }

  // Log-scale the loss axis (loss-vs-step mode only). Persisted.
  const [logY, setLogYRaw] = useState(() => {
    try { return localStorage.getItem(LOGY_KEY) === '1' } catch { return false }
  })
  const setLogY = (v: boolean) => {
    setLogYRaw(v)
    try { localStorage.setItem(LOGY_KEY, v ? '1' : '0') } catch { /* ignore */ }
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
  // Draw the highlighted trace last so it sits on top of the rest (Plotly
  // z-order = data order). Array.sort is stable → others keep their order.
  if (activeTrace) {
    data.sort((a, b) =>
      Number(a.name === activeTrace) - Number(b.name === activeTrace))
  }

  const gridcolor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const zerolinecolor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)'
  const fg = isDark ? '#bbb' : '#444'
  const muted = isDark ? '#888' : '#666'

  const xaxis = xMode === 'clock'
    ? { type: 'date' as const, gridcolor, zerolinecolor, linecolor: gridcolor }
    : {
      type: 'linear' as const,
      title: {
        text: xMode === 'rel' ? 'elapsed (h)'
          : xMode === 'active' ? 'active (h)' : 'step',
      },
      gridcolor, zerolinecolor, linecolor: gridcolor,
    }

  // Loss mode: clip the y-axis to a robust percentile so spike outliers don't
  // crush the trend band. A percentile (not median+σ — σ is inflated by the
  // very spikes we want to exclude) is robust + predictable; plotly drag-zoom
  // still gives continuous adjustment from there.
  let lossLo = 1
  let lossHi = 10
  if (xMode === 'loss') {
    const vals: number[] = []
    for (const d of data) for (const v of d.y) if (Number.isFinite(v)) vals.push(v)
    vals.sort((a, b) => a - b)
    if (vals.length) {
      lossLo = Math.max(vals[0], 1e-4)
      lossHi = Math.max(quantile(vals, 0.99), lossLo * 1.05)
    }
  }
  const yaxis = xMode === 'loss'
    ? {
      type: (logY ? 'log' : 'linear') as 'log' | 'linear',
      title: { text: 'train loss' },
      // Plotly log axes take `range` in log10 units.
      range: (logY ? [Math.log10(lossLo), Math.log10(lossHi)] : [lossLo, lossHi]) as [number, number],
      gridcolor, zerolinecolor, linecolor: gridcolor,
    }
    : {
      type: 'linear' as const,
      title: { text: 'step' },
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
            <Tooltip key={m.id} content={m.help}>
              <button
                onClick={() => setXMode(m.id)}
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
            </Tooltip>
          )
        })}
        {xMode === 'loss' && (
          <Tooltip content="log-scale the loss axis">
            <button
              onClick={() => setLogY(!logY)}
              style={{
                background: logY ? (isDark ? '#2a2a2a' : '#e8e8e8') : 'transparent',
                border: `1px solid ${logY ? (isDark ? '#444' : '#bbb') : 'transparent'}`,
                borderRadius: 4, cursor: 'pointer', padding: '2px 8px',
                fontSize: '0.72rem', fontFamily: 'inherit',
                color: logY ? fg : muted, marginLeft: 8,
              }}
            >
              log y
            </button>
          </Tooltip>
        )}
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
          yaxis,
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
            // Grid (not flex-wrap) so the columns line up like a table
            // instead of raggedly tracking each label's width.
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(330px, 1fr))',
            columnGap: 14, rowGap: 1,
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
