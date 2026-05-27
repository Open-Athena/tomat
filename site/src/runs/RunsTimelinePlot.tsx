// Cross-run timeline: one line per run, x-axis mode selectable.
//
// - clock:   y = running-max global_step vs absolute wallclock (when runs ran)
// - elapsed: same, but x = hours since each run's own start (aligns at t=0)
// - loss:    y = train/loss vs global_step (overlaid training curves)
//
// Intended as the at-a-glance "what's happening across all my training jobs?"
// visual at the top of the /runs index.

import { useEffect, useRef, useState } from 'react'
import { LegendItem, Plot, useTheme } from 'pltly/react'
import { Tooltip } from '../Tooltip'
import type { UseTraceHighlightReturn } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'
import { ALL_TAGS, tagsFor, type RunTag } from './tags'
import { compileMultiTermFilter } from './filter'
// Re-export so RunsPage can `import { ... } from './RunsTimelinePlot'` —
// keeps callers off the helper module's internal path.
export { compileMultiTermFilter, runHaystack, FILTER_EXAMPLES } from './filter'
export type { MultiTermFilter } from './filter'

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
  /** Per-run-id haystack string for the name-filter regex. The parent owns
   *  manifest + iris + history (the inputs to the haystack) so it builds the
   *  string per run; the plot just keys into the map. When omitted, the plot
   *  falls back to `[shortLabel, ...tagsFor(label)].join(' ')` (the haystack
   *  this component computed before the multi-term filter landed). */
  runHaystacks?: Map<string, string>
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
const NAME_FILTER_KEY = 'tomat:runs-name-filter'
const NAME_FILTER_EVT = 'tomat:runs-name-filter-change'
const TAG_FILTER_KEY = 'tomat:runs-tag-filter'

/**
 * Cross-component reactive store for the (multi-term regex) name-filter input.
 *
 * Sources of truth (precedence on load):
 *   1. URL `?regex=…` query param — shareable / linkable
 *   2. localStorage — survives page reloads
 *
 * The query-param name `regex` is preserved for backward compatibility with
 * pre-existing shareable links; the value's grammar is whatever the current
 * filter compiler accepts (today: whitespace=AND multi-term regex via
 * `compileMultiTermFilter` in ./filter.ts).
 *
 * On change: writes both back (URL via replaceState, no extra history entry).
 * Same-tab `localStorage`/popstate events don't always fire, so the setter
 * also dispatches a CustomEvent so every hook instance stays in sync.
 *
 * Read the URL via hash-router-aware extractor: the app uses HashRouter
 * (`#/runs?regex=…`), so we look at `window.location.hash`, not `.search`.
 */
function readUrlRegex(): string | null {
  if (typeof window === 'undefined') return null
  // Hash is like `#/runs?regex=cont33k|mg-4`. Split on `?` and parse the rest.
  const hash = window.location.hash || ''
  const qIdx = hash.indexOf('?')
  if (qIdx < 0) return null
  try {
    const params = new URLSearchParams(hash.slice(qIdx + 1))
    return params.get('regex')
  } catch { return null }
}

function writeUrlRegex(v: string): void {
  if (typeof window === 'undefined') return
  const hash = window.location.hash || '#/'
  const qIdx = hash.indexOf('?')
  const path = qIdx < 0 ? hash : hash.slice(0, qIdx)
  let params: URLSearchParams
  try { params = new URLSearchParams(qIdx < 0 ? '' : hash.slice(qIdx + 1)) }
  catch { params = new URLSearchParams() }
  if (v) params.set('regex', v); else params.delete('regex')
  const qs = params.toString()
  const next = qs ? `${path}?${qs}` : path
  if (next !== hash) {
    try { window.history.replaceState(null, '', next) } catch { /* ignore */ }
  }
}

export function useNameFilter(): readonly [string, (v: string) => void] {
  const [filter, setFilter] = useState(() => {
    // Precedence: URL ?regex= → localStorage → empty.
    const fromUrl = readUrlRegex()
    if (fromUrl != null) return fromUrl
    try { return localStorage.getItem(NAME_FILTER_KEY) ?? '' } catch { return '' }
  })
  useEffect(() => {
    // If the page loaded with a URL filter, sync it to localStorage on mount
    // so it persists across tabs / reloads that drop the query string.
    const fromUrl = readUrlRegex()
    if (fromUrl != null) {
      try { localStorage.setItem(NAME_FILTER_KEY, fromUrl) } catch { /* ignore */ }
    }
    const onEvt = (e: Event) => {
      const detail = (e as CustomEvent<string>).detail
      setFilter(typeof detail === 'string' ? detail : '')
    }
    // Hash change can fire from manual URL edits / back-forward navigation —
    // re-read so the input stays in sync.
    const onHash = () => {
      const v = readUrlRegex()
      if (v != null) setFilter(v)
    }
    window.addEventListener(NAME_FILTER_EVT, onEvt)
    window.addEventListener('hashchange', onHash)
    return () => {
      window.removeEventListener(NAME_FILTER_EVT, onEvt)
      window.removeEventListener('hashchange', onHash)
    }
  }, [])
  const update = (v: string) => {
    setFilter(v)
    try { localStorage.setItem(NAME_FILTER_KEY, v) } catch { /* ignore */ }
    writeUrlRegex(v)
    window.dispatchEvent(new CustomEvent(NAME_FILTER_EVT, { detail: v }))
  }
  return [filter, update] as const
}

/** Compile the persisted name-filter to a single case-insensitive regex.
 *  @deprecated Use `compileMultiTermFilter` for the new whitespace=AND syntax.
 *  Retained as a back-compat shim so any straggling import still typechecks;
 *  internally it just delegates to the multi-term compiler and exposes the
 *  same `{re, error}` shape the original returned (with `re` matching the
 *  whole input as one regex when there's a single term). */
export function compileNameFilter(filter: string): { re: RegExp | null; error: boolean } {
  if (filter.trim() === '') return { re: null, error: false }
  try { return { re: new RegExp(filter, 'i'), error: false } }
  catch { return { re: null, error: true } }
}

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

export function RunsTimelinePlot({ runs, hoursBack, highlight, runHaystacks }: Props) {
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

  // Regex filter on run name. Persisted + cross-component-synced (RunsPage
  // reads the same hook to sort matching cards to top + fade non-matches).
  // New multi-term syntax: whitespace = AND across regexes, `|` keeps regex-OR
  // semantics within a term. Invalid regex in ANY term → whole filter invalid
  // (red border + no filtering applied), same UX as the old single-regex.
  const [nameFilter, setNameFilter] = useNameFilter()
  const filterCompiled = compileMultiTermFilter(nameFilter)
  const nameReError = filterCompiled.error
  // Haystack per run: parent-supplied (rich: includes dates, hardware,
  // lineage) when present, else fall back to label+tags (what we did before
  // the multi-term filter landed).
  const haystackFor = (label: string): string =>
    runHaystacks?.get(label) ?? [label, ...tagsFor(label)].join(' ')
  const nameFilteredRuns = filterCompiled.empty || filterCompiled.error
    ? runs
    : runs.filter((r) => filterCompiled.matches(haystackFor(r.label)))

  // Tag-chip filter. AND across selected tags: a run is visible iff EVERY
  // selected tag is in its tag set. Empty set = no tag constraint (show all).
  // Untagged runs are hidden whenever any tag is selected (since they can't
  // satisfy an "AND" with a tag they don't have). Persisted as JSON array.
  const [selectedTags, setSelectedTagsRaw] = useState<Set<RunTag>>(() => {
    try {
      const raw = localStorage.getItem(TAG_FILTER_KEY)
      if (raw) return new Set(JSON.parse(raw) as RunTag[])
    } catch { /* ignore */ }
    return new Set()
  })
  const toggleTag = (t: RunTag) => setSelectedTagsRaw((prev) => {
    const next = new Set(prev)
    if (next.has(t)) next.delete(t); else next.add(t)
    try { localStorage.setItem(TAG_FILTER_KEY, JSON.stringify([...next])) } catch { /* ignore */ }
    return next
  })
  // Only show chips for tags that ANY currently-visible (name-filtered) run
  // carries — keeps the chip strip short and relevant.
  const visibleTagSet = new Set<RunTag>()
  for (const r of nameFilteredRuns) for (const t of tagsFor(r.label)) visibleTagSet.add(t)
  const chipTags = ALL_TAGS.filter((t) => visibleTagSet.has(t))

  const filteredRuns = selectedTags.size === 0
    ? nameFilteredRuns
    : nameFilteredRuns.filter((r) => {
      const ts = tagsFor(r.label)
      for (const sel of selectedTags) if (!ts.includes(sel)) return false
      return true
    })

  const cutoffSec = hoursBack ? (Date.now() / 1000 - hoursBack * 3600) : null

  const activeTrace = highlight?.activeTrace ?? null
  // When a run is pinned, plot ONLY that run so Plotly autoranges (x + y) to
  // its extent — short/small runs are invisible squished against the shared
  // axis. Pinning is driven by clicking a run card or legend item.
  const pinnedTrace = highlight?.pinnedTrace ?? null
  const plotted = pinnedTrace
    ? filteredRuns.filter((r) => r.label === pinnedTrace)
    : filteredRuns

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

  // User-driven x-zoom selection. Sticky until the user double-clicks the plot
  // (Plotly's auto-range gesture). Without this, every parent re-render
  // recomputes the auto-ranged `xaxis` and overwrites Plotly's internal zoom.
  // Reset when xMode changes (different x scale = different range units).
  const [userXRange, setUserXRange] = useState<[number | string, number | string] | null>(null)
  useEffect(() => { setUserXRange(null) }, [xMode])

  // Closest-trace-on-hover: with `hovermode: 'x unified'` Plotly hands us every
  // trace's value at the cursor's x; we pick the one whose y is closest to the
  // cursor's y (in data coords) and route it through `highlight.setHoverTrace`.
  // That paints the legend item active, fades the others, and lets the parent
  // float the matching run card to the top of the list (see RunsPage.displayed).
  //
  // `closestTraceRef` mirrors the latest closest-trace name without re-rendering
  // on every mousemove — the plotly_click handler reads it to know what to pin.
  const closestTraceRef = useRef<string | null>(null)
  const plotWrapperRef = useRef<HTMLDivElement | null>(null)
  const handlePlotHover = (event: { points?: Array<Record<string, unknown>>; event?: MouseEvent }) => {
    const points = event.points
    if (!points || points.length === 0) return
    // Cursor pixel y → data y via the first point's y-axis. All points share
    // the same axis here (single subplot), so any point's `yaxis` works.
    const first = points[0] as Record<string, unknown> & {
      yaxis?: { p2d?: (p: number) => number; l2p?: (l: number) => number; _offset?: number; d2l?: (v: number) => number }
    }
    const yaxis = first.yaxis
    const mouseEvt = event.event
    if (!yaxis || !mouseEvt) return
    const plotEl = (mouseEvt.target as HTMLElement | null)?.closest('.plotly') as HTMLElement | null
    if (!plotEl) return
    const rect = plotEl.getBoundingClientRect()
    const cursorPx = mouseEvt.clientY - rect.top
    const yOffset = yaxis._offset ?? 0
    // Plotly's `p2d` converts pixel-within-axis to a data value (axis-aware:
    // handles log scales etc.). We pass cursor-y relative to the axis origin.
    const cursorY = yaxis.p2d ? yaxis.p2d(cursorPx - yOffset) : null
    if (cursorY == null || !Number.isFinite(cursorY)) return
    let bestName: string | null = null
    let bestDist = Infinity
    for (const p of points) {
      const py = (p as { y?: number | null }).y
      const data = (p as { data?: { name?: string } }).data
      if (py == null || !Number.isFinite(py) || !data?.name) continue
      // On a log y-axis, comparing |y_trace − y_cursor| in data space biases
      // toward small values (a 10×-distant trace near y=1 looks closer than a
      // 1.5×-distant trace near y=10). For loss-vs-step (the only log-capable
      // mode) we compare in log space when logY is on, linear otherwise.
      const dist = (xMode === 'loss' && logY)
        ? Math.abs(Math.log(Math.max(py, 1e-9)) - Math.log(Math.max(cursorY as number, 1e-9)))
        : Math.abs(py - (cursorY as number))
      if (dist < bestDist) {
        bestDist = dist
        bestName = data.name
      }
    }
    if (bestName !== closestTraceRef.current) {
      closestTraceRef.current = bestName
      highlight?.setHoverTrace(bestName)
    }
  }
  const handlePlotUnhover = () => {
    // Don't immediately clear — Plotly fires unhover when the cursor moves
    // between samples on the same trace. The hook's debounce (debounceMs in
    // useTraceHighlight) absorbs the brief gap when the next hover arrives.
    closestTraceRef.current = null
    highlight?.setHoverTrace(null)
  }

  // Plot click → pin the closest trace. Plot doesn't expose `onClick` as a
  // prop, but Plotly's emitter mounted on `.js-plotly-plot` is reachable from
  // our wrapper ref. Re-bind on every mount; the cleanup pulls the listener
  // back off.
  useEffect(() => {
    const root = plotWrapperRef.current
    if (!root || !highlight) return
    // Plotly's div has class `js-plotly-plot` AND its emitter (`.on/.removeListener`).
    // The element appears after Plotly's first react() call; poll briefly.
    let cancelled = false
    let plotEl: (HTMLElement & {
      on?: (evt: string, fn: (e: unknown) => void) => void
      removeListener?: (evt: string, fn: (e: unknown) => void) => void
    }) | null = null
    const handler = (_e: unknown) => {
      const name = closestTraceRef.current
      if (name) highlight.togglePin(name)
    }
    const attach = () => {
      if (cancelled) return
      const el = root.querySelector('.js-plotly-plot') as typeof plotEl
      if (!el?.on) {
        // Not ready yet; try again on next frame. Bail after ~1s of trying.
        requestAnimationFrame(attach)
        return
      }
      plotEl = el
      el.on('plotly_click', handler)
    }
    attach()
    return () => {
      cancelled = true
      plotEl?.removeListener?.('plotly_click', handler)
    }
    // Re-attach when highlight changes (different `togglePin` closure).
  }, [highlight])
  const handleRelayout = (ev: Record<string, unknown>) => {
    // Double-click resets to auto-range.
    if (ev['xaxis.autorange'] === true) {
      setUserXRange(null)
      return
    }
    const x0 = ev['xaxis.range[0]']
    const x1 = ev['xaxis.range[1]']
    if (x0 != null && x1 != null) {
      setUserXRange([x0 as number | string, x1 as number | string])
    }
  }
  const xaxisBase = xMode === 'clock'
    ? { type: 'date' as const, gridcolor, zerolinecolor, linecolor: gridcolor }
    : {
      type: 'linear' as const,
      title: {
        text: xMode === 'rel' ? 'elapsed (h)'
          : xMode === 'active' ? 'active (h)' : 'step',
      },
      gridcolor, zerolinecolor, linecolor: gridcolor,
    }
  const xaxis = userXRange
    ? { ...xaxisBase, range: userXRange, autorange: false as const }
    : xaxisBase

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
      {/* Tag chips. Click to AND-filter; visible tags only. */}
      {chipTags.length > 0 && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 4,
          alignItems: 'center', justifyContent: 'flex-end',
        }}>
          <span style={{ fontSize: '0.7rem', color: muted, marginRight: 4 }}>
            tags:
          </span>
          {chipTags.map((t) => {
            const on = selectedTags.has(t)
            return (
              <Tooltip key={t} content={`${on ? 'remove' : 'add'} "${t}" filter (AND across selected)`}>
                <button
                  onClick={() => toggleTag(t)}
                  style={{
                    background: on ? (isDark ? '#2a4a7a' : '#cbe0f5') : 'transparent',
                    border: `1px solid ${on ? (isDark ? '#3a6ab0' : '#7aa7d9') : (isDark ? '#333' : '#ccc')}`,
                    borderRadius: 10, cursor: 'pointer', padding: '1px 8px',
                    fontSize: '0.7rem', fontFamily: 'inherit',
                    color: on ? (isDark ? '#cfe2ff' : '#1d3a64') : muted,
                  }}
                >
                  {t}
                </button>
              </Tooltip>
            )
          })}
          {selectedTags.size > 0 && (
            <button
              onClick={() => { setSelectedTagsRaw(new Set()); try { localStorage.removeItem(TAG_FILTER_KEY) } catch { /* ignore */ } }}
              title="clear all tag filters"
              style={{
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: muted, fontSize: '0.7rem', padding: '1px 4px',
                fontFamily: 'inherit',
              }}
            >
              ✕ clear
            </button>
          )}
        </div>
      )}
      {/* x-axis mode toggle + name-filter regex */}
      <div style={{
        display: 'flex', justifyContent: 'flex-end', gap: 4, marginBottom: 4,
        alignItems: 'center',
      }}>
        <Tooltip content={
          'multi-term regex filter (case-insensitive). Whitespace = AND across terms; '
          + '`|` is regex OR within a term. Haystack includes shortLabel + tags + '
          + 'YYMMDD (created + last activity) + hardware (TPU / GPU / modal / v6e-16 / h100x8) '
          + '+ lineage (resume / scratch). Examples: '
          + '`noprm` · `SS-sweep 260527` · `TPU resume|scratch`.'
        }>
          <input
            type="text"
            value={nameFilter}
            onChange={(e) => setNameFilter(e.target.value)}
            placeholder="filter (e.g. SS-sweep 260527)"
            style={{
              background: 'transparent',
              border: `1px solid ${nameReError ? '#c44' : (isDark ? '#444' : '#bbb')}`,
              borderRadius: 4, padding: '2px 8px',
              fontSize: '0.72rem', fontFamily: 'inherit',
              color: nameReError ? '#c44' : fg,
              width: '220px', marginRight: 8,
            }}
          />
        </Tooltip>
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
      <div ref={plotWrapperRef}>
      <Plot
        data={data}
        // Fade is applied in `data` above (true grey), not via pltly's
        // highlight fade. `disableSoloTrace` stops a stray click on a plotted
        // line from creating a solo state disconnected from our `highlight`.
        disableSoloTrace
        onHover={handlePlotHover}
        onUnhover={handlePlotUnhover}
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
        onRelayout={handleRelayout as never}
      />
      </div>
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
          {collapsed ? '▸' : '▾'} legend · {filteredRuns.length}
          {filteredRuns.length !== runs.length ? ` of ${runs.length}` : ''} run
          {filteredRuns.length === 1 ? '' : 's'}
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
            {filteredRuns.map((r) => (
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
