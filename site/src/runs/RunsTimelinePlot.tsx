// Cross-run timeline: one line per run, x-axis mode selectable.
//
// - clock:   y = running-max global_step vs absolute wallclock (when runs ran)
// - elapsed: same, but x = hours since each run's own start (aligns at t=0)
// - loss:    y = train/loss vs global_step (overlaid training curves)
//
// Intended as the at-a-glance "what's happening across all my training jobs?"
// visual at the top of the /runs index.

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactElement } from 'react'
import { LegendItem, Plot, useTheme } from 'pltly/react'
import { rolling } from 'pltly/core'
import { Tooltip } from '../Tooltip'
import type { UseTraceHighlightReturn } from 'pltly/react'
import { themedHoverlabel } from '../theme'
import type { RunHistory } from './parquet'
import { ancestorsOf } from './lineage'
import { ALL_TAGS, cycleTagFilter, effectiveTagState, parseTagFiltersLegacyLs, runPassesTagFilters, tagsFor, TAG_FILTER_PARAM, type RunTag, type TagFilters } from './tags'
import { stringParam, useUrlState, type Param } from 'use-prms'
import { compileMultiTermFilter } from './filter'
import {
  applySmoothing, bandsParam, DEFAULT_EMA_ALPHA, DEFAULT_ROLLING_WINDOW,
  smoothKey, smoothParam, type SmoothMode,
} from './smoothing'
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
  /** Fires when the closest trace under the cursor changes — passes the trace
   *  label (= `RunTimelineSeries.label`) or `null` on unhover. Distinct from
   *  `highlight.setHoverTrace` (which is also driven by card hover): use this
   *  to do plot-hover-only side effects, e.g. floating the matching card to
   *  the top without creating a card-hover feedback loop. */
  onPlotHover?: (label: string | null) => void
  /** Controlled tag-filter state. When omitted the plot owns its own state +
   *  localStorage persistence. The parent passes both to mirror the chip
   *  selection into the card list so tag filters can hard-filter rows, not
   *  just trace plotting.
   *
   *  Tri-state per tag: `'in'` = run must have the tag, `'out'` = run must
   *  not have it, absent = don't care. See `runPassesTagFilters`. */
  tagFilters?: TagFilters
  onTagFiltersChange?: (next: TagFilters) => void
  /** Controlled "include ancestors" toggle. When omitted the chip rail
   *  doesn't render the toggle (it's only meaningful when the parent is
   *  applying the toggle to its card-list filter). */
  ancestorsOn?: boolean
  onAncestorsChange?: (next: boolean) => void
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
export const TAG_FILTER_KEY = 'tomat:runs-tag-filter'

// ── URL params (via use-prms; see `urlStrategy.ts` for the HashRouter strategy)
//
//   ?regex=…   multi-term regex name-filter (string)
//   ?tags=…    tri-state tag-chip filter (use-prms `tagFilterParam`)
//   ?anc=1     include-ancestors toggle (boolean; legacy `?anc=1` encoding
//              preserved for back-compat — `boolParam`'s default ?anc form
//              wouldn't break old links but reads less obviously to users
//              who already learned the `=1` form)
//
// `useNameFilter` keeps a localStorage mirror of `?regex=` so the filter
// survives reloads that drop the query string; the URL stays canonical.

/** `?anc=0|1` boolean. Preserves the existing `=1` encoding (vs use-prms'
 *  built-in `boolParam`, which encodes true as a valueless `?anc`). */
const ancParam: Param<boolean> = {
  encode: (v) => (v ? '1' : undefined),
  decode: (s) => s === '1' || s === 'true' || s === '',  // '' = valueless ?anc
}

/** Hook owning the `anc=1` URL toggle for "include ancestors of matching runs".
 *  Default `false`. */
export function useAncestorsToggle(): readonly [boolean, (v: boolean) => void] {
  const [on, setOn] = useUrlState('anc', ancParam)
  return [on, setOn] as const
}

/** Hook owning the `?smooth=…` smoothing mode. Default `raw`. */
export function useSmoothMode(): readonly [SmoothMode, (v: SmoothMode) => void] {
  const [mode, setMode] = useUrlState('smooth', smoothParam)
  return [mode, setMode] as const
}

/** Hook owning the `?bands=1` ±σ-band toggle. Default `false`. */
export function useBandsToggle(): readonly [boolean, (v: boolean) => void] {
  const [on, setOn] = useUrlState('bands', bandsParam)
  return [on, setOn] as const
}

/** React hook owning the tag-filter OVERRIDE map. URL is canonical; legacy
 *  `localStorage` is migrated on first read then cleared. The Map holds
 *  only overrides — per-tag defaults (see `TAG_DEFAULTS`) apply for any
 *  tag absent from the Map, so an empty Map is the all-defaults state.
 *  No first-visit-default constant needed: defaults live in `TAG_DEFAULTS`
 *  and `effectiveTagState` reads them on demand. */
export function useTagFilters(): readonly [TagFilters, (v: TagFilters) => void] {
  const [filters, setFilters] = useUrlState('tags', TAG_FILTER_PARAM)
  // One-time legacy-localStorage migration. On the first render after the
  // URL upgrade, if the URL has no `?tags=` AND localStorage carries the
  // pre-URL JSON payload, push it into the URL then drop the LS entry.
  useEffect(() => {
    if (typeof window === 'undefined') return
    if (filters.size > 0) return  // URL already has overrides — nothing to do
    let raw: string | null = null
    try { raw = localStorage.getItem(TAG_FILTER_KEY) } catch { /* ignore */ }
    if (!raw) return
    const migrated = parseTagFiltersLegacyLs(raw)
    if (migrated.size > 0) setFilters(migrated)
    try { localStorage.removeItem(TAG_FILTER_KEY) } catch { /* ignore */ }
    // Empty deps — fire once on mount. `setFilters` is stable per the
    // useUrlState contract; including it would just re-run on render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return [filters, setFilters] as const
}

export function useNameFilter(): readonly [string, (v: string) => void] {
  const [urlVal, setUrlVal] = useUrlState('regex', stringParam())
  // localStorage mirror: persists across reloads that drop the query string.
  // URL is canonical when present; LS is the fallback at first paint.
  const [filter, setFilter] = useState<string>(() => {
    if (urlVal != null) return urlVal
    try { return localStorage.getItem(NAME_FILTER_KEY) ?? '' } catch { return '' }
  })
  useEffect(() => {
    if (urlVal != null) {
      setFilter(urlVal)
      try { localStorage.setItem(NAME_FILTER_KEY, urlVal) } catch { /* ignore */ }
    } else {
      // URL has no `?regex=` — keep the local state showing whatever the
      // LS-restored / explicitly-cleared value is. Don't reset to '' on
      // every render: the user may have just typed and the URL write
      // hasn't landed yet (use-prms uses replaceState; same-render reads
      // are fine, but a chain of writes can briefly see the old URL).
    }
  }, [urlVal])
  const update = (v: string) => {
    setFilter(v)
    try { localStorage.setItem(NAME_FILTER_KEY, v) } catch { /* ignore */ }
    setUrlVal(v === '' ? undefined : v)
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

/** `#rrggbb` (or `#rgb`) → `"R, G, B"` for use inside `rgba(…)` strings.
 *  Falls back to mid-grey on parse failure so band fills stay visible. */
function hexToRgb(hex: string): string {
  let h = hex.startsWith('#') ? hex.slice(1) : hex
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  if (h.length !== 6) return '128, 128, 128'
  const n = parseInt(h, 16)
  if (!Number.isFinite(n)) return '128, 128, 128'
  return `${(n >> 16) & 0xff}, ${(n >> 8) & 0xff}, ${n & 0xff}`
}

/** q-th quantile (0–1) of an ascending-sorted array. */
function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))
  return sorted[i]
}

/** One run's (x, y) series for the given x-axis mode.
 *  clock/rel: y = running-max `global_step` (flats = idle/preempt).
 *  loss:      y = `train/loss` vs `global_step`.
 *  `x` is ALWAYS numeric: ms-since-epoch for `clock`, hours for `rel`/`active`,
 *  step for `loss`. Date strings (when needed for the Plotly date axis) are
 *  built downstream from the ms values via `localDateStr`. */
function traceFor(history: RunHistory, mode: XMode, cutoffSec: number | null): {
  x: number[]; y: number[]
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
  const x: number[] = []
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
    if (mode === 'clock') x.push(tsec * 1000)        // ms since epoch
    else if (mode === 'rel') x.push((tsec - t0) / 3600)
    else x.push(activeCum / 3600)
    y.push(runningMax)
    prevTs = tsec
  }
  return { x, y }
}

interface SmoothingChipsProps {
  mode: SmoothMode
  setMode: (m: SmoothMode) => void
  bandsOn: boolean
  setBandsOn: (v: boolean) => void
  isDark: boolean
  fg: string
  muted: string
}

/** Compact 3-button chip rail for the smoothing mode: `raw | EMA(α) | rolling(N)`,
 *  plus a ±σ toggle (disabled while in raw mode). Numeric inputs commit on
 *  Enter/blur — no debounce, no per-keystroke URL writes. Used by both the
 *  cross-run timeline and the per-run wallclock plots. */
export function SmoothingChips({
  mode, setMode, bandsOn, setBandsOn, isDark, fg, muted,
}: SmoothingChipsProps): ReactElement {
  const [emaDraft, setEmaDraft] = useState(() =>
    String(mode.kind === 'ema' ? mode.alpha : DEFAULT_EMA_ALPHA))
  const [rollDraft, setRollDraft] = useState(() =>
    String(mode.kind === 'rolling' ? mode.window : DEFAULT_ROLLING_WINDOW))
  // Re-seed drafts when the mode changes from outside (URL paste, e.g.).
  useEffect(() => {
    if (mode.kind === 'ema') setEmaDraft(String(mode.alpha))
    if (mode.kind === 'rolling') setRollDraft(String(mode.window))
  }, [mode])

  const chipStyle = (on: boolean): CSSProperties => ({
    background: on ? (isDark ? '#2a2a2a' : '#e8e8e8') : 'transparent',
    border: `1px solid ${on ? (isDark ? '#444' : '#bbb') : 'transparent'}`,
    borderRadius: 4, cursor: 'pointer', padding: '2px 8px',
    fontSize: '0.72rem', fontFamily: 'inherit',
    color: on ? fg : muted,
  })
  const numInputStyle = (active: boolean): CSSProperties => ({
    background: 'transparent',
    border: `1px solid ${active ? (isDark ? '#666' : '#999') : (isDark ? '#333' : '#ddd')}`,
    borderRadius: 3, padding: '1px 4px',
    fontSize: '0.7rem', fontFamily: 'inherit',
    color: active ? fg : muted,
    width: 44, marginLeft: 4,
  })

  const commitEma = () => {
    const a = Number(emaDraft)
    if (Number.isFinite(a) && a > 0 && a <= 1) setMode({ kind: 'ema', alpha: a })
    else setEmaDraft(String(mode.kind === 'ema' ? mode.alpha : DEFAULT_EMA_ALPHA))
  }
  const commitRoll = () => {
    const n = Math.round(Number(rollDraft))
    if (Number.isFinite(n) && n >= 1) setMode({ kind: 'rolling', window: n })
    else setRollDraft(String(mode.kind === 'rolling' ? mode.window : DEFAULT_ROLLING_WINDOW))
  }

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, marginLeft: 8 }}>
      <span style={{ fontSize: '0.7rem', color: muted, marginRight: 2 }}>smooth:</span>
      <Tooltip content="no smoothing (raw history)">
        <button onClick={() => setMode({ kind: 'raw' })} style={chipStyle(mode.kind === 'raw')}>
          raw
        </button>
      </Tooltip>
      <Tooltip content="exponential moving average: y[t] = α·x[t] + (1-α)·y[t-1]. Smaller α = smoother. Click chip to toggle off. URL: ?smooth=ema:α">
        <button
          onClick={() => setMode(
            mode.kind === 'ema'
              ? { kind: 'raw' }
              : { kind: 'ema', alpha: Number(emaDraft) || DEFAULT_EMA_ALPHA }
          )}
          style={chipStyle(mode.kind === 'ema')}
        >
          EMA α
          <input
            type="number" step="0.05" min="0.01" max="1"
            value={emaDraft}
            onChange={(e) => setEmaDraft(e.target.value)}
            onBlur={commitEma}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur() }}
            onClick={(e) => e.stopPropagation()}
            style={numInputStyle(mode.kind === 'ema')}
          />
        </button>
      </Tooltip>
      <Tooltip content="centered rolling mean over N samples. Click chip to toggle off. URL: ?smooth=rolling:N">
        <button
          onClick={() => setMode(
            mode.kind === 'rolling'
              ? { kind: 'raw' }
              : { kind: 'rolling', window: Math.max(1, Math.round(Number(rollDraft))) || DEFAULT_ROLLING_WINDOW }
          )}
          style={chipStyle(mode.kind === 'rolling')}
        >
          roll N
          <input
            type="number" step="10" min="1"
            value={rollDraft}
            onChange={(e) => setRollDraft(e.target.value)}
            onBlur={commitRoll}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur() }}
            onClick={(e) => e.stopPropagation()}
            style={numInputStyle(mode.kind === 'rolling')}
          />
        </button>
      </Tooltip>
      <Tooltip content={
        mode.kind === 'rolling'
          ? '±σ bands around the rolling mean — pltly’s within-window stddev. URL: ?bands=1'
          : mode.kind === 'ema'
            ? 'EMA has no natural σ companion; switch to rolling for ±σ bands'
            : '±σ bands require rolling smoothing (raw σ would just be raw vs raw)'
      }>
        <button
          onClick={() => setBandsOn(!bandsOn)}
          disabled={mode.kind !== 'rolling'}
          style={{
            ...chipStyle(bandsOn && mode.kind === 'rolling'),
            opacity: mode.kind === 'rolling' ? 1 : 0.4,
            cursor: mode.kind === 'rolling' ? 'pointer' : 'not-allowed',
            marginLeft: 4,
          }}
        >
          ±σ
        </button>
      </Tooltip>
    </span>
  )
}

export function RunsTimelinePlot({ runs, hoursBack, highlight, runHaystacks, onPlotHover, tagFilters: tagFiltersExternal, onTagFiltersChange, ancestorsOn, onAncestorsChange }: Props) {
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

  // Smoothing (URL-backed). `raw` (default) renders unchanged. `ema` / `rolling`
  // replace each trace's y with the smoothed series; with `bands` on we add
  // two ±σ fill traces per run, grouped under the line's name so the existing
  // closest-trace highlight fades them together.
  const [smooth, setSmooth] = useSmoothMode()
  const [bandsOn, setBandsOn] = useBandsToggle()

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
  // the multi-term filter landed). `tagsFor` is keyed by the FULL id, not
  // the prefix-stripped shortLabel — so pass `r.id` not `r.label`.
  const haystackFor = (r: RunTimelineSeries): string =>
    runHaystacks?.get(r.label) ?? [r.label, ...tagsFor(r.id)].join(' ')
  const regexMatched = filterCompiled.empty || filterCompiled.error
    ? runs
    : runs.filter((r) => filterCompiled.matches(haystackFor(r)))
  // When `+ancestors` is on, walk lineage upward from each matched run and
  // add any ancestor that's also present in `runs` (so the plot shows the
  // parent baseline alongside its sweep). Mirrors `matchedIdsExpanded` in
  // RunsPage.tsx — the plot was previously missing this expansion, so the
  // parent trace was absent even when the parent card was rendered.
  const nameFilteredRuns = !ancestorsOn || regexMatched.length === runs.length
    ? regexMatched
    : (() => {
        const have = new Set(runs.map((r) => r.id))
        const idToRun = new Map(runs.map((r) => [r.id, r]))
        const out = new Set(regexMatched)
        for (const r of regexMatched) {
          for (const aid of ancestorsOf(r.id)) {
            if (have.has(aid)) {
              const ancestor = idToRun.get(aid)
              if (ancestor) out.add(ancestor)
            }
          }
        }
        return Array.from(out)
      })()

  // Tag-chip filter. Tri-state per tag (off / 'in' / 'out'): a run is visible
  // iff it has every `'in'` tag AND none of the `'out'` tags. Untagged runs
  // are hidden whenever any `'in'` tag is selected (no way to satisfy "must
  // have X" without tags).
  //
  // Controlled when `tagFiltersExternal` is passed (parent owns state and can
  // hard-filter the card list against it); falls back to the URL-backed
  // `useTagFilters` so the plot still works standalone.
  const [tagFiltersFallback, setTagFiltersFallback] = useTagFilters()
  const tagFilters = tagFiltersExternal ?? tagFiltersFallback
  const setTagFiltersRaw = (next: TagFilters) => {
    if (onTagFiltersChange) onTagFiltersChange(next)
    else setTagFiltersFallback(next)
  }
  const toggleTag = (t: RunTag) => setTagFiltersRaw(cycleTagFilter(tagFilters, t))
  // Only show chips for tags that ANY currently-visible (name-filtered) run
  // carries — keeps the chip strip short and relevant.
  const visibleTagSet = new Set<RunTag>()
  for (const r of nameFilteredRuns) for (const t of tagsFor(r.id)) visibleTagSet.add(t)
  const chipTags = ALL_TAGS.filter((t) => visibleTagSet.has(t))

  // No `.size === 0` fast-path: per-tag defaults (e.g. `bunk = out`) still
  // constrain even when the override Map is empty. `runPassesTagFilters`
  // short-circuits via its empty-constraint set when neither the Map nor
  // `TAG_DEFAULTS` carry anything.
  const filteredRuns = nameFilteredRuns.filter((r) => runPassesTagFilters(tagsFor(r.id), tagFilters))

  const cutoffSec = hoursBack ? (Date.now() / 1000 - hoursBack * 3600) : null

  const activeTrace = highlight?.activeTrace ?? null
  // When a run is pinned, plot ONLY that run so Plotly autoranges (x + y) to
  // its extent — short/small runs are invisible squished against the shared
  // axis. Pinning is driven by clicking a run card or legend item.
  const pinnedTrace = highlight?.pinnedTrace ?? null
  const plotted = pinnedTrace
    ? filteredRuns.filter((r) => r.label === pinnedTrace)
    : filteredRuns

  // Smoothing cache: keyed by (history identity, smoothKey, xMode, cutoff).
  // Built ONCE per render via useMemo; the trace-data .map below reads from it
  // so re-renders (hover-driven `activeTrace` changes) don't recompute on
  // every mousemove. xMode + cutoff baked into the key because `traceFor`
  // itself depends on them.
  //
  // `rolling` mode delegates to pltly's `rolling()` (Welford's, O(n)) — the
  // per-sample-index `smoothed[i].smoothed.y.stddev` gives a real σ band that
  // matches what the line is smoothing over. `ema` mode uses tomat's local
  // EMA (pltly ships rolling but not EMA); EMA has no natural σ companion, so
  // bands are unavailable for it (the chip rail's ±σ button is disabled there).
  const smoothK = smoothKey(smooth)
  const seriesCache = useMemo(() => {
    type Cached = {
      x: number[]
      ySmoothed: (number | null)[]
      yStd: (number | null)[] | null
    }
    const out = new Map<string, Cached>()
    for (const r of plotted) {
      const { x, y } = traceFor(r.history, xMode, cutoffSec)
      if (x.length === 0) continue
      const yRaw: (number | null)[] = y as (number | null)[]
      if (smooth.kind === 'rolling') {
        // Sample-index window (preserves the legacy `?smooth=rolling:N`
        // semantics — N is "samples on either side of i within ±N/2", not a
        // window in current-x-axis-units which would mean different things in
        // clock-ms vs step-vs-loss modes).
        const idx = yRaw.map((_, i) => i)
        const sm = rolling(idx, {
          getX: (i) => i,
          metrics: ['y'],
          getValue: (i) => yRaw[i],
          windowSize: smooth.window,
        })
        const ySmoothed: (number | null)[] = []
        const yStd: (number | null)[] = []
        for (const pt of sm) {
          const m = pt.smoothed.y
          ySmoothed.push(Number.isNaN(m.mean) ? null : m.mean)
          yStd.push(Number.isNaN(m.stddev) ? null : m.stddev)
        }
        out.set(r.id, { x, ySmoothed, yStd: bandsOn ? yStd : null })
      } else {
        // raw / ema: no pltly involvement; no σ companion for these modes.
        const ySmoothed = applySmoothing(yRaw, smooth)
        out.set(r.id, { x, ySmoothed, yStd: null })
      }
    }
    return out
    // `plotted` identity changes on every render (it's a new array), but its
    // contents are stable as long as runs / filters / pin don't change — so we
    // key on the run-id join + smoothing/x-mode/cutoff. Same idiom used by
    // `labelToId` in RunsPage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotted.map((r) => r.id).join('|'), smoothK, bandsOn, xMode, cutoffSec])

  // Fade non-highlighted traces to a true neutral grey (pltly's built-in fade
  // only desaturates partway / keeps a tint); the highlighted run keeps full
  // colour + a thicker stroke so it pops.
  //
  // Band traces (when smoothing + `bandsOn`) share the line's `name` so the
  // closest-trace hover routine matches all three together → fade-in-sync.
  // They use `hoverinfo: 'skip'` so the hover-y-distance code (which iterates
  // `event.points`) ignores them when picking the closest trace.
  //
  // Only `rolling` mode supplies a real σ companion (via pltly); `ema` keeps
  // its existing chip but has no band — its proxy-from-fake-rolling-σ helper
  // was dropped along with `rollingStd` when pltly took over the rolling path.
  const wantBands = bandsOn && smooth.kind === 'rolling'
  const data = plotted.flatMap((r) => {
    const cached = seriesCache.get(r.id)
    if (!cached) return []
    const { x: xNum, ySmoothed, yStd } = cached
    // For the clock x-axis, format ms → tz-naive local string so Plotly's date
    // axis renders in the viewer's timezone (Plotly's `type:'date'` with raw
    // numeric ms renders in UTC). Other modes use the numeric values directly.
    const x: (string | number)[] = xMode === 'clock'
      ? xNum.map((ms) => localDateStr(ms))
      : xNum
    const isActive = r.label === activeTrace
    const faded = activeTrace != null && !isActive
    const lineColor = faded ? '#666' : r.color
    const shape = (xMode === 'loss' ? 'linear' : 'hv') as 'linear' | 'hv'

    const lineTrace = {
      x,
      y: ySmoothed,
      name: r.label,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: {
        color: lineColor,
        width: isActive ? 3 : 2,
        shape,
      },
      // x-unified TT row layout: `<color> <hovertemplate-result>`. The
      // `<extra>` (right-side annotation) holds the trace name in non-unified
      // mode but is omitted from unified-mode rows, so we have to embed
      // `fullData.name` in the template ourselves. Without this, every row
      // reads "loss 3.093" with no way to tell which run is which.
      hovertemplate: xMode === 'loss'
        ? `%{fullData.name} · %{y:.3f}<extra></extra>`
        : `%{fullData.name} · %{y:,}<extra></extra>`,
    }

    if (!wantBands || !yStd) return [lineTrace]

    // ±σ bands: lower edge (no fill), upper edge (`fill: 'tonexty'`). Order
    // matters — `tonexty` fills *down to the previous trace*, so the lower
    // edge MUST precede the upper. We emit [lower, upper, line] so the line
    // draws on top of the band; the band's z-order doesn't matter for hover
    // (it's `hoverinfo: 'skip'`).
    const yLower: (number | null)[] = []
    const yUpper: (number | null)[] = []
    for (let i = 0; i < ySmoothed.length; i++) {
      const m = ySmoothed[i]
      const s = yStd[i]
      if (m == null || s == null) { yLower.push(null); yUpper.push(null); continue }
      yLower.push(m - s)
      yUpper.push(m + s)
    }
    // Band fill colour matches the line. When faded, use the same grey as the
    // line at lower opacity so the trio reads as one group.
    const bandRGB = faded ? '102, 102, 102' : hexToRgb(r.color)
    const bandAlpha = isActive ? 0.18 : 0.10
    const fillcolor = `rgba(${bandRGB}, ${bandAlpha})`
    const edgeCommon = {
      x,
      name: r.label,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: 'rgba(0,0,0,0)', width: 0, shape },
      hoverinfo: 'skip' as const,
      showlegend: false,
    }
    return [
      { ...edgeCommon, y: yLower },
      { ...edgeCommon, y: yUpper, fill: 'tonexty' as const, fillcolor },
      lineTrace,
    ]
  })
  // Draw the highlighted trace last so it sits on top of the rest (Plotly
  // z-order = data order). Array.sort is stable → others keep their order.
  // Sort by name; bands have the same name as their line so they stay grouped.
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
      onPlotHover?.(bestName)
    }
  }
  const handlePlotUnhover = () => {
    // Don't immediately clear — Plotly fires unhover when the cursor moves
    // between samples on the same trace. The hook's debounce (debounceMs in
    // useTraceHighlight) absorbs the brief gap when the next hover arrives.
    closestTraceRef.current = null
    highlight?.setHoverTrace(null)
    onPlotHover?.(null)
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
    for (const d of data) for (const v of d.y) {
      if (v != null && Number.isFinite(v)) vals.push(v)
    }
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
      {/* Tag chips + ancestors toggle. The chip rail renders whenever ANY of
          its contents is present: visible tag chips, the clear-tag-filters
          button, or `onAncestorsChange` (the +ancestors toggle should be
          available regardless of whether the current regex/tag selection
          surfaces any tag chips — that was the prior bug). */}
      {(chipTags.length > 0 || onAncestorsChange) && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 4,
          alignItems: 'center', justifyContent: 'flex-end',
        }}>
          {chipTags.length > 0 && (
            <span style={{ fontSize: '0.7rem', color: muted, marginRight: 4 }}>
              tags:
            </span>
          )}
          {chipTags.map((t) => {
            // Tri-state visual driven by the *effective* state (override
            // when present, else per-tag default). Cycle: in → out → off →
            // in; the cycle is implemented over effective state in
            // `cycleTagFilter`, so e.g. `bunk` (default `out`) starts at
            // its default-red and cycles out → off → in → back to default.
            const state = effectiveTagState(tagFilters, t)
            const styleFor = (s: typeof state) => {
              if (s === 'in') return {
                bg: isDark ? '#2a4a7a' : '#cbe0f5',
                border: isDark ? '#3a6ab0' : '#7aa7d9',
                text: isDark ? '#cfe2ff' : '#1d3a64',
                decoration: 'none' as const,
              }
              if (s === 'out') return {
                bg: isDark ? '#5a2a2a' : '#f5cbcb',
                border: isDark ? '#b04a4a' : '#d97a7a',
                text: isDark ? '#ffd0d0' : '#641d1d',
                decoration: 'line-through' as const,
              }
              return {
                bg: 'transparent',
                border: isDark ? '#333' : '#ccc',
                text: muted,
                decoration: 'none' as const,
              }
            }
            const sty = styleFor(state)
            const nextLabel = state === 'in' ? 'exclude' : state === 'out' ? 'remove' : 'include'
            return (
              <Tooltip key={t} content={`${nextLabel} "${t}" filter (in → out → off; AND across)`}>
                <button
                  onClick={() => toggleTag(t)}
                  style={{
                    background: sty.bg,
                    border: `1px solid ${sty.border}`,
                    borderRadius: 10, cursor: 'pointer', padding: '1px 8px',
                    fontSize: '0.7rem', fontFamily: 'inherit',
                    color: sty.text,
                    textDecoration: sty.decoration,
                  }}
                >
                  {state === 'out' ? `−${t}` : t}
                </button>
              </Tooltip>
            )
          })}
          {tagFilters.size > 0 && (
            <button
              onClick={() => setTagFiltersRaw(new Map())}
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
          {onAncestorsChange && (
            <Tooltip content={
              'when on, the lineage-ancestors of every matching run are also '
              + 'shown — so filtering to a sweep auto-includes the parent it '
              + 'fine-tuned off of, as a baseline. URL: ?anc=1'
            }>
              <button
                onClick={() => onAncestorsChange(!ancestorsOn)}
                style={{
                  background: ancestorsOn
                    ? (isDark ? '#2a4a7a' : '#cbe0f5') : 'transparent',
                  border: `1px solid ${ancestorsOn
                    ? (isDark ? '#3a6ab0' : '#7aa7d9')
                    : (isDark ? '#333' : '#ccc')}`,
                  borderRadius: 10, cursor: 'pointer', padding: '1px 8px',
                  fontSize: '0.7rem', fontFamily: 'inherit',
                  color: ancestorsOn
                    ? (isDark ? '#cfe2ff' : '#1d3a64') : muted,
                  marginLeft: 2,
                }}
              >
                {ancestorsOn ? '← +ancestors' : '+ancestors'}
              </button>
            </Tooltip>
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
        <SmoothingChips
          mode={smooth} setMode={setSmooth}
          bandsOn={bandsOn} setBandsOn={setBandsOn}
          isDark={isDark} fg={fg} muted={muted}
        />
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
            // No gaps — adjacent LegendItems own the inter-item space via their
            // own padding so cursor sweeps across the legend never land in a
            // dead zone (which caused flicker: leave A → 120ms debounce → enter
            // B). With flush hover zones the leave/enter pair always overlaps.
            columnGap: 0, rowGap: 0,
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
                // Overrides LegendItem's default `0 0.4em` — pads vertically too
                // (so vertically-adjacent rows are flush) and absorbs the
                // ex-columnGap horizontally.
                style={{ fontSize: '0.72rem', padding: '2px 7px' }}
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
