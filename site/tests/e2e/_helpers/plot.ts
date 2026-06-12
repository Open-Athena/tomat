// Plot-introspection helpers for the e2e suite.
//
// Plotly attaches the live trace array at `.js-plotly-plot.data`; the
// rendered SVG is whatever it derived from there. For shape-of-data
// assertions we read .data (the source of truth); for tick-format
// assertions we read SVG text.
//
// The same "extract per-trace stats" probe that has been pasted into the
// DT console for manual CIC sweeps is centralised here so test assertions
// stay readable and the parsing logic only lives in one place.

import { expect, type Locator, type Page } from '@playwright/test'

export interface TraceSummary {
  name: string
  /** legendgroup — used for "is this an ancestor or current-run trace" checks */
  lg: string
  /** number of finite x points actually plotted */
  n: number
  x_min: number | null
  x_max: number | null
  y_min: number | null
  y_max: number | null
  /** `'solid' | 'dot' | 'dash' | …` — `'solid'` when unset. */
  dash: string
  /** mode field — eg. `'lines'`, `'lines+markers'`. */
  mode: string
  /** primary color, when extractable from line.color or marker.color. */
  color: string | null
}

declare global {
  interface Window {
    Plotly?: {
      Plots?: { resize?: (gd: unknown) => void }
    }
  }
}

/** Read all plotly plot data on the page into `TraceSummary[][]` (one
 *  array per `.js-plotly-plot` div, in DOM order). Each entry summarises
 *  the trace's shape. Designed to be runnable inside `page.evaluate`.
 *
 *  Internally normalises a bunch of plotly quirks:
 *    - x may be a `TypedArray` (Float64Array) or `number[]`; both reduce
 *    - `dash` defaults to `'solid'` when undefined
 *    - non-finite values (`null`, `undefined`, `NaN`) are filtered before
 *      min/max so a trailing-null pad doesn't poison the range */
export async function readAllPlots(page: Page): Promise<TraceSummary[][]> {
  return await page.evaluate(() => {
    const plots = Array.from(document.querySelectorAll('.js-plotly-plot')) as Array<
      HTMLElement & { data?: unknown[] }
    >
    function summariseArray(arr: unknown): { n: number; lo: number | null; hi: number | null } {
      if (!arr || typeof arr !== 'object') return { n: 0, lo: null, hi: null }
      const it = arr as ArrayLike<unknown>
      let n = 0
      let lo = Infinity
      let hi = -Infinity
      for (let i = 0; i < it.length; i++) {
        const raw = it[i] as unknown
        // Coerce date-strings ("2026-06-10 10:36:59") to epoch ms — the
        // wallclock x-axis uses ISO-ish strings, and we need a numeric
        // range to assert "non-degenerate". Number-typed values (epoch
        // ms, step, FLOPs) pass through unchanged.
        let v: number
        if (typeof raw === 'number') {
          v = raw
        } else if (typeof raw === 'string') {
          // Plotly accepts space-separated date strings; Date.parse needs
          // a T for cross-engine consistency.
          const iso = raw.includes('T') ? raw : raw.replace(' ', 'T')
          v = Date.parse(iso)
        } else {
          continue
        }
        if (!Number.isFinite(v)) continue
        n += 1
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
      return {
        n,
        lo: n === 0 ? null : lo,
        hi: n === 0 ? null : hi,
      }
    }
    return plots.map((p) => {
      const data = Array.isArray(p.data) ? (p.data as Array<Record<string, unknown>>) : []
      return data.map((t): {
        name: string
        lg: string
        n: number
        x_min: number | null
        x_max: number | null
        y_min: number | null
        y_max: number | null
        dash: string
        mode: string
        color: string | null
      } => {
        const line = (t.line ?? {}) as Record<string, unknown>
        const marker = (t.marker ?? {}) as Record<string, unknown>
        const xs = summariseArray(t.x)
        const ys = summariseArray(t.y)
        const dashRaw = line.dash
        return {
          name: typeof t.name === 'string' ? t.name : '',
          lg: typeof t.legendgroup === 'string' ? t.legendgroup : '',
          n: xs.n,
          x_min: xs.lo,
          x_max: xs.hi,
          y_min: ys.lo,
          y_max: ys.hi,
          dash: typeof dashRaw === 'string' ? dashRaw : 'solid',
          mode: typeof t.mode === 'string' ? t.mode : '',
          color:
            typeof line.color === 'string' ? line.color
            : typeof marker.color === 'string' ? marker.color
            : null,
        }
      })
    })
  })
}

/** Convenience: read the n-th plotly plot's traces (default 0). */
export async function readPlot(page: Page, idx = 0): Promise<TraceSummary[]> {
  const all = await readAllPlots(page)
  if (idx >= all.length) {
    throw new Error(`readPlot(${idx}): only ${all.length} plot(s) on page`)
  }
  return all[idx]
}

/** Wait until the n-th plotly plot has at least one trace with `n > 0`,
 *  i.e. the run history fetch resolved AND was bridged into plotly.
 *
 *  We can't just await a single network response: tanstack-query may
 *  serve cached data, fetchRunHistory may stream parquet pages, etc. Poll
 *  the DOM instead. Times out per `expect.toPass` budget (15s). */
export async function waitForPlotPopulated(page: Page, idx = 0): Promise<void> {
  await expect.poll(
    async () => {
      const all = await readAllPlots(page)
      if (idx >= all.length) return 0
      return all[idx].reduce((acc, t) => acc + t.n, 0)
    },
    {
      message: `plot[${idx}] never populated with finite data`,
      timeout: 30_000,
      intervals: [500, 1000, 2000],
    },
  ).toBeGreaterThan(0)
}

/** Wait until at least `min` plotly plots are present on the page (each
 *  with the `.js-plotly-plot` class). Useful for waiting on the per-run
 *  detail page which renders WallclockPlot + MEvalTable + a few sparklines. */
export async function waitForAtLeastNPlots(page: Page, min: number): Promise<void> {
  await expect.poll(
    async () => (await readAllPlots(page)).length,
    { message: `expected ≥${min} plotly plots on page`, timeout: 30_000 },
  ).toBeGreaterThanOrEqual(min)
}

/** Read x-axis tick text from the rendered SVG of the n-th plot. Each
 *  string is the `.xtick text` content; empty / hidden ticks are filtered
 *  out so e.g. plotly's blank guard ticks don't break the regex assertion. */
export async function readXTicks(page: Page, plotIdx = 0): Promise<string[]> {
  return await page.evaluate((idx) => {
    const plots = document.querySelectorAll('.js-plotly-plot')
    if (idx >= plots.length) return []
    const root = plots[idx] as HTMLElement
    const ticks = Array.from(root.querySelectorAll('.xtick text')) as SVGTextElement[]
    return ticks
      .map((n) => (n.textContent ?? '').trim())
      .filter((s) => s.length > 0)
  }, plotIdx)
}

/** Read the rendered x-axis title (the plotly `xtitle` text element) of
 *  the n-th plot. Returns `null` if no title is present. */
export async function readXAxisTitle(page: Page, plotIdx = 0): Promise<string | null> {
  return await page.evaluate((idx) => {
    const plots = document.querySelectorAll('.js-plotly-plot')
    if (idx >= plots.length) return null
    const root = plots[idx] as HTMLElement
    const t = root.querySelector('.xtitle')
    const text = t?.textContent?.trim() ?? ''
    return text.length > 0 ? text : null
  }, plotIdx)
}

/** Click a chip whose visible text matches `label` exactly (after trim).
 *  Chips are <button>s with arbitrary children; matching by full-button-
 *  textContent is brittle, so we anchor on a span containing the label
 *  and walk up to the button.
 *
 *  Several callers use this pattern, factored out here so future chip-
 *  rail changes only need to touch one selector strategy. */
export async function clickChipByText(
  scope: Locator | Page,
  label: string,
): Promise<void> {
  const page = 'page' in scope ? (scope as unknown as { page: () => Page }).page() : null
  void page
  const target = (scope as Locator | Page).getByRole('button', { name: label, exact: true })
  await target.first().click()
}

/** Whether a console message was emitted at error level since the page
 *  started. Use as a secondary sanity check alongside the primary
 *  rendering assertion — `expect(consoleErrors(page)).toHaveLength(0)`.
 *
 *  The dashboard intentionally probes for several optional sidecars
 *  (`eval.json`, `cost.json`, `posts-index.json`, lineage parents that
 *  may not be cached, etc.) and gracefully renders without them. Those
 *  404s show up as `Failed to load resource: …status of 404` in the
 *  console — they're expected. We filter them so the assertion stays
 *  focused on REAL render-time bugs (uncaught exceptions, plotly
 *  warnings, react-key violations, etc.). To inspect the raw stream
 *  during debugging, set `includeBenign: true`. */
export interface ConsoleCapture {
  /** Filtered error stream — uncaught exceptions + non-404 console.error. */
  errors: string[]
  /** Unfiltered errors, for debugging only. */
  rawErrors: string[]
}
const BENIGN_PATTERNS = [
  // Failed-to-fetch on optional sidecars; the dashboard renders fine
  // without them.
  /Failed to load resource: the server responded with a status of 404/i,
  /Failed to load resource: net::ERR_FAILED/i,
  // Plotly's deprecated-Plotly.d3 warning on legacy `react-plotly.js`
  // builds — pre-existing, not a regression.
  /Plotly\.d3 is deprecated/i,
]

export function startConsoleErrorCapture(page: Page): ConsoleCapture {
  const rawErrors: string[] = []
  const errors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return
    const text = msg.text()
    rawErrors.push(text)
    if (BENIGN_PATTERNS.some((p) => p.test(text))) return
    errors.push(text)
  })
  page.on('pageerror', (err) => {
    const text = `pageerror: ${err.message}`
    rawErrors.push(text)
    errors.push(text)
  })
  return { errors, rawErrors }
}
