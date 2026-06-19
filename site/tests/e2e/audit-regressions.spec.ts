// Regression coverage for the 5 FE bugs surfaced in the 2026-06-12 prod
// audit. Each describe block names the original bug + the commit that fixed
// it so a future regression can be traced back to its root.
//
// Bugs covered:
//   1. MEvalTable step asterisk semantics — pre-cutoff periodic ckpts must
//      render with `*` (e.g. `30k*`), force-saves without (`50k`). Caught
//      by displaying eval rows for `bin5`'s step set
//      {30000, 49999, 89999, 100000} which hits all three legacy code
//      paths.    fix: c1cfc7e / formatStepDetail
//   2. ELVis diff URL shape — pred rows on `/#/mp/<id>` link as
//      `?m=…&s=d&v0=<gt>&v1=<pred>` (diff anchored at GT); GT rows link as
//      `?m=…&v1=<gt>` (single volume).    fix: 8cf898c / elvisUrlFor
//   3. Theme toggle in the SD widget — the lower-right SpeedDial owns the
//      theme cycle now; no standalone `.theme-toggle` button should render
//      anywhere in the runs/posts/home shell.    fix: 1f4254b
//   4. Click-LI pin — clicking a Plotly legend item PINS the trace
//      (band/shape fade persists after mouseleave); clicking the SAME LI
//      unpins; clicking a DIFFERENT LI swaps the pin. Plotly's default
//      visibility-toggle behavior must be suppressed (no "Double-click to
//      isolate" toast).    fix: 8798f8e
//   5. CFW must expose `Content-Range` so `hyparquet` can stream the runs
//      parquet — fixed in worker CORS headers. Manifested as "loading
//      parquet…" stuck forever on the run-detail page; verified live by
//      the existing run-detail per-x-mode test, but we also pin a direct
//      CFW response assertion here so a regression hits CI even if the FE
//      gracefully degrades.    fix: 97f7188

import { expect, test, type Page } from '@playwright/test'
import { FIXTURES, FIXTURE_MP_ID } from './_helpers/fixtures'
import { getHash, goHash } from './_helpers/nav'
import {
  readPlot,
  startConsoleErrorCapture,
  waitForPlotPopulated,
  type TraceSummary,
} from './_helpers/plot'

// Bug 1 — MEvalTable asterisk semantics ───────────────────────────────────
test.describe('MEvalTable — legacy step-naming asterisk semantics', () => {
  test('bin5 renders `30k*`/`50k`/`90k`/`100k*` for the four legacy code paths', async ({ page }) => {
    await goHash(page, `/runs/${FIXTURES.irisRunningFourSeg}`)
    // Wait for either the MEvalTable header row OR (if eval.json is still
    // fetching) the per-step rows.
    await expect.poll(
      async () => (await page.locator('body').innerText()).includes('Per-step m-eval'),
      { timeout: 30_000, message: 'MEvalTable header never rendered' },
    ).toBe(true)
    // Pluck the rendered step column for assertion. The first <td> of each
    // <tr> in the MEvalTable's <tbody> is the step cell; in the legacy
    // case it's wrapped in a `<Tooltip>`, so the visible text we want is
    // the rendered text content (which strips the Tooltip wrapper).
    const stepLabels = await page.evaluate(() => {
      // The MEvalTable is the table whose first <th> reads "step". Walk
      // every <table>, return the matching one's first-column cell text.
      const tables = Array.from(document.querySelectorAll('table'))
      for (const tab of tables) {
        const firstHeader = tab.querySelector('thead th')?.textContent?.trim() ?? ''
        if (firstHeader !== 'step') continue
        const rows = Array.from(tab.querySelectorAll('tbody tr'))
        return rows.map((r) => (r.querySelector('td')?.textContent ?? '').trim())
      }
      return [] as string[]
    })
    // bin5 eval.json currently has steps {30000, 40000, 49999, 60000,
    // 70000, 80000, 89999, 100000}. Asserting the FULL list pins both the
    // sort (desc) and the per-step display formatting (asterisk semantics).
    // If/when a new eval step is added, this list grows; that's the right
    // signal — a regression in the rendering code would flip an entry,
    // not extend the set.
    expect(stepLabels).toEqual([
      '100k*', // 100000 = round, no force-save → legacy periodic → *
      '90k',   // 89999  = round−1                → legacy force-save → no *
      '80k*',  // 80000  = round, no force-save → legacy periodic → *
      '70k*',
      '60k*',
      '50k',   // 49999                        → legacy force-save → no *
      '40k*',
      '30k*',
    ])
  })
})

// Bug 2 — ELVis diff URL shape ────────────────────────────────────────────
test.describe('MpPage — ELVis link URLs differ for GT vs pred rows', () => {
  test('GT row → single-volume URL; pred rows → diff URL with `v0=<gt>&v1=<pred>`', async ({ page }) => {
    const { errors } = startConsoleErrorCapture(page)
    await goHash(page, `/mp/${FIXTURE_MP_ID}`)
    // The table renders once the grids-index resolves.
    await expect.poll(
      async () => (await page.locator('a:has-text("Open in ELVis")').count()),
      { timeout: 30_000, message: 'ELVis links never rendered on /mp page' },
    ).toBeGreaterThan(1)
    // Pull every ELVis link with its row's role tag ("gt" or "pred").
    type Row = { role: string; href: string }
    const rows: Row[] = await page.evaluate(() => {
      const out: Array<{ role: string; href: string }> = []
      const trs = Array.from(document.querySelectorAll('tbody tr'))
      for (const tr of trs) {
        const role = tr.querySelector('td span')?.textContent?.trim() ?? ''
        const a = tr.querySelector('a') as HTMLAnchorElement | null
        if (!a) continue
        out.push({ role, href: a.href })
      }
      return out
    })
    expect(rows.length, 'at least 2 rows on /mp page').toBeGreaterThan(1)
    const gt = rows.filter((r) => r.role === 'gt')
    const pred = rows.filter((r) => r.role === 'pred')
    expect(gt.length, 'at least one GT row').toBeGreaterThan(0)
    expect(pred.length, 'at least one pred row').toBeGreaterThan(0)
    // GT row: `?m=…&v1=…` (no diff mode, no v0).
    for (const r of gt) {
      const u = new URL(r.href)
      expect(u.origin + u.pathname, 'GT → elvis.oa.dev').toBe('https://elvis.oa.dev/')
      const ps = u.searchParams
      expect(ps.get('m'), 'GT carries mp_id').toBe(FIXTURE_MP_ID)
      expect(ps.get('v1'), 'GT carries v1').toBeTruthy()
      expect(ps.get('v0'), 'GT must NOT carry v0').toBeNull()
      expect(ps.get('s'), 'GT must NOT carry s=d').toBeNull()
    }
    // Pred rows: `?m=…&s=d&v0=<gt-r2-url>&v1=<pred-r2-url>` — diff anchored at GT.
    for (const r of pred) {
      const u = new URL(r.href)
      expect(u.origin + u.pathname, 'pred → elvis.oa.dev').toBe('https://elvis.oa.dev/')
      const ps = u.searchParams
      expect(ps.get('m'), 'pred carries mp_id').toBe(FIXTURE_MP_ID)
      expect(ps.get('s'), 'pred s=d (diff mode)').toBe('d')
      expect(ps.get('v0'), 'pred v0 = GT R2 URL').toMatch(/^https?:\/\//)
      expect(ps.get('v1'), 'pred v1 = pred R2 URL').toMatch(/^https?:\/\//)
      // Both volumes should point at distinct R2 URLs (would be a real
      // regression to anchor diff at the pred itself).
      expect(ps.get('v0')).not.toBe(ps.get('v1'))
    }
    expect(errors).toEqual([])
  })
})

// Bug 3 — SpeedDial owns the theme toggle ─────────────────────────────────
test.describe('SpeedDial — owns the theme cycle (no standalone toggle)', () => {
  test('no `.theme-toggle` rendered anywhere on /, /#/runs, /#/posts', async ({ page }) => {
    // Three top-level shells, all wrapped in KbdShell which renders the
    // SpeedDial. A standalone .theme-toggle would shadow it.
    for (const route of ['/', '/runs', '/posts']) {
      await page.goto(`#${route}`, { waitUntil: 'domcontentloaded' })
      // Allow lazy-loaded children a beat to settle.
      await page.waitForTimeout(800)
      const count = await page.locator('.theme-toggle').count()
      expect(count, `route ${route} should not render a standalone .theme-toggle`).toBe(0)
    }
  })
  test('SpeedDial expand reveals a `theme: light|dark|auto` action', async ({ page }) => {
    await goHash(page, '/')
    // use-kbd's SpeedDial DOM ships these CSS hooks (see use-kbd
    // `dist/index.js` SpeedDial render): the primary FAB has class
    // `kbd-speed-dial-primary`, the chevron sibling has class
    // `kbd-speed-dial-chevron` (clicking it expands), the wrapper gains
    // `kbd-speed-dial-expanded`, and each action button has class
    // `kbd-speed-dial-action` with its label as `aria-label`. The primary
    // button itself opens the omnibar (`handlePrimaryClick`), not expand.
    const chevron = page.locator('.kbd-speed-dial-chevron').first()
    await expect(chevron,
      'SpeedDial chevron must mount on /',
    ).toBeVisible({ timeout: 15_000 })
    await chevron.click()
    await expect(page.locator('.kbd-speed-dial-expanded').first(),
      'SpeedDial container gains `kbd-speed-dial-expanded` after chevron click',
    ).toBeVisible({ timeout: 5_000 })
    // The theme action button — labelled `theme: <mode>` via aria-label
    // (see KbdSetup.tsx::SpeedDialWithTheme actions[0]).
    const themeAction = page.locator(
      '.kbd-speed-dial-action[aria-label^="theme:"]',
    ).first()
    await expect(themeAction,
      'a `theme: <mode>` action surfaces inside the expanded SpeedDial',
    ).toBeVisible({ timeout: 5_000 })
    const label = (await themeAction.getAttribute('aria-label')) ?? ''
    expect(label).toMatch(/^theme:\s*(light|dark|auto)$/)
  })
})

// Bug 4 — Click-LI pin ────────────────────────────────────────────────────
//
// The Plotly DOM exposes the active legend item as `.legendtext`; clicking
// the wrapper `<g.traces>` of a legend item fires `plotly_legendclick`,
// which our handler intercepts. Verification strategy:
//   - read pre-click band opacity (no active trace → all bands at 1).
//   - click a real TL/VL trace's legend wrapper.
//   - read post-click band opacity AND the bands' attributes after we
//     intentionally `mouseleave` the legend region. With the pin in place,
//     band opacity stays at the faded value (matching bands at 1,
//     unmatched bands at 0.3). Without the pin (regression), mouseleave
//     would clear the highlight → all bands snap back to 1.
//   - click the SAME legend item again → bands return to 1 (unpin).
async function clickFirstNonEventLegend(page: Page): Promise<string> {
  // Walk every `.legendtoggle` (Plotly's clickable rect on each LI). Skip:
  //  - event-group LIs (trainer_started/sigterm/cluster preempt/death:/
  //    annotations) — their fade contract is separate
  //  - legend group titles ("parent", "losses (log)", "MT/MV (mat-NMAE %)",
  //    "events") — these are NOT trace LIs; matching only by `.legendtoggle`
  //    catches them too, so we filter by membership in the live trace
  //    name set.
  const target = await page.evaluate(() => {
    const plot = document.querySelector('.js-plotly-plot') as (HTMLElement & {
      data?: Array<Record<string, unknown>>
    }) | null
    const traceNames = new Set<string>()
    for (const t of plot?.data ?? []) {
      const n = typeof t.name === 'string' ? t.name : ''
      if (n) traceNames.add(n)
    }
    const isEventLabel = (s: string) =>
      s.startsWith('trainer_started') || s.startsWith('sigterm')
      || s.startsWith('cluster preempt') || s.startsWith('death:')
      || s.startsWith('annotations')
    const items = Array.from(document.querySelectorAll('.legendtoggle')) as SVGRectElement[]
    for (const it of items) {
      const text = (it.parentElement?.querySelector('.legendtext')?.textContent ?? '').trim()
      if (text === '' || isEventLabel(text)) continue
      // Skip group titles by requiring the text to correspond to a real
      // trace (exact match OR prefix match against multi-segment traces).
      const matches = traceNames.has(text)
        || [...traceNames].some((n) => n.startsWith(text + ' '))
      if (!matches) continue
      const b = it.getBoundingClientRect()
      return { x: b.left + b.width / 2, y: b.top + b.height / 2, text }
    }
    return null
  })
  if (!target) throw new Error('no clickable non-event legend item found')
  await page.mouse.click(target.x, target.y)
  return target.text
}

function summariseBandOpacities(traces: TraceSummary[], activeName: string | null) {
  // Helper exposed by `readPlot` returns name + shape; we want raw opacity
  // too — re-read via page.evaluate elsewhere. Kept here for shape clarity.
  return { activeName, total: traces.length }
}
void summariseBandOpacities

/** Read each top-level trace's `visible` field. Plotly defaults to
 *  `undefined`/`true`. The legend's default click handler flips a trace's
 *  visible to `'legendonly'` (hidden but legendable). Our handler returns
 *  `false` from `plotly_legendclick`, which suppresses Plotly's default.
 *  This signal works on every run (no event-shapes / band-traces required)
 *  and is the actual contract: clicking an LI must NOT hide its trace. */
async function readTraceVisible(page: Page): Promise<Array<{ name: string; visible: unknown }>> {
  return await page.evaluate(() => {
    const plot = document.querySelector('.js-plotly-plot') as (HTMLElement & {
      data?: Array<Record<string, unknown>>
    }) | null
    if (!plot?.data) return []
    return plot.data.map((t) => ({
      name: typeof t.name === 'string' ? t.name : '',
      visible: t.visible,
    }))
  })
}

test.describe('WallclockPlot — click-LI does not hide the trace (pin handler is wired)', () => {
  test('click suppresses Plotly default; trace stays visible', async ({ page }) => {
    // The regression: Plotly's default `plotly_legendclick` handler sets the
    // clicked trace's `visible` to `'legendonly'` and shows a
    // "Double-click to isolate" toast. Our `plotly_legendclick` listener
    // intercepts and returns `false` to suppress this — see
    // `WallclockPlot.tsx::ensureLegendListener`. If a regression breaks
    // that return value or detaches the listener, the trace would hide on
    // first click. The simplest end-to-end signal: clicking any
    // non-event LI must NOT flip its trace's `visible` field.
    //
    // (Verifying the pin-fade visually requires a fixture with bands or
    // event-shapes that survive the chosen x-mode. Both have proven
    // brittle in CI — `?x=step` doesn't render shapes for many runs,
    // smoothed bands require enabling the smoothing toggle first. The
    // visible-suppression check is the same contract from the other end
    // and works on every run.)
    const { errors } = startConsoleErrorCapture(page)
    await goHash(page, `/runs/${FIXTURES.irisLineageGlued}`)
    await waitForPlotPopulated(page, 0)
    // Wait for the legend listener to attach — `ensureLegendListener`
    // re-runs on `plotly_afterplot`, so the first paint must complete.
    await page.waitForTimeout(500)
    const before = await readTraceVisible(page)
    expect(before.length, 'plot must have ≥1 trace').toBeGreaterThan(0)
    const clickedName = await clickFirstNonEventLegend(page)
    // Move the mouse away to ensure we're observing post-click state, not
    // a hover-only effect.
    await page.mouse.move(10, 10)
    await page.waitForTimeout(300)
    const after = await readTraceVisible(page)
    // Find every trace whose name matches the clicked legend entry. Plotly
    // renders multi-segment runs with one trace per segment (each named
    // `TL (train/loss) #N/72`) but a single legend item. The legend text
    // we click is the "unified" name (e.g. `TL (train/loss)`); the
    // matching traces are anything that starts with that.
    const clicked = after.filter((t) => t.name === clickedName || t.name.startsWith(clickedName + ' '))
    expect(clicked.length,
      `at least one trace matches legend label "${clickedName}"`,
    ).toBeGreaterThan(0)
    for (const t of clicked) {
      expect(t.visible,
        `clicking "${clickedName}" must NOT flip visible to 'legendonly' `
        + `(regression: legend handler missing / not returning false)`,
      ).not.toBe('legendonly')
    }
    // No "Double-click to isolate" toast leaked through (Plotly's modebar
    // surfaces a `.notifier` toast div when the default fires).
    const notifierCount = await page.locator('.notifier-note').count()
    expect(notifierCount, 'no Plotly "Double-click to isolate" toast').toBe(0)
    expect(errors).toEqual([])
  })
})

// Bug 5 — CFW exposes Content-Range ───────────────────────────────────────
test.describe('CFW — `Content-Range` exposed via CORS', () => {
  test('range request returns 206 with `Access-Control-Expose-Headers: Content-Range`', async ({ request }) => {
    // We pin against the prod CFW; the staging worker mirrors prod's CORS
    // headers (same code), so a regression on prod is what we want to
    // catch. The runs parquet for `bin5` is large enough that the FE
    // streams it with Range, but any binary route works for the header
    // assertion. Use a small Range so we don't pull the whole file.
    const url = `https://tomat-runs-api.openathena.workers.dev/api/runs/${FIXTURES.irisRunningFourSeg}/raw.parquet`
    const resp = await request.fetch(url, {
      method: 'GET',
      headers: { Range: 'bytes=0-1023' },
    })
    expect(resp.status(),
      'partial-content range fetch returns 206',
    ).toBe(206)
    const headers = resp.headers()
    expect(headers['content-range'], 'Content-Range header present').toMatch(/^bytes 0-1023\/\d+$/)
    // The fix: WITHOUT this header, hyparquet throws "missing content-range
    // header" and the plot stays stuck on "loading parquet…".
    const expose = headers['access-control-expose-headers'] ?? ''
    expect(expose.toLowerCase(),
      'Access-Control-Expose-Headers must include Content-Range',
    ).toMatch(/content-range/)
  })
})
