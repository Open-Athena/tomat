// Cross-run timeline (`/#/runs`) regression coverage.
//
// Spec items covered (see worktree prompt):
//   3. Per-x-mode rendering on the multi-run timeline
//   4. URL state persistence (`?regex=`, `?x=…`)
//   5. Card-header field presence on the listing card (sanity check; the
//      detail-page version is in run-detail.spec.ts).

import { expect, test } from '@playwright/test'
import { FIXTURES } from './_helpers/fixtures'
import { getHash, goHash } from './_helpers/nav'
import {
  readAllPlots,
  readPlot,
  readXAxisTitle,
  startConsoleErrorCapture,
  waitForAtLeastNPlots,
  waitForPlotPopulated,
} from './_helpers/plot'

// Cross-run X modes: see RunsTimelinePlot.tsx:108. `loss` reads as
// `loss vs step` in the chip rail.
const CROSS_RUN_URL_X_MODES = ['clock', 'elapsed', 'active', 'loss', 'flop'] as const

test.describe('cross-run timeline — per-x-mode rendering', () => {
  for (const xmode of CROSS_RUN_URL_X_MODES) {
    test(`renders a non-degenerate trace under x=${xmode}`, async ({ page }) => {
      const { errors } = startConsoleErrorCapture(page)
      // Filter to a known-good subset so the timeline doesn't try to render
      // hundreds of runs and gum up the assertion. The regex matches the
      // KL-σ0.5 lineage which we know is present + populated.
      await goHash(page, `/runs?x=${xmode}&regex=kl-s05`)
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      const traces = await readPlot(page, 0)
      const populated = traces.filter((t) => t.n > 0 && t.x_min != null && t.x_max != null)
      expect(populated.length, `x=${xmode}: at least one trace populated`).toBeGreaterThan(0)
      const biggest = populated.reduce((a, b) => (a.n > b.n ? a : b))
      expect(biggest.x_max!, `x=${xmode}: x_max>x_min`).toBeGreaterThan(biggest.x_min!)
      expect(errors).toEqual([])
    })
  }
})

test.describe('cross-run timeline — FLOP-axis per unit', () => {
  // Sanity-check the cross-run plot under each FLOP unit too — the
  // dedicated flops-axis.spec.ts file goes deeper on tick-text format;
  // this just confirms the cross-run plot renders under all units
  // without console errors.
  for (const unit of ['EF', 'PF', 'TF', 'sci'] as const) {
    test(`x=flop&fopu=${unit} renders without console errors`, async ({ page }) => {
      const { errors } = startConsoleErrorCapture(page)
      await goHash(page, `/runs?x=flop&fopu=${unit}&regex=kl-s05`)
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      const title = (await readXAxisTitle(page, 0)) ?? ''
      const unitLabel = unit === 'sci' ? 'FLOP' : unit
      expect(title, `xtitle mentions ${unitLabel}`).toContain(unitLabel)
      expect(errors).toEqual([])
    })
  }
})

test.describe('cross-run timeline — URL state persistence', () => {
  test('?regex=kl-s05&x=elapsed survives reload', async ({ page }) => {
    await goHash(page, `/runs?regex=kl-s05&x=elapsed`)
    await waitForPlotPopulated(page, 0)
    await page.reload({ waitUntil: 'domcontentloaded' })
    await waitForPlotPopulated(page, 0)
    const hash = await getHash(page)
    expect(hash).toContain('regex=kl-s05')
    expect(hash).toContain('x=elapsed')
  })
})

test.describe('cross-run listing — card-header fields', () => {
  test('finished Modal run card shows step/MFU/EF/tok in the listing', async ({ page }) => {
    await goHash(page, `/runs?regex=${FIXTURES.modalFinishedFull}`)
    // Wait for the run's card to appear in the listing.
    await page.getByText(FIXTURES.modalFinishedFull).first().waitFor({ timeout: 30_000 })
    // Allow card metadata fetches to settle. Same poll-the-body pattern
    // as run-detail.spec.ts so a flapping field doesn't false-fail.
    const requiredTokens = ['MFU', 'EF', 'tok']
    for (const token of requiredTokens) {
      await expect.poll(
        async () => (await page.locator('body').innerText()).includes(token),
        { timeout: 30_000, message: `listing card missing token: ${token}` },
      ).toBe(true)
    }
    await expect(page.getByText(/step\s+[\d,]+/).first()).toBeVisible()
  })
})
