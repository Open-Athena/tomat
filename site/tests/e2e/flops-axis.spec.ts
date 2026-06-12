// FLOP-axis tick-format coverage (commit 71c4dd7 regression suite).
//
// Spec item 2: each `fopu` chip (`EF` / `PF` / `TF` / `sci`) must
//   1. update the x-axis title to mention the unit, AND
//   2. emit tick labels in the chosen unit's scale (not raw `2e19` when
//      EF is selected).
//
// The exact tick text depends on the run's FLOP range, so the per-unit
// regex below is the loosest possible "is this in the right magnitude /
// format" check that still catches the historical bug (raw scientific
// where fixed-point was expected, etc.).

import { expect, test } from '@playwright/test'
import { FIXTURES } from './_helpers/fixtures'
import { goHash } from './_helpers/nav'
import {
  readXAxisTitle,
  readXTicks,
  startConsoleErrorCapture,
  waitForAtLeastNPlots,
  waitForPlotPopulated,
} from './_helpers/plot'

interface UnitSpec {
  unit: 'EF' | 'PF' | 'TF' | 'sci'
  /** Title must contain this string. For 'sci' it's just `FLOP`; for the
   *  others, the unit code. */
  titleNeedle: string
  /** Pattern at least one tick must match. The patterns are tuned to the
   *  `flopTickformat` switch in src/runs/flops.format.ts:
   *    EF  → `.2~f`     → `\d+(?:\.\d+)?`
   *    PF  → `,.0f`     → `[\d,]+`
   *    TF  → `.2e`      → `\d+\.\d+e[+-]?\d+`
   *    sci → `.2e`      → `\d+\.\d+e[+-]?\d+`
   *  The trailing `~` on EF strips trailing zeros, so `240` is legal too. */
  tickPattern: RegExp
  /** A pattern that MUST NOT match any tick (the regression guard). For
   *  EF/PF, ticks must NOT be in scientific notation (the original bug). */
  forbiddenPattern: RegExp | null
}

const UNIT_SPECS: UnitSpec[] = [
  {
    unit: 'EF',
    titleNeedle: 'EF',
    tickPattern: /^\d+(?:\.\d+)?$/,
    forbiddenPattern: /e[+-]?\d+/i,
  },
  {
    unit: 'PF',
    titleNeedle: 'PF',
    // Plotly's `,.0f` renders integers, possibly with comma grouping.
    tickPattern: /^[\d,]+$/,
    forbiddenPattern: /e[+-]?\d+/i,
  },
  {
    unit: 'TF',
    titleNeedle: 'TF',
    tickPattern: /^\d+(?:\.\d+)?e[+-]?\d+$/i,
    forbiddenPattern: null,
  },
  {
    unit: 'sci',
    titleNeedle: 'FLOP',
    tickPattern: /^\d+(?:\.\d+)?e[+-]?\d+$/i,
    forbiddenPattern: null,
  },
]

test.describe('FLOP-axis tick scaling — per-run detail page', () => {
  for (const spec of UNIT_SPECS) {
    test(`x=flop&fopu=${spec.unit} renders correct title + tick format`, async ({ page }) => {
      const { errors } = startConsoleErrorCapture(page)
      await goHash(
        page,
        `/runs/${FIXTURES.modalFinishedFull}?x=flop&fopu=${spec.unit}`,
      )
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      // Allow the title + tick layout one render cycle to apply.
      await page.waitForTimeout(500)

      const title = (await readXAxisTitle(page, 0)) ?? ''
      expect(title, `title mentions ${spec.titleNeedle}`).toContain(spec.titleNeedle)

      const ticks = await readXTicks(page, 0)
      expect(ticks.length, 'at least one tick rendered').toBeGreaterThan(0)
      // At least one tick must match the unit's expected format.
      const matching = ticks.filter((t) => spec.tickPattern.test(t))
      expect(
        matching.length,
        `at least one tick matches ${spec.tickPattern} (got: ${JSON.stringify(ticks)})`,
      ).toBeGreaterThan(0)
      if (spec.forbiddenPattern) {
        const forbidden = ticks.filter((t) => spec.forbiddenPattern!.test(t))
        expect(
          forbidden,
          `no tick matches forbidden ${spec.forbiddenPattern} (got: ${JSON.stringify(ticks)})`,
        ).toEqual([])
      }
      expect(errors).toEqual([])
    })
  }
})

test.describe('FLOP-axis tick scaling — cross-run timeline', () => {
  for (const spec of UNIT_SPECS) {
    test(`cross-run x=flop&fopu=${spec.unit} renders correct title + tick format`, async ({
      page,
    }) => {
      const { errors } = startConsoleErrorCapture(page)
      await goHash(page, `/runs?x=flop&fopu=${spec.unit}&regex=kl-s05`)
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      await page.waitForTimeout(500)

      const title = (await readXAxisTitle(page, 0)) ?? ''
      expect(title, `title mentions ${spec.titleNeedle}`).toContain(spec.titleNeedle)

      const ticks = await readXTicks(page, 0)
      expect(ticks.length).toBeGreaterThan(0)
      const matching = ticks.filter((t) => spec.tickPattern.test(t))
      expect(
        matching.length,
        `at least one tick matches ${spec.tickPattern} (got: ${JSON.stringify(ticks)})`,
      ).toBeGreaterThan(0)
      if (spec.forbiddenPattern) {
        const forbidden = ticks.filter((t) => spec.forbiddenPattern!.test(t))
        expect(forbidden).toEqual([])
      }
      expect(errors).toEqual([])
    })
  }
})
