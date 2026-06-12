// Smoothing-mode coverage (spec item 9).
//
// `?smooth=raw`, `?smooth=ema:0.1`, `?smooth=rolling:50` must each render
// without throwing console errors. We also confirm each lands a plot with
// finite data — a smoothing helper that silently nulls the trace would
// pass a "no console errors" check but isn't actually working.

import { expect, test } from '@playwright/test'
import { FIXTURES } from './_helpers/fixtures'
import { goHash } from './_helpers/nav'
import {
  readPlot,
  startConsoleErrorCapture,
  waitForAtLeastNPlots,
  waitForPlotPopulated,
} from './_helpers/plot'

const SMOOTH_MODES = ['raw', 'ema:0.1', 'rolling:50'] as const

test.describe.parallel('smoothing modes — per-run page', () => {
  for (const mode of SMOOTH_MODES) {
    test(`smooth=${mode} renders without console errors`, async ({ page }) => {
      const { errors } = startConsoleErrorCapture(page)
      await goHash(page, `/runs/${FIXTURES.modalFinishedFull}?smooth=${mode}`)
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      const traces = await readPlot(page, 0)
      const populated = traces.filter((t) => t.n > 0)
      expect(populated.length, `smooth=${mode}: at least one trace populated`).toBeGreaterThan(0)
      expect(errors, `smooth=${mode}: no console errors`).toEqual([])
    })
  }
})

test.describe.parallel('smoothing modes — cross-run timeline', () => {
  for (const mode of SMOOTH_MODES) {
    test(`cross-run smooth=${mode} renders without console errors`, async ({ page }) => {
      const { errors } = startConsoleErrorCapture(page)
      await goHash(page, `/runs?smooth=${mode}&regex=kl-s05`)
      await waitForAtLeastNPlots(page, 1)
      await waitForPlotPopulated(page, 0)
      const traces = await readPlot(page, 0)
      const populated = traces.filter((t) => t.n > 0)
      expect(populated.length, `smooth=${mode}: at least one trace populated`).toBeGreaterThan(0)
      expect(errors, `smooth=${mode}: no console errors`).toEqual([])
    })
  }
})
