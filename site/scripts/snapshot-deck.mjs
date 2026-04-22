#!/usr/bin/env node
/**
 * Headless deck → PDF snapshot.
 *
 * Flow: spin up `vite preview` in-process against `dist/` (assumed built),
 * drive Playwright Chromium to /#/deck, force @media print, call page.pdf().
 * Output: `snapshots/YYYY-MM-DD[-<label>]-deck.pdf`.
 *
 * Usage:
 *   pnpm --dir site build
 *   pnpm --dir site snapshot                 # default dated filename
 *   pnpm --dir site snapshot -- --label weekly-2026-04-21
 */

import { preview } from 'vite'
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
  options: {
    label: { type: 'string' },
    out: { type: 'string' },
    port: { type: 'string', default: '4427' },
  },
})

const siteRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const date = new Date().toISOString().slice(0, 10)
const suffix = values.label ? `-${values.label}` : ''
const outPath = values.out ?? join(siteRoot, 'snapshots', `${date}${suffix}-deck.pdf`)

mkdirSync(dirname(outPath), { recursive: true })

const server = await preview({
  root: siteRoot,
  preview: { port: Number(values.port), strictPort: true },
})
const url = server.resolvedUrls?.local?.[0]
if (!url) {
  await new Promise(r => server.httpServer.close(r))
  throw new Error('vite preview did not report a local URL')
}
console.log(`[snapshot] preview server at ${url}`)

try {
  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })

  // Arbitrary viewport; @page size in print CSS is what the PDF renderer uses.
  await page.goto(`${url}#/deck`, { waitUntil: 'networkidle' })

  // Plots are lazy-rendered on scroll-into-view. Walk every slide so each
  // Plotly chart initializes before we emulate print media.
  const slideCount = await page.locator('.deck-slide').count()
  console.log(`[snapshot] ${slideCount} slides; prewarming plots…`)
  for (let i = 0; i < slideCount; i++) {
    await page.locator('.deck-slide').nth(i).scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
  }
  await page.waitForTimeout(1500)  // let any trailing Plotly layout settle

  // Force dark color scheme so the PDF matches the dark-mode site styling;
  // without this, `@media (prefers-color-scheme: dark)` in our CSS doesn't
  // fire inside Chromium's print rendering and we get a washed-out light-mode
  // PDF with dark-mode content (e.g. light text on white page).
  await page.emulateMedia({ media: 'print', colorScheme: 'dark' })
  await page.pdf({
    path: outPath,
    printBackground: true,
    preferCSSPageSize: true,
  })

  await browser.close()
  console.log(`[snapshot] wrote ${outPath}`)
} finally {
  await new Promise(resolve => server.httpServer.close(resolve))
}
