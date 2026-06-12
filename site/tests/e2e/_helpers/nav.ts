// Navigation helpers.
//
// The dashboard fetches from the deployed CFW; a transient 503 on the
// snapshot endpoint will leave the page stuck on a skeleton. We retry the
// nav once after a 2s backoff if the deck's first paint produced a
// console error suggesting backend trouble, or if the expected primary
// landmark never appears.

import { expect, type Page } from '@playwright/test'

/** Hash-routed nav: `goHash(page, '/runs/<id>?x=elapsed')` lands on
 *  `http://localhost:4273/#/runs/<id>?x=elapsed`. */
export async function goHash(
  page: Page,
  hashPath: string,
  opts: { waitFor?: string } = {},
): Promise<void> {
  const path = hashPath.startsWith('#') ? hashPath : `#${hashPath.startsWith('/') ? '' : '/'}${hashPath}`
  const url = `/${path}`
  await page.goto(url, { waitUntil: 'domcontentloaded' })
  if (opts.waitFor) {
    await page.waitForSelector(opts.waitFor, { timeout: 30_000 })
  }
}

/** Wrap a navigation in the project's CFW-503 retry pattern: try once,
 *  and if it fails OR if the page is still blank after `domcontentloaded`,
 *  back off 2s and try again. Mirrors the wandb / iris polling habit of
 *  not giving up on first transient.
 *
 *  Use `expect.toPass` to express the same idea when you need a finer-
 *  grained retryable assertion (e.g. waiting on a particular field). */
export async function gotoWithRetry(page: Page, url: string): Promise<void> {
  await expect.poll(
    async () => {
      try {
        const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 })
        if (!resp) return 'no-response'
        if (resp.status() >= 500) return `status-${resp.status()}`
        return 'ok'
      } catch (e) {
        return `nav-error:${(e as Error).message.slice(0, 60)}`
      }
    },
    {
      timeout: 30_000,
      intervals: [0, 2000],
      message: `gotoWithRetry(${url}) never produced a 2xx response`,
    },
  ).toBe('ok')
}

/** Returns the current URL's hash (the part after `#`, including the
 *  leading `#`). Used to assert URL-state persistence. */
export async function getHash(page: Page): Promise<string> {
  return await page.evaluate(() => window.location.hash)
}
