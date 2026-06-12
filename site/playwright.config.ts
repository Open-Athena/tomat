import { defineConfig, devices } from '@playwright/test'

// Single-worker, chromium-only config. The runs dashboard fans out a lot of
// fetches on first paint; running tests in parallel browsers compounds the
// load on the deployed CFW + tanstack-query cache and tends to surface
// transient 503s. One worker keeps it boring.
//
// The dev server is started on demand (`reuseExistingServer: true`), which
// also lets the developer just `pnpm dev` in a sibling shell and have the
// suite attach instead of double-starting.
//
// TODO: wire this into CI. Out of scope for the initial worktree — see
// site/tests/e2e/README.md for the open items.
export default defineConfig({
  testDir: './tests/e2e',
  // Parallelism: most tests are read-only against the deployed CFW.
  // `fullyParallel: true` lets each describe block run in its own browser
  // context concurrently, cutting the suite from ~8min to ~4min on a
  // 2-worker laptop. Increase `workers` cautiously — too many concurrent
  // tanstack-query fanouts will start hitting CFW rate-limit responses.
  fullyParallel: true,
  workers: 3,
  retries: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'tests/e2e/.report' }]],
  use: {
    baseURL: 'http://localhost:4273',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:4273',
    reuseExistingServer: true,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
})
