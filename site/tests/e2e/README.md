# tomat dashboard — Playwright e2e suite

Browser-driven regression tests for the `/runs` dashboard. Catches
render-time bugs that unit tests can't: bad plot data shape, missing card
fields, FLOP-axis unit mismatches, dotted gap-bridge blowups, URL state
that doesn't survive reload, etc.

## Why a browser suite

The class of bug we keep catching by manual CIC sweeps:
- a Plotly trace renders with `n=0` because we accidentally fed it
  pre-scaling data;
- the FLOP-axis title says `EF` but the ticks say `2e19`;
- a checkbox toggle stops persisting across reloads;
- a 4-segment run's dotted bridge balloons to thousands of points and
  the legend turns into noise.

None of these are catchable in jsdom + vitest. The trace data only
materialises after Plotly mounts a real `<svg>` against real history
parquet data; the URL state only matters when the back/forward buttons
and `?param=` ride along the hash. We need a real browser.

## How to run

```bash
# from repo root
pnpm --dir site exec playwright test
# specific file
pnpm --dir site exec playwright test run-detail
# headed (for debugging)
pnpm --dir site exec playwright test --headed
# specific test by title regex
pnpm --dir site exec playwright test -g "phantom bridge"
# open the HTML report after a run
pnpm --dir site exec playwright show-report tests/e2e/.report
```

The Playwright config (`site/playwright.config.ts`) auto-starts
`pnpm dev` on `http://localhost:4273` if it isn't already running, and
reuses an existing dev server if one is. The dashboard fetches from the
deployed CFW (`https://tomat-runs-api.openathena.workers.dev`) in dev mode
too — no local backend required.

## Adding a fixture run

`tests/e2e/_helpers/fixtures.ts` lists the canonical run ids used across
the suite. Criteria for adding one:

1. **Stable**: the data shape (segment count, lineage parent, eval set
   keys, summary completeness) shouldn't change week to week. Finished
   runs are best; running runs only when the test specifically needs
   "running" semantics.
2. **Varied**: each fixture exists because it's the canonical example of
   *something*. Don't add a second "finished Modal with full summary";
   pick one and reuse it.
3. **Backend-attested**: the run must appear in
   `https://tomat-runs-api.openathena.workers.dev/api/runs.json` (and
   have a manifest + history sidecar). Fixtures that only exist in
   wandb will 404 the snapshot fetch and break every test that names
   them.

When adding a fixture, also document *why* it was chosen (segment count,
lineage entry, etc.) in the comment block above its key — the
"adding a fixture run" criteria above only catches the easy mistakes.

## Asserting plot trace data

Use `tests/e2e/_helpers/plot.ts`:

```ts
import { readPlot, waitForPlotPopulated, readXTicks, readXAxisTitle } from './_helpers/plot'

await waitForPlotPopulated(page, 0)
const traces = await readPlot(page, 0)
// traces is TraceSummary[] — { name, lg, n, x_min, x_max, y_min, y_max, dash, mode, color }
expect(traces.filter((t) => t.n > 0).length).toBeGreaterThan(0)

const ticks = await readXTicks(page, 0)             // visible tick text
const title = await readXAxisTitle(page, 0)         // x-axis label
```

`readAllPlots(page)` returns every `.js-plotly-plot` div's traces in DOM
order. Use it when the page renders more than one plot (e.g. the run
detail page renders WallclockPlot + EvalsPanel sparklines).

The summary mirrors the shape of the inline JS probes we paste into the
DevTools console for manual CIC; if a new bug class needs a probe,
extend `TraceSummary` rather than `page.evaluate`-ing ad-hoc.

## CFW transient 503 retry

The deployed CFW occasionally 503s on cold/scaled-down workers. The nav
helper `gotoWithRetry` (in `_helpers/nav.ts`) retries the navigation once
with a 2s backoff:

```ts
import { gotoWithRetry } from './_helpers/nav'
await gotoWithRetry(page, '/#/runs/<id>')
```

For finer-grained retries against the same backend, use Playwright's
built-in `expect.poll` / `expect.toPass`. Both honour the project-wide
`expect.timeout` from `playwright.config.ts` (15s default), and both
retry transient failures while letting permanent failures fail fast.

Most navigations in this suite use the simpler `goHash(page, hashPath)`
which doesn't retry. If a CFW outage starts causing flakes, swap the
`page.goto` inside `goHash` for `gotoWithRetry` — the helpers are deliberately
co-located so the change is one line.

## TODO — CI integration

Currently this suite runs locally only. To wire it into CI we'd want:

1. Install Chromium in the CI image (Playwright's `--with-deps` flag).
2. Build the dashboard once (`pnpm --dir site build`), serve via
   `vite preview`, and point the e2e config at the preview port. This
   matches what users see in prod (post-treeshake, post-minify).
3. Decide on a CFW data policy: either let CI hit prod (cheap, real
   data, occasional flake) or stand up a fixture mirror in R2 for
   determinism. Probably start with prod + a retry budget.
4. Upload the HTML report (`tests/e2e/.report/`) as an artifact on
   failure so the trace viewer is available without re-running.

Out of scope for the initial worktree.
