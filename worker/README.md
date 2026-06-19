# `worker/` — `tomat-runs-api` Cloudflare Worker

R2-backed JSON/binary API for the `/runs` dashboard. Routes are defined
in `src/index.ts`; CORS is shared via `corsHeaders()`.

## Two deployments

| name | URL | branch |
|---|---|---|
| **prod** | `https://tomat-runs-api.openathena.workers.dev` | `main` |
| **staging** | `https://tomat-runs-api-staging.openathena.workers.dev` | iteration |

Both bind the same `openathena` R2 bucket and the same `tomat/runs`
prefix — data is shared; only the worker code is independently
deployed. Iterate on staging; promote to prod when stable.

## Commands

From the repo root:

```bash
pnpm --dir worker run dev              # localhost wrangler dev (no R2)
pnpm --dir worker run deploy           # → prod
pnpm --dir worker run deploy:staging   # → staging
pnpm --dir worker run tail             # tail prod logs
pnpm --dir worker run tail:staging     # tail staging logs
```

## Pointing the FE dev-server at staging

Two routes:

1. **Build-time env var** (recommended for `pnpm dev` sessions). Either
   export it inline or drop a `.env.local` next to `site/package.json`:
   ```bash
   echo 'VITE_RUNS_API_BASE=https://tomat-runs-api-staging.openathena.workers.dev' \
     > site/.env.local
   pnpm --dir site dev
   ```
   `site/.env.local` is gitignored by Vite's defaults.
2. **Runtime query param** (one-off): append `?api=<url>` to any page
   URL. Note that `api.ts:13` reads `window.location.search` (the part
   before `#`), so the override goes BEFORE the hash:
   `https://tomat.oa.dev/?api=https://tomat-runs-api-staging.openathena.workers.dev/#/runs/bin5`.

## Iteration loop for CFW-touching changes

Before pushing a worker change to `main`:

```bash
pnpm --dir worker run deploy:staging
# CIC / curl against tomat-runs-api-staging.workers.dev
pnpm --dir worker run deploy   # only when staging is green
```

`prod` is shared infrastructure — please don't iterate on it.
