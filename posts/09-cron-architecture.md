# How `/runs` stays fresh: cron architecture

**Status**: draft

---

## What it does

`tomat.oa.dev/runs` shows a card per training run with the current iris
state, latest TL / VL / mat-NMAE, MFU, runtime, preempt + failure
counts, and a small chip with MSRP-equivalent cost. None of that data
*lives* in the dashboard or the Cloudflare Worker (CFW) backing it —
both are read-only views over a few JSON / parquet sidecars on an R2
bucket. Two cron jobs keep those sidecars within ~60s of reality.

This post is the architecture, why it's structured the way it is, and
one cautionary tale about schema drift across two writers of the same
file.

---

## Why crons (instead of "just hit wandb / iris from the CFW")

The naive shape would be: the SPA hits the CFW, the CFW hits wandb +
iris on every request. Three problems killed it:

- **wandb is slow and not always trustworthy.** A `runs(project_path)`
  call regularly takes 5–20s; `scan_history` over a 80k-step run takes
  40–95s cold. The CFW has a hard 30s CPU-time budget per invocation
  and a 50/req subrequest soft cap. Even if it fit, paying that
  latency on every dashboard load is a non-starter.
- **iris controller auth is gated.** `iris job list` shells out to
  `gcloud compute ssh --tunnel-through-iap` to reach Marin's
  controller VM. The CFW has no Python, no gcloud, no SSH. Even with
  a service account, the IAP tunnel is the wrong machine shape for an
  edge function.
- **N clients × M panels × every render = controller hammering.** Spec
  37 (`feedback_iris_controller_load`) is explicit: ad-hoc queries to
  the shared iris controller add up, and we don't want the runs
  dashboard to be a load source. The crons run as one client.

The structural answer: pull *into* R2 on a schedule, serve *from* R2
behind the CFW. R2 reads are cheap, fast, and edge-cacheable; the slow
+ gated stuff happens out-of-band.

---

## Three layers

```
┌───────────────────────────────┐  every minute
│ Cron workers                  │ ──┐
│  - GCE VM   (tomat-iris-cron) │   │  iris RPC (IAP tunnel)
│  - Modal    (tomat-iris-sync) │   │  wandb GraphQL
│  - laptop   (tomat {iris,runs} sync)
└───────────────────────────────┘   ▼
                            ┌─────────────────────────┐
                            │ R2 bucket: openathena/  │
                            │  tomat/                 │
                            │   iris-state.json       │
                            │   iris-attempts/*.json  │
                            │   runs/<id>/            │
                            │     manifest.json       │
                            │     raw.parquet         │
                            │     evals/index.json    │
                            │     evals/<key>.json    │
                            │     cost.json           │
                            │   modal-state.json      │
                            └────────────┬────────────┘
                                         │  R2 reads (Range-supported)
                                         ▼
                            ┌─────────────────────────┐
                            │ Cloudflare Worker       │
                            │  tomat-runs-api         │
                            │  - /api/runs-snapshot   │
                            │  - /api/iris-state.json │
                            │  - /api/runs/:id/*      │
                            │  edge-cache 30s         │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ React SPA (Vite)        │
                            │  tomat.oa.dev/runs      │
                            │  TanStack Query 30s     │
                            └─────────────────────────┘
```

1. **Crons** poll iris's controller (RPC) and wandb's GraphQL,
   normalize the response into our wire format, and `PUT` the result
   to R2.
2. **R2** is the only durable layer. Sidecars are byte-identical
   regardless of which cron wrote them (modulo the schema-bump
   footnote below) — the dashboard doesn't know or care.
3. **The CFW** ([`worker/src/index.ts`]) serves R2 with CORS,
   Range-request support (for the parquet footer-then-column read
   pattern hyparquet uses), and a single aggregated
   `/api/runs-snapshot.json` endpoint that fans out ~50 parallel
   `r2.get()` calls and inlines them into one response. That
   aggregated payload is edge-cached for 30s with stale-while-
   revalidate.

[`worker/src/index.ts`]: ../worker/src/index.ts

---

## What goes in each artifact

| artifact | writer(s) | role |
|---|---|---|
| `tomat/iris-state.json` | `tomat iris sync` (laptop), `cron_iris_sync_modal.py` (Modal), `scripts/iris-sync.py` (GCE VM) | parent-job snapshot: state, preempts, failures, task histograms — drives the iris badge on each card |
| `tomat/iris-attempts/<label>.json` | `tomat iris sync` (only) | per-attempt death timestamps + classified causes — drives `CRASH-LOOP` badge + `RecentEvents` causation sub-lines |
| `tomat/runs/<id>/manifest.json` | `tomat runs sync` (laptop), `scripts/runs-sync.py` (GCE VM) | wandb run-level metadata: config, summary, history range, last sync time |
| `tomat/runs/<id>/raw.parquet` | `tomat runs sync` (laptop), `scripts/runs-sync.py` (GCE VM) | full wandb history table — train/eval loss, MFU, mat-NMAE, lifecycle flags, cluster preempts/failures |
| `tomat/runs/<id>/evals/index.json` | `tomat evals sync` / `tomat evals backfill` | eval-records index (spec 43): `{key, fired_at, state}` per fired eval |
| `tomat/runs/<id>/evals/<key>.json` | `tomat evals sync` / `tomat evals backfill` | one eval record: per-mat-set NMAE/NEMD, plus optional per-task breakdown |
| `tomat/runs/<id>/cost.json` | `tomat runs cost` | MSRP-equivalent compute estimate (spec 46) — TPU hours × per-chip rate |
| `tomat/modal-state.json` | `tomat modal sync` | Modal app + function-call snapshot — backs the `ModalBadge` for runs without an iris job (replaces the wandb-session-state fallback) |
| `tomat/voxel-corr/<label>.{json,bin.gzip}` | `tomat analyze voxel-corr-blob --r2` | per-run voxel-position correlation matrix (immutable per label, cache 1h at edge) |

The naming convention is "one artifact = one file"; the CFW route map
is mostly a thin wrapper that maps `/api/<thing>` → R2 key + caching
policy. Lazy-fetching is reserved for the bigger blobs (parquet,
voxel-corr); everything an `/runs` first-paint needs is in the one
`runs-snapshot.json`.

---

## Two parallel crons doing the same job

There are **two** copies of the iris+wandb cron in production right
now. Both write to the same R2 prefixes; only one needs to win each
tick.

### GCE VM (`tomat-iris-cron`, since the project's early days)

Setup: [`scripts/setup-iris-cron-vm.sh`]. An e2-small in
us-east1-d, ~$5/mo. The VM has `iris` installed in a venv
(`~/iris-sync-venv`), ADC + R2 creds dropped to `~`, and two crontab
lines:

```bash
*/5 * * * * $VENV/bin/python $HOME/iris-sync.py >> $HOME/iris-sync.log 2>&1
* * * * *   /usr/bin/flock -n /tmp/runs-sync.lock \
              $VENV/bin/python $HOME/runs-sync.py >> $HOME/runs-sync.log 2>&1
```

- `iris-sync.py` every 5 min: each `iris` invocation re-establishes
  its own IAP tunnel to Marin's controller (~60s typical, can spike
  to 200s); 3 prefixes serially fits comfortably in a 5-min cycle.
- `runs-sync.py` every 1 min, `flock`-guarded: no tunnel needed (just
  wandb API + R2 PUTs), but a sync of a large run's full history can
  take 40–95s, so `flock -n` skips the tick if the prior one's still
  running.

The 5-minute iris cadence is the latency floor for this writer. On
top of that, the IAP tunnel cold-start dominates: from the time the
controller's state changes to the time R2 sees it, ~30s–5min is the
honest window.

### Modal (`tomat-iris-sync-cron`, added 2026-06-02)

[`scripts/cron_iris_sync_modal.py`] — a single Modal app with a
`@app.function(schedule=modal.Cron("* * * * *"))` decorator.

```python
@app.function(
    cpu=1,
    memory=1024,
    timeout=120,
    secrets=[adc_secret, r2_secret],
    schedule=modal.Cron("* * * * *"),
)
def cron_sync() -> dict:
    return _sync()
```

Modal pulls a fresh `marin-iris` build from the GitHub repo at a
pinned SHA, materializes the GCP ADC JSON to `/tmp/adc.json`,
sub-process-shells `iris --cluster=marin job list --prefix … --json`
across 3 prefixes, normalizes, and `PUT`s.

Latency from request → R2 PUT in steady state: ~15s for the iris
RPC + ~50ms for the R2 write. Cold-start adds a few seconds (~5s) on
the first call after a container expires, which is rare with a
1/minute schedule.

Cost: ~15s × 1440/day × Modal's ~$0.000038/CPU-s ≈ **$0.80/day** for
the iris-state sync, paid on demand. No always-on VM bill.

### Comparison

| | GCE VM | Modal cron |
|---|---|---|
| iris-state cadence | every 5 min | every 1 min |
| iris-state freshness window | 30s–5min (tunnel cold-start dominates) | ~50ms (warm container) – ~15s (cold) |
| runs sync (wandb) | every 1 min (`flock`-guarded) | not yet — laptop / GCE still own this |
| baseline cost | $5/mo (always-on e2-small) | $0.80/day (~$24/mo) for iris-only |
| operational dep | gcloud + iris + boto3 inside the VM | Modal secrets + pinned `marin-iris` SHA |
| ssh-key story | gcloud's `~/.ssh/google_compute_engine` registered with project metadata on first IAP use | Open — Modal containers need a stable key path to register; currently still under bring-up |
| failure mode | VM gets OOM-wedged if pip-installs grow (don't switch to e2-micro — wandb + pyarrow swap-thrash so deep that sshd can't get a shell) | clean container per call; failures are visible in Modal's UI |

The cost columns are noisy in both directions. GCE is a fixed bill
regardless of cron frequency; the VM also runs `runs-sync.py` every
minute, which Modal isn't doing yet. Modal's $0.80/day figure is for
the iris-state cron alone; adding runs-sync at the same cadence would
bump it but still well under the GCE always-on cost for a workload
this small.

The real argument for Modal isn't dollars. It's the **per-minute vs
per-5-min cadence** + much lower steady-state latency. Modal's cron
sees an iris state change and has it in R2 in seconds. The GCE
`iris-sync.py` polls 3 prefixes serially with a fresh IAP tunnel each
time — a fast tick is ~60s and a slow one is 200s+, so cron has to
fire every 5min just to avoid stacking. That difference matters
because we're trying to detect cascade-restart loops (spec 45)
quickly enough to act on them, not 5min after the fact.

[`scripts/setup-iris-cron-vm.sh`]: ../scripts/setup-iris-cron-vm.sh
[`scripts/cron_iris_sync_modal.py`]: ../scripts/cron_iris_sync_modal.py

---

## The `task_state_counts` schema bump — cautionary tale

Spec 45 (`dashboard-tz11-surfacing`) added a `task_state_counts` field
to the iris-state payload so the dashboard could detect crash-loop
pathology: `state == 'RUNNING' ∧ failures > 0 ∧
task_state_counts.running == 0`. Without that field, a job that's
crash-looped 13× over 36h reads as green `RUNNING (4p)` on the badge,
and we lose 21h between the fix landing and noticing the new failure
mode. The whole post-mortem (`specs/done/31-tz11-postmortem.md`) is
the cost of *not* seeing through the lie.

The bump was added to two of the three writers:

```python
# tomat (laptop) iris_sync — has it
tsc_raw = row.get("task_state_counts") or {}
tsc = {k: int(v) for k, v in tsc_raw.items() if int(v) > 0}
jobs[jid] = {
    "state": state,
    ...
    "task_state_counts": tsc,
}

# scripts/cron_iris_sync_modal.py — has it (added today)
tsc_raw = row.get("task_state_counts") or {}
tsc = {k: int(v) for k, v in tsc_raw.items() if int(v) > 0}
jobs[jid] = {
    "state": state,
    ...
    "task_state_counts": tsc,
}

# scripts/iris-sync.py (GCE VM) — DOES NOT have it
jobs[jid] = {
    "state": state,
    "state_code": 0,
    "preempts": int(row.get("preemption_count") or 0),
    ...
    # ← no task_state_counts
}
```

R2's `iris-state.json` is a single key; whichever cron writes last
wins. Three writers, two of them with the new field, one without
means the dashboard's crash-loop detection works **only when the
freshest write came from a v2 writer**. When the GCE VM's 5-min tick
lands after the Modal cron's 1-min tick, the freshest snapshot loses
the field, the dashboard regresses to green `RUNNING`, and we're back
to the tz-11 failure mode.

Both the file's `schema_version: 1` and the "last writer wins"
semantics conspire here. Bumping the schema number would have let the
CFW detect "GCE VM wrote v1 over a v2" and refuse, or at least surface
it. We didn't, and the field is optional from the dashboard's
perspective, so the regression is silent.

**The rule going forward**: any field added to a multi-writer
artifact has to land in *every* writer in the same commit, or the
schema version has to be bumped + readers updated to refuse old
writes. Spec 45 documents the dashboard-side invariant; the
cron-side invariant is the symmetric one.

(Pragmatic fix in flight: deprecate the GCE VM writer. Below.)

---

## Future direction

The Modal cron is the obvious replacement target:

- 60× higher cadence for iris-state (1min vs 5min) — material for
  crash-loop detection.
- ~30× lower steady-state latency (50ms warm vs 30s tunnel cold).
- Cleaner operational model: no VM to babysit, secrets in Modal's
  vault, container rebuilds on `MARIN_SHA` bump and that's it.
- Modal's UI is straightforward for debugging — a failed tick shows
  up red with stderr inline.

Two paths from here:

1. **Consolidate.** Move `runs-sync.py` to Modal too, retire the GCE
   VM. Settles the `task_state_counts`-class schema-drift problem
   structurally (one writer = one schema). Pending: SSH-key handling
   for the IAP tunnel from inside a Modal container, which today still
   uses a fresh key per invocation — fine for occasional runs, but
   needs a stable key registered with the project metadata before this
   is the only writer (otherwise gcloud has to re-register on every
   cold start). Spec slot reserved for this.
2. **Keep both as redundancy.** Modal is the primary, GCE is a
   warm-standby that catches outages. Costs ~$5/mo + ongoing
   schema-drift discipline. The dashboard never goes blank.

Path 1 is the right answer eventually; path 2 is the safer default
while we work through Modal-side issues.

---

## TODO

- File a spec for the SSH-key + IAP-tunnel handling in Modal
  containers (path 1 prerequisite).
- Add `runs-sync` Modal counterpart (`cron_runs_sync_modal.py`)
  matching the iris-only deploy pattern.
- Decide on a schema-versioning bump policy for iris-state when fields
  are added — readers refuse stale writers, or writers all bump
  together?
- Make the CFW surface the `synced_at` timestamp prominently when it's
  more than ~3min stale (currently the dashboard quietly serves
  whatever's there).
- Drop the GCE VM (path 1) or document the redundancy posture
  (path 2). Either decision unblocks deleting one of the two
  iris-sync.py copies.
