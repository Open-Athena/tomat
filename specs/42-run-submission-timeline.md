# 42 — Run submission → terminal timeline (full wallclock)

## Problem

`tomat runs status` and the `/runs` run-detail page derive `runtime` from
wandb's `_runtime` field. That clock only starts when the first worker
process imports wandb and emits a metric — typically several minutes after
the user fired `tomat train`:

```
[tomat train fired locally]
   ↓ ~5-60s   iris cli → controller, RPC
[iris job submitted]
   ↓ minutes-hours    queued: pool scale-up / preemption-backoff / quota
[iris job accepted, workers provisioning]
   ↓ ~1-3min  sync-deps, install pip, activate venv
[user command runs on worker]
   ↓ ~30-90s  JAX init, train_lm config build, data cache attach
[first wandb metric arrives]   ← `_runtime` clock starts here
   ↓ ~30-60s  JIT compile
[first train step]
   ↓ N hours  training
[terminal: completed / failed / preempted]
```

The currently-displayed `runtime` collapses everything before "first wandb
metric arrives" to zero. So a run that took 4 hours to leave the queue and
12 minutes to crash looks like "12-minute run" — and queue churn (a real
operational signal) is invisible in our reporting.

## Goal

Run-detail page shows the **honest temporal story** — from the user's
local `tomat train` invocation to the iris terminal transition — split
into named phases the reader can interpret.

## Design

Two complementary capture points, joined at render time.

### A. Local-side submission record

`tomat train` writes a `SubmissionRecord` at the moment it hands off to
`iris job submit`:

```python
@dataclass
class SubmissionRecord:
    label: str
    submitted_ts: datetime         # wall clock at fire
    cluster: str                   # marin
    tpu: str                       # v6e-16
    fire_argv: list[str]           # full CLI for reproducibility
    fire_host: str                 # local hostname (which laptop fired it)
    git_sha: str | None            # tomat repo HEAD at fire
    marin_pin: str | None          # pyproject.toml's `Open-Athena/marin@...`
```

Persistence: the same D1 the dashboard CFW poller already uses (task #111).
Schema: a new `submissions` table keyed by `label`. The poller already has
write access; `tomat train` POSTs to a new CFW route
`/api/submissions` with the record.

Failure modes — what if D1 write fails at fire time? Do NOT block the fire:
log a warning and continue. The local-side record is best-effort; the
server-side state-transition record (B) backfills the missing `submitted_ts`
as `iris.first_seen_ts` (a slight over-estimate by the RPC latency).

### B. Server-side iris state transitions

The CFW cron poller (task #111) already polls iris for in-flight tracked
jobs. Extend its loop to record every state transition into a new
`iris_transitions` table:

```sql
CREATE TABLE iris_transitions (
  label TEXT,
  state TEXT,            -- pending|accepted|provisioning|running|preempted|failed|completed
  ts INTEGER,            -- unix seconds
  detail TEXT,           -- e.g. "cosched_failed=3, preempts=2"
  PRIMARY KEY (label, ts)
);
```

iris exposes `state` and (for some states) timestamps via `job summary`.
The poller diffs the latest summary against the last recorded state per
label and writes a new row on transition. To keep the controller load low
(see memory `feedback_iris_controller_load.md`), poll only labels that
are tracked + non-terminal, at the existing poll cadence.

### C. Render

Add a `<SubmissionTimeline label={…} />` component on the run-detail page:

```
Submitted   2026-05-31 11:18:24Z  by ryan@laptop
   │ 2m 14s   ↓ iris ingest + capacity wait
Accepted    11:20:38Z
   │ 1m 47s   ↓ provisioning workers
Running     11:22:25Z
   │ 0m 52s   ↓ deps + JAX init + JIT
First step  11:23:17Z   (wandb `_runtime` = 0)
   │ 4h 12m  ↓ training
Completed   15:35:01Z
            Wallclock total: 4h 16m 37s   (wandb-runtime: 4h 11m 44s)
```

Each segment is a duration chip with a clear label. Hover shows the exact
timestamps. The trailing line shows the gap between the honest wallclock
and the current `_runtime`-based display, so we can quantify how much
queue/setup time has been hidden.

## Open questions

- **`tomat train` without D1 write**: should fire be a hard fail if the
  POST 5xx's, or best-effort? Recommend best-effort — the alternative
  is "dashboard outage blocks training submissions," which is worse.
- **Multi-attempt runs** (`failures=N preemptions=K`): each iris retry
  fires a new pending→running cycle. Record all of them and surface the
  pattern (e.g., 3 preempt-restarts in the first 30min is worth a glance).
- **Pre-existing runs**: backfill `submitted_ts` for runs we already
  track? Skip — accept that pre-spec runs have wandb-runtime only, and
  the dashboard renders a single segment for those.
- **Modal runs**: same idea, different state machine. Modal app states
  map cleanly: `created → enqueued → running → terminated`. Out of
  scope for v1; track in a follow-up (task #112 already covers Modal
  state on the dashboard).

## Out of scope

- New metrics or plots beyond the timeline view (e.g., "average queue
  time by pool" — that lives downstream of this data).
- iris fork / upstream changes — we only need read access via existing
  `job summary`.
- Replacing wandb's `_runtime` everywhere; the new wallclock is
  *additive*. wandb-runtime stays as the "training-time" metric.

## Phasing

1. **Schema + poller**: extend the CFW poller to write `iris_transitions`.
   Renders nothing yet, but starts capturing data.
2. **Submission record**: `tomat train` POSTs at fire time. Backfill old
   runs is N/A; we accept the gap.
3. **Render**: `<SubmissionTimeline />` on the run-detail page.
4. **Follow-ups**: Modal state machine (task #112), per-pool queue-time
   aggregates, retry-pattern heuristics.

## Related

- Task #111 (Phase 2 /runs dashboard: cron CFW poller + D1 + multi-team)
- Task #112 (Add Modal run/app state to /runs)
- Task #195 (RunsTimelinePlot: end-of-trace markers for running runs)
- Task #197 (runs-sync: drop stale trajectories on global_step reset)
- Memory: `feedback_iris_controller_load.md` (limit polling rate)
- Memory: `runs-sync-trim-stale-trajectories.md`
