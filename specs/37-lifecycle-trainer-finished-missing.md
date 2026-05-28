# Spec 37: `lifecycle/trainer_finished` not being emitted on natural completion

> Status: **fix landed** in `marin/train_tomat_tpu.py` —
> `_log_lifecycle_event` now logs terminal events (`trainer_finished`,
> `sigterm_received`) **inline** with an explicit `wandb.run.log({}, commit=True)`
> flush, instead of via fire-and-forget daemon thread. Pre-fix runs
> still show "running" in wandb forever; backfill is manual or via a
> dashboard heuristic (`iris=SUCCEEDED && wandb=running → finished`).


## Symptom

Six runs from the last 48h ran to their target step and exited cleanly
(iris reports `state=SUCCEEDED exit=0`, no error, ckpts present), but
**none** of them ever logged `lifecycle/trainer_finished` to wandb.
The wandb-side `state` is therefore stuck at `running` indefinitely
for each.

Concretely (from snapshot 2026-05-28T01:28 UTC):

| Run                                                    | Final step | iris state | wandb state | trainer_finished logged? |
|--------------------------------------------------------|-----------:|------------|-------------|-------------------------:|
| `train-full-v3-200M-bs128-ce-10k-v5p16-noprm`          |       9999 | SUCCEEDED  | running     | NO                       |
| `train-full-v3-200M-bs128-ce-10k-v5p16-paired-base`    |       9999 | SUCCEEDED  | running     | NO                       |
| `train-ss-cont80k-emax050-1`                           |      84999 | SUCCEEDED  | running     | NO                       |
| `train-ss-cont80k-emax075-1`                           |      84999 | SUCCEEDED  | running     | NO                       |
| `train-ss-cont80k-emax100-1`                           |      84999 | SUCCEEDED  | running     | NO                       |
| `train-ss-cont80k-eps1const-1`                         |      84999 | SUCCEEDED  | running     | NO                       |
| `train-ss-cont80k-hi-argmax-1`                         |      84999 | SUCCEEDED  | running     | NO                       |
| `train-mg-4-cos-ce`                                    |       9999 | SUCCEEDED  | running     | NO                       |

Every parquet for these has exactly one lifecycle row:
`lifecycle/trainer_started: 1`. The
`OPTIONAL INT32 lifecycle/trainer_finished (INTEGER(8,true));` column
exists in the schema but is null for every row.

## Why it matters

Two cascading consequences:

1. **`/runs` dashboard misleads.** Cards for finished runs render with
   `state: running`. The dev who came back this morning to "check how
   the runs are doing" had to dig into the parquet history to see they
   were done.
2. **FR-eval-on-completion watchers don't fire.** Per
   `memory/first-real-mg-and-ss-results.md` the eval watchdog hooks on
   `trainer_finished`. Of the 5 finished SS-sweep cells, only
   `emax025-1` got auto-fired evals (which had a different completion
   path; see below). The other 4 sat for ~16h with their final ckpt
   un-evaluated until manual `tomat evals fire` was kicked.

## Where the signal is supposed to come from

`marin/train_tomat_tpu.py:155-196` — `_log_lifecycle_event(event,
**fields)`. It's a fire-and-forget wandb logger guarded by a
"wait-for-wandb.run" loop. Three call sites:

- `_handle_sigterm` → `_log_lifecycle_event("sigterm_received", ...)`
  (line 199–200). Inline; fires from the signal handler.
- `_log_lifecycle_event("trainer_started", ...)` (location: before
  Levanter `main()`).
- `_log_lifecycle_event("trainer_finished", ...)` — **should** fire
  on natural completion. Need to confirm this call site exists and
  what guards it.

Hypothesis: either the `trainer_finished` call site is in a code path
that only runs on the sigterm-init-only branch (i.e. it never gets
reached when Levanter's `train` returns normally), or it's behind a
`try` that swallows the success path. Audit `train_tomat_tpu.py` for:

- After Levanter's `train` call returns: is there an unconditional
  `_log_lifecycle_event("trainer_finished")`?
- If yes, what's the threading model? `_log_lifecycle_event` spawns a
  daemon thread (line 195–196). If the main process exits before the
  daemon flushes, the wandb POST never lands.
- If the trainer process is being torn down by iris immediately after
  the trainer function returns, the daemon thread may be killed
  before its 0.5s poll loop completes a successful `wandb.run.log`.

## Why emax025-1 worked

Looking at the snapshot: emax025-1's training row count is 2501
(2499 fine-tune steps + lifecycle starts), and it does have eval jobs
fired. But its parquet shows only `trainer_started`, no
`trainer_finished` (same as the others). So the eval fire wasn't
gated on `trainer_finished`; it was manual or watcher-fired via a
different signal. We need to confirm.

## Proposed investigation

1. **Locate the `trainer_finished` call site** in `train_tomat_tpu.py`
   (and any other trainer entry points — `train_tomat_modal.py` if
   that exists). If absent, that's the bug — add it.
2. **Verify thread flush.** Make the trainer-finished log path
   synchronous (or `wandb.run.log(..., commit=True)` followed by a
   short `wandb.run.flush()` / `wandb.run.finish()`). Daemon-thread
   fire-and-forget on the way-out is the suspect pattern.
3. **Backfill the affected runs.** For the 8 runs above, log
   `lifecycle/trainer_finished: 1` retroactively via a one-shot
   `wandb.run.log({"lifecycle/trainer_finished": 1})` patch script.
   Decide whether that's worth the time vs just letting the next
   sync-from-iris flip the dashboard via an `iris_state: SUCCEEDED` →
   `effectiveState: finished` heuristic.
4. **Dashboard fallback.** Independent of the trainer fix, the `/runs`
   card's `state` should derive from `(wandb.state, iris.state,
   lifecycle events)` jointly. If `iris.state == SUCCEEDED &&
   wandb.state == "running"`, treat as `finished`. That way even
   pre-fix runs render correctly.

## Out of scope

- Eval-watcher rewire (depends on this fix). Track separately if
  needed; for now manual `tomat evals fire` covers the gap.

## Owner / priority

Medium. The training results are intact; it's a state-reporting bug.
But it bit the SS-sweep harvest (5 cells silently waiting for evals
that never fired) and will keep biting until fixed.
