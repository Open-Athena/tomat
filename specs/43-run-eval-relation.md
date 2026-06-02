# 43. Model "eval job ↔ training run" relation explicitly

## Motivation

Eval jobs (`tomat evals fire …`) are conceptually **children** of a
training run: they read one of its checkpoints, run inference on a mat
set, write a result. Today, the parent ↔ child link is reconstructible
only by:

- parsing iris job names (`tomat-eval-<run_label>-<set>-step-<N>(-taskI)`)
- parsing Modal call descriptors (`<fc_id>` + the spawn-side function
  call arg dict)
- ls'ing GCS at predictable paths
  (`gs://…/tomat/eval/results/<run_label_or_leaf>/<set>/<mode>/step-N.json`)

Three concrete problems with the string-parse approach:

1. **Renames break the dashboard silently.** If we ever change the
   eval-job naming pattern (we already have edge cases — `-bs128-seed42`
   leaf suffixes for Modal training runs, `-task<i>` suffix when
   `--num-tasks > 1`, the upcoming `--backend modal` path with its own
   identifier shape), every consumer of the relation has to update.
2. **In-flight state is invisible per child.** The current
   `eval.json` artifact only carries *completed* result numbers. There's
   no record of "step-15000 / val_200 / maskgit was submitted at T,
   currently running, ETA in ~12 min." That state lives in iris /
   Modal job-state dumps that the dashboard doesn't tie back to the
   parent run.
3. **Backfilling state is racey.** A new eval job is created by the
   CLI; if the cron poller next runs *before* that job logs anything,
   the relation is invisible until the next pass — and if the job dies
   before logging at all (e.g. ckpt-load OOM), the relation is
   *never* recorded.

## Goal

Make the run ↔ eval-job relation a **first-class persisted record**,
written by the CLI at submit time and updated as the job progresses,
so the dashboard and harvest both consume from the same authoritative
source instead of re-deriving it from job names.

## Non-goals

- Replacing `eval.json` (the per-step aggregated result series) — it
  stays as the consumer-friendly per-(set, mode) trajectory, but is now
  *derived* from the relation records, not the primary store.
- A general "jobs DB" — only eval children are in scope here. Training
  runs are already cards on `/runs`; we're not refactoring that.
- Adding a SQL DB just for this. The existing R2-sidecar pattern is
  the simplest fit; D1 can come later as a queryable index if needed.

## Data model

One record per *(training_run × step × mat_set × eval_mode × task_idx)*
combination. Each record carries enough to reconstruct the GCS result
path, the iris/Modal job state, and the displayable result.

```jsonc
{
  "run_label": "train-mg-modal-h200x8-tz-v4",     // training-run id (= the dashboard run card)
  "ckpt_leaf": "train-mg-modal-h200x8-tz-v4-bs128-seed42",  // leaf dir; null when == run_label
  "step": 15000,
  "set": "val_200",                                // val_200 | train_200 | …
  "mode": "maskgit",                               // teacher | free | maskgit
  "task_idx": 0,                                   // 0 when --num-tasks=1; absent otherwise
  "n_tasks": 1,                                    // total number of fanout tasks
  "n_mats": 50,                                    // cap requested with -n
  "eval_label": "val-full-v3",                     // TOMAT_LABEL (t10n must match ckpt)
  "model_preset": "200M",
  "mg_k_steps": 12,                                // (mode=maskgit) iterative-refinement K
  "free_batch": null,                              // (mode=free) HBM-knob batch
  "eval_batch": 32,
  "decoder": "median",

  "backend": "modal",                              // iris | modal
  "job_ref": {                                     // backend-specific job identifier
    "kind": "modal",
    "app_id": "ap-tdp56ytfY0DIvLTwOKQJe4",
    "call_id": "fc-01KT2X9C967342N4PKRG9CQE0J",
    "function_name": "eval_checkpoint_h200x8"
    // for kind: "iris" → { kind:"iris", job_name:"tomat-eval-…-step-15000", task_id:1 }
  },

  "fired_at": "2026-06-01T20:10:00Z",
  "state": "running",                              // pending | running | succeeded | failed | killed
  "state_synced_at": "2026-06-01T20:14:32Z",       // last refresh from backend
  "started_at": "2026-06-01T20:11:13Z",            // null until backend reports
  "finished_at": null,
  "exit_code": null,
  "error_msg": null,                               // populated on failure (truncated stderr)

  "gcs_result_path": "gs://marin-eu-west4/tomat/eval/results/train-mg-modal-h200x8-tz-v4-bs128-seed42/val_200-maskgit/step-15000.json",
  "result": null,                                  // populated when the result JSON lands; copy of the relevant scalar(s)
  "result_seen_at": null
}
```

### Where the record lives

**Per-run R2 sidecar**, alongside `manifest.json` + `eval.json`:

- New: `runs/<run_label>/evals/<step>-<set>-<mode>(-task<i>).json`
  — one file per record. Idempotent overwrite from any path that updates
  state.
- New: `runs/<run_label>/evals/index.json` — pointer list,
  `[{ "key": "<step>-<set>-<mode>", "fired_at": "…", "state": "…" }, …]`,
  sorted by `(step, set, mode, task)`. Cheap for the dashboard to fetch
  to enumerate children without N R2 GETs.

The CFW serves these via:

- `GET /api/runs/<run_label>/evals` — the index
- `GET /api/runs/<run_label>/evals/<key>` — one record

The aggregated `/api/runs-snapshot.json` (line ~252 of `worker/src/index.ts`)
gains a `runs[i].evals: [<index entry>, …]` array so the dashboard's
single hot-cached fetch carries everything it needs to render the
eval matrix per run card on first paint.

### Why not D1 yet

The R2-sidecar pattern is the same one we use for manifests + the iris
attempts dump; the CFW already knows how to serve it; rebuilds are
idempotent from the underlying GCS truth. A D1 index would be useful
for cross-run queries ("all maskgit-mode 5k evals across all tz-*
runs") and pending task #111 ("Phase 2 /runs dashboard: cron CFW
poller + D1 + multi-team") is already on the roadmap for that. When D1
lands, the records here ship into it 1:1 with no schema change.

## Lifecycle / who writes when

### Author 1: `tomat evals fire` (submit time)

For each (step, set, mode, task) the CLI submits, write the initial
record with `state="pending"` and the backend job_ref (call_id /
iris-job-name). This happens *before* the actual `iris job run` /
Modal `spawn` call — so a record exists even if the backend submit
itself fails (in which case we move it to `state="failed"` immediately
with the error).

The write goes to R2 via a new endpoint on the CFW:
`POST /api/runs/<run_label>/evals` body=`<record>`. CFW writes the
sidecar + appends to the index. Auth: the same scheme as `tomat runs
sync` (R2 write creds in the CLI's env). If the CFW isn't reachable,
the CLI falls back to writing a `tmp/evals-pending-<key>.json`
file and surfaces a warning; `tomat evals sync` flushes them on next
run.

### Author 2: `tomat evals sync` (state + result harvest)

Existing CLI already aggregates per-step result JSONs into
`eval.json`. Extend it to:

1. **Update state**: for each record with `state ∈ {pending, running}`,
   poll the backend (`iris job summary` for iris, `modal call status`
   for Modal) and update `state` + `state_synced_at` + the timing
   fields.
2. **Detect result-landed**: ls the `gcs_result_path`; if present,
   parse the JSON, populate `result` with the canonical scalar(s) for
   `(set, mode)`, set `result_seen_at`, transition `state` to
   `succeeded`. (A record can be `succeeded` even if the backend
   reports `failed`/`killed` post-result — the result file landing is
   the ground truth.)
3. **Detect dead jobs**: when the backend reports a terminal failure
   AND no result file exists, transition to `failed` with the truncated
   error.

This sync runs in two contexts: on-demand via `tomat evals sync` (as
today), and from the existing cron at `scripts/cron_iris_sync_modal.py`
(extended with one extra call per active run).

### Backfill

`tomat evals backfill <run_label>` reconstructs records for historical
evals by:

1. ls'ing `gs://…/tomat/eval/results/<run_label or leaf>/<set>/<mode>/`
   for all extant result JSONs.
2. For each, fabricate a record with `state="succeeded"`,
   `gcs_result_path` + parsed `result`, and `fired_at` ≈
   `result_seen_at` ≈ GCS object mtime. `job_ref` = `null` (we lost
   the backend handle).
3. Idempotent: skip records that already exist.

Run once per training run we care about to populate the relation
without re-firing.

## Read paths

### Dashboard run-detail page

A new `EvalsPanel` component below `WallclockPlot` on the run-detail
page:

```
┌─ Evals ─────────────────────────────────────────────────────────────┐
│  step     val_200          train_200         val_200-free   …       │
│           (maskgit/MG)     (maskgit/MG)      (free/AR)              │
│  ──────  ──────────────   ──────────────   ──────────────           │
│  5000    ⏳ running (8m)   ✅ 0.84%          —                       │
│  10000   ✅ 0.62%           ✅ 0.71%          —                       │
│  15000   ⏳ running (3m)   ⏳ pending         —                       │
│  9969    ✅ 0.65% (a2a)    ✅ 0.69%          —                       │
└──────────────────────────────────────────────────────────────────────┘
```

- Rows = step (sorted desc, latest first).
- Columns = `(set, mode)` tuple. Show whichever combos have any
  records; don't pre-allocate columns for unfired combos.
- Cell content = state badge + result number (when succeeded).
- Click a cell → opens the result JSON link / Modal app dashboard /
  iris log tail.

### Existing `eval.json`

Stays as-is (the per-step `(NMAE, NEMD)` series for the line plot
overlay on `WallclockPlot`). It's now *derived* — `tomat evals sync`
builds it from the records' `result` payloads. No backwards-compat
break.

## UI placement

- Run-detail page: `EvalsPanel` between `WallclockPlot` and
  `RecentEvents` (so the chronological context flows: plot → eval matrix
  → events list).
- Run card on `/runs`: a one-line `evals: 6/8 succeeded, 2 running`
  summary in the card header, only when the run has fired evals.

## Open questions

- **Per-task records**: should we keep one record per task index, or
  collapse to one record per (step, set, mode) with a `tasks: [{idx,
  state, …}]` substructure? Current proposal: one per task. Pro: less
  state mutation. Con: 4× records when `--num-tasks=4`. Resolution:
  keep flat, group at display time.
- **Hooking `--backend modal` properly**: the Modal call_id is only
  known after `spawn()` returns. The pending-record write must happen
  *after* spawn returns successfully, otherwise we have records with
  `job_ref=null`. (Currently OK — the existing code logs the call_ids
  right after spawn, so add the record-write there.)
- **Cross-run "find all evals fired against ckpt X"**: requires
  either D1 or a global secondary index. Out of scope for v1; can be
  built on top of the per-run sidecars later.

## Phases

1. **Phase A**: Define the record schema + R2 layout. Add CFW endpoints
   (read-side only — list + get). Migrate existing `eval.json` to be
   built from records via the backfill path. Run backfill on
   `train-mg-modal-h200x8-tz-v4`, `train-mg-tz-11`, `cont33k`,
   `train-mg-3-cos-emd`, `train-mg-4-cos-ce` to validate.
2. **Phase B**: `tomat evals fire` writes pending records before /
   alongside backend submission. Add the CFW write endpoint. Verify
   with the 4 in-flight Modal eval jobs from today.
3. **Phase C**: `tomat evals sync` polls backend state + populates the
   result-side fields. Extend the cron at
   `scripts/cron_iris_sync_modal.py` to invoke per active run.
4. **Phase D**: Dashboard `EvalsPanel`. Plug into `runs-snapshot.json`
   for hot-cache loading.

Phases A → D are independently shippable. A landed alone gives the
dashboard a useful matrix view of historical evals. D-without-state-sync
shows only `succeeded` records.

## Memories worth linking from any impl PRs

- [[feedback_pull_task_logs_on_first_cascade]] — "iris's 'sibling
  bounced' is the effect; the real failure is in some task's stderr"
  — argues for first-class job-state surfaces vs derived-by-name.
- [[iris-build-date-empty-bug]], [[marin-dev-wheel-rotation]] — the
  iris flakiness that motivates Modal as a backend; the record model
  is backend-agnostic.
- [[cross-region-eval-egress]] — when eval ckpt + eval region differ,
  `gcs_result_path` may live in the mirror bucket. Record schema must
  carry the bucket explicitly (don't assume default).
- Spec 42 — submission→terminal wallclock timeline. Same approach
  applied to training runs; this spec is its eval-child analog.
