# Marin cluster incidents

Running log of crashes, hangs, and counter-bugs observed in tomat work on
the Marin iris/TPU cluster. Intended for periodic batch-surfacing to the
Marin team — keep entries terse but concrete enough that whoever reads
this can repro or correlate without further context.

Format per entry:
- **Date**: first observed.
- **Cluster**: TPU slice + zone where reproducible.
- **Symptom**: what visibly broke (crash, hang, counter wrong, etc.).
- **Signature**: stderr / log snippet / iris-summary state — short.
- **Repro**: minimum config / command if known.
- **Workaround**: what we did. If "kill + retry on a different slice"
  is the only thing, say that.
- **Status**: `open` / `resolved` / `WAI`.

Cross-region GCS transfer budget (`MARIN_MIRROR_BUDGET_GB=10` default
crashing on the first ckpt save) is **NOT** tracked here — that's
working-as-intended once you write back to a region-local bucket.

---

### 2026-05-09 v6e-32 1B post-JAX-init silent stall — `open`

- **Cluster**: v6e-32, eu-west4-b (`marin-tpu-v6e-preemptible-32-…`).
- **Symptom**: From-scratch 1B training run reaches end of JAX
  coordination init, prints process-rank lines, then **stops emitting any
  log output for >1 h**. No ckpts written. No SIGTERM. iris reports the
  parent job as `running` but no train-step metrics ever flush.
- **Signature**: last stderr line is from `levanter.tracker.wandb`
  ("Synced wandb run information from process 0: …"); silence after.
- **Repro**: `tomat train -T v6e-32 -m 1B -s 26000 -b 256 …
  train-full-v3-1B-bs256-emd-do-26k-tpu32-shuf1k`. Observed twice
  (BS=256 and BS=512 attempts) before we gave up.
- **Workaround**: switched to v5p (separate run on v5p-16 worked once
  cross-region issue was resolved). 1B on v6e-32 deferred.
- **Status**: `open` — no clear repro instructions beyond "use a 1B
  workload from scratch on v6e-32"; would benefit from Marin-side
  instrumentation on the worker pool.

### 2026-05-09 v6e-32 zephyr cache-build worker crashloop — `open`

- **Cluster**: v6e-32, eu-west4-b.
- **Symptom**: from-scratch tomat runs (with per-run cache_dir) crashloop
  in the Zephyr cache-build worker pool — workers get preempted during
  cache-build, restart, never converge. Even with `--share-cache` (one
  cache_dir per data label), the cache-rebuild path is hit if the cache
  isn't fully populated.
- **Repro**: any new tomat training run on v6e-32 with a not-yet-built
  levanter cache. Observed across model sizes (200M, 1B).
- **Workaround**: seed the cache by running a tiny prep step on
  preemption-resistant hardware first, then re-fire on v6e-32.
- **Status**: `open` — local draft at
  `gh/drafts/cache-build-brittleness/DESCRIPTION.md` (not yet filed
  upstream). Proposes a per-worker retry budget so a single preempt
  doesn't reset the whole cache-build cohort.

### 2026-05-10 v4 chip-config errors (secondhand) — `open`

- **Cluster**: v4 (zone unclear).
- **Symptom**: chip-config errors on job launch.
- **Source**: Tim mentioned this in Slack; we have not directly
  reproduced it on tomat workloads (we've been avoiding v4 since).
- **Status**: `open` — secondhand; needs first-party repro before
  surfacing.

### 2026-05-11 iris preempt counter inflated by post-crash auto-relaunches — `open`

- **Cluster**: v5p-16, us-central1-a.
- **Symptom**: `iris job summary` reports `preemptions=1002` for a job
  that almost certainly experienced ≪10 actual preempt events. The
  counter looks like it's incrementing once per worker-pod replacement,
  including replacements that happen because the *trainer itself*
  crashed (e.g. due to a rigging budget exception at step 1000), not
  because GCP preempted the slice.
- **Signature**:
  ```
  Job: /ryan/train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k
  State: failed  exit=0  failures=1  preemptions=1002
  Tasks: 0/2 completed  failed=1  worker_failed=1
  ```
  where the actual `failures=1` field correctly counts 1 trainer crash,
  but `preemptions` ballooned.
- **Repro**: hit a fast-loop trainer-side crash (rigging exception, OOM,
  etc.) on a preemptible TPU slice — every retry-and-instantly-crash
  cycle adds to `preemptions`.
- **Workaround**: trust `failures` over `preemptions` when diagnosing.
  We surface `cluster/preempts_total` in wandb via `tomat preempts watch`
  and have to mentally subtract crash-relaunch noise.
- **Status**: `open` — iris-side counter-semantics bug. Worth a one-line
  patch in iris to distinguish "GCP preempted us" from "trainer
  exit-non-zero, we restarted".

### 2026-05-11 v6e-16-preemptible 48 h stall mid-training — `open`

- **Cluster**: v6e-16, us-east1-d (`marin-tpu-v6e-preemptible-16-us-east1-d-*`).
- **Symptom**: cont7k-ext training reached step 18000 then **flatlined for
  ~48 h** (2026-05-09 12:00 → 2026-05-11 10:00 UTC) before any further
  step metrics flushed. iris reported the job alternately as `running`
  and `pending` during this window with coscheduled-sibling waits. No
  manual intervention — recovered on its own.
- **Signature**: `tomat runs wallclock-plot` shows a flat step curve over
  a 2-day window with no trainer_started / sigterm events landing in
  wandb during the gap. iris-side: `preemptions` counter advanced (was 0,
  ended ~15 by recovery), suggesting many preempt-restart cycles in a
  row where the trainer never reached the first metric flush.
- **Repro**: long-running preemptible v6e-16 job during a period of
  apparent high tenant pressure in us-east1-d.
- **Workaround**: none — let it recover. Heavy use of
  `--share-cache` reduces re-init time but didn't prevent the stall.
- **Status**: `open` — possibly tenant-pressure-related rather than a
  Marin bug per se, but the iris-side preempt counter not advancing in a
  way that tracks wallclock pain is observability-painful.

---

## Backlog / vague

- **v5p JAX `CoordinationService/RegisterTask` deadline-exceeded** —
  observed on v5p-16 and v5p-32 in an earlier session; smoke runs on
  v5p-16 in this session worked fine, so likely resolved or
  zone-transient. Re-open if seen again.
