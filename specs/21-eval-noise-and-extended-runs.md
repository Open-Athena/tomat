# 2026-04-29 — Eval-noise calibration + extended runs

Living doc, updated as numbers land. Goal: figure out how seriously to take
the step-7000 → step-7999 NMAE regression in the 8k EMD-density-only run,
and set up the next phase of training with proper instrumentation.

## TL;DR

Final 8k headline (val_200 / train_200, mat-NMAE %, n=200 mats each):

| step | val mean | val median | val p99 | train mean | train median | train p99 | eval/loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 272  | 35.78 | 33.03 | 122.35 | 37.56 | 33.36 | 121.90 | — |
| 1000 | 60.25 | 59.33 | 156.11 | 60.96 | 60.46 | 145.58 | 1.66 |
| 2000 | 30.48 | 30.41 |  71.94 | 32.04 | 31.11 |  87.22 | — |
| 3000 | 13.13 | 11.91 |  46.51 | 13.92 | 12.70 |  52.57 | — |
| 4000 |  6.74 |  5.89 |  33.67 |  7.30 |  6.01 |  32.62 | — |
| 5000 |  6.47 |  5.52 |  31.74 |  7.25 |  5.71 |  29.10 | — |
| 6000 |  5.29 |  4.72 |  26.00 |  5.80 |  4.87 |  29.48 | — |
| 7000 |  3.65 |  3.11 |  15.02 |  4.24 |  3.25 |  33.13 | 1.13 |
| **7999** | **4.04** | **3.91** | **12.94** | **4.56** | **4.13** | **12.73** | **0.93** |

Both train and val mean/median NMAE *worsen* in the cooldown phase (step
7000 → 7999) while EMD eval/loss keeps improving (1.13 → 0.93). Direction
matches across train + val — not overfit. Hypothesis: loss-vs-decoder
mismatch sharpening — EMD pulls the predicted distribution closer in W₁
without pulling the median (the L₁-optimal point estimator we read out)
closer to the true voxel.

Ref: ChargE3Net SOTA 0.53%, electrAI 1.02%, codec floor 0.18% (P=14, V=16k).

## In-flight experiments

### (1) Disjoint val_400b + train_400b sample
- Sampler: `scripts/sample_disjoint_eval_mats.py` (reads task_id from every
  worker-XX/shard-XX.parquet via thread-pool, dedupes mat IDs, excludes
  existing val_200/train_200, samples 400 with seed=42).
- Subtle bug found and fixed mid-session: shards within a worker contain
  DISJOINT mats (worker-00/shard-00000 ∩ shard-00001 = ∅), so reading just
  shard-00000 covers only ~1/5 of the universe. Now reads all shards in
  parallel.
- Adds `val_400b` + `train_400b` keys to `data/eval_mat_ids.json`. Note
  suffix `b` — keeps old `val_200`/`train_200` available indefinitely.
- After sampling: upload `data/eval_mat_ids.json` to
  `gs://marin-eu-west4/tomat/eval/eval_mat_ids.json` for iris workers.
- Status: **WIP** (sampler running).

### (2) Re-eval all 8k ckpts on val_400b + train_400b
- 9 ckpts × 2 mat-sets = 18 jobs; each ~10 min on v6e-8.
- Launch: `tmp/fire-disjoint-evals.sh` — fires 18 jobs, pre-populates
  watchdog FIRED files, starts harvest watchdogs (which auto-log NMAE to
  the existing wandb run via `eval/mat_nmae/{val,train}_400b/{mean,...}`).
- Disjoint sample → independent confirmation of the curve. If shape matches
  val_200 / train_200, NMAE wiggles are real signal; if it smooths out,
  n=200 was undersampled.
- **Step-272 unrecoverable**: Levanter's `keep: [{every: 1000}]` retention
  deleted it long ago. val_200/train_200 step-272 was eval'd while it still
  existed. Disjoint set will have 8 points (steps 1000…7999), not 9.
- Status: **firing** (val_400b + train_400b watchdogs harvesting).

### (3) Continuation from step-7999 on v6e-16
- Warm-start from step-7999 ckpt: load **model weights only** from ckpt;
  let optimizer state init fresh (so its internal step counter == 0 →
  LR schedule starts at warmup, not parked in source ckpt's cooldown
  tail at LR=0).
- Same recipe as 8k run otherwise: 200M Qwen3, BS=128, EMD-density-only,
  full 77k-mat data, gradient_checkpointing=True. New env-var
  `TOMAT_INIT_FROM_CHECKPOINT` plumbed through `train_tomat_tpu.py`.
- `TOMAT_PROFILE=1` (default) → JAX profiler trace at steps 20-45, saved
  as wandb artifact `jax-profile-step-20-45`.
- Launch: `tmp/launch-train-cont-from-7999-tpu16.sh`.
- **First attempt killed** — naive
  `TrainLmConfig.initialize_from_checkpoint_path` loads the FULL
  TrainerState (incl. optimizer state + step=7999), and the LR schedule
  reads from optimizer's internal step → LR=0 from step 1, no weight
  movement. NMAEs identical to source 7999. Diagnosed via wandb (run
  `train-cont-from-7999-…` shows `learning_rate=0` from step 1).
- **Fix**: monkey-patch `levanter.main.train_lm.load_checkpoint` in
  `train_tomat_tpu.py` to do `load_checkpoint(state.model, ckpt,
  subpath="model")` and stitch back, so optimizer state stays fresh.
  Same pattern as `eval_mat_nmae.py` and `train_dpo.py`.
- Status: **pending re-launch** with fixed warm-start.

### (4) 16k from-scratch on v6e-16
- Same recipe as the 8k run, doubled. Settles whether the NMAE curve
  saturates (→ underfit confirmation) or oscillates around ~4% (→ loss-vs-
  decoder mismatch confirmation).
- Launch: `tmp/launch-train-16k-emd-do-tpu16.sh`.
- Status: **pending** v6e-16 capacity.

## Wandb instrumentation
- Mat-NMAE points logged to the same wandb run via
  `scripts/log_nmae_to_wandb.py`, using a custom `eval/mat_nmae/step` axis
  so retroactive (post-finish) writes don't trip wandb's monotonic-step
  guard. Watchdog harvest path now calls this script automatically.
- Throughput / MFU already in wandb (`throughput/mfu`, `tokens_per_second`,
  `duration`, `loading_time`, `hook_time`, plus `p10/p50/p90_mfu`). 8k run
  shows steady **~8.7% MFU**, ~502k tok/s, NOT data-bound (loading_time =
  0.5 ms), NOT hook-bound (hook_time = 8 ms). Pure compute bottleneck.

## MFU debug
- Profiler wired in `train_tomat_tpu.py`. Knobs:
  - `TOMAT_PROFILE` (default `1`) — gate.
  - `TOMAT_PROFILE_START` (default `20`) — first traced step.
  - `TOMAT_PROFILE_NUM_STEPS` (default `25`) — trace window.
- Trace lands as wandb artifact (no extra plumbing).
- Likely root causes ranked: small per-chip BS (128/16=8 ex/chip @ 8k tok),
  gradient checkpointing on (~30% tax, needed to fit), multi-host comm
  overhead (v6e-16 = 4 hosts), wide LM head (16k codebook).
- Cheap experiments: BS={64,128,256} × {grad_ckpt on/off}, 200 steps each;
  v6e={8,16,32} at fixed per-chip BS.

## Wheel-rotation workaround (unblocked all of the above)
- Daily marin-* dev wheels rotate; today's marin-iris dev20260429 declares
  a runtime dep on `marin-finelog` but no `marin-finelog-latest` GH release
  has been published (404), and yesterday's wheels are gone.
- Pinned `[tool.uv.sources]` in `marin/pyproject.toml` to git+SHA
  `117de4f18f1ca2b180f93a218d6eb92c517ed88c` (parent of the
  `[iris] Lift log store and log server into new lib/finelog package` PR).
- Tradeoff: workers build marin-* from source on first launch (~1 min,
  uv-cached after).
- Revert when upstream's daily release pipeline is healthy again.

## Open questions
- Is the step-7000 → step-7999 NMAE regression real signal or n=200 noise?
  Will be answered by (2).
- Can we get to ChargE3Net's 0.53% with this approach, or is there a hard
  ceiling from the EMD-loss-vs-median-decoder gap? Will be informed by
  (4)'s saturation behavior.
- What's our actual MFU ceiling on v6e-16 for this model size? Will be
  informed by profiler traces from (3) + (4).

## Decisions log
- Use disjoint-sample (val_400b/train_400b) rather than expanding val_200
  to val_400 — preserves apples-to-apples comparability of the existing
  curve.
- Default `TOMAT_PROFILE=1` (opt-out, not opt-in) — cost is ~50s of slowed
  compute on a multi-hour run; benefit is having a trace whenever you wish
  you had one.
- Use levanter's built-in profiler (`ProfilerConfig`) — auto-uploads as a
  wandb artifact, no GCS plumbing needed.

## Compute usage (MSRP-equivalent)

Track chip-hours × public TPU rate per run. Useful for sense-of-scale and
sharing experiment scope with the team. v6e public preemptible: ~$1.08-1.19
/chip-hour (use $1.15 midpoint); on-demand: $2.70-2.97. Both are list, not
necessarily what gets billed.

| run | TPU | wall-clock | chip-hours | preempt $ | on-demand $ |
|---|---|---:|---:|---:|---:|
| `train-full-lmq-v2-200M-bs128-emd-do-8k-tpu16` (orig 8k) | v6e-16 | ~7 hr | 112 | ~$130 | ~$330 |
| `train-full-lmq-v2-200M-bs128-emd-do-16k-tpu16` (16k from-scratch) | v6e-16 | ~13 hr (est) | ~210 | ~$240 | ~$620 |
| `train-cont-from-7999-…-8k-tpu16` (broken cont, killed) | v6e-16 | 1.6 hr (killed) | 26 | ~$30 | ~$77 |
| `train-cont-from-7999-…-8k-tpu16-v2` (re-launch w/ fix) | v6e-16 | ~7 hr (est) | 112 | ~$130 | ~$330 |
| **`train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32`** (in flight) | v6e-32 | ~7 hr (est) | 224 | **~$260** | ~$665 |
| evals (~18 jobs × ~10 min on v6e-8) | v6e-8 | ~3 hr total | 24 | ~$30 | ~$72 |

**Running total** for spec-21 work (incl. the just-launched v6e-32 1B):
~$820 preempt-MSRP, ~$2.1k on-demand-MSRP.
