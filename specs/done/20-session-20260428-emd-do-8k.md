# 2026-04-28 — EMD-density-only 8k run + eval-pipeline overhaul

Informal session notes. Not for the team meeting yet (next mtg ≈ 1 wk out).

## Headline

**8k-step 200M EMD-density-only run on v6e-16 with the full 77k-mat train data → val_200 mean NMAE 3.65% at step 7000** (step-7999 final eval pending). Yesterday's framing was 12.20%; the bulk of that improvement comes from fixing the eval-tiling bug.

## What we found / did

### Eval pipeline (the big one)

The mat-NMAE eval was systematically inflating numbers. Two issues:

1. **Tiling didn't cover the grid.** `tile_disjoint_offsets` (in `marin/eval_mat_nmae.py:245-253`) strode by P=14 from origin, dropping any final partial-patch slab on each axis. For a 96×96×180 grid: `(84/96)² × (168/180) ≈ 71.5%` of voxels evaluated; the rest silently dropped from both numerator and denominator. **Fixed** by adding boundary patches with per-patch local slices (full coverage, every voxel evaluated exactly once).
2. **Trailing partial batch dropped.** `n_full_batches = n_patches // eval_batch` truncated up to B-1 patches per mat at B=32. Replaced with pad-and-mask.

Side effect: same checkpoint, same 20 mats, same decoder gave **12.20% (old tiling) → 5.28% (new)** at step-3999 of the prior 4k run. The earlier curve we'd been quoting was an artifact of cherry-picking interior 71% of voxels, where boundary slabs (which the model predicts well, since they're often low-density) were excluded from both num + denom and inflated NMAE.

### Eval mat IDs pinned + train-mat eval

Snapshotted **200 val + 200 train** mat IDs to `gs://marin-eu-west4/tomat/eval/eval_mat_ids.json`. Eval reads via `TOMAT_EVAL_MAT_SET=val_200|train_200`. Watchdog fires both per ckpt, giving train-vs-val NMAE pair — same diagnostic role as train-vs-val loss.

Train-mat eval required uploading the train-split zarrs to GCS (only validation/ was up there). One-shot: ~390GB, ~25 min via `scripts/upload_zarrs_to_gcs.py --split train`. 1 mat failed (mp-error not yet identified) — 77,497/77,498 OK.

### λ knob deprecation

`TOMAT_DENSITY_L1_WEIGHT` was vestigial under `density_only=True` — it just multiplies the EMD term, equivalent to LR scaling, no semantic role. Now defaults to 1.0. Density-loss class is gated on `TOMAT_LMQ_PATH` presence. Also stripped dead CE compute (`log_softmax` + `take_along_axis`) when `density_only=True`. Unit tests still pass.

### Dataset recovery: 38k → 77k mats

`train-full-lmq-v2` had only **32 of 64 worker dirs in GCS** (~38k unique mats). Tokenize ran with `--n-workers 64` (workers 0-63 each had a disjoint slice), all 64 produced parquets in the Modal volume `tomat-rho-gga-train`, but only workers 0-31 were synced to GCS. Recovered the missing half via `scripts/sync_parquets_modal.py` (~13.4 GB sync, ~13 min). All 77,498 mats × 32 patches = 2.48M sequences now in GCS.

### Training run

Launched fresh **8000-step EMD-density-only on v6e-16** with full 77k-mat data. First attempt OOM'd at compile time (used 127GB / 31GB available per chip) with `gradient_checkpointing=False` — Be of envelope was off. Re-launched with `gradient_checkpointing=True` and it ran clean. ~4h45m wall clock, 2 preemptions auto-recovered. MFU 8.6% (vs 9.8% on v6e-8 — slight multi-host overhead).

Throughput: 502k tok/s vs 283k tok/s on v6e-8 = 1.78× from chip count. Lost ~1.5-2× from grad-ckpt going back on, but still net better.

### NMAE curve

| step | val_200 mean | train_200 mean | val median |
|---:|---:|---:|---:|
| 272 | 35.78% | 37.56% | 33.03% |
| 1000 | 60.25% | 60.96% | 59.33% |
| 2000 | 30.48% | 32.04% | 30.41% |
| 3000 | 13.13% | 13.92% | 11.91% |
| 4000 | 6.74% | 7.30% | 5.89% |
| 5000 | 6.47% | 7.25% | 5.52% |
| 6000 | 5.29% | 5.80% | 4.72% |
| 7000 | **3.65%** | **4.24%** | **3.11%** |

- **No overfitting**: val ≈ train throughout. Even slight under-fit (val sometimes < train) suggests room for more data/steps.
- **Step-1000 hump**: NMAE went UP from step 272 (35%) to step 1000 (60%) before recovering. Train loss was monotonically dropping (2.22 → 1.62) the whole time. Hypothesis: model went through a transition where it became confident on wrong modes (median decoder picks them, NMAE spikes), then corrected. Worth re-checking on future runs but didn't block progress.
- The eval/loss curve from wandb: 1.66 (step 1000) → 1.13 (step 7000) → **0.93 (step 7999)** — big drop in the last 1000 steps as cooldown kicks in. Step-7999 eval pending (mat-NMAE).
- ChargE3Net SOTA: 0.53%. We're at ~7× off (vs ~22× off where we were yesterday).

### Bugs/quirks found

- **Final-step OBO**: `num_train_steps=N` → steps run 0..N-1, so the final step is N-1 (3999, 7999). The watchdog's `target_step` only fires on `step % 1000 == 0`, missing the final ckpt. Need to add a "fire-on-final-step" trigger when training reaches `succeeded`. Levanter does preserve step-7999 ckpt — `gsutil ls` was truncating the listing earlier; recursive ls confirmed it's there.
- **Levanter ckpt retention**: the `keep: [{every: 1000}]` rule does work — step-1000 through step-7000 are all in GCS. The recursive listing also shows step-7849 (final temp pre-cooldown-end?) and step-7999.

## What's next (planned)

1. Recover step-7999 numbers (in flight).
2. Fix watchdog: fire eval on final step regardless of step%1000.
3. Run continuation from step-7999 with fresh LR schedule (initialize-from-weights, new warmup+cooldown). In parallel, fresh 16k-step run from scratch.
4. Eventually: HP sweeps + scaling-law characterization. First model the user has trained, so building intuition on what the curves look like is high-value.
