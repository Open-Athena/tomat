# Ablation runbook: P=15, balls, M=256

Status: **ready to execute**. Written 2026-04-23 after 208M/1B training
launches.

Prepared during Ryan's nap — `specs/10-ball-patches.md` has the design;
this is the list of specific tokenize + train commands to kick off
when ready.

## Goals

Produce matched-compute ablation data vs the baseline P=14/M=32 cube
tokenization to answer:

1. **Bigger P helps?** (P=15 / M=256 vs P=14 / M=256)
2. **Balls vs cubes at matched voxel count?** (ball r²≤75 / M=256 vs
   cube P=14 / M=256)
3. **More M (8×) helps?** (P=14 / M=256 vs the existing P=14 / M=32
   baseline)

Voxel counts:
- cube P=14 → 2,744 voxels/patch, ~5,700 non-pad tokens
- cube P=15 → 3,375 voxels, ~6,970 tokens (fits 8k)
- ball r²≤75 → 2,777 voxels, ~5,780 tokens (fits 8k; validated 2026-04-23)
- ball r²≤86 → 3,407 voxels, ~7,040 tokens (fits 8k)

Coverage math (77k train-full, 135 G voxel corpus, C ≈ M·V_patch·N / ΣV):
- M=32 P=14 → C ≈ 0.05 (5% avg voxel multiplicity) — current baseline
- M=256 P=14 → C ≈ 0.40 (8× current) — proposed bump
- M=256 P=15 → C ≈ 0.50
- M=512 P=14 → C ≈ 0.81 (next horizon)

User goal was C ≈ 2–3; M=256 gets us to the middle of the ramp.

## Tokenize commands

All use the existing `tokenize_patches_modal.py::parallel` entry point
(not yet migrated to tokenize-to-GCS per spec 12 — do that migration
separately). Output goes to `/vol/tokenized/<label>/worker-NN/` on
`tomat-rho-gga-train` volume.

**Baseline M=256 cube (P=14):**

```bash
cd /Users/ryan/c/oa/tomat
TOMAT_VOLUME=tomat-rho-gga-train modal run scripts/tokenize_patches_modal.py::parallel \
  --label train-full-m256 \
  --split train \
  --patches-per-material 256 \
  --patch-size 14 \
  --n-workers 64 \
  --seed 42 \
  --pad-to 8192
```

Expected output: ~77k mats × 256 = 19.8M rows. ~8× data vs current
train-full (2.48M rows). Throughput est.: 8× wall-clock of current
train-full tokenize → ~3–4 hours at 64 workers. Consider bumping to 128
workers if quota allows.

**P=15/M=256 cube:**

```bash
TOMAT_VOLUME=tomat-rho-gga-train modal run scripts/tokenize_patches_modal.py::parallel \
  --label train-full-P15-m256 \
  --split train \
  --patches-per-material 256 \
  --patch-size 15 \
  --n-workers 64 \
  --seed 42 \
  --pad-to 8192
```

**Ball r²≤75/M=256 (matches P=14 voxel count):**

```bash
TOMAT_VOLUME=tomat-rho-gga-train modal run scripts/tokenize_patches_modal.py::parallel \
  --label train-full-ball75-m256 \
  --split train \
  --patches-per-material 256 \
  --shape ball \
  --r2-max 75 \
  --n-workers 64 \
  --seed 42 \
  --pad-to 8192
```

**(Optional) Ball r²≤86/M=256 (matches P=15 voxel count):**

```bash
TOMAT_VOLUME=tomat-rho-gga-train modal run scripts/tokenize_patches_modal.py::parallel \
  --label train-full-ball86-m256 \
  --split train \
  --patches-per-material 256 \
  --shape ball \
  --r2-max 86 \
  --n-workers 64 \
  --seed 42 \
  --pad-to 8192
```

After each tokenize finishes, sync to GCS:

```bash
scripts/sync_parquets_to_gcs.py \
  -v tomat-rho-gga-train \
  -l <label>
```

(Or skip the sync once spec 12's GCS-direct pipeline lands.)

## Cost estimate

Per variant, ~77k mats × 256 patches × 8k tokens:
- Tokenize CPU: ~3–4 hrs × 64 workers × 1 core × $0.04 = ~$8
- Modal volume I/O: negligible
- GCS write/read within region: negligible
- Total: ~$8–10 per variant × 3–4 variants = **$30–40 total**

## Training-run commands

Once a label is tokenized + GCS-synced, train the 208M Qwen3 against
it. Use the env-var knobs from `marin/train_tomat_tpu.py`:

```bash
cd /Users/ryan/c/oa/tomat/marin
uv run iris --cluster=marin job run \
  --tpu v6e-8 --enable-extra-resources --cpu 32 --memory 64GB \
  --env-vars WANDB_API_KEY "$WANDB_API_KEY" \
  --env-vars TOMAT_LABEL train-full-m256 \
  --env-vars TOMAT_MODEL 200M \
  --env-vars TOMAT_BATCH_SIZE 128 \
  --env-vars TOMAT_STEPS 8000 \
  --env-vars TOMAT_VAL_SEQS 256 \
  --env-vars TOMAT_LR_SCHEDULE constant \
  --env-vars TOMAT_WARMUP 0.02 \
  --env-vars TOMAT_COOLDOWN 0.1 \
  --env-vars TOMAT_RESULTS_LABEL train-full-m256-200M-bs128-val \
  --no-wait --no-terminate-on-exit \
  --job-name train-full-m256-200M \
  -- python train_tomat_tpu.py
```

Note: **constant LR schedule with WSD-style cooldown** (warmup=0.02,
cooldown=0.1) is the default for ablations going forward — makes
resume-and-extend painless (no cosine-decay LR restart bumps). Per
conversation with Ryan, 2026-04-23.

Matched-compute ablation: train each variant for **same token budget**
(not same step count, since different P / M changes tokens per step).
Current 208M on train-full hit 2.10 B tokens = 2000 steps × bs=128 ×
8k. For matched-compute ablations, target 4 B tokens (4000 steps) so
we have 2× Chinchilla for each variant.

## Matched-compute table

Given 4 B total training tokens per variant, at bs=128 × 8k = ~1M tokens/step:

| label                     | shape   | P  | r²   | M   | steps | tokens |
|---------------------------|---------|----|------|-----|-------|--------|
| train-full                | cube    | 14 | —    | 32  | 2000  | 2.1 B (baseline, done) |
| train-full-m256           | cube    | 14 | —    | 256 | 4000  | 4.2 B |
| train-full-P15-m256       | cube    | 15 | —    | 256 | 4000  | 4.2 B |
| train-full-ball75-m256    | ball    | —  | 75   | 256 | 4000  | 4.2 B |
| train-full-ball86-m256    | ball    | —  | 86   | 256 | 4000  | 4.2 B |

Compute per variant: 208M × 4.2 B × 6 + attention ≈ **10 EF** (2× the
current 208M run's 5.2 EF). Across 4 variants: ~40 EF. Prior cumulative
project spend: ~16 EF. So this doubles-plus our total compute, but
gives us 4 clean ablation data points.

## Reporting

Each completed variant produces:
- W&B run with `train/loss`, `eval/tomat/loss`, `eval/tomat/bpb` at
  step boundaries.
- Headline delta vs baseline: `(final_loss, final_bpb, tokens_trained)`.
- Once spec 11 lands: per-material NMAE on the held-out test split.

Compile into a single results table in `docs/ablations.md` after all
variants finish.
