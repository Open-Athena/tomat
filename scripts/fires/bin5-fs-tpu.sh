#!/bin/bash
# train-mg-kl-bin5-fs-tpu — recovered fire script (config from wandb after-
# the-fact: wandb run config for `optimizer.*`, `trainer.*`, `data.*` + run
# name knobs for the TOMAT_MG_* env vars that aren't logged to wandb yet
# (#269)). Captured so future cos-r / σ-ablation forks have an apples-to-
# apples baseline. Not fired here — bin5 already ran to step-100000 on
# 2026-06-11..-17 (us-east5-b, v6e-16).
#
# Inferred TOMAT_MG_* from the run name:
#   - `mg` → TOMAT_MG_MODE=1 (MaskGIT replacement for LM head)
#   - `kl-bin5` → TOMAT_MG_LOSS_TYPE=kl_gauss, TOMAT_MG_KL_SIGMA=5
#   - `fs` → from-scratch (no warm-start)
#   - r=1 / absorbing prior is the bin5 spec convention (see memory
#     `tz-runs-no-tf-no-gt` + `tomat-arch-key-facts`).
#
# Verified from wandb (post-hoc):
#   - 200M model (hidden=1024, n_layers=12, n_heads=16, max_seq_len=8192)
#   - LR 3e-4 cosine, warmup 0.1, BS 256 train, seed 42
#   - mp.compute_dtype=bfloat16, num_train_steps=100000
#   - data.cache_dir=gs://marin-us-east5/tomat/results/train-mg-kl-bin5-fs-tpu/cache
#   - data.shuffle window_blocks=310113 (epochwin over train-full-v3)
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null
cd marin

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

/Users/ryan/c/oa/marin/.venv/bin/iris --cluster=marin job run \
  --tpu v6e-16 \
  --zone us-east5-b \
  --enable-extra-resources \
  --cpu 32 --memory 64GB \
  --max-retries 20 \
  --env-vars WANDB_API_KEY "$WANDB_API_KEY" \
  --env-vars TOMAT_WANDB_ENTITY open-athena \
  --env-vars TOMAT_BUCKET gs://marin-us-east5/tomat \
  --env-vars TOMAT_LABEL train-full-v3 \
  --env-vars TOMAT_RESULTS_LABEL train-mg-kl-bin5-fs-tpu \
  --env-vars TOMAT_MODEL 200M \
  --env-vars TOMAT_STEPS 100000 \
  --env-vars TOMAT_BATCH_SIZE 256 \
  --env-vars TOMAT_SEED 42 \
  --env-vars TOMAT_LMQ_PATH gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz \
  --env-vars TOMAT_DENSITY_LOSS_TYPE emd \
  --env-vars TOMAT_DENSITY_ONLY_LOSS 1 \
  --env-vars TOMAT_LR 0.0003 \
  --env-vars TOMAT_LR_SCHEDULE cosine \
  --env-vars TOMAT_WARMUP 0.1 \
  --env-vars TOMAT_PROFILE 0 \
  --env-vars TOMAT_GRADIENT_CHECKPOINTING 1 \
  --env-vars TOMAT_SHUFFLE_WINDOW_BLOCKS 310113 \
  --env-vars TOMAT_SHARE_CACHE 1 \
  --env-vars TOMAT_VAL_SEQS 0 \
  --env-vars TOMAT_STEPS_PER_EVAL 0 \
  --env-vars TOMAT_MG_MODE 1 \
  --env-vars TOMAT_MG_MASK_PRIOR absorbing \
  --env-vars TOMAT_MG_LOSS_TYPE kl_gauss \
  --env-vars TOMAT_MG_KL_SIGMA 5 \
  --no-wait \
  --no-terminate-on-exit \
  --priority interactive \
  --job-name train-mg-kl-bin5-fs-tpu \
  -- python train_tomat_tpu.py
