#!/bin/bash
# train-mg-cos-r-fs-tpu — 10k from-scratch MaskGIT with the canonical
# cosine mask schedule (vs bin5's absorbing r=1). Same model + loss as bin5;
# only the mask prior changes. Smoke-then-extend pattern: 10k first to gate
# whether to continue to 100k.
#
# Key delta vs `tmp/fire-bin5-fs-tpu.sh`:
#   TOMAT_MG_MASK_PRIOR absorbing  →  cosine        (the headline change)
#   TOMAT_STEPS         100000     →  10000         (smoke length)
#   TOMAT_LR_SCHEDULE   cosine     →  constant      (per memory
#                                  `lr-schedule-default-constant-for-
#                                  extensions` — keeps extensions clean,
#                                  avoids LR-spike on resume past 10k)
#   TOMAT_WARMUP        0.1        →  0.05          (cheaper on a 10k smoke;
#                                  500 step warmup, matches LF-ablation arms)
#
# Everything else mirrors bin5 to keep apples-to-apples on the schedule
# comparison: 200M / BS=256 / σ=5 / KL-Gauss / TS0 / epochwin shuffle.
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
  --env-vars TOMAT_RESULTS_LABEL train-mg-cos-r-fs-tpu \
  --env-vars TOMAT_MODEL 200M \
  --env-vars TOMAT_STEPS 10000 \
  --env-vars TOMAT_BATCH_SIZE 256 \
  --env-vars TOMAT_SEED 42 \
  --env-vars TOMAT_LMQ_PATH gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz \
  --env-vars TOMAT_DENSITY_LOSS_TYPE emd \
  --env-vars TOMAT_DENSITY_ONLY_LOSS 1 \
  --env-vars TOMAT_LR 0.0003 \
  --env-vars TOMAT_LR_SCHEDULE constant \
  --env-vars TOMAT_WARMUP 0.05 \
  --env-vars TOMAT_PROFILE 0 \
  --env-vars TOMAT_GRADIENT_CHECKPOINTING 1 \
  --env-vars TOMAT_SHUFFLE_WINDOW_BLOCKS 310113 \
  --env-vars TOMAT_SHARE_CACHE 1 \
  --env-vars TOMAT_VAL_SEQS 0 \
  --env-vars TOMAT_STEPS_PER_EVAL 0 \
  --env-vars TOMAT_MG_MODE 1 \
  --env-vars TOMAT_MG_MASK_PRIOR cosine \
  --env-vars TOMAT_MG_LOSS_TYPE kl_gauss \
  --env-vars TOMAT_MG_KL_SIGMA 5 \
  --no-wait \
  --no-terminate-on-exit \
  --priority interactive \
  --job-name train-mg-cos-r-fs-tpu \
  -- python train_tomat_tpu.py
