#!/bin/bash
# Extend `train-mg-kl-bin5-fs-tpu` from step-100000 → step-110000 (+10k).
# No config changes — answers "does bin5 keep improving past 100k under
# its current recipe?". We're still well under 1 epoch of TS0 (`train-full-v3`)
# so the data isn't repeating yet.
#
# Mechanism: same TOMAT_RESULTS_LABEL → Levanter auto-discovers
# step-100000 ckpt and continues; wandb resumes the same run ID
# (`resume="allow"`). TL trajectory should pick up smoothly without the
# σ-shift spike the cont-s{3,10,20} runs showed (those changed σ).
#
# LR + warmup match bin5's *actual* trajectory (constant 3e-4 + warmup
# 0.05) — see `scripts/fires/bin5-fs-tpu.sh` for the wandb-vs-actual
# config discrepancy.
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
  --env-vars TOMAT_STEPS 110000 \
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
  --env-vars TOMAT_MG_MASK_PRIOR absorbing \
  --env-vars TOMAT_MG_LOSS_TYPE kl_gauss \
  --env-vars TOMAT_MG_KL_SIGMA 5 \
  --no-wait \
  --no-terminate-on-exit \
  --priority interactive \
  --job-name train-mg-kl-bin5-fs-tpu \
  -- python train_tomat_tpu.py
