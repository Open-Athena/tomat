#!/bin/bash
# 3-way σ-ablation FROM SCRATCH (vs bin5-sigma-cont.sh which extended bin5).
# Each arm is a fresh 10k-step v6e-16 fire with the same bin5 recipe except
# `TOMAT_MG_KL_SIGMA` — directly comparable to bin5 + cos-r at 10k. Per
# memory `feedback_smoke_before_sweep`: each is its own smoke; if any arm
# looks interesting we extend it. Constant LR + warmup 0.05 matches
# bin5's actual LR trajectory (the wandb config said cosine, the trace
# is flat — see `scripts/fires/bin5-fs-tpu.sh`).
#
# Labels:
#   train-mg-kl-bin3-fs-tpu   (σ=3, sharper than bin5)
#   train-mg-kl-bin10-fs-tpu  (σ=10, wider than bin5)
#   train-mg-kl-bin20-fs-tpu  (σ=20, much wider)
#
# Together with `train-mg-cos-r-fs-tpu` (already running, cos prior at σ=5)
# and an eventual `bin5`-at-10k baseline (subset of the existing run) these
# give a 4-way σ at 10k steps + a prior-schedule comparison.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null
cd marin

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

for SIGMA in 3 10 20; do
  LABEL="train-mg-kl-bin${SIGMA}-fs-tpu"
  echo "[fire] σ=${SIGMA} → ${LABEL}"
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
    --env-vars TOMAT_RESULTS_LABEL "${LABEL}" \
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
    --env-vars TOMAT_MG_MASK_PRIOR absorbing \
    --env-vars TOMAT_MG_LOSS_TYPE kl_gauss \
    --env-vars TOMAT_MG_KL_SIGMA "${SIGMA}" \
    --no-wait \
    --no-terminate-on-exit \
    --priority interactive \
    --job-name "${LABEL}" \
    -- python train_tomat_tpu.py
done

echo
echo "[fired 3 σ-fs jobs]"
echo "Watch: ./tomat iris ls | grep train-mg-kl-bin"
echo "Dashboard: https://tomat.oa.dev/#/runs?f=train-mg-kl-bin"
