#!/bin/bash
# 3-way σ-ablation continuation from bin5@step-100000.
# Fires train-mg-kl-bin5-cont-s{3,10,20} on v6e-16, each a 5k-step
# continuation with a different TOMAT_MG_KL_SIGMA (bin5 trained σ=5).
#
# Mechanism: Levanter auto-discovers the latest step-* ckpt under
# `<TOMAT_BUCKET>/results/<TOMAT_RESULTS_LABEL>/`, so we mirror bin5's
# step-100000 ckpt into each new label's results dir up-front (cheap
# server-side gsutil cp within us-east5), then fire 3 jobs against the
# new labels. Levanter resumes from step-100000 with the new wandb run
# name + new σ in the loss; TL discontinuity is expected (σ change rescales
# the KL target).
#
# Per memory `lr-schedule-default-constant-for-extensions`: continuations
# stay on constant LR. bin5 trained on cosine; switching to constant for
# the continuation avoids a cosine-decay TL spike at the resume boundary
# (Levanter would otherwise re-stretch the cosine schedule over the new
# num_train_steps and effectively warm-down the LR right after resume).
#
# Per `feedback_smoke_before_sweep` — this is 3 parallel fires of an
# untested recipe (σ != 5 mid-training). Each fire is a smoke; we'll
# extend the winners. Total = 15k step-equivalents × 3 = ~45k v6e-16
# step-equivalents (~3-5h wall-clock each).
set -euo pipefail

SRC_BUCKET="gs://marin-us-east5/tomat"
SRC_LABEL="train-mg-kl-bin5-fs-tpu"
# Levanter writes ckpts at <bucket>/results/<label>/checkpoints/<label>/step-N
# (nested-label subdir is how Levanter scopes the run_id under save_dir).
SRC_CKPT_DIR="${SRC_BUCKET}/results/${SRC_LABEL}/checkpoints/${SRC_LABEL}"
SRC_STEP="step-100000"
RESUME_STEPS=105000  # bin5@100k + 5k continuation

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

for SIGMA in 3 10 20; do
  DST_LABEL="train-mg-kl-bin5-cont-s${SIGMA}"
  DST_CKPT_DIR="${SRC_BUCKET}/results/${DST_LABEL}/checkpoints/${DST_LABEL}"
  echo "[spawn-sigma-cont] σ=${SIGMA} → ${DST_LABEL}"

  # 1. Mirror bin5's step-100000 into the new label's nested ckpt dir
  # (server-side, same bucket → fast). Skip if already mirrored.
  # NOTE: use gcsfs not `gsutil -m cp -r` — gsutil deadlocked for 50+ min
  # on bin5's ocdbt `d/` subdir (~50 hash-named tree files) producing zero
  # output. gcsfs walks `fs.find()` + per-file `fs.cp()` and finishes in
  # under 5 s for the same set.
  if ! gsutil -q stat "${DST_CKPT_DIR}/${SRC_STEP}/metadata.json" >/dev/null 2>&1; then
    echo "  [mirror] ${SRC_CKPT_DIR}/${SRC_STEP}/ → ${DST_CKPT_DIR}/${SRC_STEP}/"
    SRC_NO_PROTO="${SRC_CKPT_DIR#gs://}/${SRC_STEP}" \
    DST_NO_PROTO="${DST_CKPT_DIR#gs://}/${SRC_STEP}" \
    python3 -c "
import fsspec, os
fs = fsspec.filesystem('gcs')
src, dst = os.environ['SRC_NO_PROTO'], os.environ['DST_NO_PROTO']
files = fs.find(src)
for f in files:
    fs.cp(f, f'{dst}/{f[len(src)+1:]}')
print(f'  mirrored {len(files)} files')
"
  else
    echo "  [mirror] already present, skipping"
  fi

  # 2. Fire the continuation job.
  cd marin
  /Users/ryan/c/oa/marin/.venv/bin/iris --cluster=marin job run \
    --tpu v6e-16 \
    --zone us-east5-b \
    --enable-extra-resources \
    --cpu 32 --memory 64GB \
    --max-retries 20 \
    --env-vars WANDB_API_KEY "$WANDB_API_KEY" \
    --env-vars TOMAT_WANDB_ENTITY open-athena \
    --env-vars TOMAT_BUCKET "${SRC_BUCKET}" \
    --env-vars TOMAT_LABEL train-full-v3 \
    --env-vars TOMAT_RESULTS_LABEL "${DST_LABEL}" \
    --env-vars TOMAT_MODEL 200M \
    --env-vars TOMAT_STEPS "${RESUME_STEPS}" \
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
    --job-name "${DST_LABEL}" \
    -- python train_tomat_tpu.py
  cd ..
done

echo
echo "[fired 3 σ-cont jobs]"
echo "Watch: ./tomat iris ls | grep train-mg-kl-bin5-cont-s"
echo "Dashboard: https://tomat.oa.dev/#/runs?f=train-mg-kl-bin5-cont"
