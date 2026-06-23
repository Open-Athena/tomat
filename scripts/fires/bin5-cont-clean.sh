#!/bin/bash
# bin5 SANITY-CHECK resume: train-mg-kl-bin5-cont-clean from bin5@step-100000.
#
# Tests H1 (data mix change as cause of bin5-extend-10k's TL spike at the
# resume boundary). Matches `bin5-extend-10k.sh` exactly except:
#   - TOMAT_LABEL: train-full-v3,…shard1,…shard2,…shard3   (TS0123 union)
#     bin5-extend used TS0 single-shard (`train-full-v3`).
#   - TOMAT_WARMUP=0 (explicit no-warmup intent; behaviorally a no-op since
#     Levanter's lr_scheduler is indexed by global step — step-100000 is
#     past the 5250-step warmup window either way).
#   - TOMAT_STEPS=102000 (just 2k steps post-resume — enough to see whether
#     TL settles cleanly or spikes. ~2 hours on v6e-16.)
#   - TOMAT_RESULTS_LABEL=train-mg-kl-bin5-cont-clean (new label, mirror
#     bin5@step-100000 → new ckpt dir first so Levanter picks it up.)
#
# Why H2 (LR jump) is ruled out: per `bin-sigma-fs.sh:7` and
# `bin5-extend-10k.sh:12`, bin5's *actual* mid-life LR trajectory was
# constant 3e-4 + warmup 0.05 (despite the originating wandb config saying
# cosine). bin5-extend matched that; no LR jump at resume.
#
# What we expect:
#   - If H1 is correct → TL stays at ~6.5 (where bin5 left off), no spike.
#     Implication: the bin5-extend spike was caused by TS0-instead-of-TS0123
#     data mix change. Open puzzle: WHY would TS0 (a strict subset of TS0123)
#     cause a spike? Hypotheses include (a) different LMQ codec baked into
#     TS0 cache vs. TS0123 caches at different build times, (b) sub-shard
#     tokenization drift.
#   - If TL still spikes → resume is broken at the trainer layer
#     (optimizer state? ckpt load? codec ID drift?) and Phase 1's resume
#     guard alone won't be sufficient.
#
# Cleanup after observation: this is a one-shot diagnostic fire; no
# downstream training pipeline depends on its ckpts. Can be killed once we
# see the first ~500 post-resume TL points.
set -euo pipefail

SRC_BUCKET="gs://marin-us-east5/tomat"
SRC_LABEL="train-mg-kl-bin5-fs-tpu"
SRC_CKPT_DIR="${SRC_BUCKET}/results/${SRC_LABEL}/checkpoints/${SRC_LABEL}"
SRC_STEP="step-100000"
DST_LABEL="train-mg-kl-bin5-cont-clean"
DST_CKPT_DIR="${SRC_BUCKET}/results/${DST_LABEL}/checkpoints/${DST_LABEL}"
RESUME_STEPS=102000

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

# 1. Mirror bin5's step-100000 → new label's nested ckpt dir (server-side
#    gcsfs walk + per-file cp). Same approach as `bin5-sigma-cont.sh`.
if ! gsutil -q stat "${DST_CKPT_DIR}/${SRC_STEP}/metadata.json" >/dev/null 2>&1; then
  echo "[mirror] ${SRC_CKPT_DIR}/${SRC_STEP}/ → ${DST_CKPT_DIR}/${SRC_STEP}/"
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
  echo "[mirror] already present, skipping"
fi

# 2. Fire the continuation job.
cd marin
/Users/ryan/c/oa/marin/.venv/bin/iris --cluster=marin job run \
  --tpu v5p-16 \
  --zone us-east5-a \
  --enable-extra-resources \
  --cpu 32 --memory 64GB \
  --max-retries 20 \
  --env-vars WANDB_API_KEY "$WANDB_API_KEY" \
  --env-vars TOMAT_WANDB_ENTITY open-athena \
  --env-vars TOMAT_BUCKET "${SRC_BUCKET}" \
  --env-vars TOMAT_LABEL train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
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
  --env-vars TOMAT_WARMUP 0 \
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
  --job-name "${DST_LABEL}" \
  -- python train_tomat_tpu.py

echo
echo "[fired bin5-cont-clean sanity check]"
echo "Watch: ./tomat iris ls | grep ${DST_LABEL}"
echo "Dashboard: https://tomat.oa.dev/#/runs/${DST_LABEL}"
