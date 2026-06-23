#!/bin/bash
# bin5 SANITY-CHECK resume: train-mg-kl-bin5-cont-from-99k from bin5@step-99000.
#
# Tests whether the bin5@step-100000 ckpt specifically is broken (vs the
# resume mechanism being universally broken for bin5).
#
# bin5-cont-clean (from step-100000) spiked TL 6.5 → 13 → settled ~7.55.
# This run resumes from bin5@step-99000 with the SAME recipe + SAME data
# (TS0123 union) + SAME LR + SAME post-resume horizon (+2k = 101k target).
#
# Expected:
#   - Same spike → resume mechanism is broken for bin5 in general (look
#     for a config delta we missed: maybe the optimizer's
#     trace-of-precision state, the data-iter epoch position, etc.)
#   - Clean resume → bin5@step-100000 specifically is corrupted (some
#     bit-rot or tensor-store inconsistency at exactly that step). Less
#     likely but interesting.
#
# Per Ryan: "What if we try a resume from bin5@90k? Maybe something is
# wrong with the 100k ckpt specifically?"
set -euo pipefail

SRC_BUCKET="gs://marin-us-east5/tomat"
SRC_LABEL="train-mg-kl-bin5-fs-tpu"
SRC_CKPT_DIR="${SRC_BUCKET}/results/${SRC_LABEL}/checkpoints/${SRC_LABEL}"
SRC_STEP="step-99000"
DST_LABEL="train-mg-kl-bin5-cont-from-99k"
DST_CKPT_DIR="${SRC_BUCKET}/results/${DST_LABEL}/checkpoints/${DST_LABEL}"
RESUME_STEPS=101000

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

# Mirror bin5's step-99000 → new label's nested ckpt dir.
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
echo "[fired bin5-cont-from-99k sanity check]"
echo "Watch: ./tomat iris ls | grep ${DST_LABEL}"
echo "Dashboard: https://tomat.oa.dev/#/runs/${DST_LABEL}"
