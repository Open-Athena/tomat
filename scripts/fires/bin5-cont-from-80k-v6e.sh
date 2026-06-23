#!/bin/bash
# bin5 PARENT-HW resume: train-mg-kl-bin5-cont-from-80k-v6e from bin5@step-80000 on v6e-16.
#
# Tests Betsy's HW-switch hypothesis. The four cont-clean fires (cont-clean,
# cont-from-99k, cont-s3, cont-s10, cont-s20) all ran on v5p-16 us-east5-a and
# all produced a TL 6.5 → 13 → ~7.55 settled (+1 nat above parent). bin5's
# own ~73 intra-run preempt restarts during steps 40k-100k were all on
# v6e-16 and were seamless — TL never spiked at any of them.
#
# Variable being tested: HW (v6e-16 — same chip + mesh as parent).
# If this resume is clean → cross-HW (v6e → v5p) is the cause.
# If still spikes → something else is at play.
#
# Resume target: step-80000 (mid-converged; parent's TL was ~6.6 around then,
# i.e. close enough to "converged" to be a stress test, far enough from 100k
# to rule out a step-100000-specific corruption hypothesis).
set -euo pipefail

SRC_BUCKET="gs://marin-us-east5/tomat"
SRC_LABEL="train-mg-kl-bin5-fs-tpu"
SRC_CKPT_DIR="${SRC_BUCKET}/results/${SRC_LABEL}/checkpoints/${SRC_LABEL}"
SRC_STEP="step-80000"
DST_LABEL="train-mg-kl-bin5-cont-from-80k-v6e"
DST_CKPT_DIR="${SRC_BUCKET}/results/${DST_LABEL}/checkpoints/${DST_LABEL}"
RESUME_STEPS=82000  # +2k post-resume

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

export WANDB_API_KEY="${WANDB_API_KEY:-$(grep -E '^password' ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')}"

# Mirror bin5's step-80000 → new label's nested ckpt dir.
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
  --tpu v6e-16 \
  --zone us-east5-b \
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
echo "[fired bin5-cont-from-80k on v6e-16 us-east5-b]"
echo "Dashboard: https://tomat.oa.dev/#/runs/${DST_LABEL}"

# Self-record this fire to R2 (spec 61 §2.1). Idempotent on the manifest
# body — re-running this script after iris submit-fail produces the same
# fire-id (and a no-op overwrite if the prior PUT landed).
cd "$(dirname "$0")/../.."  # tomat root
./tomat fires record \
  --run-id "${DST_LABEL}" \
  --parent "${SRC_LABEL}" \
  --substrate iris \
  --iris-job-id "/ryan/${DST_LABEL}" \
  --iris-tpu v6e-16 --iris-zone us-east5-b \
  --wandb-entity open-athena --wandb-project tomat-lmq-P19 \
  --resume-from-step 80000 --target-steps "${RESUME_STEPS}" \
  --intended-delta 'hardware.tpu=v5p-16 us-east5-a->v6e-16 us-east5-b' \
  --notes 'controlled HW test: parent v6e-16 → resume on v6e-16 (vs cont-clean v5p-16). If TL clean → cross-HW v6e->v5p caused the +1 nat spike.'
