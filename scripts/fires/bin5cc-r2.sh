#!/bin/bash
# bin5cc-r2: 2nd-gen resume of bin5-cont-clean@step-101000.
# Tests whether bin5's resume-spike propagates through descendants once
# the model has recovered from the initial spike. bin5-cont-clean@step-101000
# is fully recovered (TL settled ~7.5 by step ~100140; 101000 is well past).
#
# Naming: bin5 (r0) → bin5-cont-clean (r1, spiked at 100k) → bin5cc-r2 (this).
# Distinguished from `bin5r2` which resumes bin5@100000 directly (not via
# cont-clean's post-spike state).
#
# Hypothesis:
#   - SPIKE → there's something in bin5's lineage that propagates indefinitely.
#     Every cross-fire from anything bin5-descended will spike. This would
#     indicate the saved bytes themselves carry the latent defect.
#   - CLEAN → the spike is a one-time pain of loading bin5's specific saved
#     state. The post-spike recovered state cross-fires cleanly. This would
#     indicate the saved bytes are mostly OK; something about bin5's
#     specific state at save-time is the trigger, but it's recoverable.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-kl-bin5-cont-clean
DST_LABEL=train-mg-bin5cc-r2
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-101000"

if ! gsutil -q stat "${FROM_CKPT}/metadata.json" >/dev/null 2>&1; then
  echo "ERROR: ${FROM_CKPT}/metadata.json missing." >&2
  exit 1
fi

./tomat train \
  --resume --parent "${PARENT_LABEL}" \
  --from-ckpt "${FROM_CKPT}" \
  --bucket "${SRC_BUCKET}" \
  --region-strategy explicit --zone us-east5-a \
  -T v5p-16 \
  -D train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
  -m 200M -b 256 --seed 42 \
  --lr 3e-4 --lr-schedule constant --warmup 0 \
  --val-seqs 0 --steps-per-eval 0 \
  --mg-mode --mask-prior absorbing --mg-loss kl_gauss --mg-kl-sigma 5 \
  --shuffle-window-blocks 310113 \
  -s 102000 \
  "${DST_LABEL}"
