#!/bin/bash
# bin5-from-step: parameterized cross-fire resume of bin5@step-N (N from arg).
# Tests whether bin5's resume-spike is universal across its training history
# or limited to specific steps. Recipe MIRRORS bin5-cont-clean exactly.
#
# Usage: bin5-from-step.sh <step>
#
# 2026-06-27 sequence:
#   - bin5-from-step.sh 30000   # well before TS0->TS0123 cutover @ step 67k
#   - bin5-from-step.sh 65000   # just before cutover
#   - bin5-from-step.sh 70000   # just after cutover
#
# Hypothesis tests:
#   - if step-30000 ALSO spikes: every bin5 save is broken. The cross-fire
#     resume bug is bin5-wide regardless of step or data label.
#   - if step-30000 is CLEAN but 65k/70k spike: corruption arose at/after the
#     data-label cutover; bracket more tightly.
#   - if all 3 spike: bin5 has been bad since at least step-30000.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "ERROR: missing step arg. Usage: $0 <step>" >&2
  exit 1
fi
STEP=$1

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-kl-bin5-fs-tpu
DST_LABEL=train-mg-bin5-from${STEP}
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-${STEP}"

if ! gsutil -q stat "${FROM_CKPT}/metadata.json" >/dev/null 2>&1; then
  echo "ERROR: ${FROM_CKPT}/metadata.json missing." >&2
  exit 1
fi

# Steps for the fire = STEP + 1000 (just enough post-resume TL to see whether
# the trajectory is clean or spikes). ~30 min on v5p-16.
TARGET=$((STEP + 1000))

./tomat train \
  --resume --parent "${PARENT_LABEL}" \
  --from-ckpt "${FROM_CKPT}" \
  --allow-config-change=optimizer.warmup \
  --bucket "${SRC_BUCKET}" \
  --region-strategy explicit --zone us-east5-a \
  -T v5p-16 \
  -D train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
  -m 200M -b 256 --seed 42 \
  --lr 3e-4 --lr-schedule constant --warmup 0 \
  --val-seqs 0 --steps-per-eval 0 \
  --mg-mode --mask-prior absorbing --mg-loss kl_gauss --mg-kl-sigma 5 \
  --shuffle-window-blocks 310113 \
  -s ${TARGET} \
  "${DST_LABEL}"
