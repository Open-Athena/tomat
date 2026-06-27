#!/bin/bash
# cos-r 2nd-gen resume (cos-r-r2): train-mg-cos-r-r2 from cos-r-cont-clean@step-19000.
#
# Naming: cos-r r0 = train-mg-cos-r-fs-tpu (from-scratch on TS0),
#         cos-r r1 = train-mg-cos-r-cont-clean (iris, +10k),
#         cos-r r2 = THIS run (iris, +10k more).
#
# Hypothesis test (pin-drift): cos-r-cont-clean was the cleanest resume we've
# observed (parent + child both at pin e20bdd18). Another +10k tests whether
# the clean recipe stays clean across yet another save/load boundary —
# the "production-grade extensions" sanity check.
#
# Recipe mirrors cos-r-cont-clean exactly.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-cos-r-cont-clean
DST_LABEL=train-mg-cos-r-r2
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-19000"

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
  -D train-full-v3 \
  -m 200M -b 256 --seed 42 \
  --lr 3e-4 --lr-schedule constant --warmup 0 \
  --val-seqs 0 --steps-per-eval 0 \
  --mg-mode --mask-prior cosine --mg-loss kl_gauss --mg-kl-sigma 5 \
  --shuffle-window-blocks 310113 \
  -s 29000 \
  "${DST_LABEL}"
