#!/bin/bash
# cos-r SANITY-CHECK resume: train-mg-cos-r-cont-clean +10k from cos-r@step-9000.
#
# cos-r was the first idiomatic MG from-scratch run (cosine-prior mask
# schedule, KL-Gauss loss σ=5). It ended at the Levanter OBO step-9999 with
# the last save at step-9000. This continuation extends +10k to step-19000
# on the same recipe and data (TS0, since cos-r was trained on TS0 alone).
#
# TOMAT_WARMUP: explicit 0 (intent-explicit; behaviorally a no-op since
# Levanter's lr_scheduler is global-step-indexed past the warmup window).
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-cos-r-fs-tpu
DST_LABEL=train-mg-cos-r-cont-clean
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-9000"

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
  -s 19000 \
  "${DST_LABEL}"
