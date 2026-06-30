#!/bin/bash
# cos-r 3rd-gen +1k extension (cos-r-r3): get a /10k MT/MV datapoint at step-30000.
#
# Naming: cos-r r0 = train-mg-cos-r-fs-tpu (from-scratch on TS0),
#         cos-r r1 = train-mg-cos-r-cont-clean (iris, +10k → step-19000),
#         cos-r r2 = train-mg-cos-r-r2        (iris, +10k → step-29000),
#         cos-r r3 = THIS run (iris, +1k → step-30000).
#
# Rationale: original r2 target was 30k (clean /10k boundary), but it was
# fired as +10k from 19k = 29k. r3 grabs the missing /10k MT/MV (and now,
# under #C of session VL plan, /10k VL via auto-fire).
#
# Recipe mirrors cos-r-r2 exactly (same data label, same MG cosine KL-Gauss
# σ=5 recipe, same v5p-16 us-east5-a pin).
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-cos-r-r2
DST_LABEL=train-mg-cos-r-r3
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-29000"

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
  -s 30000 \
  "${DST_LABEL}"
