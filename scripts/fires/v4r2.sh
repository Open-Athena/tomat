#!/bin/bash
# v4 2nd-gen resume (v4r2): train-mg-v4r2 from v4-cont-clean@step-110000.
#
# Naming: v4r0 = train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42 (Modal),
#         v4r1 = train-mg-modal-v4-cont-clean (iris, +10k),
#         v4r2 = THIS run (iris, +10k more).
#
# Hypothesis test (pin-drift): save-pin == load-pin == e20bdd18, so save/load
# RT should be clean and TL should continue smoothly past 110k with no spike.
# Spike → falsifies pin-drift hypothesis. See tmp/pin-drift-hypothesis.md.
#
# Recipe mirrors v4-cont-clean exactly: same TPU/zone/bucket/loss/data/HPs.
# Includes the pre-existing 78125 shuffle_window_blocks (inherited from
# v4-modal-epochwin's TS1-epochwin) even though we're on TS0123 — fidelity
# to the parent is what isolates the resume mechanism.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-modal-v4-cont-clean
DST_LABEL=train-mg-v4r2
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-110000"

if ! gsutil -q stat "${FROM_CKPT}/metadata.json" >/dev/null 2>&1; then
  echo "ERROR: ${FROM_CKPT}/metadata.json missing." >&2
  exit 1
fi

./tomat train \
  --resume --parent "${PARENT_LABEL}" \
  --from-ckpt "${FROM_CKPT}" \
  --bucket "${SRC_BUCKET}" \
  --region-strategy explicit --zone us-east5-b \
  -T v6e-16 \
  -D train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
  -m 200M -b 128 --seed 42 \
  --lr 4e-5 --lr-schedule constant --warmup 0 \
  --val-seqs 0 --steps-per-eval 0 \
  --mg-mode --mask-prior absorbing --mg-loss ce \
  --shuffle-window-blocks 78125 \
  -s 120000 \
  "${DST_LABEL}"
