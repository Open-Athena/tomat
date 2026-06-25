#!/bin/bash
# modal-v4-epochwin SANITY-CHECK resume: +10k from step-100000 on TS0123.
#
# Reference resume test — modal-v4 reached lower TL than bin5 (~7.0 vs 7.55
# settled), and is non-KL (absorbing prior + plain CE). Tests whether a
# non-KL resume avoids the spike. Per Ryan: "+10k of TS0123 at it".
#
# Parent recipe (from wandb config + scripts/preamble_vl_modal.py):
#   loss_type = "ce" (bidirectional CE on masked positions; not KL)
#   prior = absorbing
#   LR = 4e-5 constant, warmup = 0
#   shuffle window_blocks = 78125 (epochwin auto for TS1)
#   batch_size = 128
#
# Deltas vs parent (intentional):
#   data: TS1 only (modal-local /tmp cache) → TS0123 union
#     This is the headline test variable. Phase 1 guard would refuse this
#     without --allow-config-change=data.cache_dir.
#   bucket: eu-west4 (parent's) → us-east5 (this run); ckpt mirrored.
#   TPU: H200×8 (Modal) → v6e-16 (TPU)
#
# NOTE: parent ckpt step-100000 must be mirrored under us-east5 before
# this script runs (modal-v4 trained on Modal-local volume; ckpts were
# pulled out by `tomat ckpt mirror` post-training).
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42
DST_LABEL=train-mg-modal-v4-cont-clean
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${DST_LABEL}/checkpoints/${DST_LABEL}/step-100000"

# Pre-flight: refuse to fire if ckpt isn't mirrored. (tomat train would
# also catch this via --from-ckpt, but a head-of-script bail is friendlier.)
if ! gsutil -q stat "${FROM_CKPT}/metadata.json" >/dev/null 2>&1; then
  echo "ERROR: ${FROM_CKPT}/metadata.json missing — mirror modal-v4@step-100000 first." >&2
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
  --allow-config-change data.cache_dir,trainer.train_batch_size,optimizer.learning_rate \
  -s 110000 \
  "${DST_LABEL}"
