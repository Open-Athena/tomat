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
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-kl-bin5-fs-tpu
DST_LABEL=train-mg-kl-bin5-cont-from-99k
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-99000"

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
  -s 101000 \
  "${DST_LABEL}"
