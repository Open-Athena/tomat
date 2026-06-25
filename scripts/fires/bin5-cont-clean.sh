#!/bin/bash
# bin5 SANITY-CHECK resume: train-mg-kl-bin5-cont-clean from bin5@step-100000.
#
# Tests H1 (data mix change as cause of bin5-extend-10k's TL spike at the
# resume boundary). Matches `bin5-extend-10k.sh` exactly except:
#   - TOMAT_LABEL: train-full-v3,…shard1,…shard2,…shard3   (TS0123 union)
#     bin5-extend used TS0 single-shard (`train-full-v3`).
#   - TOMAT_WARMUP=0 (explicit no-warmup intent; behaviorally a no-op since
#     Levanter's lr_scheduler is indexed by global step — step-100000 is
#     past the 5250-step warmup window either way).
#   - TOMAT_STEPS=102000 (just 2k steps post-resume — enough to see whether
#     TL settles cleanly or spikes. ~2 hours on v6e-16.)
#   - TOMAT_RESULTS_LABEL=train-mg-kl-bin5-cont-clean (new label, mirror
#     bin5@step-100000 → new ckpt dir first so Levanter picks it up.)
#
# Why H2 (LR jump) is ruled out: per `bin-sigma-fs.sh:7` and
# `bin5-extend-10k.sh:12`, bin5's *actual* mid-life LR trajectory was
# constant 3e-4 + warmup 0.05 (despite the originating wandb config saying
# cosine). bin5-extend matched that; no LR jump at resume.
#
# This is the .sh→tomat-train migration follow-up; explicit zone pin
# preserves the original v5p-16 us-east5-a intent (testing HW-switch
# hypothesis).
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-kl-bin5-fs-tpu
DST_LABEL=train-mg-kl-bin5-cont-clean
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-100000"

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
