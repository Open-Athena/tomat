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
# This is the .sh→tomat-train migration pilot. Note the `explicit` region
# strategy: we deliberately pin us-east5-b (parent's v6e zone) so the test
# isolates HW from any cross-region resume noise. `auto` strategy would
# correctly pick us-east5 here but we're being defensive.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
scripts/stamp_iris_build_date.py >/dev/null

PARENT_LABEL=train-mg-kl-bin5-fs-tpu
DST_LABEL=train-mg-kl-bin5-cont-from-80k-v6e
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-80000"

./tomat train \
  --resume --parent "${PARENT_LABEL}" \
  --from-ckpt "${FROM_CKPT}" \
  --bucket "${SRC_BUCKET}" \
  --region-strategy explicit --zone us-east5-b \
  -T v6e-16 \
  -D train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
  -m 200M -b 256 --seed 42 \
  --lr 3e-4 --lr-schedule constant --warmup 0 \
  --val-seqs 0 --steps-per-eval 0 \
  --mg-mode --mask-prior absorbing --mg-loss kl_gauss --mg-kl-sigma 5 \
  --shuffle-window-blocks 310113 \
  --allow-config-change hardware.tpu \
  -s 82000 \
  "${DST_LABEL}"
