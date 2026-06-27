#!/bin/bash
# bin5 pin-controlled re-fire (bin5r2): train-mg-bin5r2 from bin5@step-100000.
#
# **Pin-drift hypothesis test.** Mirrors bin5-cont-clean.sh EXACTLY (same
# HW/zone/bucket/loss/data/HPs/seed/BS/LR/shuffle), with one knob varied:
# the marin pin in tomat/marin/uv.lock must be downgraded to 51f17e5f
# (bin5 step-100000's save-pin per the metadata.json timestamp =
# 2026-06-17T11:17:49 UTC, which falls in the [b52ab82d, e20bdd18) window
# where pin = 51f17e5f).
#
# Workflow (PRE-FIRE):
#   1. `./tomat marin bump 51f17e5f --no-smoke`   # downgrade pin
#   2. `scripts/stamp_iris_build_date.py`         # restamp BUILD_DATE
# Then run THIS script. POST-FIRE (after iris submission):
#   3. `./tomat marin bump <e20bdd18 SHA> --no-smoke`  # restore pin
#   4. `scripts/stamp_iris_build_date.py`              # restamp again
#
# Hypothesis: pin-drift is the cause of bin5's resume spike. Same-pin
# save→load is byte-clean (verified 2026-06-27 via v4r2 + cos-r-r2 — both
# clean at e20bdd18→e20bdd18). If bin5r2 at save-pin 51f17e5f also lands
# clean, hypothesis confirmed; if it still spikes, fall back to the
# 'dc0edbbee2 vs e20bdd1892' bisect or look for a 3rd drift source.
set -euo pipefail

cd "$(dirname "$0")/../.."  # tomat root
# DO NOT stamp BUILD_DATE here — the workflow's step-2 stamp must already
# match the active pin (51f17e5f). Re-stamping mid-script could clobber.

PARENT_LABEL=train-mg-kl-bin5-fs-tpu
DST_LABEL=train-mg-bin5r2
SRC_BUCKET=gs://marin-us-east5/tomat
FROM_CKPT="${SRC_BUCKET}/results/${PARENT_LABEL}/checkpoints/${PARENT_LABEL}/step-100000"

if ! gsutil -q stat "${FROM_CKPT}/metadata.json" >/dev/null 2>&1; then
  echo "ERROR: ${FROM_CKPT}/metadata.json missing." >&2
  exit 1
fi

# Sanity-check the active marin pin matches bin5's save-pin.
PIN=$(grep -oE 'rev=[a-f0-9]+' marin/uv.lock | head -1 | cut -c5-12)
if [ "$PIN" != "51f17e5f" ]; then
  echo "ERROR: marin pin is '${PIN}', expected '51f17e5f'." >&2
  echo "Run: ./tomat marin bump 51f17e5f --no-smoke" >&2
  exit 1
fi
echo "[bin5r2] marin pin: ${PIN} (matches bin5's save-pin) ✓"

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
  -s 102000 \
  "${DST_LABEL}"
