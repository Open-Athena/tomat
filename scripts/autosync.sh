#!/usr/bin/env bash
# Auto-sync sidecars for the /runs dashboard.
#
# Run periodically (via launchd; see `launchd/com.openathena.tomat-autosync.plist`)
# to keep MSRP / segments / annotations / m-eval-result sidecars fresh
# without the user (or Claude) having to invoke them manually. See
# tomat memory `#280` for the rationale.
#
# Phase A (this script): MSRP via `tomat cost compute --all`.
# Phase B (todo): segments + annotations from `~/.tomat/submissions.jsonl`.
# Phase C (todo): m-eval auto-fire for missing (run, step, set, mode) tuples.
#
# Output goes to /tmp/tomat-autosync.log (rotated weekly).

set -euo pipefail

TOMAT="${TOMAT:-/Users/ryan/c/oa/tomat/tomat}"
LOG="/tmp/tomat-autosync.log"
LOCK="/tmp/tomat-autosync.lock"

# Single-instance lock (so a launchd over-fire doesn't pile up jobs).
# macOS doesn't ship `flock` — use a flag file. Race window is tiny vs
# the 15-minute cadence so simple-is-fine.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date -u +%FT%TZ)] already running (pid $(cat "$LOCK")), exiting" >> "$LOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

{
  echo "==== $(date -u +%FT%TZ) autosync start ===="

  # Phase A: MSRP. Idempotent; rewrites every run's cost.json from
  # iris-state.json + per-run parquet history.
  echo "[$(date -u +%FT%TZ)] tomat cost compute --all"
  "$TOMAT" cost compute --all 2>&1 || echo "  cost compute exit=$?"

  # Phase B: segments + annotations from submissions.jsonl. Stub until
  # next pass. The CLI would read ~/.tomat/submissions.jsonl, group by
  # run label, detect consecutive fires whose TOMAT_LABEL or
  # TOMAT_BATCH_SIZE changed, and write `runs/<label>/segments.json` +
  # `runs/<label>/annotations.json` on R2.
  # TODO:
  #   "$TOMAT" runs build-segments --all
  #   "$TOMAT" runs build-annotations --all

  # Phase C: m-eval auto-fire. Stub. Would walk each running/finished
  # run's ckpts and fire missing (val_200|train_200) × (K=1|K=12) evals.
  # TODO:
  #   "$TOMAT" evals auto-fire --all --modes oneshot,maskgit \
  #            --sets val_200,train_200

  echo "==== $(date -u +%FT%TZ) autosync done ===="
} >> "$LOG" 2>&1
