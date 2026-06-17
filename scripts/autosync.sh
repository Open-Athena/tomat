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

# Use the project venv's python so deps (`click`, `fsspec`, …) are in
# scope. `./tomat`'s shebang `#!/usr/bin/env python3` resolves to the
# system python under launchd's PATH (no `direnv` activation), which
# doesn't have project deps → autosync silently no-op'd for days until
# noticed (MSRP chip stayed $0).
TOMAT_PY="${TOMAT_PY:-/Users/ryan/c/oa/tomat/.venv/bin/python}"
TOMAT_SCRIPT="${TOMAT_SCRIPT:-/Users/ryan/c/oa/tomat/tomat}"
TOMAT="$TOMAT_PY $TOMAT_SCRIPT"
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
  $TOMAT cost compute --all 2>&1 || echo "  cost compute exit=$?"

  # Phase A.2: m-eval results aggregation. Per-step JSONs on GCS
  # (`gs://marin-eu-west4/tomat/eval/results/<run>/<set>/step-N.json`)
  # → per-run `eval.json` on R2. Without this, the dashboard's MEvalTable
  # shows stale rows: the iris jobs finish, the JSONs land on GCS, but
  # `eval.json` on R2 doesn't auto-rebuild. Bin5's "13 done · 1 dot
  # rendered" symptom on 2026-06-14 traced to this.
  echo "[$(date -u +%FT%TZ)] tomat evals sync (all recently-synced runs)"
  # `evals sync` takes a SUBSTR. Empty string = match all → walk every
  # run that has a parquet sidecar. Each run takes ~1s of GCS scans;
  # the dominant cost is the eu-west4 bucket walk.
  $TOMAT evals sync 2>&1 || echo "  evals sync exit=$?"

  # Phase B: segments + annotations from submissions.jsonl. Stub until
  # next pass. The CLI would read ~/.tomat/submissions.jsonl, group by
  # run label, detect consecutive fires whose TOMAT_LABEL or
  # TOMAT_BATCH_SIZE changed, and write `runs/<label>/segments.json` +
  # `runs/<label>/annotations.json` on R2.
  # TODO:
  #   $TOMAT runs build-segments --all
  #   $TOMAT runs build-annotations --all

  # Phase C: m-eval auto-fire. Stub. Would walk each running/finished
  # run's ckpts and fire missing (val_200|train_200) × (K=1|K=12) evals.
  # TODO:
  #   $TOMAT evals auto-fire --all --modes oneshot,maskgit \
  #            --sets val_200,train_200

  # Phase D: process pending eval-fire requests posted from the dashboard.
  # FE → CFW (`POST /api/eval/fire`) writes a small JSON request to R2 at
  # `tomat/eval-fire-requests/<run>/<step>-<set>-<mode>-<iso>.json`. We list
  # those, dispatch `tomat evals fire` for each, then delete the consumed
  # request blob so the dashboard's "queued" cell flips back to "missing" /
  # "in-flight" on the next snapshot cycle. The whole pass is idempotent — a
  # failed `tomat evals fire` leaves the request in R2 to retry next pass.
  echo "[$(date -u +%FT%TZ) eval-fire-requests] processing pending fires"
  $TOMAT evals process-fire-requests 2>&1 || echo "  process-fire-requests exit=$?"

  echo "==== $(date -u +%FT%TZ) autosync done ===="
} >> "$LOG" 2>&1
