#!/usr/bin/env bash
# tomat-runs-sync-active: sync the small set of "currently moving" runs.
#
# Active = wandb state == "running" OR last log emitted < 15 min ago
# (cheap to determine via the already-cached snapshot the dashboard reads).
# Runs that fail either test are picked up by the stale-sync timer instead
# (every 30 min), so we don't hammer wandb for every dead run every 2 min.
#
# Logs every per-run sync's stdout/stderr prefixed with `[<run>] ` so
# `journalctl -u tomat-runs-sync-active.service` is parseable.
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
SNAPSHOT_URL="${TOMAT_SNAPSHOT_URL:-https://tomat.oa.dev/api/runs-snapshot.json}"
ACTIVE_MAX_LOG_AGE_SEC="${TOMAT_ACTIVE_MAX_LOG_AGE_SEC:-900}"  # 15 min

cd "$REPO_DIR"

# Pull active-run names from the live snapshot. If the snapshot endpoint is
# down (or returns malformed JSON) we silently sync nothing for this tick —
# the stale-sync timer is the safety net.
SNAPSHOT_TMP="$(mktemp -t tomat-snapshot.XXXXXX.json)"
trap 'rm -f "$SNAPSHOT_TMP"' EXIT

if ! curl -sfL --max-time 20 -o "$SNAPSHOT_TMP" "$SNAPSHOT_URL"; then
    echo "snapshot fetch failed: $SNAPSHOT_URL" >&2
    exit 0
fi

# Active filter:
#   - iris reports state RUNNING / PENDING / BUILDING (uppercase per worker.ts)
#   - OR snapshot-side last log < 15 min ago (via `last_log_sec_ago`
#     when the snapshot exposes it; falls back to 9999 if absent so we
#     don't accidentally treat every run as active).
ACTIVE_RUNS=$(jq -r --argjson maxAge "$ACTIVE_MAX_LOG_AGE_SEC" '
    ((.iris.jobs // {}) | to_entries[]
        | select(.value.state == "RUNNING" or .value.state == "PENDING" or .value.state == "BUILDING")
        | .key
        | capture("^/[^/]+/(?<label>.+)$").label),
    ((.runs // {}) | to_entries[]
        | select(.value != null)
        | select((.value.last_log_sec_ago // 9999) < $maxAge)
        | .key)
' < "$SNAPSHOT_TMP" | sort -u)

LAST_RUN_COUNT=0
LAST_FAILURE_COUNT=0

if [[ -z "$ACTIVE_RUNS" ]]; then
    echo "no active runs"
else
    while IFS= read -r run; do
        [[ -z "$run" ]] && continue
        LAST_RUN_COUNT=$((LAST_RUN_COUNT + 1))
        # `tomat runs sync` writes the manifest + parquet to R2 for this run.
        # Per-run failure shouldn't kill the whole cron tick — count it and
        # move on; the next tick will retry.
        if ! ./tomat runs sync "$run" 2>&1 | sed "s/^/[$run] /"; then
            LAST_FAILURE_COUNT=$((LAST_FAILURE_COUNT + 1))
            echo "[$run] sync failed (rc=${PIPESTATUS[0]})" >&2
        fi
    done <<<"$ACTIVE_RUNS"
    echo "synced $LAST_RUN_COUNT runs, $LAST_FAILURE_COUNT failures"
fi

# Heartbeat → R2: dashboard reads this to surface "cron last fired Xm ago".
# Idle ticks still write a heartbeat so we can distinguish "cron alive but
# idle" from "cron down".
if [[ -x "$REPO_DIR/gce/sync/heartbeat.sh" ]]; then
    "$REPO_DIR/gce/sync/heartbeat.sh" "$LAST_RUN_COUNT" "$LAST_FAILURE_COUNT" || true
fi
