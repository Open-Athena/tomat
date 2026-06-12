#!/usr/bin/env bash
# tomat-runs-sync-stale: full-sweep sync (every 30 min).
#
# The active-sync timer touches "currently moving" runs every 2 min; this
# unit picks up GC + late-arriving evals on everything else. We iterate
# the runs in the live snapshot (same source as runs-sync-active.sh) and
# call `tomat runs sync <name>` per name — the CLI subcmd is one-run-at-a-
# time, so a literal `tomat runs sync ""` would bail with "multiple
# distinct runs match" instead of looping.
#
# `tomat runs sync` short-circuits no-op runs cheaply (manifest unchanged
# → no wandb history fetch), so a 30-min sweep across the full fleet is
# bounded by wandb-list-runs cost, not per-run sync cost.
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
SNAPSHOT_URL="${TOMAT_SNAPSHOT_URL:-https://tomat-runs-api.openathena.workers.dev/api/runs-snapshot.json}"

cd "$REPO_DIR"

SNAPSHOT_TMP="$(mktemp -t tomat-snapshot.XXXXXX.json)"
trap 'rm -f "$SNAPSHOT_TMP"' EXIT

if ! curl -sfL --max-time 30 -o "$SNAPSHOT_TMP" "$SNAPSHOT_URL"; then
    echo "snapshot fetch failed: $SNAPSHOT_URL" >&2
    exit 0
fi

ALL_RUNS=$(jq -r '(.runs // {}) | to_entries[] | select(.value != null) | .key' < "$SNAPSHOT_TMP" | sort -u)

n_ok=0
n_fail=0
while IFS= read -r run; do
    [[ -z "$run" ]] && continue
    if ! ./tomat runs sync "$run" 2>&1 | sed "s/^/[$run] /"; then
        n_fail=$((n_fail + 1))
        echo "[$run] sync failed (rc=${PIPESTATUS[0]})" >&2
    else
        n_ok=$((n_ok + 1))
    fi
done <<<"$ALL_RUNS"

echo "stale-sync: $n_ok ok, $n_fail failed"
