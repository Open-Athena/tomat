#!/usr/bin/env bash
# tomat-runs-sync-stale: full-sweep sync (every 30 min).
#
# The active-sync timer touches "currently moving" runs every 2 min; this
# unit picks up GC + late-arriving evals on everything else. Empty filter
# means "all runs"; `tomat runs sync` short-circuits no-op runs cheaply so
# it's safe to fire across the entire fleet every half hour.
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
cd "$REPO_DIR"

./tomat runs sync
