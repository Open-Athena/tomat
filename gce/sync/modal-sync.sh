#!/usr/bin/env bash
# tomat-modal-sync: write `tomat/modal-state.json` to R2 (every 1 min).
#
# Drives the Modal-state badge on /runs. Tight cadence because that badge
# is the only signal that a Modal-spawned training fire is actually moving
# (no iris job slot for those).
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
cd "$REPO_DIR"

./tomat modal sync --all-pending
