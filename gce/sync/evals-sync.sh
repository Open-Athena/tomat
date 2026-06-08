#!/usr/bin/env bash
# tomat-evals-sync: pull new m-eval JSONs from GCS → R2 (every 10 min).
#
# Empty filter = all runs; `tomat evals sync` short-circuits when nothing
# new has landed.
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
cd "$REPO_DIR"

./tomat evals sync
