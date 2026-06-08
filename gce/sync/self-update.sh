#!/usr/bin/env bash
# tomat-self-update: pull latest main and re-install systemd units.
#
# `git pull --ff-only` is the deploy gate: if the local clone has diverged
# (uncommitted edits, hand-applied hotfixes), this fails loud rather than
# silently rewriting the work tree. Recover by ssh'ing in and resolving by
# hand — the cron is meant for clean fast-forwards from main only.
#
# systemd-safety note: this script is itself launched by
# `tomat-self-update.service`. When install.sh restarts a unit whose
# binary just changed on disk, systemd respects the in-flight invocation
# (this oneshot keeps running with the old script's bytes already mmap'd)
# and the next timer tick picks up the new binary. So it's safe to call
# install.sh from inside itself; we don't try to "swap mid-flight".
set -euo pipefail

REPO_DIR="${TOMAT_REPO_DIR:-$HOME/tomat}"
cd "$REPO_DIR"

# Fail fast on a dirty work tree — we don't want a hand-edited clone to
# silently get overwritten by an --ff-only that suddenly succeeds.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "self-update: work tree dirty, refusing to pull" >&2
    git status --short >&2
    exit 1
fi

BEFORE_SHA="$(git rev-parse HEAD)"
git pull -q --ff-only
AFTER_SHA="$(git rev-parse HEAD)"

if [[ "$BEFORE_SHA" == "$AFTER_SHA" ]]; then
    # Nothing new on main — skip the install.sh round-trip.
    exit 0
fi

echo "self-update: $BEFORE_SHA → $AFTER_SHA"
exec "$REPO_DIR/gce/install.sh"
