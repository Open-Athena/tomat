#!/usr/bin/env bash
# tomat-self-update: pull latest main and re-install systemd units.
#
# Uses `git fetch && git reset --hard @{u}` rather than `git pull` so we
# can never get wedged by a dirty work tree on the VM — `pull` fails on
# any local mutation and would freeze the auto-deploy loop until someone
# SSH'd in, but `reset --hard` blows past it. The VM clone is meant to be
# strictly a mirror of `origin/main`; if you need to hand-edit the
# checkout for debugging, accept that the next 5-min tick will wipe it.
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

BEFORE_SHA="$(git rev-parse HEAD)"

# `git fetch` is the only network call this script makes; if it fails
# (network glitch, GH ratelimit), we exit cleanly and the next tick will
# retry. We do NOT want a transient fetch failure to leave the work tree
# in some half-updated state.
if ! git fetch --quiet; then
    echo "self-update: git fetch failed; will retry next tick" >&2
    exit 0
fi

# Hard-reset to the tracking branch's tip. Wipes any local edits — see
# header comment for rationale.
git reset --quiet --hard '@{u}'

AFTER_SHA="$(git rev-parse HEAD)"

if [[ "$BEFORE_SHA" == "$AFTER_SHA" ]]; then
    # Nothing new on main — skip the install.sh round-trip.
    exit 0
fi

echo "self-update: $BEFORE_SHA → $AFTER_SHA"
exec "$REPO_DIR/gce/install.sh"
