#!/usr/bin/env bash
# tomat-iris-sync: pull iris controller state → `tomat/iris-state.json` in R2.
#
# Run every 1 min so the dashboard's iris-state badges (RUNNING / PENDING
# / BUILDING / FAILED / etc.) stay live. The legacy `~/iris-sync.py` ran
# this from crontab; we now invoke it from a systemd timer for consistency
# with the rest of the gce/ stack.
#
# Keeps `~/iris-sync.py` as the executable rather than a `tomat iris sync`
# CLI call because the legacy script has hand-rolled R2 + ADC auth + an
# empty-result safety check (`refusing R2 upload (probable iris/auth
# failure)`) that the tomat CLI doesn't replicate today.
set -euo pipefail

SCRIPT="${TOMAT_IRIS_SYNC_SCRIPT:-$HOME/iris-sync.py}"

if [[ ! -x "$SCRIPT" ]]; then
    echo "iris-sync.py not found / not executable at: $SCRIPT" >&2
    exit 1
fi

"$SCRIPT"
