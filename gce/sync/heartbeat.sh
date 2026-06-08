#!/usr/bin/env bash
# tomat cron heartbeat → R2.
#
# Called at the tail of `runs-sync-active.sh` (every 2 min). Writes a tiny
# JSON to `s3://openathena/tomat/cron-heartbeat.json` so the dashboard can
# render "cron last fired Xm ago". Idle ticks still write (with
# `last_run_count: 0`) so we can tell "alive but idle" from "cron down".
#
# Usage: heartbeat.sh <last_run_count> <last_failure_count>
set -euo pipefail

LAST_RUN_COUNT="${1:-0}"
LAST_FAILURE_COUNT="${2:-0}"

R2_ENDPOINT="${TOMAT_R2_ENDPOINT:-https://43a6f2d588b1483733189d39418ec5be.r2.cloudflarestorage.com}"
R2_BUCKET="${TOMAT_R2_BUCKET:-openathena}"
R2_KEY="${TOMAT_HEARTBEAT_KEY:-tomat/cron-heartbeat.json}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(hostname -s 2>/dev/null || hostname)"

PAYLOAD_TMP="$(mktemp -t tomat-heartbeat.XXXXXX.json)"
trap 'rm -f "$PAYLOAD_TMP"' EXIT

cat > "$PAYLOAD_TMP" <<JSON
{
  "ts": "$TS",
  "host": "$HOST",
  "last_run_count": $LAST_RUN_COUNT,
  "last_failure_count": $LAST_FAILURE_COUNT
}
JSON

# aws-cli is already on the VM (used by other tomat sync paths). Endpoint
# URL routes to R2 instead of S3.
aws s3 cp --quiet \
    --endpoint-url "$R2_ENDPOINT" \
    --content-type application/json \
    --cache-control "max-age=30" \
    "$PAYLOAD_TMP" "s3://$R2_BUCKET/$R2_KEY"
