#!/usr/bin/env bash
# tomat GCE-VM systemd installer.
#
# Copies `gce/systemd/*.{service,timer}` into `/etc/systemd/system/`,
# reloads the daemon, enables timers, and restarts services whose unit
# files actually changed on disk. Idempotent — safe to re-run every
# `tomat-self-update.timer` tick (every 5 min).
#
# Modes:
#   `--check` — validate prerequisites (creds, tools, repo state) without
#               touching /etc/systemd. Exit 0 = READY, exit 1 = MISSING.
#   no args   — full install: validate, copy units, reload, enable timers,
#               restart changed services. Requires `sudo` for the
#               /etc/systemd/system writes + `systemctl` calls.
#
# Authentication is expected to come from `/etc/default/tomat-monitor`
# (the EnvironmentFile= on every service). That file is sensitive
# (WANDB_API_KEY, R2 creds, GCS-ADC path) so it lives ONLY on the VM and
# is not written by this script. We validate its keys exist; we never
# print their values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_SRC="$SCRIPT_DIR/systemd"
SYSTEMD_DST="/etc/systemd/system"
ENV_FILE="${TOMAT_ENV_FILE:-/etc/default/tomat-monitor}"

# Timers we enable + their paired services. (Each *.timer wakes its
# *.service of the same basename via `Unit=` default.) Listed explicitly
# so we don't accidentally enable a half-staged unit dropped into
# gce/systemd/ during dev.
TIMERS=(
    tomat-runs-sync-active.timer
    tomat-runs-sync-stale.timer
    tomat-modal-sync.timer
    tomat-evals-sync.timer
    tomat-self-update.timer
)
SERVICES=(
    tomat-runs-sync-active.service
    tomat-runs-sync-stale.service
    tomat-modal-sync.service
    tomat-evals-sync.service
    tomat-self-update.service
)
# Required cred env keys (presence check only — never log values).
REQUIRED_ENV_KEYS=(
    WANDB_API_KEY
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    GOOGLE_APPLICATION_CREDENTIALS
)

CHECK_ONLY=0
case "${1:-}" in
    --check) CHECK_ONLY=1 ;;
    "") ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

err() { echo "$@" >&2; }
ok()  { echo "  OK: $*"; }
fail() { echo "  FAIL: $*" >&2; FAILED=$((FAILED + 1)); }

FAILED=0

echo "== check: env file =="
if [[ ! -r "$ENV_FILE" ]]; then
    fail "$ENV_FILE missing or unreadable"
else
    ok "$ENV_FILE exists"
    for key in "${REQUIRED_ENV_KEYS[@]}"; do
        if grep -qE "^${key}=" "$ENV_FILE"; then
            ok "$key set"
        else
            fail "$key missing from $ENV_FILE"
        fi
    done
fi

echo "== check: PATH tools =="
for tool in curl jq aws git gcloud systemctl; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool: $(command -v "$tool")"
    else
        fail "$tool not on PATH"
    fi
done

echo "== check: repo state =="
if [[ ! -d "$REPO_DIR/.git" ]]; then
    fail "$REPO_DIR is not a git clone"
else
    ok "git clone: $REPO_DIR"
    if git -C "$REPO_DIR" pull --dry-run --ff-only >/dev/null 2>&1; then
        ok "git pull --ff-only would succeed"
    else
        fail "git pull --dry-run --ff-only failed (diverged or no remote)"
    fi
fi

echo "== check: ./tomat CLI =="
if (cd "$REPO_DIR" && ./tomat --help >/dev/null 2>&1); then
    ok "./tomat --help runs"
else
    fail "./tomat --help failed (venv / direnv broken?)"
fi

echo "== check: gcloud ADC =="
if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    ok "gcloud ADC token mints"
else
    fail "gcloud auth application-default print-access-token failed"
fi

echo "== check: R2 access =="
# Source env so AWS_* + endpoint vars are available for the test call.
# Subshell so we don't leak creds into the parent.
R2_ENDPOINT="${TOMAT_R2_ENDPOINT:-https://43a6f2d588b1483733189d39418ec5be.r2.cloudflarestorage.com}"
R2_BUCKET="${TOMAT_R2_BUCKET:-openathena}"
if ( set -a; . "$ENV_FILE" 2>/dev/null; set +a;
     aws s3 ls "s3://$R2_BUCKET/tomat/" --endpoint-url "$R2_ENDPOINT" >/dev/null 2>&1 ); then
    ok "s3 ls s3://$R2_BUCKET/tomat/ (R2) returned"
else
    fail "s3 ls against R2 failed"
fi

if (( FAILED > 0 )); then
    err ""
    err "$FAILED check(s) failed."
    exit 1
fi

if (( CHECK_ONLY == 1 )); then
    echo ""
    echo "READY"
    exit 0
fi

# ------------------------------------------------------------------
# Install phase
# ------------------------------------------------------------------
echo ""
echo "== install: copy units =="

# Track which services have a new unit file → restart them after reload.
CHANGED_SERVICES=()
CHANGED_TIMERS=()

copy_unit() {
    local name="$1"
    local src="$SYSTEMD_SRC/$name"
    local dst="$SYSTEMD_DST/$name"
    if [[ ! -f "$src" ]]; then
        fail "$src missing"
        return
    fi
    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        echo "  unchanged: $name"
        return
    fi
    sudo install -m 0644 "$src" "$dst"
    echo "  copied:    $name"
    case "$name" in
        *.service) CHANGED_SERVICES+=("$name") ;;
        *.timer)   CHANGED_TIMERS+=("$name") ;;
    esac
}

for svc in "${SERVICES[@]}"; do copy_unit "$svc"; done
for tmr in "${TIMERS[@]}";   do copy_unit "$tmr"; done

# Also make sync scripts executable (in case `git pull` dropped them
# without the +x bit, e.g. on a fresh clone via gh API or zip).
chmod +x "$SCRIPT_DIR/sync/"*.sh "$SCRIPT_DIR/install.sh" 2>/dev/null || true

if (( ${#CHANGED_SERVICES[@]} == 0 && ${#CHANGED_TIMERS[@]} == 0 )); then
    echo "  no unit changes; skipping reload"
else
    echo ""
    echo "== install: daemon-reload =="
    sudo systemctl daemon-reload
fi

echo ""
echo "== install: enable + start timers =="
for tmr in "${TIMERS[@]}"; do
    # `enable --now` is idempotent: makes it survive reboot AND starts it
    # immediately if not already running.
    sudo systemctl enable --now "$tmr" >/dev/null
    echo "  enabled:   $tmr"
done

# Restart any services whose .service file changed. Timers themselves
# don't need restart on file change (daemon-reload picks it up), but a
# .service-file change means the *next* oneshot run should use the new
# unit; explicit restart is a no-op for oneshot Type=oneshot units
# unless one is currently mid-run. We touch them defensively.
if (( ${#CHANGED_SERVICES[@]} > 0 )); then
    echo ""
    echo "== install: restart changed services (best-effort) =="
    for svc in "${CHANGED_SERVICES[@]}"; do
        # `try-restart` skips if not active; for oneshot units that just
        # finished, it's effectively a no-op until the next timer fire,
        # which is exactly what we want.
        sudo systemctl try-restart "$svc" >/dev/null 2>&1 || true
        echo "  try-restart: $svc"
    done
fi

echo ""
echo "INSTALLED"
echo ""
echo "Next steps:"
echo "  journalctl -f -u 'tomat-*' --since '5 min ago'"
echo "  systemctl list-timers 'tomat-*'"
