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
# Authentication is read from the files the legacy crontab-based
# iris-sync/runs-sync already provisioned on the VM:
#   ~/.aws/credentials                              ([cfo] profile for R2)
#   ~/.config/gcloud/application_default_credentials.json   (GCS + iris)
#   ~/.wandb-api-key                                (single-line raw key)
# `--check` validates each one is present; no values are ever printed.
#
# For systemd's `EnvironmentFile=` (which requires KEY=VAL lines, not the
# raw single-line `~/.wandb-api-key` the existing Python scripts read),
# this installer synthesizes `~/.wandb-api-key.env` as a sibling file
# whenever the raw key changes. The raw file is NOT modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_SRC="$SCRIPT_DIR/systemd"
SYSTEMD_DST="/etc/systemd/system"

# VM-local auth paths. Hard-coded to match the live `tomat-iris-cron` VM
# (Debian 12, user `ryan_williams_openathena_ai`). If we ever provision a
# second monitor VM with a different user, factor these out — for now,
# baking them in matches what the systemd units use.
VM_HOME="${TOMAT_VM_HOME:-/home/ryan_williams_openathena_ai}"
WANDB_KEY_RAW="$VM_HOME/.wandb-api-key"
WANDB_KEY_ENV="$VM_HOME/.wandb-api-key.env"
GCLOUD_ADC="$VM_HOME/.config/gcloud/application_default_credentials.json"
AWS_CREDS="$VM_HOME/.aws/credentials"
IRIS_VENV="$VM_HOME/iris-sync-venv"

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

echo "== check: platform =="
# systemd is Linux-only. macOS dev shells should run `--check` and bail
# with a clear message rather than crashing later on missing systemctl.
case "$(uname -s)" in
    Linux) ok "kernel: $(uname -sr)" ;;
    *) fail "requires Linux + systemd (got $(uname -s)); macOS dev hosts cannot install" ;;
esac

echo "== check: repo state =="
# `.git` may be a directory (regular clone) or a file (git worktree
# pointer); `rev-parse` handles both.
if ! git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    fail "$REPO_DIR is not a git clone (need: git clone https://github.com/Open-Athena/tomat.git $VM_HOME/tomat)"
else
    ok "git clone: $REPO_DIR"
fi

echo "== check: ./tomat CLI =="
if [[ -x "$REPO_DIR/tomat" ]]; then
    ok "$REPO_DIR/tomat is executable"
else
    fail "$REPO_DIR/tomat missing or not executable"
fi

echo "== check: iris-sync-venv =="
if [[ ! -d "$IRIS_VENV" ]]; then
    fail "$IRIS_VENV missing (legacy crontab provisions this; see scripts/setup-iris-cron-vm.sh)"
elif [[ ! -x "$IRIS_VENV/bin/python" ]]; then
    fail "$IRIS_VENV/bin/python not executable"
elif ! "$IRIS_VENV/bin/python" -c 'import wandb' >/dev/null 2>&1; then
    fail "wandb not importable from $IRIS_VENV"
else
    ok "$IRIS_VENV/bin/python imports wandb"
fi

echo "== check: wandb api key =="
if [[ ! -r "$WANDB_KEY_RAW" ]]; then
    fail "$WANDB_KEY_RAW missing or unreadable"
elif [[ ! -s "$WANDB_KEY_RAW" ]]; then
    fail "$WANDB_KEY_RAW exists but is empty"
else
    ok "$WANDB_KEY_RAW present (non-empty)"
fi

echo "== check: gcloud ADC =="
if [[ ! -r "$GCLOUD_ADC" ]]; then
    fail "$GCLOUD_ADC missing (run: gcloud auth application-default login)"
else
    ok "$GCLOUD_ADC present"
fi

echo "== check: aws creds + [cfo] profile =="
if [[ ! -r "$AWS_CREDS" ]]; then
    fail "$AWS_CREDS missing"
elif ! AWS_SHARED_CREDENTIALS_FILE="$AWS_CREDS" aws configure list --profile cfo >/dev/null 2>&1; then
    fail "[cfo] profile missing from $AWS_CREDS"
else
    ok "[cfo] profile present in $AWS_CREDS"
fi

echo "== check: PATH tools =="
for tool in curl jq aws git gcloud systemctl; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool: $(command -v "$tool")"
    else
        fail "$tool not on PATH"
    fi
done

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

# Synthesize the systemd-style env file from the raw single-line key, IFF
# the raw key has changed (or the .env doesn't exist yet). The .env file
# is `WANDB_API_KEY=<value>` — what systemd `EnvironmentFile=` expects.
# The raw file is left untouched so the legacy `runs-sync.py` /
# `iris-sync.py` Python scripts (which read it raw) keep working.
echo ""
echo "== install: synthesize ~/.wandb-api-key.env =="
RAW_KEY="$(tr -d '\r\n ' < "$WANDB_KEY_RAW")"
DESIRED_LINE="WANDB_API_KEY=$RAW_KEY"
if [[ -r "$WANDB_KEY_ENV" ]] && [[ "$(cat "$WANDB_KEY_ENV")" == "$DESIRED_LINE" ]]; then
    echo "  unchanged: $WANDB_KEY_ENV"
else
    umask 077
    printf '%s\n' "$DESIRED_LINE" > "$WANDB_KEY_ENV"
    chmod 0600 "$WANDB_KEY_ENV"
    echo "  wrote:     $WANDB_KEY_ENV (mode 0600)"
fi

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
