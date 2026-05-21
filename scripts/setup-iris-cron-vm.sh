#!/usr/bin/env bash
# Idempotent setup for the `tomat-iris-cron` GCE VM (e2-small, us-east1-d).
#
# NB: must be e2-small (2GB), not e2-micro (1GB) — installing wandb +
# pyarrow into the venv OOM-wedges a 1GB VM (swap-thrash so deep that
# even sshd can't get a login shell).
#
# Prereqs (perform from the laptop before SSHing in):
#   1. `gcloud compute scp` the artifacts to ~/ on the VM:
#        - ~/.config/gcloud/application_default_credentials.json  (ADC)
#        - ~/.aws/credentials                                     ([cfo] profile for R2)
#        - ~/.wandb-api-key                                       (single-line WANDB_API_KEY)
#        - scripts/iris-sync.py                                   (iris-state cron worker)
#        - scripts/runs-sync.py                                   (wandb-runs cron worker)
#        - scripts/setup-iris-cron-vm.sh                          (this script)
#   2. `gcloud compute ssh` into the VM and `bash ~/setup-iris-cron-vm.sh`
#
# What this does:
#   - apt-installs python3-venv, git, cron
#   - creates a venv at ~/iris-sync-venv with marin-iris @ pinned SHA
#   - ssh-keygens the key gcloud expects for IAP tunneling
#   - drops the cron entry: minute-by-minute calls to iris-sync.py
#   - leaves logs at ~/iris-sync.log
#
# Re-run safely: each step is idempotent (won't clobber existing keys, venv,
# or crontab entry).
set -euo pipefail

# Pin marin to the same commit the laptop venv has — daily-rotating dev
# wheels in marin's release index drift in ways that broke us in Modal.
MARIN_SHA="674eb2a4f06ab6daf5478ebdceddcd76a277a5eb"
MARIN_REPO="https://github.com/marin-community/marin"

# --- apt deps --------------------------------------------------------------
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-pip git cron

# --- python venv with marin-iris -------------------------------------------
VENV="$HOME/iris-sync-venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip

# marin-iris's dev wheels have a strict `==0.99` pin on marin-finelog that
# the dev-only release index can't satisfy. Install from git at the pinned
# SHA (matching laptop's `direct_url.json`) with --no-deps to dodge the pin.
"$VENV/bin/pip" install --quiet --no-deps \
    "marin-iris @ git+${MARIN_REPO}@${MARIN_SHA}#subdirectory=lib/iris" \
    "marin-finelog @ git+${MARIN_REPO}@${MARIN_SHA}#subdirectory=lib/finelog" \
    "marin-rigging @ git+${MARIN_REPO}@${MARIN_SHA}#subdirectory=lib/rigging"

# Runtime deps for marin-iris + boto3 for the R2 PUT + wandb/pyarrow for runs-sync.
# fsspec/gcsfs/s3fs MUST share a minor version (their pins are strict).
"$VENV/bin/pip" install --quiet \
    boto3 wandb \
    click cloudpickle connect-python google-auth google-cloud-tpu \
    grpcio httpx humanfriendly pydantic pyjwt pyyaml starlette tabulate \
    "typing-extensions>=4.0" "uvicorn[standard]" zstandard \
    duckdb protobuf pyarrow numpy \
    "fsspec==2025.3.0" "gcsfs==2025.3.0" "s3fs==2025.3.0"

# --- gcloud SSH key (for IAP tunneling into Marin's controller VM) ---------
# `iris job list` shells out to `gcloud compute ssh --tunnel-through-iap …`
# which requires this exact key path. gcloud auto-registers the .pub with
# the project metadata on first use.
if [ ! -f "$HOME/.ssh/google_compute_engine" ]; then
    mkdir -p "$HOME/.ssh"
    ssh-keygen -t rsa -f "$HOME/.ssh/google_compute_engine" \
        -C "tomat-iris-cron@$(hostname)" -N ''
fi

# --- script placements + perms ---------------------------------------------
chmod +x "$HOME/iris-sync.py" "$HOME/runs-sync.py"

# wandb API key file (single-line) — runs-sync.py reads this if WANDB_API_KEY
# isn't already in env. Cron strips env, so the file is the path of least
# resistance.
if [ ! -f "$HOME/.wandb-api-key" ]; then
    echo "[setup] WARN: ~/.wandb-api-key missing — runs-sync.py will fail."
    echo "[setup]       scp it from your laptop: gcloud compute scp ~/.wandb-api-key \$VM:~/"
fi
chmod 600 "$HOME/.wandb-api-key" 2>/dev/null || true

# --- cron entries ----------------------------------------------------------
# Replace any prior `iris-sync.py` / `runs-sync.py` lines, idempotently.
TMP_CRON=$(mktemp)
crontab -l 2>/dev/null | grep -vE 'iris-sync\.py|runs-sync\.py' > "$TMP_CRON" || true
# iris-sync every 5 min: each iris call rebuilds its IAP tunnel (~60s typical,
# can spike to 200s); 3 prefixes serially fits comfortably in a 5-min cycle.
echo "*/5 * * * * $VENV/bin/python $HOME/iris-sync.py >> $HOME/iris-sync.log 2>&1" >> "$TMP_CRON"
# runs-sync every 1 min, flock-guarded: no IAP tunnel needed (wandb API +
# R2 PUT only), but a sync of a large run's full history can take 40-95s
# (it re-fetches all rows each tick), so `flock -n` skips the tick if the
# prior runs-sync is still running — otherwise overlapping procs pile up
# and can re-OOM the VM once several runs are active at once.
echo "* * * * * /usr/bin/flock -n /tmp/runs-sync.lock $VENV/bin/python $HOME/runs-sync.py >> $HOME/runs-sync.log 2>&1" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

# Make sure the cron daemon is running on first boot.
sudo systemctl enable --now cron 2>/dev/null || sudo service cron start

# --- sanity-check by running each script once now --------------------------
echo "[setup] one-off iris-sync.py verification:"
"$VENV/bin/python" "$HOME/iris-sync.py" || {
    echo "[setup] iris-sync.py failed on first run — see output above"
    exit 1
}
if [ -f "$HOME/.wandb-api-key" ]; then
    echo "[setup] one-off runs-sync.py verification:"
    "$VENV/bin/python" "$HOME/runs-sync.py" || {
        echo "[setup] runs-sync.py failed on first run — see output above"
        exit 1
    }
fi

echo
echo "[setup] DONE. crontab:"
crontab -l
echo
echo "[setup] tail -f ~/iris-sync.log ~/runs-sync.log to watch cron runs"
