#!/usr/bin/env bash
# First-time bring-up for the `tomat-iris-cron` GCE VM (e2-small, us-east1-d).
#
# NB: must be e2-small (2GB), not e2-micro (1GB) — installing wandb +
# pyarrow into the venv OOM-wedges a 1GB VM (swap-thrash so deep that
# even sshd can't get a login shell).
#
# Scope: this script handles ONLY the host-level + venv-level provisioning
# that has to happen before the cron jobs themselves can be installed.
# The actual cron units (systemd timers + sync scripts) live under
# `gce/` in the tomat repo and are installed from there — see the
# `gce/README.md` quickstart. This script's parting message hands off to
# that flow.
#
# Prereqs (perform from the laptop before SSHing in):
#   1. `gcloud compute scp` the artifacts to ~/ on the VM:
#        - ~/.config/gcloud/application_default_credentials.json  (ADC)
#        - ~/.aws/credentials                                     ([cfo] profile for R2)
#        - ~/.wandb-api-key                                       (single-line WANDB_API_KEY)
#        - scripts/setup-iris-cron-vm.sh                          (this script)
#   2. `gcloud compute ssh` into the VM and `bash ~/setup-iris-cron-vm.sh`
#
# What this does:
#   - apt-installs python3-venv, git, jq, curl, awscli, cron
#   - creates a venv at ~/iris-sync-venv with marin-iris @ pinned SHA + the
#     wandb / pyarrow / fsspec deps that the new `tomat runs sync` subcmd needs
#   - ssh-keygens the key gcloud expects for IAP tunneling
#
# After this finishes, the next steps are MANUAL (the operator clones the
# repo and runs `gce/install.sh`):
#
#     git clone https://github.com/Open-Athena/tomat.git ~/tomat
#     ~/tomat/gce/install.sh --check
#     ~/tomat/gce/install.sh
#
# Re-run safely: each step is idempotent (won't clobber existing keys, venv,
# or apt-installed packages).
set -euo pipefail

# Pin marin to the same commit the laptop venv has — daily-rotating dev
# wheels in marin's release index drift in ways that broke us in Modal.
MARIN_SHA="674eb2a4f06ab6daf5478ebdceddcd76a277a5eb"
MARIN_REPO="https://github.com/marin-community/marin"

# --- apt deps --------------------------------------------------------------
# `cron` is kept in the install list because the legacy ~/iris-sync.py and
# ~/runs-sync.py crontab entries still run alongside the new systemd timers
# during the cutover. Once those entries are removed (see gce/README.md
# "Coexistence" section) `cron` can drop off this list.
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-pip git jq curl awscli cron

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

# --- wandb API key file ----------------------------------------------------
# wandb API key file (single-line) — both legacy runs-sync.py and the
# new gce/install.sh's `--check` look for this path. `gce/install.sh`
# additionally synthesizes a sibling `~/.wandb-api-key.env` (KEY=VAL
# form) for systemd's EnvironmentFile=.
if [ ! -f "$HOME/.wandb-api-key" ]; then
    echo "[setup] WARN: ~/.wandb-api-key missing — runs-sync.py and the new"
    echo "[setup]       gce/install.sh --check will fail. scp it from your"
    echo "[setup]       laptop: gcloud compute scp ~/.wandb-api-key \$VM:~/"
fi
chmod 600 "$HOME/.wandb-api-key" 2>/dev/null || true

# Make sure the cron daemon is running on first boot (still needed for the
# legacy iris-sync / runs-sync entries until they're removed).
sudo systemctl enable --now cron 2>/dev/null || sudo service cron start

echo
echo "[setup] DONE. Host-level provisioning complete."
echo
echo "[setup] Next: clone the tomat repo and install the systemd cron units."
echo
echo "    git clone https://github.com/Open-Athena/tomat.git ~/tomat"
echo "    ~/tomat/gce/install.sh --check"
echo "    ~/tomat/gce/install.sh"
echo
echo "[setup] See ~/tomat/gce/README.md for the journalctl recipes and the"
echo "[setup] cutover plan for the legacy ~/iris-sync.py / ~/runs-sync.py"
echo "[setup] crontab entries (additive coexistence during rollout)."
