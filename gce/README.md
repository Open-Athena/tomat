# `gce/` — VM-hosted cron units for tomat dashboard freshness

systemd timer + sync-script bundle that lives on the `tomat-monitor` GCE
VM and keeps the runs dashboard's R2 sidecars (`raw.parquet`,
`manifest.json`, `cost.json`, `eval.json`, `pending-fires.json`,
`modal-state.json`) up to date without anyone hitting `tomat runs sync`
by hand. See [`specs/55-gce-cron-and-vm-repo-layout.md`][spec] for the
rationale + cadence rubric.

## What's in here

```
gce/
├── install.sh                # one-shot installer; idempotent; --check mode
├── systemd/
│   ├── tomat-runs-sync-active.{service,timer}    # every 2 min
│   ├── tomat-runs-sync-stale.{service,timer}     # every 30 min
│   ├── tomat-modal-sync.{service,timer}          # every 1 min
│   ├── tomat-evals-sync.{service,timer}          # every 10 min
│   └── tomat-self-update.{service,timer}         # every 5 min
└── sync/
    ├── runs-sync-active.sh   # iterates active runs (curl + jq on snapshot)
    ├── runs-sync-stale.sh    # full-sweep `tomat runs sync`
    ├── modal-sync.sh         # `tomat modal sync`
    ├── evals-sync.sh         # `tomat evals sync`
    ├── self-update.sh        # `git pull --ff-only && install.sh`
    └── heartbeat.sh          # writes cron-heartbeat.json to R2
```

## Cadence rubric

| Timer | Cadence | What it does |
|---|---|---|
| `tomat-runs-sync-active.timer` | 2 min | sync the small set of "currently moving" runs — wandb `state == running` OR last log < 15 min ago |
| `tomat-runs-sync-stale.timer` | 30 min | full-sweep `tomat runs sync` — catches GC + late evals on finished runs |
| `tomat-modal-sync.timer` | 1 min | refresh `modal-state.json` for the Modal-state badge |
| `tomat-evals-sync.timer` | 10 min | pull m-eval JSONs newly landed in GCS |
| `tomat-self-update.timer` | 5 min | `git pull --ff-only` + re-exec `install.sh` (zero-ssh deploy loop) |

## First-time install on the VM

```bash
# 1. SSH to the monitor VM (service user is `ryan`).
ssh tomat-monitor

# 2. Clone the repo (skip if already present at ~/tomat).
git clone https://github.com/Open-Athena/tomat.git ~/tomat

# 3. Sanity-check creds + tools before touching /etc/systemd.
~/tomat/gce/install.sh --check

# 4. If `--check` prints `READY`, install + enable the timers.
~/tomat/gce/install.sh

# 5. Watch the first ~30 min of journal output to confirm syncs land.
journalctl -f -u 'tomat-*'
```

After that, edits land via the laptop's normal `git push` → main →
`tomat-self-update.timer` (5 min) pulls + reinstalls — no manual ssh
required for routine deploys.

## Required env file (`/etc/default/tomat-monitor`)

`install.sh --check` will refuse to proceed unless this file exists and
contains all four required keys:

```ini
# /etc/default/tomat-monitor — sensitive; never checked into git.
WANDB_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/home/ryan/.config/gcloud/application_default_credentials.json
```

The file is loaded by every `*.service` unit via
`EnvironmentFile=-/etc/default/tomat-monitor`. The leading `-` makes the
EnvironmentFile optional from systemd's PoV; `install.sh --check`
enforces presence so we don't ship silent-partial-sync units.

## Debugging — journalctl recipes

```bash
# Tail every tomat unit, live:
journalctl -f -u 'tomat-*'

# Last hour of one unit:
journalctl -u tomat-runs-sync-active.service --since '1 hour ago'

# Just the most-recent run of one unit (oneshot units emit one block per fire):
journalctl -u tomat-runs-sync-active.service -n 100 --no-pager

# Failures only (units that exited non-zero):
journalctl -u 'tomat-*' --since '1 day ago' -p err

# When did each timer last fire, and when's the next one?
systemctl list-timers 'tomat-*'

# Force-fire one unit manually (doesn't reset the timer):
sudo systemctl start tomat-runs-sync-active.service

# Disable one unit without disabling siblings:
sudo systemctl disable --now tomat-runs-sync-stale.timer
```

## Adding a new timer

1. Drop `gce/systemd/tomat-foo.{service,timer}` (use one of the existing
   pairs as a template — same `User=ryan`, `EnvironmentFile=`,
   `WorkingDirectory=`).
2. Drop `gce/sync/foo.sh` doing the actual work (must be `+x`).
3. Append `tomat-foo.timer` to `TIMERS=()` and `tomat-foo.service` to
   `SERVICES=()` in `install.sh`.
4. Commit + push to main. Within 5 min, `tomat-self-update.timer` pulls
   it and `install.sh` enables it automatically — no ssh needed.

## Heartbeat → dashboard chip

`runs-sync-active.sh`'s tail invokes `heartbeat.sh`, which writes
`{ts, host, last_run_count, last_failure_count}` JSON to
`s3://openathena/tomat/cron-heartbeat.json`. The runs dashboard surfaces
this as a "cron Xm ago" chip in the header paragraph (crimson once the
heartbeat is > 5 min stale). If the chip disappears or goes red, the VM
is silent and badges / sparklines will start drifting.

## Why not GitHub Actions / Cloudflare Cron / a Lambda?

- **GHA scheduled workflows**: 5-min minimum cadence + scheduling slop
  + per-job container startup (~30s) makes 2-min runs impractical.
- **Cloudflare Cron Triggers**: 60s execution cap; our `tomat runs sync`
  routinely spans tens of seconds per run and the active-sync iterates a
  variable list.
- **Lambda + EventBridge**: viable, but the VM already has gcloud +
  iris-sync wired and we want one stable host for the iris-sync unit to
  share with the new units rather than three-way splitting state across
  AWS + GCP + CFW. Single GCE VM with systemd is the lowest-overhead
  option that hits all the cadence + observability goals.

## Out of scope (already covered elsewhere)

- The existing `tomat-iris-sync` unit on the VM is untouched — additive.
- `tomat evals fire` is still manual; only the post-fire **sync** of
  completed-eval JSONs is automated here.

[spec]: https://github.com/Open-Athena/tomat/blob/main/specs/55-gce-cron-and-vm-repo-layout.md
