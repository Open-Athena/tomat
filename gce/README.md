# `gce/` — VM-hosted cron units for tomat dashboard freshness

systemd timer + sync-script bundle that lives on the `tomat-iris-cron`
GCE VM (Debian 12, us-east1-d, user `ryan_williams_openathena_ai`) and
keeps the runs dashboard's R2 sidecars (`raw.parquet`, `manifest.json`,
`cost.json`, `eval.json`, `pending-fires.json`, `modal-state.json`)
up to date without anyone hitting `tomat runs sync` by hand. See
[`specs/55-gce-cron-and-vm-repo-layout.md`][spec] for the rationale +
cadence rubric.

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
    ├── modal-sync.sh         # `tomat modal sync --all-pending`
    ├── evals-sync.sh         # `tomat evals sync`
    ├── self-update.sh        # `git fetch + reset --hard @{u} && install.sh`
    └── heartbeat.sh          # writes cron-heartbeat.json to R2
```

## Cadence rubric

| Timer | Cadence | What it does |
|---|---|---|
| `tomat-runs-sync-active.timer` | 2 min | sync runs whose wandb `run.state == "running"` OR whose iris job is `RUNNING`/`PENDING`/`BUILDING` |
| `tomat-runs-sync-stale.timer` | 30 min | full-sweep `tomat runs sync` (empty substr = all runs); catches GC + late evals on finished runs |
| `tomat-modal-sync.timer` | 1 min | refresh `modal-state.json` for the Modal-state badge |
| `tomat-evals-sync.timer` | 10 min | pull m-eval JSONs newly landed in GCS |
| `tomat-self-update.timer` | 5 min | `git fetch && git reset --hard @{u}` + re-exec `install.sh` (zero-ssh deploy loop) |

The active/stale split is a load-shed optimization, not a correctness fix.
The current (pre-systemd) crontab runs the equivalent of `runs-sync-stale`
every minute over **all** runs and that works fine — if the split turns
out to misbehave (e.g. an "active" run that nobody syncs because both the
wandb state and the iris state look idle), fall back to running
`runs-sync-stale.sh` at a tighter cadence and disable
`tomat-runs-sync-active.timer`.

## First-time install on the VM

```bash
# 1. SSH to the monitor VM. User is `ryan_williams_openathena_ai`.
gcloud compute ssh tomat-iris-cron --zone us-east1-d

# 2. Clone the repo (skip if already present at ~/tomat). HTTPS works —
#    the tomat repo is public, no GH auth needed on the VM.
git clone https://github.com/Open-Athena/tomat.git ~/tomat

# 3. Sanity-check creds + tools before touching /etc/systemd.
~/tomat/gce/install.sh --check

# 4. If `--check` prints `READY`, install + enable the timers.
~/tomat/gce/install.sh

# 5. Watch the first ~30 min of journal output to confirm syncs land.
journalctl -f -u 'tomat-*'
```

After that, edits land via the laptop's normal `git push` → main →
`tomat-self-update.timer` (5 min) `git fetch && git reset --hard @{u}`
+ re-installs — no manual ssh required for routine deploys. (We use
`git reset --hard @{u}` instead of `git pull` because pull fails on a
dirty work tree, which on a cron VM could wedge the self-update loop
forever; `reset --hard` recovers automatically from any local crud.)

## Coexistence with the legacy crontab (additive cutover)

The pre-existing `crontab -e` has two entries that this work does NOT
remove:

```cron
*/5 * * * * /home/ryan_williams_openathena_ai/iris-sync-venv/bin/python \
            /home/ryan_williams_openathena_ai/iris-sync.py \
            >> /home/ryan_williams_openathena_ai/iris-sync.log 2>&1
*   * * * * /usr/bin/flock -n /tmp/runs-sync.lock \
            /home/ryan_williams_openathena_ai/iris-sync-venv/bin/python \
            /home/ryan_williams_openathena_ai/runs-sync.py \
            >> /home/ryan_williams_openathena_ai/runs-sync.log 2>&1
```

These were `scp`'d as bare files (no `~/tomat` clone). They continue to
run alongside the new systemd timers during the cutover — there's no
duplicate-write hazard since both write idempotent R2 objects whose
content is determined by upstream wandb / GCS state, not by which path
wrote first.

When the systemd timers have been observed healthy for a week or two,
manually clean up:

```bash
crontab -e   # delete the two */5 and * * * lines above
rm ~/iris-sync.py ~/runs-sync.py ~/iris-sync.log ~/runs-sync.log
```

(Log rotation is the reason this cleanup matters: the legacy `>> *.log`
files are already 13 MB / 1.6 MB. The new systemd units use
`StandardOutput=journal`, so disk usage is bounded by `journald`'s
config and we don't need a `logrotate.d` snippet.)

## Required auth on the VM

`install.sh --check` will refuse to proceed unless all of the following
already exist. These paths match what the legacy `iris-sync.py` /
`runs-sync.py` already read; no `scp` required if those scripts work.

| Path | What | How `--check` validates |
|---|---|---|
| `~/.aws/credentials` | R2 creds in `[cfo]` profile | `aws configure list --profile cfo` |
| `~/.config/gcloud/application_default_credentials.json` | GCS + iris ADC | file readable |
| `~/.wandb-api-key` | single-line raw wandb key | file non-empty |
| `~/iris-sync-venv` | Python venv with wandb, pyarrow, gcsfs, … | `python -c 'import wandb'` |
| `~/tomat/.git` | this repo, cloned | dir present |

No values are ever printed by `--check`.

### Why a derivative `~/.wandb-api-key.env`?

systemd's `EnvironmentFile=` expects `KEY=VAL` lines, but the existing
Python scripts read `~/.wandb-api-key` as a raw single-line key. To avoid
breaking either, `install.sh` synthesizes `~/.wandb-api-key.env`
(`WANDB_API_KEY=<value>\n`, mode 0600) on every run, and the systemd
services point their `EnvironmentFile=` at the derivative. The raw file
is never modified.

### Why not `/etc/default/tomat-monitor`?

An earlier draft of this installer wanted a sysadmin-style
`/etc/default/tomat-monitor` env file holding the whole bag of creds.
Skipped because the VM already has the creds in the right places on
disk for the legacy scripts — duplicating them in `/etc/default/`
would just be a third place to rotate when the wandb key changes.

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

## What to do when a timer is failing

1. `systemctl list-timers 'tomat-*'` to confirm which units are failing
   (look for "PASSED" rows with NEXT in the future = healthy; absent =
   not enabled; failed status visible via `systemctl status <unit>`).
2. `journalctl -u <unit>.service --since '30 min ago'` for the last few
   stderr blocks. Each oneshot run is a self-contained log block.
3. Common causes + remediation:
   - **`wandb` 401 / auth refused**: `~/.wandb-api-key` got rotated.
     Update the file, re-run `~/tomat/gce/install.sh` (which rewrites
     `~/.wandb-api-key.env`).
   - **`gcloud` ADC token expired**:
     `gcloud auth application-default login` (interactive — needs
     `--no-browser` flow over SSH).
   - **`./tomat` exits with `ModuleNotFoundError`**: someone updated the
     `tomat/` deps without bumping `~/iris-sync-venv`. SSH in and
     `~/iris-sync-venv/bin/pip install -r` the missing wheel. Track the
     real fix as repo follow-up; venv mgmt is a known gap (see "Tech
     debt" below).
   - **`runs-sync-active.sh` says "no active runs" every tick**: snapshot
     endpoint may be stale or the iris/wandb-state schema may have
     drifted. Hit `curl
     https://tomat-runs-api.openathena.workers.dev/api/runs-snapshot.json`
     by hand and inspect the `iris.jobs[*].state` and
     `runs[*].run.state` fields against the jq filter in
     `runs-sync-active.sh`.
   - **Self-update loop refuses to pull**: it's `git fetch && git reset
     --hard @{u}`, which is robust against dirty work trees, but a
     `git fetch` failure (network, GH-rate-limit, bad credential helper)
     will block the loop. Check `journalctl -u
     tomat-self-update.service --since '1 hour ago'`.
4. If `tomat-self-update.timer` is the failing unit, the auto-deploy
   loop is broken and a hand-fix won't propagate. SSH in,
   `cd ~/tomat && git fetch && git reset --hard @{u}` by hand, then
   `~/tomat/gce/install.sh` to re-stamp the units.

## Adding a new timer

1. Drop `gce/systemd/tomat-foo.{service,timer}` (use one of the existing
   pairs as a template — same `User=`, `Environment=` block,
   `EnvironmentFile=`, `WorkingDirectory=`).
2. Drop `gce/sync/foo.sh` doing the actual work (must be `+x`).
3. Append `tomat-foo.timer` to `TIMERS=()` and `tomat-foo.service` to
   `SERVICES=()` in `install.sh`.
4. Commit + push to main. Within 5 min, `tomat-self-update.timer` pulls
   it and `install.sh` enables it automatically — no ssh needed.

## Heartbeat → dashboard chip

`runs-sync-active.sh`'s tail invokes `heartbeat.sh`, which writes
`{ts, host, last_run_count, last_failure_count}` JSON to
`s3://openathena/tomat/cron-heartbeat.json` (via the R2 endpoint, with
`--profile cfo`). The runs dashboard surfaces this via
`fetchCronHeartbeat()` in `site/src/runs/api.ts` and renders a
"cron Xm ago" chip in the header paragraph (crimson once the heartbeat
is > 5 min stale). If the chip disappears or goes red, the VM is silent
and badges / sparklines will start drifting.

## Tech debt — follow-up

- `scripts/runs-sync.py` and `tomat runs sync` (subcmd) re-implement
  the same wandb-history-to-R2 path. The legacy crontab fires the
  Python script; the new systemd unit fires the CLI subcmd. Pick one
  and delete the other — recommended direction is "subcmd, since it's
  what humans run and shares code with the rest of the CLI", with
  `scripts/runs-sync.py` graduated to a single-line wrapper or
  removed. Out of scope for this systemd-units PR.
- `~/iris-sync-venv` is provisioned by `scripts/setup-iris-cron-vm.sh`
  with a hand-curated pip list. Anytime `tomat/pyproject.toml` adds a
  runtime dep, the VM needs a manual `pip install`. Long-term, the VM
  should `uv sync` against the same lockfile the laptop uses.

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

- The legacy `iris-sync.py` / `runs-sync.py` crontab entries are left in
  place — this work is additive. See "Coexistence" above for cleanup
  guidance.
- `tomat evals fire` is still manual; only the post-fire **sync** of
  completed-eval JSONs is automated here.

[spec]: https://github.com/Open-Athena/tomat/blob/main/specs/55-gce-cron-and-vm-repo-layout.md
