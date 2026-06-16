#!/usr/bin/env python3
"""Cron-invoked iris-state sync — runs on the `tomat-iris-cron` GCE VM.

Standalone (no `tomat` / `marin/` working-dir requirement) so the VM doesn't
need the full repo. Mirrors the data-shape that `tomat iris sync` writes to
`s3://openathena/tomat/iris-state.json`, so the dashboard reads identically.

Deployed by `scripts/setup-iris-cron-vm.sh`. Crontab runs this every minute.

R2 creds: pulled from `~/.aws/credentials [cfo]` (so we don't depend on
cron's stripped env).
GCP creds: `~/.config/gcloud/application_default_credentials.json` (ADC).
"""

from __future__ import annotations

import configparser
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3

HOME = Path.home()
ADC_PATH = HOME / ".config/gcloud/application_default_credentials.json"
AWS_CREDS = HOME / ".aws/credentials"
IRIS_BIN = HOME / "iris-sync-venv/bin/iris"  # populated by setup-iris-cron-vm.sh

# Fan across every known runner's namespace (jobs land under the submitting
# user). Keep in sync with IRIS_KNOWN_USERS in the `tomat` CLI.
USERS = ["ryan", "betsy"]
PREFIXES = [f"/{u}/{p}" for u in USERS for p in ("tomat", "train", "eval", "kl")]

# Bound on simultaneous iris RPCs (each opens its own IAP tunnel to the
# controller). Caps connection spikes while still cutting the every-minute
# cron's wall-clock from len(PREFIXES)×~60s to ~ceil(len/MAX)×~60s.
MAX_SYNC_WORKERS = 4

R2_ACCOUNT_ID = "43a6f2d588b1483733189d39418ec5be"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "openathena"
R2_KEY = "tomat/iris-state.json"


def err(*a):
    print(*a, file=sys.stderr)


def iris_job_list_json(prefix: str, timeout: int = 240) -> list[dict]:
    # 240s ceiling: each `iris` invocation re-establishes its own IAP tunnel
    # to Marin's controller (laptop: 5-6s warm; e2-micro fresh tunnel: ~60s
    # typical, can spike to 200s). The crontab fires every minute and we fan
    # across len(PREFIXES) namespaces, so main() runs these concurrently (see
    # MAX_SYNC_WORKERS) to keep wall-clock under the interval; a slow call just
    # skips the upload — the safety guard in main() keeps the last-good R2
    # snapshot.
    cmd = [
        str(IRIS_BIN),
        "--cluster=marin",
        "job",
        "list",
        "--prefix",
        prefix,
        "--json",
    ]
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    err(
        f"[iris-rpc {prefix} {elapsed_ms}ms]" + (" SLOW" if elapsed_ms > 10_000 else "")
    )
    if r.returncode != 0:
        err(f"iris job list failed: rc={r.returncode}\n{r.stderr[-500:]}")
        return []
    # iris CLI emits log lines before the JSON; locate the array start.
    i = r.stdout.find("[")
    return json.loads(r.stdout[i:].strip()) if i >= 0 else []


def build_payload(rows_by_prefix: dict[str, list[dict]]) -> dict:
    jobs: dict[str, dict] = {}
    for rows in rows_by_prefix.values():
        for row in rows:
            jid = row["job_id"]
            if jid.count("/") > 2:
                # Skip task rows (`/ryan/<job>/<task>`); parents only.
                continue
            state_full = row.get("state", "")
            state = state_full.removeprefix("JOB_STATE_") if state_full else "UNKNOWN"
            # `task_state_counts` is iris's per-state task histogram (keys are
            # `task_state_friendly` strings: 'running', 'pending', 'building',
            # 'completed', 'failed', 'killed', ...). Surface it raw — the
            # dashboard uses it to disambiguate `RUNNING (1r/3p)` cascade-
            # restart loops from genuine all-healthy RUNNING (spec 45). Drop
            # zero entries to keep the payload tight; consumers treat missing
            # keys as zero.
            tsc_raw = row.get("task_state_counts") or {}
            tsc = {k: int(v) for k, v in tsc_raw.items() if int(v) > 0}
            jobs[jid] = {
                "state": state,
                "state_code": 0,
                "preempts": int(row.get("preemption_count") or 0),
                "failures": int(row.get("failure_count") or 0),
                "error": row.get("error") or None,
                "exit_code": row.get("exit_code"),
                "submitted_at_ms": int(
                    (row.get("submitted_at") or {}).get("epoch_ms") or 0
                )
                or None,
                "started_at_ms": int((row.get("started_at") or {}).get("epoch_ms") or 0)
                or None,
                "finished_at_ms": int(
                    (row.get("finished_at") or {}).get("epoch_ms") or 0
                )
                or None,
                "num_tasks": int(row.get("task_count") or 0),
                "task_state_counts": tsc,
            }
    return {
        "schema_version": 1,
        "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "count": len(jobs),
        "jobs": jobs,
    }


def r2_put_json(payload: dict) -> int:
    cp = configparser.ConfigParser()
    cp.read(AWS_CREDS)
    data = json.dumps(payload, indent=1).encode("utf-8")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=cp.get("cfo", "aws_access_key_id"),
        aws_secret_access_key=cp.get("cfo", "aws_secret_access_key"),
        region_name="auto",
    )
    s3.put_object(
        Bucket=R2_BUCKET, Key=R2_KEY, Body=data, ContentType="application/json"
    )
    return len(data)


def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ADC_PATH)
    # Concurrent fan-out (bounded) — sequential 8×~60s would blow the 1-min
    # cron interval. ex.map preserves input order, so zip pairs cleanly.
    with ThreadPoolExecutor(max_workers=MAX_SYNC_WORKERS) as ex:
        rows_by_prefix = dict(zip(PREFIXES, ex.map(iris_job_list_json, PREFIXES)))
    payload = build_payload(rows_by_prefix)
    # Safety: a broken iris CLI or auth failure would zero out the R2 snapshot;
    # the dashboard should keep showing last-good state until the cron is fixed.
    if payload["count"] == 0:
        err(
            "[iris sync] 0 jobs across all prefixes — refusing R2 upload (probable iris/auth failure)"
        )
        sys.exit(1)
    n_bytes = r2_put_json(payload)
    print(
        f"[iris sync] {datetime.datetime.now().isoformat(timespec='seconds')} "
        f"{payload['count']} jobs → s3://{R2_BUCKET}/{R2_KEY} ({n_bytes} bytes)"
    )


if __name__ == "__main__":
    main()
