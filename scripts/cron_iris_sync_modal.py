#!/usr/bin/env python
"""Modal cron: refresh `s3://openathena/tomat/iris-state.json` every minute.

Mirrors `tomat iris sync` (RPC-only, no SQL on the controller) and uploads
the resulting JSON to OA's R2 bucket so /runs always has fresh data even
when the laptop is asleep.

Why Modal and not CFW: `iris job list` talks to Marin's controller via a
gRPC RPC that requires GCP application-default credentials. CFW workers
have no Python/gcloud/SSH access — they can't make these calls. Modal can.

Cost: ~15s per run × 1440/day = ~21,600 CPU-seconds/day. At Modal's CPU
rate (~$0.000038/s) that's ~$0.80/day. Dial back to */3 * * * * if it's
ever an issue.

================
Setup (one-off):
================

1. **GCP service account** (needs read access to the Marin iris controller's
   `JobService.ListJobs` RPC):

   The simplest path is to use the same identity that already has access
   from the laptop — `ryan@runsascoded.com` — by storing its
   application-default refresh token as a secret. Pros: no Marin ops ask
   needed. Cons: tied to one user.

   ```bash
   # On laptop, after `gcloud auth application-default login`:
   cat ~/.config/gcloud/application_default_credentials.json | \
     modal secret create marin-iris-adc \
       --from-stdin=GOOGLE_APPLICATION_CREDENTIALS_JSON
   ```

   Long-term, request a dedicated read-only SA from Marin ops and swap.

2. **R2 access keys** (Cloudflare R2 token with write to the `openathena`
   bucket — same creds as the local `aws --profile cfo`):

   ```bash
   modal secret create oa-r2-write \
     AWS_ACCESS_KEY_ID=<...> \
     AWS_SECRET_ACCESS_KEY=<...>
   ```

3. **Deploy**:

   ```bash
   modal deploy scripts/cron_iris_sync_modal.py
   ```

   Or invoke once for testing:

   ```bash
   modal run scripts/cron_iris_sync_modal.py::sync_once
   ```
"""

from __future__ import annotations

import modal

# --- Image -----------------------------------------------------------------
# Marin publishes per-package wheels (`marin-iris`, `marin-rigging`,
# `marin-finelog`, ...) as GitHub release assets that pip can treat as
# find-links indexes. `marin-iris` is the only direct dep we need; the
# others are transitive (`marin-iris` → `marin-finelog`, `marin-rigging`).
# Wheels rotate daily (see memory: marin-dev-wheel-rotation), so the build
# pulls fresh each time the image rebuilds — no version pinning here.
# Pin marin's three sub-packages to the same commit the local laptop venv
# was built from (see `direct_url.json` under each `.dist-info/`). Daily-dev
# wheels in marin's GitHub releases drift in ways that broke us — pinning
# gives us identical behavior to the laptop, and the cron only needs to
# rebuild when *we* bump the SHA.
MARIN_SHA = "674eb2a4f06ab6daf5478ebdceddcd76a277a5eb"
MARIN_REPO = "https://github.com/marin-community/marin"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # marin-iris's runtime deps + boto3 for the R2 PUT. We install the
    # marin packages with --no-deps (next step), so the dep set has to be
    # complete here.
    .pip_install(
        "boto3",
        "click>=8.3.1",
        "cloudpickle>=3.1.2",
        "connect-python>=0.9.0",
        # fsspec/gcsfs/s3fs pin to matching minor versions — keep all three
        # in sync (mirror what the laptop venv has).
        "fsspec==2025.3.0",
        "gcsfs==2025.3.0",
        "google-auth>=2.0",
        "google-cloud-tpu>=1.18.0",
        "grpcio>=1.76.0",
        "httpx>=0.28.1",
        "humanfriendly>=10.0",
        "pydantic>=2.0",
        "pyjwt>=2.12.0",
        "pyyaml>=6.0",
        "s3fs>=2024.0.0",
        "starlette>=0.50.0",
        "tabulate>=0.9.0",
        "typing-extensions>=4.0",
        "uvicorn[standard]>=0.23.0",
        "zstandard>=0.22.0",
        "s3fs==2025.3.0",
        # marin-finelog's deps not yet covered above:
        "duckdb>=1.0.0",
        "protobuf>=5.0",
        "pyarrow>=19.0.0",
        # finelog/store/log_namespace.py imports numpy at module level:
        "numpy>=1.26",
    )
    .pip_install(
        f"marin-iris @ git+{MARIN_REPO}@{MARIN_SHA}#subdirectory=lib/iris",
        f"marin-finelog @ git+{MARIN_REPO}@{MARIN_SHA}#subdirectory=lib/finelog",
        f"marin-rigging @ git+{MARIN_REPO}@{MARIN_SHA}#subdirectory=lib/rigging",
        extra_options="--no-deps",
    )
    # Ship the canonical runner roster (scripts/iris_runners.py) into the
    # image so `_sync` can import it at runtime (it isn't a pip dep).
    .add_local_python_source("iris_runners")
)

adc_secret = modal.Secret.from_name("marin-iris-adc")  # GCP ADC for iris RPC
r2_secret = modal.Secret.from_name("oa-r2-write")  # AWS_ACCESS_KEY_ID/SECRET

app = modal.App("tomat-iris-sync-cron", image=image)

# The runner roster / prefix set comes from scripts/iris_runners.py (shipped
# into the image above); `_sync` builds the prefixes from it at runtime.

# Bound on simultaneous iris RPCs (each opens its own tunnel to the
# controller). The cron fires every minute, so fan the per-prefix RPCs out
# concurrently rather than sequentially (8×~60s would overrun the interval).
MAX_SYNC_WORKERS = 4

R2_ACCOUNT_ID = "43a6f2d588b1483733189d39418ec5be"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "openathena"
R2_KEY = "tomat/iris-state.json"


def _materialize_adc() -> str:
    """Drop the ADC JSON to a file and point env vars at it.

    Modal exposes the secret value as the env var name we set when we
    created the secret (`GOOGLE_APPLICATION_CREDENTIALS_JSON`).
    """
    import os

    raw = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    path = "/tmp/adc.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    return path


def _iris_job_list_json(prefix: str, timeout: int = 60) -> list[dict]:
    """Subprocess `iris --cluster=marin job list --prefix PFX --json`.

    Mirrors `tomat:_iris_job_list_json`. RPC-only; no controller SQL.
    """
    import json
    import subprocess
    import sys
    import time

    cmd = ["iris", "--cluster=marin", "job", "list", "--prefix", prefix, "--json"]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # The function docstring promises a slow call just skips this prefix's
        # upload — honor that instead of crashing the whole Modal cron run.
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        print(
            f"[iris-rpc {prefix} {elapsed_ms}ms] TIMEOUT (>{timeout}s, skip)",
            file=sys.stderr,
        )
        return []
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    tag = "SLOW" if elapsed_ms > 10_000 else ""
    print(f"[iris-rpc {prefix} {elapsed_ms}ms] {tag}", file=sys.stderr)
    if r.returncode != 0:
        print(
            f"iris job list failed: rc={r.returncode}\n{r.stderr[-500:]}",
            file=sys.stderr,
        )
        return []
    # Iris CLI emits log lines before the JSON; find the JSON array start.
    out = r.stdout
    i = out.find("[")
    return json.loads(out[i:].strip()) if i >= 0 else []


def _build_payload(rows_by_prefix: dict[str, list[dict]]) -> dict:
    """Mirrors `tomat:iris_sync` payload-building.

    Includes `task_state_counts` (per spec 45) so the dashboard can
    detect the crash-loop pathology where `state == RUNNING` ∧ all tasks
    pending ∧ failures > 0. Without this field the dashboard mis-labels
    crash-loops as healthy RUNNING jobs.
    """
    import datetime

    jobs: dict[str, dict] = {}
    for rows in rows_by_prefix.values():
        for row in rows:
            jid = row["job_id"]
            # Skip task rows (`/ryan/<job>/<task>`); we want parents.
            if jid.count("/") > 2:
                continue
            state_full = row.get("state", "")
            state = state_full.removeprefix("JOB_STATE_") if state_full else "UNKNOWN"
            # Per-state task histogram (keys are iris's friendly task-state
            # strings: 'running', 'pending', 'building', 'completed',
            # 'failed', 'killed', ...). Drop zero entries to keep the payload
            # tight; the dashboard treats missing keys as zero.
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


def _r2_put_json(payload: dict) -> int:
    import json
    import os
    import boto3

    data = json.dumps(payload, indent=1).encode("utf-8")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.put_object(
        Bucket=R2_BUCKET, Key=R2_KEY, Body=data, ContentType="application/json"
    )
    return len(data)


def _sync() -> dict:
    """The actual work — runs in both the cron and one-shot entrypoints."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from iris_runners import iris_prefixes  # shipped via add_local_python_source

    _materialize_adc()
    prefixes = iris_prefixes()
    # Concurrent fan-out (bounded) — sequential per-prefix RPCs would overrun
    # the 1-min cron. ex.map preserves input order, so zip pairs cleanly.
    with ThreadPoolExecutor(max_workers=MAX_SYNC_WORKERS) as ex:
        rows_by_prefix = dict(zip(prefixes, ex.map(_iris_job_list_json, prefixes)))
    payload = _build_payload(rows_by_prefix)
    # Safety: never upload an empty state. A broken iris CLI / missing dep /
    # auth failure should NOT clobber the existing R2 snapshot — let the
    # dashboard keep showing the last-good state until we fix the cron.
    if payload["count"] == 0:
        print(
            "[iris sync] 0 jobs across all prefixes — refusing to "
            "overwrite R2 (likely iris/auth failure)",
            file=sys.stderr,
        )
        return {"jobs": 0, "bytes": 0, "skipped": True}
    n_bytes = _r2_put_json(payload)
    print(
        f"[iris sync] {payload['count']} jobs → "
        f"s3://{R2_BUCKET}/{R2_KEY} ({n_bytes} bytes)",
        file=sys.stderr,
    )
    return {"jobs": payload["count"], "bytes": n_bytes}


@app.function(
    cpu=1,
    memory=1024,
    # 240s ceiling: at MAX_SYNC_WORKERS=4 and len(PREFIXES)=8 the worst-case
    # wall-clock is ceil(8/4) × per-prefix_timeout = 2 × 60s = 120s with zero
    # overhead headroom — equal to the function's own timeout, so a sibling
    # prefix would always race the cron's own cutoff. 240s gives us a clean
    # 2× margin and still leaves the safety check (`count == 0` → skip
    # upload) free to bail if iris is wedged. Bump again if we add more
    # runners (the worst-case scales with ceil(len(PREFIXES) / 4)).
    timeout=240,
    secrets=[adc_secret, r2_secret],
    schedule=modal.Cron("* * * * *"),  # every minute UTC
)
def cron_sync() -> dict:
    return _sync()


@app.function(
    cpu=1,
    memory=1024,
    # 240s ceiling: at MAX_SYNC_WORKERS=4 and len(PREFIXES)=8 the worst-case
    # wall-clock is ceil(8/4) × per-prefix_timeout = 2 × 60s = 120s with zero
    # overhead headroom — equal to the function's own timeout, so a sibling
    # prefix would always race the cron's own cutoff. 240s gives us a clean
    # 2× margin and still leaves the safety check (`count == 0` → skip
    # upload) free to bail if iris is wedged. Bump again if we add more
    # runners (the worst-case scales with ceil(len(PREFIXES) / 4)).
    timeout=240,
    secrets=[adc_secret, r2_secret],
)
def sync_once() -> dict:
    """One-off invocation for setup verification.

    Run: `modal run scripts/cron_iris_sync_modal.py::sync_once`.
    """
    return _sync()


@app.local_entrypoint()
def main():
    print(sync_once.remote())
