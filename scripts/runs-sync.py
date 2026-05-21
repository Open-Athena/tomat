#!/usr/bin/env python3
"""Cron-invoked wandb-runs sync — runs on the `tomat-iris-cron` GCE VM.

Mirrors the data-shape `tomat runs sync` writes (raw.parquet + manifest.json)
to `s3://openathena/tomat/runs/<run_name>/`, so the dashboard reads
identically. Only syncs runs that are currently `state=running` or were
active within the last hour — finished runs don't need re-polling.

History is fetched incrementally: each tick pulls only the rows past the
`step_max` already on R2, so a steady-state sync is a few seconds rather
than a full re-scan of the whole run.

Standalone (no `tomat` / `marin/` working-dir requirement). Deploy via
`scripts/setup-iris-cron-vm.sh`. Crontab runs this every minute.

Creds:
- R2: `~/.aws/credentials [cfo]` (boto3, same as iris-sync.py).
- wandb: `~/.wandb-api-key` (single-line) or `WANDB_API_KEY` env.
- GCS not needed — wandb API doesn't touch GCS.
"""
from __future__ import annotations

import configparser
import datetime
import io
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HOME = Path.home()
AWS_CREDS = HOME / ".aws/credentials"
WANDB_API_KEY_PATH = HOME / ".wandb-api-key"

R2_ACCOUNT_ID = "43a6f2d588b1483733189d39418ec5be"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "openathena"
R2_PREFIX = "tomat/runs"

# Mirrors `tomat:64` — primary + archive projects. Add to the list when
# wandb-team migrations happen; finished archive runs don't need polling
# but keeping them here lets us re-sync if their state changes.
WANDB_PROJECTS = [
    "open-athena/tomat-lmq-P19",
    "open-athena/tomat-lmq-P14",
    "open-athena/tomat-lmq-v2-P14",
    "PrinceOA/tomat-lmq-P19",
    "PrinceOA/tomat-lmq-P14",
    "PrinceOA/tomat-lmq-v2-P14",
]

# Only re-sync runs we believe are still progressing. A finished/failed/
# crashed run's history is immutable — re-syncing wastes wandb API calls
# and R2 PUT cost. Heartbeat-fresh runs that aren't state=running are
# rare (just-crashed before the state-flush) but worth catching.
ACTIVE_HEARTBEAT_WINDOW_SEC = 60 * 60  # 1 hour

# Schema mirrors `tomat:76-92`. Bump version on incompatible changes.
RUN_PARQUET_SCHEMA_VERSION = 1
RUN_PARQUET_KEYS = [
    "_step", "_timestamp", "_runtime",
    "global_step",
    "train/loss", "eval/loss",
    "throughput/mfu", "throughput/tokens_per_second", "throughput/duration",
    "eval/mat_nmae/val_200/mean", "eval/mat_nmae/val_200/median", "eval/mat_nmae/val_200/p99",
    "eval/mat_nmae/train_200/mean", "eval/mat_nmae/train_200/median", "eval/mat_nmae/train_200/p99",
    "eval/mat_nemd/val_200/mean", "eval/mat_nemd/val_200/median", "eval/mat_nemd/val_200/p99",
    "eval/mat_nemd/train_200/mean", "eval/mat_nemd/train_200/median", "eval/mat_nemd/train_200/p99",
    "lifecycle/trainer_started", "lifecycle/sigterm_received", "lifecycle/trainer_finished",
    "cluster/preemptions", "cluster/failures",
    "cluster/preempts_delta", "cluster/failures_delta",
    "cluster/preempts_per_hour", "cluster/elapsed_min",
]


def err(*a, **k):
    print(*a, file=sys.stderr, **k)


def setup_wandb_creds():
    """Load WANDB_API_KEY into env from ~/.wandb-api-key if not already set."""
    if os.environ.get("WANDB_API_KEY"):
        return
    if WANDB_API_KEY_PATH.is_file():
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY_PATH.read_text().strip()
        return
    err("WANDB_API_KEY not set and ~/.wandb-api-key missing — aborting")
    sys.exit(1)


def run_parquet_schema():
    """pyarrow schema for the merged history table — matches `tomat:486`."""
    import pyarrow as pa
    return pa.schema([
        ("_step",                            pa.int64()),
        ("_timestamp",                       pa.float64()),
        ("_runtime",                         pa.float64()),
        ("global_step",                      pa.int64()),
        ("train/loss",                       pa.float32()),
        ("eval/loss",                        pa.float32()),
        ("throughput/mfu",                   pa.float32()),
        ("throughput/tokens_per_second",     pa.float32()),
        ("throughput/duration",              pa.float32()),
        ("eval/mat_nmae/val_200/mean",       pa.float32()),
        ("eval/mat_nmae/val_200/median",     pa.float32()),
        ("eval/mat_nmae/val_200/p99",        pa.float32()),
        ("eval/mat_nmae/train_200/mean",     pa.float32()),
        ("eval/mat_nmae/train_200/median",   pa.float32()),
        ("eval/mat_nmae/train_200/p99",      pa.float32()),
        ("eval/mat_nemd/val_200/mean",       pa.float32()),
        ("eval/mat_nemd/val_200/median",     pa.float32()),
        ("eval/mat_nemd/val_200/p99",        pa.float32()),
        ("eval/mat_nemd/train_200/mean",     pa.float32()),
        ("eval/mat_nemd/train_200/median",   pa.float32()),
        ("eval/mat_nemd/train_200/p99",      pa.float32()),
        ("lifecycle/trainer_started",        pa.int8()),
        ("lifecycle/sigterm_received",       pa.int8()),
        ("lifecycle/trainer_finished",       pa.int8()),
        ("cluster/preemptions",              pa.int32()),
        ("cluster/failures",                 pa.int32()),
        ("cluster/preempts_delta",           pa.int32()),
        ("cluster/failures_delta",           pa.int32()),
        ("cluster/preempts_per_hour",        pa.float32()),
        ("cluster/elapsed_min",              pa.float32()),
    ])


def collect_active_runs(api):
    """Yield (entity, run_name, copies) tuples for runs to sync this cycle.

    A `(entity, name)` may have copies across multiple projects (the same
    display_name across the team migration). We group by (entity, name)
    and merge histories — matches `tomat runs sync` semantics, including
    its refusal to merge when copies have incompatible `data.cache_dir`.
    """
    now = time.time()
    # entity → name → list[Run]
    groups: dict[tuple[str, str], list] = {}
    for path in WANDB_PROJECTS:
        try:
            for r in api.runs(path):
                # Active = state=running OR heartbeat fresh.
                is_running = r.state == "running"
                hb = r.heartbeatAt
                hb_fresh = False
                if hb:
                    try:
                        ts = datetime.datetime.fromisoformat(hb.replace("Z", "+00:00")).timestamp()
                        hb_fresh = (now - ts) < ACTIVE_HEARTBEAT_WINDOW_SEC
                    except (ValueError, AttributeError):
                        pass
                if not (is_running or hb_fresh):
                    continue
                groups.setdefault((r.entity, r.name), []).append(r)
        except ValueError as e:
            if "Could not find project" in str(e):
                continue
            raise
    return groups


def _r2_get_json(s3, key: str):
    """GET + JSON-parse an R2 object, or None if it doesn't exist."""
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read())


def _r2_get_parquet_rows(s3, key: str) -> list[dict]:
    """GET an R2 parquet object and return its rows as dicts."""
    import pyarrow.parquet as pq
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read())).to_pylist()


def _history_block(table) -> dict:
    """The `history` sub-dict of a manifest, derived from the parquet table."""
    if table.num_rows == 0:
        return {"rows": 0, "step_min": None, "step_max": None,
                "ts_min": None, "ts_max": None}
    step_col = table.column("_step")
    ts_col = table.column("_timestamp")
    return {
        "rows": table.num_rows,
        "step_min": int(step_col[0].as_py()),
        "step_max": int(step_col[-1].as_py()),
        "ts_min": float(ts_col[0].as_py()) if ts_col[0].is_valid else None,
        "ts_max": float(ts_col[-1].as_py()) if ts_col[-1].is_valid else None,
    }


def _build_manifest(primary, history: dict) -> dict:
    """Assemble the manifest dict from a wandb Run + a `history` block."""
    summary = dict(primary.summary)
    return {
        "schema_version": RUN_PARQUET_SCHEMA_VERSION,
        "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "id": primary.id, "name": primary.name,
            "project": primary.project, "entity": primary.entity,
            "state": primary.state, "url": primary.url,
            "created_at": primary.created_at,
            "tags": list(primary.tags or []), "group": primary.group,
            "config": dict(primary.config),
        },
        "summary": {k: v for k, v in summary.items()
                    if isinstance(v, (int, float, bool, str)) or v is None},
        "history": history,
    }


def sync_one_run(s3, run_name: str, copies: list) -> tuple[str, int, int]:
    """Fetch history incrementally, build manifest + parquet, upload to R2.

    Returns `(status, rows, parquet_bytes)` where status is one of:
    - `"ok"`       — parquet + manifest re-uploaded with fresh history
    - `"nochange"` — no history past R2's `step_max`; only manifest refreshed
    - `"skip"`     — cross-tokenizer collision; nothing uploaded

    `wandb`'s `scan_history` is the slow part (~1s per ~1k rows), so we
    pass `min_step` to pull only the tail past whatever R2 already holds.
    A full re-scan happens only when R2 has nothing for this run, or its
    parquet schema version changed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _cache_dir(rn):
        return (rn.config.get("data") or {}).get("cache_dir")

    cache_dirs = {_cache_dir(r) for r in copies}
    if len(cache_dirs) > 1:
        # Cross-tokenizer collision — same display name but different
        # data.cache_dir. `tomat runs sync` refuses; we should too.
        err(f"  [{run_name}] CROSS-TOKENIZER collision: cache_dirs={cache_dirs}; SKIP")
        return ("skip", 0, 0)

    primary = max(copies, key=lambda r: float(r.summary.get("_timestamp") or 0))
    keep = set(RUN_PARQUET_KEYS)
    parquet_key = f"{R2_PREFIX}/{run_name}/raw.parquet"
    manifest_key = f"{R2_PREFIX}/{run_name}/manifest.json"

    # What does R2 already have? `step_max` from a schema-compatible
    # manifest is the watermark we fetch past; anything else → full scan.
    prev_manifest = _r2_get_json(s3, manifest_key)
    since_step = None
    if prev_manifest and prev_manifest.get("schema_version") == RUN_PARQUET_SCHEMA_VERSION:
        since_step = (prev_manifest.get("history") or {}).get("step_max")

    scan_kwargs = {} if since_step is None else {"min_step": since_step + 1}
    new_rows: list[dict] = []
    for r in copies:
        for row in r.scan_history(**scan_kwargs):
            projected = {k: v for k, v in row.items() if k in keep}
            if any(k for k in projected if not k.startswith("_")):
                new_rows.append(projected)

    if since_step is not None and not new_rows:
        # No history past R2's watermark — the parquet is still current.
        # Run state/summary may still have moved (e.g. running → finished),
        # so refresh the (tiny) manifest and leave the parquet alone.
        prev_history = prev_manifest.get("history") or {}
        manifest = _build_manifest(primary, prev_history)
        s3.put_object(
            Bucket=R2_BUCKET, Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json", CacheControl="public, max-age=60",
        )
        return ("nochange", prev_history.get("rows", 0), 0)

    if since_step is None:
        rows = new_rows
    else:
        # Splice the new tail onto the rows already on R2; dedup on `_step`
        # (defensive — `min_step` already excludes the watermark) keeping
        # the freshly-fetched copy.
        by_step = {row.get("_step"): row
                   for row in _r2_get_parquet_rows(s3, parquet_key)}
        for row in new_rows:
            by_step[row.get("_step")] = row
        rows = list(by_step.values())

    rows.sort(key=lambda d: d.get("_step") if d.get("_step") is not None else -1)

    schema = run_parquet_schema()
    if rows:
        cols = {f.name: [d.get(f.name) for d in rows] for f in schema}
        table = pa.Table.from_arrays(
            [pa.array(cols[f.name], type=f.type) for f in schema], schema=schema,
        )
    else:
        table = pa.Table.from_pylist([], schema=schema)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy", row_group_size=10**7)
    parquet_bytes = buf.getvalue()

    manifest = _build_manifest(primary, _history_block(table))
    s3.put_object(
        Bucket=R2_BUCKET, Key=parquet_key, Body=parquet_bytes,
        ContentType="application/octet-stream", CacheControl="public, max-age=60",
    )
    s3.put_object(
        Bucket=R2_BUCKET, Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json", CacheControl="public, max-age=60",
    )
    return ("ok", table.num_rows, len(parquet_bytes))


def main():
    setup_wandb_creds()
    import wandb  # late-import: requires WANDB_API_KEY already in env

    cp = configparser.ConfigParser()
    cp.read(AWS_CREDS)
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=cp.get("cfo", "aws_access_key_id"),
        aws_secret_access_key=cp.get("cfo", "aws_secret_access_key"),
        region_name="auto",
    )

    api = wandb.Api()
    t0 = time.time()
    groups = collect_active_runs(api)
    err(f"[runs-sync] {len(groups)} active runs to sync "
        f"({(time.time()-t0):.1f}s to enumerate)")

    n_ok = 0
    n_nochange = 0
    n_rows_total = 0
    n_bytes_total = 0
    for (entity, name), copies in groups.items():
        t_run = time.time()
        try:
            status, rows, sz = sync_one_run(s3, name, copies)
        except Exception as e:
            err(f"  [{name}] FAIL: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t_run
        if status == "skip":
            continue
        if status == "nochange":
            n_nochange += 1
            err(f"  [{name}] no new rows ({rows} on R2) ({dt:.1f}s)")
            continue
        n_ok += 1
        n_rows_total += rows
        n_bytes_total += sz
        err(f"  [{name}] {rows} rows, {sz/1e6:.2f} MB ({dt:.1f}s)")

    print(f"[runs-sync] {datetime.datetime.now().isoformat(timespec='seconds')} "
          f"{n_ok} updated, {n_nochange} unchanged / {len(groups)} active runs; "
          f"{n_rows_total} rows, {n_bytes_total/1e6:.1f} MB ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
