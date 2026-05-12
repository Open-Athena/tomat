#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb", "pyarrow", "boto3", "click", "utz"]
# ///
"""Pull a single wandb run's history → parquet → R2.

Phase-A scaffolding for the `tomat.oa.dev/runs` dashboard
(`specs/23-runs-dashboard.md`).

Layout in R2:
    s3://openathena/tomat/runs/<run_id>/raw.parquet
    s3://openathena/tomat/runs/<run_id>/manifest.json

The parquet has one row per logged-step (sparse columns for metrics that
weren't logged that step). The manifest tracks schema version + step_max
+ last-poll timestamp, used later by the CFW for cache-staleness checks.

Usage:
    scripts/sync_run_to_r2.py <run_substr>
    scripts/sync_run_to_r2.py cont7k-ext --dry-run
    scripts/sync_run_to_r2.py cont7k-ext --bucket openathena --prefix tomat/runs
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import boto3
import click
import pyarrow as pa
import pyarrow.parquet as pq
import wandb

err = partial(print, file=sys.stderr)

# Schema version — bump on incompatible parquet-schema changes.
SCHEMA_VERSION = 1

# Wandb keys we pull. wandb's `scan_history(keys=…)` returns rows where
# at least one of these is set. Order here is the parquet column order.
METRIC_KEYS = [
    # implicit
    "_step",
    "_timestamp",
    "_runtime",
    # core training
    "train/loss",
    "eval/loss",
    # throughput
    "throughput/mfu",
    "throughput/tokens_per_second",
    "throughput/duration",
    # NMAE / NEMD (val_200, val_400b for any older runs)
    "eval/mat_nmae/val_200/mean",
    "eval/mat_nmae/val_200/median",
    "eval/mat_nmae/val_200/p99",
    "eval/mat_nemd/val_200/mean",
    "eval/mat_nemd/val_200/median",
    "eval/mat_nemd/val_200/p99",
    # lifecycle (sparse, spikes on each event)
    "lifecycle/trainer_started",
    "lifecycle/sigterm_received",
    "lifecycle/trainer_finished",
    # cluster (sparse, from `tomat preempts watch`)
    "cluster/preemptions",
    "cluster/failures",
    "cluster/preempts_delta",
    "cluster/failures_delta",
    "cluster/preempts_per_hour",
    "cluster/elapsed_min",
]

# pyarrow schema. _step is non-null; everything else nullable.
ARROW_SCHEMA = pa.schema([
    ("_step",                            pa.int64()),
    ("_timestamp",                       pa.float64()),
    ("_runtime",                         pa.float64()),
    ("train/loss",                       pa.float32()),
    ("eval/loss",                        pa.float32()),
    ("throughput/mfu",                   pa.float32()),
    ("throughput/tokens_per_second",     pa.float32()),
    ("throughput/duration",              pa.float32()),
    ("eval/mat_nmae/val_200/mean",       pa.float32()),
    ("eval/mat_nmae/val_200/median",     pa.float32()),
    ("eval/mat_nmae/val_200/p99",        pa.float32()),
    ("eval/mat_nemd/val_200/mean",       pa.float32()),
    ("eval/mat_nemd/val_200/median",     pa.float32()),
    ("eval/mat_nemd/val_200/p99",        pa.float32()),
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


def _r2_client():
    """boto3 S3 client wired to OA's R2, via the `cfo` AWS profile."""
    session = boto3.Session(profile_name="cfo")
    return session.client(
        "s3",
        endpoint_url=(
            f"https://43a6f2d588b1483733189d39418ec5be"
            f".r2.cloudflarestorage.com"
        ),
        region_name="auto",
    )


def _find_run(api: wandb.Api, substr: str) -> "wandb.apis.public.Run":
    """Find a unique run whose name contains SUBSTR, across our tomat projects."""
    for project in ("PrinceOA/tomat-lmq-P19", "PrinceOA/tomat-lmq-P14"):
        matches = [r for r in api.runs(project) if substr in r.name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            err(f"multiple runs match {substr!r} in {project}:")
            for r in matches:
                err(f"  {r.name}")
            sys.exit(1)
    err(f"no run matching {substr!r} in known projects")
    sys.exit(1)


def _scan_history_rows(run) -> list[dict]:
    """Pull every history row, projected to METRIC_KEYS.

    `scan_history(keys=…)` returns rows where **all** keys are set (not
    at-least-one as one might hope) — which would mean ~zero rows with
    our heterogeneous metric list. Pull unfiltered and project."""
    keep = set(METRIC_KEYS)
    rows: list[dict] = []
    for row in run.scan_history():
        projected = {k: v for k, v in row.items() if k in keep}
        # Keep the row if it has anything beyond just `_step`/`_timestamp`/`_runtime`.
        if any(k for k in projected if not k.startswith("_")):
            rows.append(projected)
    return rows


def _rows_to_table(rows: list[dict]) -> pa.Table:
    """Convert a list of (sparse) wandb-history dicts into an arrow Table.

    Sorts ascending by `_step`. Missing values become null."""
    if not rows:
        return pa.Table.from_pylist([], schema=ARROW_SCHEMA)
    # Normalize: every row has every column (null where missing).
    cols: dict[str, list] = {f.name: [] for f in ARROW_SCHEMA}
    rows = sorted(rows, key=lambda r: r.get("_step") or -1)
    for r in rows:
        for name in cols:
            cols[name].append(r.get(name))
    arrays = []
    for f in ARROW_SCHEMA:
        arr = pa.array(cols[f.name], type=f.type)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=ARROW_SCHEMA)


def _table_to_parquet_bytes(table: pa.Table) -> bytes:
    """Serialize an arrow Table to in-memory parquet bytes.

    Single row group, ZSTD-compressed. Sorted by _step in the source rows,
    so per-step seeks via row-group statistics are cheap."""
    buf = io.BytesIO()
    pq.write_table(
        table, buf,
        compression="zstd",
        compression_level=6,
        # Force a single row-group: row count is small (≤30k for current
        # runs) and consumers want to read the whole run anyway.
        row_group_size=10**7,
    )
    return buf.getvalue()


def _build_manifest(run, table: pa.Table) -> dict:
    """Run-level metadata snapshot. Mirrors the spec's D1 columns so the
    eventual CFW polling logic can reuse this JSON 1:1."""
    summary = dict(run.summary)
    step_col = table.column("_step")
    ts_col = table.column("_timestamp")
    return {
        "schema_version": SCHEMA_VERSION,
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "id": run.id,
            "name": run.name,
            "project": run.project,
            "entity": run.entity,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
            "tags": list(run.tags or []),
            "group": run.group,
            "config": dict(run.config),
        },
        "summary": {
            # numeric subset only (drop dicts / nested structures wandb
            # sometimes parks under summary keys like `_wandb`, etc.)
            k: v for k, v in summary.items()
            if isinstance(v, (int, float, bool, str)) or v is None
        },
        "history": {
            "rows": table.num_rows,
            "step_min": int(step_col[0].as_py()) if table.num_rows else None,
            "step_max": int(step_col[-1].as_py()) if table.num_rows else None,
            "ts_min": float(ts_col[0].as_py()) if table.num_rows and ts_col[0].is_valid else None,
            "ts_max": float(ts_col[-1].as_py()) if table.num_rows and ts_col[-1].is_valid else None,
        },
    }


@click.command()
@click.option("-b", "--bucket", default="openathena", help="R2 bucket")
@click.option("-p", "--prefix", default="tomat/runs",
              help="R2 key prefix (run id appended)")
@click.option("-n", "--dry-run", is_flag=True,
              help="fetch + assemble, but skip the upload")
@click.option("-o", "--out-dir", default=None,
              help="if set, also write a local copy of the parquet + manifest")
@click.argument("substr")
def main(bucket: str, prefix: str, dry_run: bool, out_dir: str | None, substr: str):
    """Sync wandb run matching SUBSTR → R2."""
    api = wandb.Api()
    run = _find_run(api, substr)
    err(f"[sync] run: {run.name} (state={run.state}, project={run.project})")

    rows = _scan_history_rows(run)
    err(f"[sync] pulled {len(rows)} history rows")
    table = _rows_to_table(rows)
    parquet_bytes = _table_to_parquet_bytes(table)
    err(f"[sync] parquet: {len(parquet_bytes) / 1e6:.2f} MB"
        f" ({table.num_rows} rows × {len(ARROW_SCHEMA)} cols)")

    manifest = _build_manifest(run, table)
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    err(f"[sync] manifest: {len(manifest_bytes)} bytes,"
        f" step range [{manifest['history']['step_min']}, {manifest['history']['step_max']}]")

    if out_dir:
        d = Path(out_dir) / run.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "raw.parquet").write_bytes(parquet_bytes)
        (d / "manifest.json").write_bytes(manifest_bytes)
        err(f"[sync] wrote local copy: {d}/")

    if dry_run:
        err("[sync] dry-run — skipping R2 upload")
        return

    s3 = _r2_client()
    parquet_key = f"{prefix}/{run.name}/raw.parquet"
    manifest_key = f"{prefix}/{run.name}/manifest.json"
    s3.put_object(
        Bucket=bucket, Key=parquet_key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
        CacheControl="public, max-age=60",
    )
    s3.put_object(
        Bucket=bucket, Key=manifest_key,
        Body=manifest_bytes,
        ContentType="application/json",
        CacheControl="public, max-age=60",
    )
    err(f"[sync] uploaded to s3://{bucket}/{prefix}/{run.name}/{{raw.parquet,manifest.json}}")
    print(f"https://{bucket}.43a6f2d588b1483733189d39418ec5be.r2.cloudflarestorage.com/{parquet_key}")


if __name__ == "__main__":
    main()
