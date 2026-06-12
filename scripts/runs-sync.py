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
import re
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

# Spec 46 (Phase A/B): per-run MSRP estimates land at
# `runs/<label>/cost.json`. We mirror the per-GPU/per-TPU rate tables from
# `src/tomat/cost.py` here so the cron VM doesn't need the `tomat` package
# installed — `runs-sync.py` is a standalone deploy.
COST_PRICING_TABLE_VERSION = "2026-06-02"
COST_SCHEMA_VERSION = 1
TPU_RATES_USD_PER_CHIP_HR = {
    "v5p": {"on-demand": 4.20, "preemptible": 1.68},
    "v6e": {"on-demand": 2.70, "preemptible": 1.08},
}
MODAL_GPU_RATES_USD_PER_HR = {
    "h100": 3.95, "h200": 4.56, "b200": 6.25, "a100": 2.78, "l40s": 1.95,
}
MODAL_CPU_MEM_ADDER = 1.005
_MODAL_GPU_RE = re.compile(
    r"-(?P<gpu>h100|h200|b200|a100|l40s)x(?P<n>\d+)\b", re.IGNORECASE,
)
_TPU_VARIANT_RE = re.compile(
    r"(?<![a-z0-9])(v5p|v6e|tpu)(\d+)(?![a-z0-9])", re.IGNORECASE,
)
# Levanter logs `throughput/device_kind = jax.devices()[0].device_kind` —
# values like "TPU v6 lite" (v6e), "TPU v5p", "TPU v5 lite" (v5e). Used as
# the wandb-summary fallback for run labels with no chip-count suffix
# (e.g. `kl-s05-tpu-cont-v3`, `train-mg-tz-11-bs256`).
DEVICE_KIND_TO_VARIANT = {
    "tpu v6 lite": "v6e",
    "tpu v5 lite": "v5e",
    "tpu v5p": "v5p",
    "tpu v4": "v4",
}


# Schema mirrors `tomat:76-92`. Bump version on incompatible changes.
# v2 (2026-06-11): `throughput/total_gflops` column was added in v1 (commit
# 0d0a993) without bumping the version, so existing manifests stayed
# schema-compatible → incremental sync never re-fetched the full history
# from wandb → the column landed in zero parquets. Bump forces a full
# backfill on the next cron tick.
RUN_PARQUET_SCHEMA_VERSION = 2
RUN_PARQUET_KEYS = [
    "_step", "_timestamp", "_runtime",
    "global_step",
    "train/loss", "eval/loss",
    "throughput/mfu", "throughput/tokens_per_second", "throughput/duration",
    # Cumulative GFLOPs since training started — monotonic, drives the FLOP
    # x-axis option in the dashboard. Added 2026-06-07; older parquets won't
    # have it (older runs would need a forced full re-sync).
    "throughput/total_gflops",
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
        # Cumulative GFLOPs since training started — monotonic, drives the
        # FLOP x-axis option in the dashboard. Added 2026-06-07.
        ("throughput/total_gflops",          pa.float64()),
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


def _trim_to_latest_trajectory(rows: list[dict]) -> list[dict]:
    """Drop rows from abandoned trajectories before a `global_step` reset.

    Mirrors `tomat:_trim_to_latest_trajectory`. `rows` must be sorted by
    `_step` ascending (the call site does this). Walk in order, find the
    LAST index where `global_step` decreased vs the prior non-null gstep
    row — that's a fresh-from-scratch restart (e.g. `tomat train --force`
    after a previous attempt logged progress). Keep only rows from there
    onward. No restart detected → no change.
    """
    if not rows:
        return rows
    last_restart = 0
    prev_gs = None
    for i, row in enumerate(rows):
        gs = row.get("global_step")
        if gs is None:
            continue
        if prev_gs is not None and gs < prev_gs:
            last_restart = i
        prev_gs = gs
    if last_restart == 0:
        return rows
    print(f"[sync] trajectory reset at row {last_restart}/{len(rows)}: "
          f"dropping {last_restart} pre-restart rows",
          file=sys.stderr)
    return rows[last_restart:]


def _history_block(table) -> dict:
    """The `history` sub-dict of a manifest, derived from the parquet table.

    `last_train_step` = max non-null `global_step` (Levanter's training-step
    counter, logged with each train/loss row). Dashboard uses it for the
    "step X / Yk" card display — has to come from the parquet because
    `summary.global_step` only updates at ckpt boundaries, and `summary._step`
    overcounts (it's wandb's auto log-counter, bumped by cluster/* pings too).
    """
    if table.num_rows == 0:
        return {"rows": 0, "step_min": None, "step_max": None,
                "last_train_step": None,
                "ts_min": None, "ts_max": None, "last_row": {}}
    step_col = table.column("_step")
    ts_col = table.column("_timestamp")
    last_train_step: int | None = None
    if "global_step" in table.schema.names:
        gs_arr = table.column("global_step").to_pylist()
        for v in reversed(gs_arr):
            if v is not None:
                last_train_step = int(v)
                break
    # Build a "last non-null per column" row from the parquet's tail. Walks
    # backward up to N rows; this is fine even when the trailing rows are
    # cluster/* pings (train/loss is None there — keep walking). Used by
    # `_build_manifest` to backfill summary fields for runs whose wandb
    # summary-flush didn't include them.
    last_row: dict[str, float | int | str] = {}
    tail_rows = min(table.num_rows, 200)
    if tail_rows > 0:
        tail = table.slice(table.num_rows - tail_rows, tail_rows)
        for col_name in tail.schema.names:
            if col_name.startswith("_") or col_name == "global_step":
                continue
            col = tail.column(col_name).to_pylist()
            for v in reversed(col):
                if v is not None:
                    last_row[col_name] = v
                    break
    return {
        "rows": table.num_rows,
        "step_min": int(step_col[0].as_py()),
        "step_max": int(step_col[-1].as_py()),
        "last_train_step": last_train_step,
        "ts_min": float(ts_col[0].as_py()) if ts_col[0].is_valid else None,
        "ts_max": float(ts_col[-1].as_py()) if ts_col[-1].is_valid else None,
        "last_row": last_row,
    }


def _build_manifest(primary, history: dict) -> dict:
    """Assemble the manifest dict from a wandb Run + a `history` block."""
    summary = dict(primary.summary)
    # Backfill missing throughput/training-stat fields from history's last
    # non-null row. wandb flushes summary AT RUN-END; if the trainer process
    # ends in `failed` state mid-flight (host crash, OOM-kill, etc.), the
    # auto-flushed summary is missing these fields → dashboard cards lose
    # tr / MFU / EF / $/EF and the cost computer falls back to $0 because
    # it can't infer hardware from `throughput/device_kind`. Sample the last
    # history row for each missing key. (`primary.summary` always has the
    # cron-written `cluster/*` and the basic wandb `_runtime`/`_step`/
    # `_timestamp`, so those don't need backfilling.)
    _BACKFILL_KEYS = (
        "train/loss", "throughput/mfu", "throughput/p50_mfu",
        "throughput/tokens_per_second", "throughput/duration",
        "throughput/total_gflops", "throughput/total_tokens",
        "throughput/device_kind", "throughput/theoretical_flops_per_device",
        "throughput/theoretical_flops", "throughput/flops_per_example",
        "num_devices", "parameter_count",
    )
    last_row = history.get("last_row") if isinstance(history, dict) else None
    if last_row:
        for k in _BACKFILL_KEYS:
            if k not in summary and last_row.get(k) is not None:
                summary[k] = last_row[k]
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
        # Refresh cost.json too so the chip keeps ticking on in-flight runs
        # — wallclock keeps growing even when no new history rows landed.
        try:
            _compute_and_upload_cost(s3, run_name, manifest)
        except Exception as e:
            err(f"  [{run_name}] cost compute failed (non-fatal): "
                f"{type(e).__name__}: {e}")
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
    rows = _trim_to_latest_trajectory(rows)

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
    # Spec 46: keep cost.json fresh on the same cadence as manifest. Cheap
    # (Modal: no extra R2 read; TPU: one iris-state.json fetch — but we
    # accept the cost since the chip on /runs reads from runs-snapshot,
    # which itself reads cost.json). Failure here MUST not block the
    # manifest write — cost is secondary.
    try:
        _compute_and_upload_cost(s3, run_name, manifest)
    except Exception as e:
        err(f"  [{run_name}] cost compute failed (non-fatal): "
            f"{type(e).__name__}: {e}")
    return ("ok", table.num_rows, len(parquet_bytes))


# ── cost.json (spec 46) — Modal + TPU MSRP from manifest ─────────────────


def _parse_modal_gpu(label):
    """Return (gpu_lc, num_gpus) parsed from a run's `-h200x8` suffix, or None."""
    m = _MODAL_GPU_RE.search(label)
    if not m:
        return None
    return (m.group("gpu").lower(), int(m.group("n")))


def _parse_tpu_variant(label):
    """Return (family_lc, num_chips) from a TPU run-name suffix, or None.

    Matches `v5p16`, `v6e32`, `tpu16` (shorthand → v6e).
    """
    m = _TPU_VARIANT_RE.search(label)
    if not m:
        return None
    family = m.group(1).lower()
    chips = int(m.group(2))
    if family == "tpu":
        family = "v6e"
    return (family, chips)


def _parse_tpu_variant_from_manifest(manifest):
    """Read `(family, num_chips)` from `summary.throughput/device_kind` +
    `summary.num_devices`. Used when the run name has no chip-count suffix
    but the wandb run is reporting hardware (any Levanter TPU run is).
    Mirrors `tomat.cost.parse_tpu_hardware_from_manifest` — kept inline so
    the cron VM doesn't need the `tomat` package installed.
    """
    summary = (manifest or {}).get("summary") or {}
    kind = summary.get("throughput/device_kind")
    n = summary.get("num_devices")
    if not isinstance(kind, str) or not isinstance(n, int) or n <= 0:
        return None
    family = DEVICE_KIND_TO_VARIANT.get(kind.strip().lower())
    if family is None:
        return None
    return (family, n)


def _modal_segment_from_manifest(label, manifest):
    """Modal cost segment from `history.ts_max - ts_min`. None when:
       - label has no Modal GPU suffix
       - GPU family not in our rate table
       - manifest has no usable wallclock signal
    """
    parsed = _parse_modal_gpu(label)
    if parsed is None:
        return None
    gpu, n = parsed
    per_gpu = MODAL_GPU_RATES_USD_PER_HR.get(gpu)
    if per_gpu is None:
        return None
    rate = per_gpu * n * MODAL_CPU_MEM_ADDER
    hist = (manifest or {}).get("history") or {}
    ts_min, ts_max = hist.get("ts_min"), hist.get("ts_max")
    src = "wandb history ts span"
    wallclock_sec = None
    started_at_ms = finished_at_ms = None
    if (isinstance(ts_min, (int, float)) and isinstance(ts_max, (int, float))
            and ts_max > ts_min):
        wallclock_sec = float(ts_max - ts_min)
        started_at_ms = int(ts_min * 1000)
        finished_at_ms = int(ts_max * 1000)
    else:
        summary = (manifest or {}).get("summary") or {}
        rt = summary.get("_runtime")
        if isinstance(rt, (int, float)) and rt > 0:
            wallclock_sec = float(rt)
            ts = summary.get("_timestamp")
            if isinstance(ts, (int, float)):
                finished_at_ms = int(ts * 1000)
                started_at_ms = int((ts - rt) * 1000)
            src = "wandb summary._runtime"
    if wallclock_sec is None or wallclock_sec < 1.0:
        return None
    msrp = (wallclock_sec / 3600.0) * rate
    return {
        "kind": "modal", "label": f"{gpu.upper()}×{n}",
        "wallclock_sec": round(wallclock_sec, 3),
        "rate_per_hr_usd": rate,
        "msrp_usd": round(msrp, 4),
        "source": src,
        "started_at_ms": started_at_ms, "finished_at_ms": finished_at_ms,
    }


def _tpu_segment_from_manifest(label, manifest):
    """TPU cost segment using wandb-tracked wallclock × the *preemptible* rate.

    The cron has no iris-state visibility (that's iris-sync.py's job, and
    the two scripts only share R2). Assuming preemptible is the safe
    direction: it UNDERESTIMATES vs on-demand for reserved/serving slices,
    which is the right direction for a public MSRP chip — matches
    `tomat.cost.detect_allocation_class`'s default.

    Returns `(segment_dict, source_note_or_None)` so the caller can record
    a note when the hardware was inferred from the wandb summary fallback
    rather than the run name itself.
    """
    parsed = _parse_tpu_variant(label)
    source_note = None
    if parsed is None:
        parsed = _parse_tpu_variant_from_manifest(manifest)
        if parsed is not None:
            source_note = (
                f"hardware inferred from wandb summary device_kind: "
                f"{parsed[0]}-{parsed[1]}"
            )
    if parsed is None:
        return None, None
    family, chips = parsed
    rates = TPU_RATES_USD_PER_CHIP_HR.get(family)
    if rates is None:
        return None, None
    rate = rates["preemptible"]
    hist = (manifest or {}).get("history") or {}
    ts_min, ts_max = hist.get("ts_min"), hist.get("ts_max")
    if not (isinstance(ts_min, (int, float)) and isinstance(ts_max, (int, float))
            and ts_max > ts_min):
        return None, None
    wallclock_sec = float(ts_max - ts_min)
    if wallclock_sec < 1.0:
        return None, None
    msrp = (wallclock_sec / 3600.0) * chips * rate
    return {
        "kind": "tpu",
        "label": f"{family}-{chips} preemptible",
        "wallclock_sec": round(wallclock_sec, 3),
        "rate_per_hr_usd": rate * chips,
        "msrp_usd": round(msrp, 4),
        "source": "wandb history ts span (covers all iris re-fires)",
        "started_at_ms": int(ts_min * 1000),
        "finished_at_ms": int(ts_max * 1000),
    }, source_note


def _compute_and_upload_cost(s3, run_name, manifest):
    """Write `runs/<run_name>/cost.json` (spec 46 Phase A/B).

    We try Modal first (the suffix `-h200x8` etc. dominates), then TPU
    (`v5p16` / `v6e16` / `tpu16` suffixes). Runs with no recognisable
    hardware suffix get a placeholder record so the dashboard chip can
    still render — though it'll hide when `msrp_usd === 0`.
    """
    seg = _modal_segment_from_manifest(run_name, manifest)
    notes = []
    if seg is None:
        seg, tpu_note = _tpu_segment_from_manifest(run_name, manifest)
        if tpu_note:
            notes.append(tpu_note)
    else:
        notes.append(
            "Modal MSRP is an estimate from wandb-tracked wallclock × "
            "published per-GPU-hour rates (snapshot pricing); see spec 46."
        )
    breakdown = [seg] if seg is not None else []
    total = round(sum(b.get("msrp_usd") or 0.0 for b in breakdown), 2)
    state = (manifest.get("run") or {}).get("state")
    is_complete = state in {"finished", "crashed", "failed", "killed"}
    record = {
        "schema_version": COST_SCHEMA_VERSION,
        "pricing_table_version": COST_PRICING_TABLE_VERSION,
        "computed_at": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_complete": is_complete,
        "msrp_usd": total,
        "breakdown": breakdown,
        "notes": notes,
    }
    key = f"{R2_PREFIX}/{run_name}/cost.json"
    s3.put_object(
        Bucket=R2_BUCKET, Key=key,
        Body=json.dumps(record, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json", CacheControl="public, max-age=60",
    )


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
