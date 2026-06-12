"""Per-run parquet history schema + the manifest-summary backfill key set.

Canonical source for the constants that `./tomat` (the CLI) AND
`scripts/runs-sync.py` (the cron) both need to write the SAME parquet
shape + manifest to R2. Spec 56 Cluster 2 + Cluster 3.

Drift between these two writers silently invalidates each other's
manifests / cached parquets — `a02cda7` already paid that tax once
(SCHEMA_VERSION not bumped when a column was added). Keep both
writers importing from here so a new column / backfill key lands in
one diff.

Cron-side note: `scripts/runs-sync.py` runs in a venv that doesn't
necessarily have the `tomat` package pip-installed. It bootstraps
its `sys.path` to point at `src/` before importing.
"""

# Bump when the parquet shape or backfill semantics change in a way
# that invalidates already-uploaded manifests. The version is stored
# in each `manifest.json` and the writers full-rescan from wandb when
# the cached version mismatches.
RUN_PARQUET_SCHEMA_VERSION = 2

# Columns retained in the per-run history parquet. Order is meaningful
# for the arrow schema in `_arrow_schema()` (and the CLI's mirror).
RUN_PARQUET_KEYS = [
    "_step", "_timestamp", "_runtime",
    "global_step",
    "train/loss", "eval/loss",
    "throughput/mfu", "throughput/tokens_per_second", "throughput/duration",
    # Cumulative GFLOPs since trainer boot — drives the dashboard's
    # `?x=flop` x-axis. Added 2026-06-07; v2 of the schema picked it up.
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

# Summary fields the writers backfill from the parquet's history tail
# when wandb's auto-flushed summary is missing them — happens when the
# trainer ends `failed` (host crash, OOM-kill, etc.) and wandb never
# fired its at-clean-exit flush. Without these, the dashboard card
# loses `tr / MFU / EF / $/EF / epoch` and `cost.json` falls back to
# `$0` because the cost computer needs `throughput/device_kind` to
# infer hardware.
#
# bin5 (state=failed at step 50k, 2026-06-12) was the recent victim.
BACKFILL_KEYS = (
    "train/loss",
    "throughput/mfu", "throughput/p50_mfu",
    "throughput/tokens_per_second", "throughput/duration",
    "throughput/total_gflops", "throughput/total_tokens",
    "throughput/device_kind",
    "throughput/theoretical_flops_per_device", "throughput/theoretical_flops",
    "throughput/flops_per_example",
    "num_devices", "parameter_count",
)


def arrow_schema():
    """`pyarrow` schema for the per-run history parquet. Column order
    matches `RUN_PARQUET_KEYS`. Lazy-imports `pyarrow` so callers that
    only need the constants don't pay the import cost.
    """
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
