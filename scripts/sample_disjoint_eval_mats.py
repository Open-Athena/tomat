#!/usr/bin/env python3
"""Sample disjoint val_400b + train_400b mat-id sets for noise calibration.

Reads existing val_200/train_200 from data/eval_mat_ids.json, lists all
unique mat IDs in the val + train tokenized parquets on GCS, removes the
pinned 200 from each, and samples 400 fresh mats per split (deterministic
seed). Writes back to data/eval_mat_ids.json. Does NOT upload to GCS;
caller does that separately so they can review the diff.

Usage:
    sample_disjoint_eval_mats.py [--n 400] [--seed 42]
"""
from __future__ import annotations

import json
import random
import sys
from functools import partial
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from click import command, option

err = partial(print, file=sys.stderr)

JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_mat_ids.json"
BUCKET = "gs://marin-eu-west4/tomat/tokenized"


def list_mat_ids(label: str, max_shards_per_worker: int | None = None) -> list[str]:
    """Return sorted unique mat IDs from worker-*/shard-*.parquet files.

    Within a worker, shards hold *disjoint* mats (verified empirically:
    worker-00/shard-00000 ∩ worker-00/shard-00001 = ∅), so reading more
    shards per worker grows the universe. `max_shards_per_worker` caps
    that — useful for the train split, which has ~19 shards/worker × 64
    workers = 1216 shards (slow); for sampling 400, 1 shard/worker × 64
    workers = ~4000 mats is plenty.

    Uses `as_completed` for streaming progress (so a slow shard doesn't
    block reporting on faster ones).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fs = fsspec.filesystem("gs")
    if max_shards_per_worker is None:
        paths = fs.glob(f"marin-eu-west4/tomat/tokenized/{label}/worker-*/shard-*.parquet")
        err(f"[{label}] {len(paths)} shards (all)")
    else:
        # Build paths manually so we can cap per-worker. fs.glob on a
        # trailing-slash pattern returns nothing; instead glob a sentinel
        # file (meta.json) per worker to enumerate worker dirs.
        meta_paths = fs.glob(f"marin-eu-west4/tomat/tokenized/{label}/worker-*/meta.json")
        worker_dirs = sorted({p.rsplit("/", 1)[0] for p in meta_paths})
        paths = []
        for wd in worker_dirs:
            shards = sorted(fs.glob(f"{wd}/shard-*.parquet"))[:max_shards_per_worker]
            paths.extend(shards)
        err(f"[{label}] {len(paths)} shards (capped: {len(worker_dirs)} workers × ≤{max_shards_per_worker} each)")

    def read_one(p):
        with fs.open(p, "rb") as f:
            t = pq.read_table(f, columns=["task_id"])
        return set(t["task_id"].to_pylist())

    ids: set[str] = set()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(read_one, p) for p in paths]
        for fut in as_completed(futs):
            ids.update(fut.result())
            done += 1
            if done % 16 == 0 or done == len(paths):
                err(f"[{label}] read {done}/{len(paths)} shards, {len(ids)} unique IDs so far")
    err(f"[{label}] total: {len(ids)} unique mat IDs")
    return sorted(ids)


@command()
@option('-n', '--n-mats', type=int, default=400, help='Mats to sample per split.')
@option('-s', '--seed', type=int, default=42, help='Random seed.')
@option('-T', '--train-label', default='train-full-lmq-v2', help='Train tokenized label.')
@option('-V', '--val-label', default='val-full-lmq-v2', help='Val tokenized label.')
def main(n_mats, seed, train_label, val_label):
    blob = json.loads(JSON_PATH.read_text())
    val_200 = set(blob["val_200"])
    train_200 = set(blob["train_200"])

    # Cap both at 1 shard/worker. Plenty of mats for sampling 400 disjoint
    # (val: 16 workers × 64 mats = 1024; train: 64 workers × 64 = 4096).
    # Reading every val shard was 10+ min on home network — bias from
    # shard-00000-only sampling is mild and not worth that cost.
    val_universe = list_mat_ids(val_label, max_shards_per_worker=1)
    train_universe = list_mat_ids(train_label, max_shards_per_worker=1)

    val_pool = [m for m in val_universe if m not in val_200]
    train_pool = [m for m in train_universe if m not in train_200]
    err(f"val pool (excl val_200): {len(val_pool)} mats")
    err(f"train pool (excl train_200): {len(train_pool)} mats")

    if len(val_pool) < n_mats:
        err(f"FATAL: val pool size {len(val_pool)} < n_mats={n_mats}")
        sys.exit(1)
    if len(train_pool) < n_mats:
        err(f"FATAL: train pool size {len(train_pool)} < n_mats={n_mats}")
        sys.exit(1)

    rng = random.Random(seed)
    val_400b = sorted(rng.sample(val_pool, n_mats))
    train_400b = sorted(rng.sample(train_pool, n_mats))

    blob[f"val_{n_mats}b"] = val_400b
    blob[f"train_{n_mats}b"] = train_400b
    blob["_note"] = (
        blob.get("_note", "")
        + f" | val_{n_mats}b/train_{n_mats}b: disjoint from val_200/train_200; seed={seed}"
    )

    JSON_PATH.write_text(json.dumps(blob, indent=2) + "\n")
    err(f"Wrote {JSON_PATH}: added val_{n_mats}b ({len(val_400b)}) + train_{n_mats}b ({len(train_400b)})")
    err(f"First 3 val_{n_mats}b: {val_400b[:3]}")
    err(f"First 3 train_{n_mats}b: {train_400b[:3]}")


if __name__ == "__main__":
    main()
