#!/usr/bin/env python
# /// script
# dependencies = ["pyarrow", "gcsfs", "tqdm"]
# ///
"""Pull grid shape (nx, ny, nz) from each material in train-full (or any
label) by reading the first row of each 32-row block in each parquet.

Each training row's preamble starts with:
    [BOS] [GRID_START] int(nx) int(ny) int(nz) [GRID_END] ...

Int tokens occupy vocab[18+118 : 18+118+1024] = [136, 1160). grid dim =
token - 136.

Since `patches_per_material = 32` and shards are stored in order, row 0,
32, 64, ... each correspond to a distinct material's first patch. We read
only those rows and extract 3 int tokens per material.

Output: CSV with columns `mp_id,nx,ny,nz` to stdout.

Usage: pull_grid_shapes.py --label train-full [--max-mats N]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from tqdm import tqdm

INT_BASE = 18 + 118  # 136
INT_VOCAB = 1024
GRID_START_TOK = 7
GRID_END_TOK = 8
PATCHES_PER_MAT = 32


def decode_grid(tokens: list[int]) -> tuple[int, int, int]:
    if tokens[1] != GRID_START_TOK:
        raise ValueError(f"expected GRID_START at pos 1, got {tokens[:6]}")
    dims = []
    for i in range(2, 5):
        t = tokens[i]
        if not (INT_BASE <= t < INT_BASE + INT_VOCAB):
            raise ValueError(f"token[{i}]={t} not in int range [{INT_BASE}, {INT_BASE+INT_VOCAB})")
        dims.append(t - INT_BASE)
    if tokens[5] != GRID_END_TOK:
        raise ValueError(f"expected GRID_END at pos 5, got {tokens[:6]}")
    return (dims[0], dims[1], dims[2])


def process_shard(shard: str, fs: gcsfs.GCSFileSystem) -> list[tuple[str, int, int, int]]:
    """Read only row 0, 32, 64, ... (one per material) from a shard.

    Uses parquet's row-group + column selection so we only pull the
    `input_ids` and id columns, not the giant density bytes.
    """
    out = []
    with fs.open(shard, "rb") as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c in ("mp_id", "input_ids", "task_id")]
        if "input_ids" not in cols:
            return out
        tbl = pf.read(columns=cols)
    ids = tbl.column("input_ids").to_pylist()
    mp_ids = (
        tbl.column("mp_id").to_pylist()
        if "mp_id" in tbl.column_names
        else tbl.column("task_id").to_pylist()
        if "task_id" in tbl.column_names
        else [None] * len(ids)
    )
    for i in range(0, len(ids), PATCHES_PER_MAT):
        nx, ny, nz = decode_grid(ids[i][:6])
        mpid = mp_ids[i] if mp_ids[i] is not None else f"{shard}:{i}"
        out.append((mpid, nx, ny, nz))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="train-full")
    ap.add_argument("--bucket", default="gs://marin-eu-west4/tomat")
    ap.add_argument("--max-mats", type=int, default=0, help="0 = no cap")
    ap.add_argument("--workers", type=int, default=32, help="concurrent shard downloads")
    ap.add_argument("--sample-every", type=int, default=1, help="read every Nth shard (1 = all)")
    args = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    root = f"{args.bucket[5:]}/tokenized/{args.label}"
    shards = sorted(p for p in fs.glob(f"{root}/worker-*/*.parquet") if p.endswith(".parquet"))
    if args.sample_every > 1:
        shards = shards[::args.sample_every]
    print(f"found {len(shards)} shards under gs://{root}", file=sys.stderr)

    print("mp_id,nx,ny,nz")
    mats_seen = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, s, fs): s for s in shards}
        for fut in tqdm(as_completed(futs), total=len(futs), file=sys.stderr):
            shard = futs[fut]
            try:
                for mpid, nx, ny, nz in fut.result():
                    print(f"{mpid},{nx},{ny},{nz}")
                    mats_seen += 1
                    if args.max_mats and mats_seen >= args.max_mats:
                        return
            except Exception as e:
                print(f"SKIP {shard}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
