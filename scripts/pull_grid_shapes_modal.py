#!/usr/bin/env python
"""Run pull_grid_shapes over all shards of a tokenized-label on Modal.

One Modal function per shard, ~1000-wide concurrency → ~60s to scan all 77k
materials in train-full. Emits CSV to stdout.

Usage:
    modal run scripts/pull_grid_shapes_modal.py::run --label train-full
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "gcsfs")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")

app = modal.App("tomat-pull-grid-shapes", image=image)


def setup_gcp_creds():
    """Materialize GOOGLE_APPLICATION_CREDENTIALS_JSON into a file + env var.

    Modal exposes the secret as an env var with the raw JSON. gcsfs / google-cloud
    libs read a file path from GOOGLE_APPLICATION_CREDENTIALS.
    """
    import os

    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

INT_BASE = 18 + 118
GRID_START_TOK = 7
GRID_END_TOK = 8
PATCHES_PER_MAT = 32


def decode_grid(tokens: list[int]) -> tuple[int, int, int]:
    if tokens[1] != GRID_START_TOK:
        raise ValueError(f"expected GRID_START at pos 1, got {tokens[:6]}")
    dims = []
    for i in range(2, 5):
        t = tokens[i]
        dims.append(t - INT_BASE)
    if tokens[5] != GRID_END_TOK:
        raise ValueError(f"expected GRID_END at pos 5, got {tokens[:6]}")
    return (dims[0], dims[1], dims[2])


@app.function(cpu=1, memory=2048, timeout=600, secrets=[gcp_secret])
def process_shard(shard: str) -> list[tuple[str, int, int, int]]:
    import gcsfs  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()
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


@app.function(cpu=1, memory=1024, secrets=[gcp_secret])
def list_shards(label: str, bucket: str) -> list[str]:
    import gcsfs  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()
    root = f"{bucket[5:]}/tokenized/{label}"
    return sorted(p for p in fs.glob(f"{root}/worker-*/*.parquet") if p.endswith(".parquet"))


@app.local_entrypoint()
def run(label: str = "train-full", bucket: str = "gs://marin-eu-west4/tomat"):
    import sys

    shards = list_shards.remote(label, bucket)
    print(f"[modal-pull] {len(shards)} shards under {bucket}/tokenized/{label}", file=sys.stderr)

    print("mp_id,nx,ny,nz")
    seen = 0
    for rows in process_shard.map(shards, order_outputs=False):
        for mpid, nx, ny, nz in rows:
            print(f"{mpid},{nx},{ny},{nz}")
        seen += len(rows)
    print(f"[modal-pull] total mats: {seen}", file=sys.stderr)
