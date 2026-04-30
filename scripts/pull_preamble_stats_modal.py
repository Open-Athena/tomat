#!/usr/bin/env python
"""Pull (mp_id, nx, ny, nz, n_atoms) for every material in a tokenized label.

Extracts from the parquet preambles: grid shape (first 3 ints after
[GRID_START]) and atom count (tokens between [ATOMS_START]=3 and
[ATOMS_END]=4). One material = one 32-row block in the parquet;
we read only row 0 of each block.

Datasets tokenized after the lattice-aware change use INT_BASE=138 and
include a [LATTICE_START]…[LATTICE_END] block between [GRID_END] and
[ATOMS_START]. Pre-lattice datasets use INT_BASE=136 and no LATTICE
block; pass --int-base 136 + --no-skip-lattice to read those.

Usage:
    modal run scripts/pull_preamble_stats_modal.py::run --label train-full-lat
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "gcsfs")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")

app = modal.App("tomat-pull-preamble-stats", image=image)

INT_BASE_DEFAULT = 20 + 118    # 138 for lattice-aware datasets
INT_VOCAB = 1024
GRID_START, GRID_END = 7, 8
ATOMS_START, ATOMS_END = 3, 4
LATTICE_START, LATTICE_END = 18, 19  # tokens skipped between GRID_END and ATOMS_START
PATCHES_PER_MAT = 32


def setup_gcp_creds():
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


def parse_preamble(
    tokens: list[int], int_base: int = INT_BASE_DEFAULT,
) -> tuple[int, int, int, int]:
    """Return (nx, ny, nz, n_atoms) from the preamble of one row.

    Layout: [BOS] [GRID_START] nx ny nz [GRID_END]
            [LATTICE_START] qa qb qc qα qβ qγ [LATTICE_END]?  (lat-aware datasets)
            [ATOMS_START] atoms… [ATOMS_END] …
    """
    if tokens[1] != GRID_START:
        raise ValueError(f"expected GRID_START at pos 1, got {tokens[:6]}")
    nx, ny, nz = (tokens[2] - int_base, tokens[3] - int_base, tokens[4] - int_base)
    if tokens[5] != GRID_END:
        raise ValueError(f"expected GRID_END at pos 5, got {tokens[:6]}")
    # Optional LATTICE block between GRID_END and ATOMS_START.
    if tokens[6] == LATTICE_START:
        # Skip until LATTICE_END.
        for k in range(7, min(len(tokens), 7 + 32)):
            if tokens[k] == LATTICE_END:
                atoms_start_pos = k + 1
                break
        else:
            raise ValueError("no LATTICE_END after LATTICE_START")
    else:
        atoms_start_pos = 6
    if tokens[atoms_start_pos] != ATOMS_START:
        raise ValueError(
            f"expected ATOMS_START at pos {atoms_start_pos}, got {tokens[:atoms_start_pos+4]}"
        )
    # Find ATOMS_END in the next few tokens; atoms are one token each.
    for j in range(atoms_start_pos + 1, min(len(tokens), atoms_start_pos + 1 + 10000)):
        if tokens[j] == ATOMS_END:
            return nx, ny, nz, j - (atoms_start_pos + 1)
    raise ValueError(f"no ATOMS_END found within window")


@app.function(cpu=1, memory=2048, timeout=600, secrets=[gcp_secret])
def process_shard(shard: str) -> list[tuple[str, int, int, int, int]]:
    import gcsfs  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()
    out = []
    with fs.open(shard, "rb") as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c in ("task_id", "input_ids", "mp_id")]
        if "input_ids" not in cols:
            return out
        tbl = pf.read(columns=cols)
    ids = tbl.column("input_ids").to_pylist()
    id_col = "task_id" if "task_id" in tbl.column_names else "mp_id"
    mp_ids = tbl.column(id_col).to_pylist() if id_col in tbl.column_names else [None] * len(ids)
    for i in range(0, len(ids), PATCHES_PER_MAT):
        row = ids[i]
        # ATOMS block can be many tokens long; take a generous window.
        # Typical mat: <50 atoms → window of ~500 is plenty.
        window = row[:1024]
        try:
            nx, ny, nz, n_atoms = parse_preamble(window)
        except ValueError:
            # Try a bigger window for dense-atom mats
            window = row[:4096]
            nx, ny, nz, n_atoms = parse_preamble(window)
        mpid = mp_ids[i] if mp_ids[i] is not None else f"{shard}:{i}"
        out.append((mpid, nx, ny, nz, n_atoms))
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
    print(f"[modal-preamble] {len(shards)} shards under {bucket}/tokenized/{label}", file=sys.stderr)

    print("mp_id,nx,ny,nz,n_atoms")
    seen = 0
    for rows in process_shard.map(shards, order_outputs=False):
        for mpid, nx, ny, nz, n_atoms in rows:
            print(f"{mpid},{nx},{ny},{nz},{n_atoms}")
        seen += len(rows)
    print(f"[modal-preamble] total mats: {seen}", file=sys.stderr)
