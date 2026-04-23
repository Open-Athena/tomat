#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["modal", "pyarrow>=16", "click"]
# ///
"""Scan `val-full` parquet files for ZSTD / row-group corruption.

Runs inside a Modal container with the `tomat-rho-gga` volume mounted,
walks every `tokenized/val-full/worker-*/*.parquet`, and for each file
reads every row group and records its md5. Prints one JSON line per
file on the local side so we can diff against GCS.

Usage:
    modal run scripts/verify_val_full_parquet.py
"""

from __future__ import annotations

import json
from functools import partial
import sys

import modal

err = partial(print, file=sys.stderr)

VOLUME_NAME = "tomat-rho-gga"
MOUNT = "/vol"
LABEL = "val-full"

image = modal.Image.debian_slim().pip_install("pyarrow>=16")
app = modal.App("tomat-val-full-verify", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def _scan_one(path: str) -> dict:
    import hashlib
    import os
    import pyarrow.parquet as pq

    size = os.path.getsize(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    md5 = h.hexdigest()

    pf = pq.ParquetFile(path)
    num_rg = pf.num_row_groups
    num_rows = pf.metadata.num_rows
    bad_rgs: list[tuple[int, str]] = []
    for i in range(num_rg):
        try:
            pf.read_row_group(i)
        except Exception as e:
            bad_rgs.append((i, f"{type(e).__name__}: {e}"))
    return {
        "path": path,
        "size": size,
        "md5": md5,
        "num_row_groups": num_rg,
        "num_rows": num_rows,
        "bad_row_groups": bad_rgs,
    }


@app.function(volumes={MOUNT: volume}, cpu=8, memory=16 * 1024, timeout=3600)
def scan_all(label: str = LABEL) -> list[dict]:
    import concurrent.futures as cf
    import glob
    import os

    base = f"{MOUNT}/tokenized/{label}"
    parquets = sorted(glob.glob(f"{base}/worker-*/shard-*.parquet"))
    print(f"[verify] {len(parquets)} parquets under {base}", file=sys.stderr)

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_one, p): p for p in parquets}
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"path": p, "error": f"{type(e).__name__}: {e}"}
            print(f"[verify] {os.path.basename(os.path.dirname(p))}/"
                  f"{os.path.basename(p)}: "
                  f"rows={res.get('num_rows')} "
                  f"bad={len(res.get('bad_row_groups', []))} "
                  f"md5={res.get('md5', '?')[:8]}",
                  file=sys.stderr, flush=True)
            results.append(res)
    return results


@app.local_entrypoint()
def main(label: str = LABEL):
    results = scan_all.remote(label)
    bad = [r for r in results if r.get("bad_row_groups") or "error" in r]
    err(f"[verify] total={len(results)} bad={len(bad)}")
    for r in sorted(results, key=lambda x: x["path"]):
        print(json.dumps(r))
