#!/usr/bin/env python
"""Sync `gs://marin-eu-west4/tomat/tokenized/<label>/` → Modal volume.

After `shuffle_tokenized_modal.py` writes to GCS, the Modal training
entry (`train_smoke_modal.py`) still reads from the mounted volume
(`/vol/tokenized/<label>/worker-*/*.parquet`), so we need a copy on
the volume too. This script does that: lists every shard in the GCS
source label, downloads in parallel, and writes to
`/vol/tokenized/<label>/worker-NN/shard-NNNNN.parquet`.

Also pulls the `_perm.npy`, `_perm.json`, and `worker-00/meta.json`
sidecars so the volume layout matches GCS.

Usage::

    modal run scripts/sync_shuffled_to_modal_volume.py::run \\
        --label train-full-v3-shuffled

Idempotent: skips any local file that already exists with size > 0.
"""

from __future__ import annotations

import os
import sys
from functools import partial

import modal

err = partial(print, file=sys.stderr)

TRAIN_VOLUME_NAME = "tomat-rho-gga-train"
MOUNT = "/vol"
DEFAULT_BUCKET = "gs://marin-eu-west4/tomat"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gcsfs")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
app = modal.App("tomat-sync-shuffled-to-volume", image=image)
train_volume = modal.Volume.from_name(TRAIN_VOLUME_NAME)


def setup_gcp_creds():
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


@app.function(
    cpu=8,
    memory=16_384,
    volumes={MOUNT: train_volume},
    secrets=[gcp_secret],
    timeout=14_400,
)
def sync_label(label: str, bucket: str) -> dict:
    """Mirror gs://<bucket>/tomat/tokenized/<label>/ → /vol/tokenized/<label>/."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import gcsfs  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()

    train_volume.reload()

    src_root = f"{bucket[5:]}/tokenized/{label}"
    dst_root = Path(f"{MOUNT}/tokenized/{label}")
    dst_root.mkdir(parents=True, exist_ok=True)

    err(f"[sync] src=gs://{src_root}")
    err(f"[sync] dst={dst_root}")

    # Enumerate every file (parquet + sidecars).
    err("[sync] listing source files...")
    t0 = time.time()
    all_paths = sorted(fs.ls(src_root))
    # Recurse one level for worker-* dirs.
    worker_files: list[tuple[str, str, int]] = []
    sidecars: list[tuple[str, str, int]] = []
    for p in all_paths:
        info = fs.info(p)
        if info["type"] == "directory":
            # worker-NN/
            sub = sorted(fs.ls(p))
            for sp in sub:
                sinfo = fs.info(sp)
                if sinfo["type"] != "file":
                    continue
                rel = sp[len(src_root) + 1:]
                worker_files.append((f"gs://{sp}", str(dst_root / rel), sinfo["size"]))
        else:
            rel = p[len(src_root) + 1:]
            sidecars.append((f"gs://{p}", str(dst_root / rel), info["size"]))
    err(f"[sync] listed in {time.time()-t0:.1f}s: "
        f"{len(worker_files)} worker files, {len(sidecars)} top-level sidecars")

    all_files = worker_files + sidecars

    def _copy(item):
        src, dst, sz = item
        dst_path = Path(dst)
        if dst_path.exists() and dst_path.stat().st_size == sz:
            return dict(src=src, dst=dst, size=sz, status="skip")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(src[5:], "rb") as f_in:
            data = f_in.read()
        with open(dst, "wb") as f_out:
            f_out.write(data)
        return dict(src=src, dst=dst, size=len(data), status="ok")

    err(f"[sync] copying {len(all_files)} files...")
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, r in enumerate(ex.map(_copy, all_files)):
            results.append(r)
            if (i + 1) % 200 == 0:
                done = sum(rr["size"] for rr in results)
                err(f"[sync]   copied {i+1}/{len(all_files)} files, "
                    f"{done/1e9:.2f} GB, {time.time()-t0:.1f}s elapsed")
    total_bytes = sum(r["size"] for r in results)
    n_skip = sum(1 for r in results if r["status"] == "skip")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    err(f"[sync] done: {n_ok} copied, {n_skip} skipped, "
        f"{total_bytes/1e9:.2f} GB in {time.time()-t0:.1f}s")

    train_volume.commit()

    return dict(label=label, n_files=len(all_files), n_ok=n_ok, n_skip=n_skip,
                total_bytes=total_bytes)


@app.local_entrypoint()
def run(label: str = "train-full-v3-shuffled", bucket: str = DEFAULT_BUCKET):
    err(f"[run] label={label} bucket={bucket}")
    result = sync_label.remote(label=label, bucket=bucket)
    err(f"[run] done: {result}")
