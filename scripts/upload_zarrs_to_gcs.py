#!/usr/bin/env python
"""Upload raw Zarrs from Modal volume → GCS for eval-time access.

The val-full volume (tomat-rho-gga) has ~4,305 mats × ~5 MB = ~22 GB.
Each Zarr is a directory with a few hundred KB of metadata + one big chunk.

Usage:
    TOMAT_VOLUME=tomat-rho-gga modal run scripts/upload_zarrs_to_gcs.py::run \\
        --split validation
"""

from __future__ import annotations

import os

import modal

VOLUME_NAME = os.environ.get("TOMAT_VOLUME", "tomat-rho-gga")
MOUNT = "/vol"
BUCKET = "gs://marin-eu-west4/tomat"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gcsfs")
    .apt_install("rsync")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")

app = modal.App("tomat-upload-zarrs", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def setup_gcp_creds():
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


@app.function(volumes={MOUNT: volume}, cpu=2, memory=2048, timeout=600, secrets=[gcp_secret])
def list_mats(split: str) -> list[str]:
    """Resolve task_ids for the given split."""
    import json
    import pathlib

    split_path = pathlib.Path(MOUNT) / "split_limit_22M.json"
    with open(split_path) as f:
        sp = json.load(f)
    entries = sp[split]
    if entries and isinstance(entries[0], int):
        fl_path = pathlib.Path(MOUNT) / "mp_filelist.txt"
        with open(fl_path) as f:
            filelist = [l.strip() for l in f if l.strip()]
        return [filelist[i] for i in entries]
    return list(entries)


@app.function(
    volumes={MOUNT: volume},
    cpu=1,
    memory=2048,
    timeout=600,
    secrets=[gcp_secret],
    retries=3,
)
def upload_mat(mat_id: str, dest_prefix: str) -> dict:
    """Upload one Zarr directory from volume → GCS. Recurse through files."""
    import pathlib
    import gcsfs  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()
    src = pathlib.Path(MOUNT) / "label" / f"{mat_id}.zarr"
    if not src.exists():
        return {"mat_id": mat_id, "ok": False, "reason": "missing", "bytes": 0}
    total = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src).as_posix()
        gcs_path = f"{dest_prefix}/{mat_id}.zarr/{rel}"
        with open(f, "rb") as src_f:
            data = src_f.read()
        with fs.open(gcs_path, "wb") as dst_f:
            dst_f.write(data)
        total += len(data)
    return {"mat_id": mat_id, "ok": True, "bytes": total}


@app.local_entrypoint()
def run(split: str = "validation", max_mats: int = 0, dest_prefix: str = ""):
    """Upload all Zarrs in a split from Modal volume to GCS."""
    import sys

    err = lambda *a: print(*a, file=sys.stderr)

    if not dest_prefix:
        dest_prefix = f"{BUCKET[5:]}/rho_gga_raw/{split}"
    err(f"[upload-zarrs] dest = gs://{dest_prefix}")

    mat_ids = list_mats.remote(split)
    if max_mats > 0:
        mat_ids = mat_ids[:max_mats]
    err(f"[upload-zarrs] {len(mat_ids)} mats")

    inputs = [(mat_id, dest_prefix) for mat_id in mat_ids]
    n_ok = 0
    n_fail = 0
    total_bytes = 0
    for res in upload_mat.starmap(inputs, order_outputs=False):
        if res["ok"]:
            n_ok += 1
            total_bytes += res["bytes"]
        else:
            n_fail += 1
            err(f"[upload-zarrs] FAIL {res['mat_id']}: {res.get('reason')}")
        if (n_ok + n_fail) % 100 == 0:
            err(f"[upload-zarrs] {n_ok + n_fail}/{len(inputs)} done, "
                f"{total_bytes / 1e9:.2f} GB")
    err(f"[upload-zarrs] DONE: {n_ok} ok, {n_fail} fail, {total_bytes / 1e9:.2f} GB")
