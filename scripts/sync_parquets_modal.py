#!/usr/bin/env python
"""Modal-native sync of tokenized parquets from Modal volume → GCS.

Avoids the laptop-in-the-middle sync flow of sync_parquets_to_gcs.py.
Reads parquets directly from the mounted volume, uploads via gcsfs
using the tomat-gcp-sa Modal secret.

Usage (from laptop):
    TOMAT_VOLUME=tomat-rho-gga-train \\
    modal run scripts/sync_parquets_modal.py::run \\
        --label train-full-m256

Streams progress; exits nonzero on any upload failure.
"""

from __future__ import annotations

import os

import modal


VOLUME_NAME = os.environ.get("TOMAT_VOLUME", "tomat-rho-gga-train")
MOUNT = "/vol"
BUCKET = "gs://marin-eu-west4/tomat"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gcsfs")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")

app = modal.App("tomat-sync-parquets", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def setup_gcp_creds():
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


@app.function(volumes={MOUNT: volume}, secrets=[gcp_secret], timeout=3600, cpu=2)
def upload_shard(local_path: str, gcs_path: str) -> dict:
    """Upload a single parquet from mounted volume to GCS. Returns
    {'path': ..., 'bytes': ..., 'ok': bool, 'err': ...}."""
    import gcsfs  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()

    try:
        with open(local_path, "rb") as src:
            data = src.read()
        with fs.open(gcs_path, "wb") as dst:
            dst.write(data)
        return {"path": gcs_path, "bytes": len(data), "ok": True, "err": None}
    except Exception as e:
        return {"path": gcs_path, "bytes": 0, "ok": False, "err": f"{type(e).__name__}: {e}"}


@app.function(volumes={MOUNT: volume}, secrets=[gcp_secret], timeout=600, cpu=1)
def list_local_parquets(label: str) -> list[tuple[str, str, int]]:
    """List every parquet + meta.json under /vol/tokenized/<label>/.
    Returns [(local_path, gcs_relpath, size_bytes)]."""
    from pathlib import Path

    root = Path(f"{MOUNT}/tokenized/{label}")
    if not root.exists():
        raise FileNotFoundError(f"no such label dir: {root}")

    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in (".parquet", ".json"):
            continue
        rel = p.relative_to(root).as_posix()
        out.append((str(p), rel, p.stat().st_size))
    return out


@app.local_entrypoint()
def run(label: str, bucket: str = BUCKET):
    import sys

    err = lambda *a: print(*a, file=sys.stderr)

    files = list_local_parquets.remote(label)
    if not files:
        err(f"[sync] no files under /vol/tokenized/{label}")
        return

    total_bytes = sum(sz for _, _, sz in files)
    err(f"[sync] {len(files)} files, {total_bytes / 1e9:.2f} GB → {bucket}/tokenized/{label}/")

    # Build (local_path, gcs_path) pairs.
    pairs = [
        (local, f"{bucket}/tokenized/{label}/{rel}")
        for local, rel, _ in files
    ]
    inputs = [(local, gcs) for (local, gcs) in pairs]

    n_ok = 0
    n_fail = 0
    total_up = 0
    for res in upload_shard.starmap(inputs, order_outputs=False):
        if res["ok"]:
            n_ok += 1
            total_up += res["bytes"]
        else:
            n_fail += 1
            err(f"[sync] FAIL {res['path']}: {res['err']}")
        if (n_ok + n_fail) % 50 == 0:
            err(f"[sync] {n_ok + n_fail}/{len(pairs)} done ({total_up / 1e9:.2f} GB)")

    err(f"[sync] done: {n_ok} ok, {n_fail} failed, {total_up / 1e9:.2f} GB uploaded")
    if n_fail > 0:
        raise SystemExit(f"{n_fail} upload failures")
