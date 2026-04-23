#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["click"]
# ///
"""Upload a tokenized-parquet dir to GCS with md5 verification.

Pattern motivated by spec 09:
- compute md5 of every local parquet → manifest
- `gcloud storage cp --recursive` local → `gs://.../tokenized/<label>/`
- re-fetch GCS md5 per object, compare to local manifest
- exit nonzero on any mismatch (no "mostly succeeded")

Usage:
    scripts/sync_parquets_to_gcs.py tmp/train-full-pull/train-full train-full

Assumes:
- source dir layout: `<src>/worker-NN/{meta.json,shard-*.parquet}`
- GCS dest: `gs://<bucket>/tomat/tokenized/<label>/`
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import click

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@click.command()
@click.option("-b", "--bucket", default=BUCKET, help=f"GCS bucket root (default: {BUCKET})")
@click.option("-V", "--no-verify", is_flag=True, help="skip post-upload md5 verification")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("label")
def main(bucket: str, no_verify: bool, src_dir: Path, label: str) -> None:
    gcs_root = f"{bucket}/tokenized/{label}"

    err(f"[sync] computing md5 of parquets under {src_dir}")
    parquets = sorted(src_dir.rglob("*.parquet"))
    err(f"[sync] {len(parquets)} parquets found")
    local_md5 = {p.relative_to(src_dir).as_posix(): _md5(p) for p in parquets}

    err(f"[sync] uploading to {gcs_root}")
    subprocess.run(
        ["gcloud", "storage", "cp", "--recursive", str(src_dir), f"{bucket}/tokenized/"],
        check=True,
    )

    if no_verify:
        err("[sync] skipping verify (-V)")
        return

    err(f"[sync] fetching GCS md5 manifest")
    out = subprocess.check_output(
        ["gcloud", "storage", "ls", "-L", f"{gcs_root}/worker-*/*.parquet"],
        text=True,
    )
    gcs_md5: dict[str, str] = {}
    current: str | None = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("gs://") and s.endswith(".parquet:"):
            current = s[:-1]
        elif s.startswith("Hash (MD5):") and current:
            gcs_md5[current] = base64.b64decode(s.split()[-1]).hex()
            current = None

    ok = 0
    mism: list[tuple[str, str, str | None]] = []
    for relpath, m in local_md5.items():
        g = gcs_md5.get(f"{gcs_root}/{relpath}")
        if g == m:
            ok += 1
        else:
            mism.append((relpath, m, g))
    err(f"[sync] ok: {ok}, mismatched: {len(mism)}")
    if mism:
        for rp, m, g in mism[:10]:
            err(f"  ! {rp}: local={m[:8]} gcs={(g or 'MISSING')[:8]}")
        sys.exit(1)
    err(f"[sync] all {ok} md5s match GCS")


if __name__ == "__main__":
    main()
