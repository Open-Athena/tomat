#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["modal"]
# ///
"""Modal wrapper around `scripts/tokenize_patches.py`.

Runs patch tokenization inside a Modal function with the
`tomat-rho-gga` volume mounted read-write, writing parquet shards to
``/vol/tokenized/<label>/`` on the volume. The local entrypoint then
pulls the output dir back to ``data/tokenized/<label>/`` for local
DVX tracking.

Usage (defaults mirror spec 05 "smoke" settings):

    modal run scripts/tokenize_patches_modal.py \
        --label        val-smoke \
        --split        validation \
        --patches-per-material 32 \
        --patch-size   14 \
        --density-codec two_token_9_12 \
        --n-materials  128 \
        --seed         42

After the Modal function completes, this script runs
`modal volume get tomat-rho-gga /tokenized/<label> data/tokenized/` to
mirror the output into the local repo.
"""

from functools import partial
from pathlib import Path
import subprocess
import sys

import modal

err = partial(print, file=sys.stderr)

VOLUME_NAME = "tomat-rho-gga"
MOUNT = "/vol"

# Install tomat's runtime deps + make the tomat package importable in-function.
# `add_local_python_source` bundles the local src/ tree into the image so
# edits round-trip without needing a rebuild+publish cycle.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "click",
        "numpy",
        "pyarrow>=15",
        "pymatgen",
        "zarr>=3",
    )
    .add_local_python_source("tomat")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App("tomat-tokenize-patches", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={MOUNT: volume}, timeout=3600)
def tokenize(
    label: str,
    split: str,
    patches_per_material: int,
    patch_size: int,
    density_codec: str,
    density_log_min: float,
    density_log_max: float,
    n_materials: int | None,
    seed: int,
) -> dict:
    """Invoke `scripts/tokenize_patches.py` as a subprocess with the
    Modal volume mount as `--rho-gga-dir`."""
    import json as _json
    import subprocess as _sp

    output_dir = f"{MOUNT}/tokenized/{label}"

    cmd = [
        "python", "/root/scripts/tokenize_patches.py",
        "-r", MOUNT,
        "-s", f"{MOUNT}/split_limit_22M.json",
        "-k", split,
        "-m", str(patches_per_material),
        "-p", str(patch_size),
        "-c", density_codec,
        "--density-log-min", str(density_log_min),
        "--density-log-max", str(density_log_max),
        "-o", output_dir,
        "-S", str(seed),
    ]
    if n_materials is not None:
        cmd += ["-n", str(n_materials)]

    _sp.run(cmd, check=True)

    # Commit the volume writes so a subsequent `modal volume get` sees them.
    volume.commit()

    meta = _json.loads(Path(f"{output_dir}/meta.json").read_text())
    return {"output_dir": output_dir, "meta": meta}


@app.local_entrypoint()
def main(
    label: str = "val-smoke",
    split: str = "validation",
    patches_per_material: int = 32,
    patch_size: int = 14,
    density_codec: str = "two_token_9_12",
    density_log_min: float = -4.127,
    density_log_max: float = 4.967,
    n_materials: int = 128,
    seed: int = 42,
    pull: bool = True,
) -> None:
    err(f"[modal] tokenize → /vol/tokenized/{label}")
    result = tokenize.remote(
        label=label,
        split=split,
        patches_per_material=patches_per_material,
        patch_size=patch_size,
        density_codec=density_codec,
        density_log_min=density_log_min,
        density_log_max=density_log_max,
        n_materials=n_materials if n_materials > 0 else None,
        seed=seed,
    )
    meta = result["meta"]
    err(f"[modal] done: {meta['total_rows']:,} rows in {meta['n_shards']} shards")
    err(f"        codec={meta['density_codec_name']} vocab={meta['vocab']['total_size']}")

    if pull:
        local_dst = Path("data/tokenized") / label
        local_dst.parent.mkdir(parents=True, exist_ok=True)
        if local_dst.exists():
            err(f"[modal] local dst {local_dst} exists; `modal volume get` will skip existing files")
        err(f"[modal] pull → {local_dst}")
        subprocess.run(
            [
                "modal", "volume", "get", VOLUME_NAME,
                f"/tokenized/{label}",
                str(local_dst.parent),
            ],
            check=True,
        )
