#!/usr/bin/env python
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
import os
from pathlib import Path
import subprocess
import sys

import modal

err = partial(print, file=sys.stderr)

# Env-var so we can point at `tomat-rho-gga-train` (77 k structures, spec 08)
# without forking the script. Modal volumes resolve at local-entrypoint time,
# so swapping this per-invocation gives us a single tokenize pipeline for both
# val and train splits.
VOLUME_NAME = os.environ.get("TOMAT_VOLUME", "tomat-rho-gga")
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


@app.function(volumes={MOUNT: volume}, timeout=14400)  # 4h
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
    pad_to: int | None,
    worker_idx: int = 0,
    n_workers: int = 1,
) -> dict:
    """Invoke `scripts/tokenize_patches.py` as a subprocess with the
    Modal volume mount as `--rho-gga-dir`.

    When called in parallel (via the ``tokenize_parallel`` entrypoint),
    `worker_idx` / `n_workers` slice the task-id list across workers
    and the tokenizer auto-nests output under `worker-NN/`.
    """
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
        "-w", str(worker_idx),
        "-W", str(n_workers),
    ]
    if n_materials is not None:
        cmd += ["-n", str(n_materials)]
    if pad_to is not None:
        cmd += ["-L", str(pad_to)]

    _sp.run(cmd, check=True)

    # Commit the volume writes so a subsequent `modal volume get` sees them.
    volume.commit()

    # When parallel, the tokenizer wrote under worker-NN/; per-worker meta
    # lives there. The parallel entrypoint merges meta after all workers
    # finish. Return the worker-local dir for caller sanity-checking.
    worker_dir = (
        f"{output_dir}/worker-{worker_idx:02d}"
        if n_workers > 1 else output_dir
    )
    meta = _json.loads(Path(f"{worker_dir}/meta.json").read_text())
    return {"output_dir": worker_dir, "meta": meta}


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
    pad_to: int = 8192,
    pull: bool = True,
) -> None:
    err(f"[modal] tokenize → /vol/tokenized/{label} (pad_to={pad_to})")
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
        pad_to=pad_to if pad_to > 0 else None,
    )
    meta = result["meta"]
    err(f"[modal] done: {meta['total_rows']:,} rows in {meta['n_shards']} shards")
    err(f"        codec={meta['density_codec_name']} vocab={meta['vocab']['total_size']}")

    if pull:
        local_dst = Path("data/tokenized") / label
        local_dst.parent.mkdir(parents=True, exist_ok=True)
        err(f"[modal] pull → {local_dst}")
        # --force overwrites existing files; acceptable since the volume is
        # the source of truth for this label and `dvx` hashes the pulled copy.
        subprocess.run(
            [
                "modal", "volume", "get", "--force", VOLUME_NAME,
                f"/tokenized/{label}",
                str(local_dst.parent),
            ],
            check=True,
        )


@app.local_entrypoint()
def parallel(
    label: str = "val-full",
    split: str = "validation",
    patches_per_material: int = 32,
    patch_size: int = 14,
    density_codec: str = "two_token_9_12",
    density_log_min: float = -4.127,
    density_log_max: float = 4.967,
    n_materials: int = 0,
    seed: int = 42,
    pad_to: int = 8192,
    n_workers: int = 16,
    worker_indices: str = "",
    pull: bool = False,
) -> None:
    """Parallel tokenize via Modal's ``.map()`` — dispatches ``n_workers``
    containers that each process ``task_ids[i::n_workers]`` and write to
    a per-worker subdir ``/vol/tokenized/<label>/worker-NN/``.

    Training reads from the per-worker dirs directly via glob; no
    post-merge step in this MVP (per spec 07). Pull-to-local is off by
    default since train-scale parquet is large; set ``--pull`` for
    DVX-hashing the output when the label fits on local disk.

    Pass ``--worker-indices "26,27,28,29,30,31"`` to only (re)run a
    subset of the modular-stride slices, e.g. to recover from a prior
    launch where some workers never spun up. The modular-stride math
    still uses ``n_workers`` so output lands in the right dirs.
    """
    if worker_indices:
        indices = [int(s) for s in worker_indices.split(",") if s.strip()]
        err(f"[modal] parallel tokenize → /vol/tokenized/{label} "
            f"(n_workers={n_workers}, only workers={indices}, pad_to={pad_to})")
    else:
        indices = list(range(n_workers))
        err(f"[modal] parallel tokenize → /vol/tokenized/{label} "
            f"(n_workers={n_workers}, pad_to={pad_to})")

    # `tokenize.spawn(...)` returns a FunctionCall handle immediately; Modal
    # runs all N concurrently. `fc.get()` waits for that worker's result.
    calls = [
        tokenize.spawn(
            label=label, split=split,
            patches_per_material=patches_per_material, patch_size=patch_size,
            density_codec=density_codec,
            density_log_min=density_log_min, density_log_max=density_log_max,
            n_materials=n_materials if n_materials > 0 else None,
            seed=seed, pad_to=pad_to if pad_to > 0 else None,
            worker_idx=i, n_workers=n_workers,
        )
        for i in indices
    ]

    outcomes = []
    for i, fc in zip(indices, calls):
        try:
            outcomes.append(fc.get())
            err(f"[modal] worker {i} done: "
                f"{outcomes[-1]['meta']['total_rows']:,} rows")
        except Exception as e:
            err(f"[modal] worker {i} FAILED: {e}")
            raise

    total_rows = sum(o["meta"]["total_rows"] for o in outcomes)
    err(f"[modal] {len(indices)} worker(s) done: {total_rows:,} total rows "
        f"across /vol/tokenized/{label}/worker-*")

    if pull:
        local_dst = Path("data/tokenized") / label
        local_dst.parent.mkdir(parents=True, exist_ok=True)
        err(f"[modal] pull → {local_dst}")
        subprocess.run(
            [
                "modal", "volume", "get", "--force", VOLUME_NAME,
                f"/tokenized/{label}",
                str(local_dst.parent),
            ],
            check=True,
        )
