#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["modal"]
# ///
"""Verify the seeded `tomat-rho-gga` Modal volume.

Mounts the volume in a Modal function, lists `label/`, opens one Zarr
group, reads its `charge_density_total` array and `structure` JSON
attribute, and prints a one-line summary back to the local caller.
"""

from functools import partial
import sys

import modal

err = partial(print, file=sys.stderr)

VOLUME_NAME = "tomat-rho-gga"
MOUNT = "/vol"

image = (
    modal.Image.debian_slim()
    .pip_install("zarr>=3", "numpy")
)
app = modal.App("tomat-rho-gga-verify", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(volumes={MOUNT: volume})
def inspect(mp_id: str | None = None) -> dict:
    import json
    import os
    import zarr

    label_dir = f"{MOUNT}/label"
    entries = sorted(os.listdir(label_dir))
    n = len(entries)

    pick = f"{mp_id}.zarr" if mp_id else entries[0]
    path = f"{label_dir}/{pick}"

    g = zarr.open_group(path, mode="r")
    arr = g["charge_density_total"]
    structure = json.loads(g.attrs["structure"])
    metadata = json.loads(g.attrs["metadata"])

    return {
        "n_label_entries": n,
        "first_three": entries[:3],
        "last_three": entries[-3:],
        "inspected": pick,
        "task_id": metadata.get("task_id"),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "rho_min": float(arr[:].min()),
        "rho_max": float(arr[:].max()),
        "n_sites": len(structure["sites"]),
        "elements": sorted({s["species"][0]["element"] for s in structure["sites"]}),
    }


@app.local_entrypoint()
def main(mp_id: str = ""):
    result = inspect.remote(mp_id or None)
    err("=" * 60)
    err(f"Volume:           {VOLUME_NAME}")
    err(f"label/ entries:   {result['n_label_entries']}")
    err(f"first three:      {result['first_three']}")
    err(f"last three:       {result['last_three']}")
    err("-" * 60)
    err(f"inspected:        {result['inspected']}")
    err(f"task_id:          {result['task_id']}")
    err(f"shape:            {result['shape']}")
    err(f"dtype:            {result['dtype']}")
    err(f"rho range:        [{result['rho_min']:.4g}, {result['rho_max']:.4g}]")
    err(f"n_sites:          {result['n_sites']}")
    err(f"elements:         {result['elements']}")
    err("=" * 60)
