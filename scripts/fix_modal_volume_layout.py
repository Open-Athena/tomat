#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["modal"]
# ///
"""One-shot fix: move /label/label/* → /label/* in tomat-rho-gga volume.

`modal volume put` of `$STAGE/label` to remote `label/` produced
`/label/label/mp-*.zarr` instead of `/label/mp-*.zarr`. Server-side
cp -r is unsupported on V1 volumes; this fixes the layout via os.rename
inside a Modal function (same filesystem → cheap).
"""

import modal

app = modal.App("tomat-rho-gga-fix-layout")
volume = modal.Volume.from_name("tomat-rho-gga")


@app.function(volumes={"/vol": volume}, timeout=600)
def fix() -> dict:
    import os

    src = "/vol/label/label"
    dst = "/vol/label"
    entries = sorted(os.listdir(src))
    moved = 0
    for name in entries:
        os.rename(f"{src}/{name}", f"{dst}/{name}")
        moved += 1
    os.rmdir(src)
    volume.commit()
    remaining = sorted(os.listdir(dst))
    return {
        "moved": moved,
        "label_entries_after": len(remaining),
        "first_three": remaining[:3],
        "last_three": remaining[-3:],
    }


@app.local_entrypoint()
def main():
    result = fix.remote()
    print(result)
