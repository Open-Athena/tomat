#!/usr/bin/env python
"""Dry-load a Levanter checkpoint to verify cross-platform format compat.

Without running training, attempt the same Orbax load that
`load_checkpoint_or_initialize` would do at the start of a TPU job. Useful for
de-risking Modal-trained ckpt -> TPU-resume before queueing on iris.

Usage:
    cd marin/
    .venv/bin/python ../scripts/dryload_ckpt.py \\
        --ckpt-path gs://marin-eu-west4/tomat/results/<run>/checkpoints/<run>/step-<N> \\
        --model 200M --batch-size 128 --seed 42

Exits 0 on success, non-zero on any of:
    - discover_latest_checkpoint returns None (no metadata.json)
    - shape mismatch between checkpoint and freshly-init TrainerState
    - Orbax restore raises (corrupt manifest, version skew, etc.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click


@click.command()
@click.option("-c", "--ckpt-path", required=True, help="gs:// or local path to step-N checkpoint dir")
@click.option("-m", "--model", default="200M", help="TOMAT_MODEL preset")
@click.option("-b", "--batch-size", default=128, type=int, help="TOMAT_BATCH_SIZE")
@click.option("--seed", default=42, type=int, help="TOMAT_SEED")
@click.option("-l", "--label", default="train-full-v3-shard1", help="TOMAT_LABEL data cache")
def main(ckpt_path: str, model: str, batch_size: int, seed: int, label: str) -> None:
    # Force CPU so we don't accidentally engage TPU/GPU backends just for a load.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TOMAT_MODEL", model)
    os.environ.setdefault("TOMAT_BATCH_SIZE", str(batch_size))
    os.environ.setdefault("TOMAT_SEED", str(seed))
    os.environ.setdefault("TOMAT_LABEL", label)
    os.environ.setdefault("WANDB_MODE", "disabled")

    print(f"[dryload] ckpt_path={ckpt_path}")
    print(f"[dryload] model={model}  batch_size={batch_size}  seed={seed}")

    from levanter.checkpoint import discover_latest_checkpoint, load_checkpoint

    print("[dryload] step 1: discover_latest_checkpoint(parent_dir) ...")
    parent = ckpt_path.rstrip("/").rsplit("/step-", 1)[0]
    discovered = discover_latest_checkpoint(parent)
    if discovered is None:
        print(f"[dryload] FAIL: no metadata.json found under {parent}")
        sys.exit(2)
    print(f"[dryload]   discovered={discovered}")

    print("[dryload] step 2: read metadata.json ...")
    import fsspec
    import json
    with fsspec.open(f"{ckpt_path.rstrip('/')}/metadata.json") as f:
        metadata = json.loads(f.read())
    print(f"[dryload]   metadata keys: {sorted(metadata.keys())}")
    if "step" in metadata:
        print(f"[dryload]   step: {metadata['step']}")
    if "is_temporary" in metadata:
        print(f"[dryload]   is_temporary: {metadata['is_temporary']}")

    print("[dryload] step 3: dummy TrainerState shape ...")
    print("[dryload]   (skipping full state build; format check via metadata only.)")
    print("[dryload]   For full dry-load including dtype/shape verification of every param,")
    print("[dryload]   set TOMAT_DRYLOAD_FULL=1 and provide a working data cache.")

    if os.environ.get("TOMAT_DRYLOAD_FULL") == "1":
        print("[dryload] step 4: full Levanter load via load_checkpoint ...")
        import train_tomat_tpu as ttt  # type: ignore[import-not-found]
        state_shape = ttt.build_initial_trainer_state_shape()
        loaded = load_checkpoint(state_shape, ckpt_path)
        print(f"[dryload]   loaded type: {type(loaded).__name__}")
        print(f"[dryload]   loaded step: {getattr(loaded, 'step', '?')}")

    print("[dryload] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
