#!/usr/bin/env python
"""Enumerate + dispatch preprocessing runs for the (codec × patch_size) sweep.

Produces one parquet-shard directory per ``(codec, patch_size)`` combination,
named ``<output-root>/<codec>-P<patch_size>/``. Token-count estimates in
this module refer to mp-1775382 (14 atoms, grid 80×80×144); real
structures will vary by atom count + preamble length.

Run all valid combos for 8k context:

    scripts/sweep_preprocess.py \\
        --rho-gga-dir /path/to/rho_gga \\
        --split-file  /path/to/split_limit_22M.json \\
        --split       validation \\
        --patches-per-material 32 \\
        --output-root data/tokenized/val \\
        --context-budget 8192

or dry-run to just print the commands:

    scripts/sweep_preprocess.py ... --dry-run

Sweep axes:

* ``codec`` ∈ {``tomol_3byte``, ``two_token_9_12``, ``fp16_1token``} — 3 of
  our :class:`FP16Codec` variants; affects vocab size (1 k / 4.6 k / 65 k)
  and tokens-per-density-value (3 / 2 / 1).
* ``patch_size`` ∈ {12, 14, 16} — real-space cube edge; affects token count
  as ``P³ × tokens_per_value + preamble``.

Orthogonal training-time axes (not varied here, need one training config
per value):

* ``M`` (patches per material) — controlled at preprocessing by
  ``--patches-per-material``. Preprocess at the largest you want to
  sweep (e.g. 128) and sub-sample at train time via Levanter's
  ``max_train_batches``.
* ``N`` (shuffle buffer size) — pure training-time param, plugged into
  Levanter's ``BlockShuffleConfig``. No preprocessing impact.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from functools import partial
from pathlib import Path

import click
from click import command, option

from tomat.training.sweep import all_configs

err = partial(print, file=sys.stderr)


@command()
@option('-r', '--rho-gga-dir', required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@option('-s', '--split-file',  required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@option('-k', '--split',       default='validation')
@option('-m', '--patches-per-material', type=int, default=32)
@option('-o', '--output-root', required=True, type=click.Path(path_type=Path))
@option('-S', '--seed', type=int, default=42)
@option('-b', '--context-budget', type=int, default=8192,
        help='Skip configs whose estimated token count exceeds this.')
@option('-n', '--n-materials', type=int, default=None,
        help='Debug: cap number of materials per run.')
@option('-d', '--dry-run', is_flag=True, help='Print commands; do not execute.')
def main(
    rho_gga_dir: Path,
    split_file: Path,
    split: str,
    patches_per_material: int,
    output_root: Path,
    seed: int,
    context_budget: int,
    n_materials: int | None,
    dry_run: bool,
) -> None:
    configs = all_configs()
    valid = [c for c in configs if c.fits(context_budget)]
    dropped = [c for c in configs if c not in valid]

    err(f"[sweep] {len(configs)} total configs; {len(valid)} fit in context={context_budget}")
    for c in valid:
        err(f"  ✓ {c.label:25s}  ~{c.estimated_context:>5d} tokens")
    for c in dropped:
        err(f"  ✗ {c.label:25s}  ~{c.estimated_context:>5d} tokens  (over budget)")

    for cfg in valid:
        out = output_root / cfg.label
        cmd = [
            "scripts/tokenize_patches.py",
            "-r", str(rho_gga_dir),
            "-s", str(split_file),
            "-k", split,
            "-m", str(patches_per_material),
            "-p", str(cfg.patch_size),
            "-c", cfg.codec,
            "-o", str(out),
            "-S", str(seed),
        ]
        if n_materials is not None:
            cmd += ["-n", str(n_materials)]

        err(f"\n[sweep] → {cfg.label}")
        err("  " + " ".join(shlex.quote(a) for a in cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
