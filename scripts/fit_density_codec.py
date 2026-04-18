#!/usr/bin/env python
"""Fit ``(log_min, log_max)`` for the FP16-like density codec.

Mirrors ``Open-Athena/tomol:build_fp16_config.py`` (Will's config fitter):
takes per-voxel quantiles of ``log10(ρ)`` across a sample of CHGCARs,
adds one log-unit of padding on each side, and writes a JSON config the
codec loads with :meth:`FP16Codec.from_json`.

Usage:

    scripts/fit_density_codec.py -n 50 -o configs/density-fp16.json

The default 0.01 / 99.99 percentile pair matches Will's choice — robust
to the extreme-value tail while still covering essentially all voxels.
"""

import json
import sys
from functools import partial
from pathlib import Path

import click
import numpy as np
from click import command, option

from tomat.data.mp import list_mp_ids, load_chgcar

err = partial(print, file=sys.stderr)

DEFAULT_OUTPUT = Path("configs/density-fp16.json")


def compute_log_range(
    values: np.ndarray,
    *,
    percentile_low: float = 0.01,
    percentile_high: float = 99.99,
    padding: float = 1.0,
) -> tuple[float, float]:
    """``[log10(p_low) - padding, log10(p_high) + padding]`` over ``|values|``."""
    mags = np.abs(values)
    nonzero = mags[mags > 0]
    if not nonzero.size:
        raise ValueError("No non-zero values to fit over.")
    low_mag = np.percentile(nonzero, percentile_low)
    high_mag = np.percentile(nonzero, percentile_high)
    return (
        float(np.log10(low_mag) - padding),
        float(np.log10(high_mag) + padding),
    )


@command()
@option('-n', '--n-samples', type=int, default=50, help='Number of CHGCARs to sample.')
@option('-o', '--output', type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, help='Output JSON path.')
@option('-p', '--padding', type=float, default=1.0, help='Log-unit padding beyond fitted percentiles.')
@option('-l', '--percentile-low', type=float, default=0.01)
@option('-u', '--percentile-high', type=float, default=99.99)
def main(n_samples: int, output: Path, padding: float, percentile_low: float, percentile_high: float) -> None:
    ids = list_mp_ids()[:n_samples]
    err(f"Sampling {len(ids)} CHGCARs…")

    sample = []
    for i, mp_id in enumerate(ids, 1):
        chgcar = load_chgcar(mp_id)
        vox = chgcar.data['total'].ravel()
        # Density (ρ × V_cell in VASP units) is non-negative — but keep abs for
        # safety in case this fitter is ever used on Δρ-like signed inputs.
        sample.append(np.abs(vox).astype(np.float64))
        err(f"  [{i}/{len(ids)}] {mp_id}: {vox.size:,} voxels, "
            f"min={vox.min():.3e}, max={vox.max():.3e}")

    all_vals = np.concatenate(sample)
    nonzero = all_vals[all_vals > 0]
    err(f"Total: {all_vals.size:,} voxels ({nonzero.size:,} non-zero)")

    log_min, log_max = compute_log_range(
        all_vals,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        padding=padding,
    )
    err(f"p{percentile_low}  → log10 = {np.log10(np.percentile(nonzero, percentile_low)):.3f}")
    err(f"p{percentile_high} → log10 = {np.log10(np.percentile(nonzero, percentile_high)):.3f}")
    err(f"padded: log_min={log_min:.3f}, log_max={log_max:.3f} "
        f"({log_max - log_min:.2f} decades)")

    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "encoding_type": "fp16_like",
        "channel": "density",
        "n_samples_fitted": len(ids),
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "padding_log_units": padding,
        "log_min": log_min,
        "log_max": log_max,
    }
    with output.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    err(f"Wrote {output}")


if __name__ == '__main__':
    main()
