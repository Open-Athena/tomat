#!/usr/bin/env python
"""Fit ``(log_min, log_max)`` for the FP16-like codec, per channel.

Two channels:

* ``density`` — per-voxel ρ values (signed in principle; CHGCARs carry
  small negative regions from FFT aliasing).
* ``fourier`` — real and imaginary parts of ``rfftn(ρ)`` coefficients,
  fitted jointly. Fourier coefficients span a much wider dynamic range
  than real-space density: DC ≈ sum(ρ) ~ 10³–10⁴ on one end, high-|G|
  coefficients ~ 10⁻¹⁰ on the other.

Mirrors ``Open-Athena/tomol:build_fp16_config.py`` (one channel per
dimension, same 0.01 / 99.99 percentile + 1-log-unit padding).

Usage::

    scripts/fit_density_codec.py -n 50 -o configs/fp16-channels.json
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

DEFAULT_OUTPUT = Path("configs/fp16-channels.json")


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

    density_samples: list[np.ndarray] = []
    fourier_samples: list[np.ndarray] = []
    for i, mp_id in enumerate(ids, 1):
        chgcar = load_chgcar(mp_id)
        rho = np.asarray(chgcar.data['total'], dtype=np.float64)
        coefs = np.fft.rfftn(rho)
        density_samples.append(rho.ravel())
        # Fit real+imag jointly — they share a channel in the codec.
        fourier_samples.append(np.concatenate([coefs.real.ravel(), coefs.imag.ravel()]))
        err(f"  [{i}/{len(ids)}] {mp_id}: "
            f"ρ ∈ [{rho.min():.2e}, {rho.max():.2e}]  "
            f"|F| ∈ [{np.abs(coefs).min():.2e}, {np.abs(coefs).max():.2e}]")

    density_all = np.concatenate(density_samples)
    fourier_all = np.concatenate(fourier_samples)
    err(f"Density voxels: {density_all.size:,}")
    err(f"Fourier components (real+imag): {fourier_all.size:,}")

    density_range = compute_log_range(
        density_all, percentile_low=percentile_low, percentile_high=percentile_high, padding=padding,
    )
    # Fourier's DC component is a single huge value per structure (= sum ρ × V).
    # p99.99 discards those outliers; clipping them injects a massive uniform
    # spatial error on decode. Use the actual max on the high side.
    fourier_range = compute_log_range(
        fourier_all, percentile_low=percentile_low, percentile_high=100.0, padding=padding,
    )
    err(f"density: log_min={density_range[0]:.3f}, log_max={density_range[1]:.3f} "
        f"({density_range[1] - density_range[0]:.2f} decades)")
    err(f"fourier: log_min={fourier_range[0]:.3f}, log_max={fourier_range[1]:.3f} "
        f"({fourier_range[1] - fourier_range[0]:.2f} decades)")

    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "encoding_type": "fp16_like",
        "n_samples_fitted": len(ids),
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "padding_log_units": padding,
        "channels": {
            "density": {"log_min": density_range[0], "log_max": density_range[1]},
            "fourier": {"log_min": fourier_range[0], "log_max": fourier_range[1]},
        },
    }
    with output.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    err(f"Wrote {output}")


if __name__ == '__main__':
    main()
