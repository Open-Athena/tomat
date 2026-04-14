#!/usr/bin/env python
"""Reconstruction-fidelity sweep for density tokenizers.

For each configured tokenizer, round-trip ``N`` MP CHGCARs through
encode→decode and report NMAE = sum|ρ' − ρ| / sum|ρ| against the original
density grid. This is the per-structure information-loss floor — any
downstream model trained on these tokens inherits it as a lower bound on
achievable error.

Usage:

    scripts/fidelity_sweep.py                 # 5 samples, all default configs
    scripts/fidelity_sweep.py -n 20           # 20 samples
    scripts/fidelity_sweep.py -n 3 -o out.csv
"""

import csv
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import numpy as np
from click import argument, command, option

from tomato.data.classify import classify_elements
from tomato.data.mp import load_chgcar, list_mp_ids
from tomato.tokenizers import (
    CutoffEncoded,
    CutoffTokenizer,
    DeltaDensityTokenizer,
    DensityTokenizer,
    DirectTokenizer,
    FourierTokenizer,
)

err = partial(print, file=sys.stderr)


@dataclass
class SweepConfig:
    label: str
    tokenizer: DensityTokenizer


def default_configs() -> list[SweepConfig]:
    """Each cutoff/fourier entry keeps a fraction of the representation.

    Label convention reflects what's kept:

    * ``cutoff-top-Npct`` = top N% of voxels by density value.
    * ``fourier-lowg-Npct`` = lowest N% of FFT coefficients by |G| (smoothest modes).

    Both sort by the scheme's natural ranking, but the ranking criterion differs.
    """
    cfgs: list[SweepConfig] = [SweepConfig("direct", DirectTokenizer())]
    for frac in (0.01, 0.05, 0.25, 1.0):
        cfgs.append(SweepConfig(f"cutoff-top-{frac * 100:g}pct", CutoffTokenizer(top_fraction=frac)))
    for frac in (0.01, 0.05, 0.25, 1.0):
        cfgs.append(SweepConfig(f"fourier-lowg-{frac * 100:g}pct", FourierTokenizer(coefficient_fraction=frac)))
    # Δρ-Fourier variants — scheme 4 + scheme 5 composed.
    # PADS is crude Gaussian (see `src/tomato/pads.py` caveat); results
    # bound the "Δρ with a very approximate promolecule density" scenario.
    for frac in (0.01, 0.05, 0.25):
        cfgs.append(
            SweepConfig(
                f"delta-fourier-lowg-{frac * 100:g}pct",
                DeltaDensityTokenizer(FourierTokenizer(coefficient_fraction=frac)),
            )
        )
    return cfgs


def nmae(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    denom = float(np.abs(reference).sum())
    if denom == 0.0:
        raise ValueError("Reference density sum is zero — check input CHGCAR")
    return float(np.abs(reference - reconstruction).sum() / denom)


@command()
@option("-n", "--n-samples", type=int, default=5, help="Number of MP entries to evaluate")
@option("-o", "--output-csv", type=click.Path(dir_okay=False, path_type=Path), help="Write per-(sample, config) rows to CSV")
@option("-s", "--split", type=click.Choice(["data", "label"]), default="label", help="Which CHGCAR to tokenize; 'label' = DFT-converged target")
@argument("mp_ids", nargs=-1)
def main(n_samples: int, output_csv: Path | None, split: str, mp_ids: tuple[str, ...]):
    ids = list(mp_ids) if mp_ids else list_mp_ids()[:n_samples]
    err(f"Running sweep over {len(ids)} mp ids (split={split})")

    configs = default_configs()
    rows: list[dict] = []

    for mp_id in ids:
        t0 = time.perf_counter()
        chgcar = load_chgcar(mp_id, split=split)
        density = np.asarray(chgcar.data["total"], dtype=np.float64)
        category = classify_elements(el.symbol for el in chgcar.structure.composition.elements)
        load_s = time.perf_counter() - t0
        err(f"\n{mp_id} [{category}]: grid {density.shape}, sum ρ = {density.sum():.3e}, loaded in {load_s:.1f}s")
        for cfg in configs:
            t0 = time.perf_counter()
            encoded = cfg.tokenizer.encode(chgcar)
            recon = cfg.tokenizer.decode(encoded)
            elapsed = time.perf_counter() - t0
            val = nmae(density, recon)
            row = dict(mp_id=mp_id, category=category, config=cfg.label, nmae=val, seconds=elapsed, grid=str(density.shape))
            if isinstance(encoded, CutoffEncoded):
                row["mass_captured"] = encoded.mass_captured
                row["effective_threshold"] = encoded.effective_threshold
                err(f"  {cfg.label:>24s}  NMAE={val:.4e}  mass={encoded.mass_captured:.3f}  thresh={encoded.effective_threshold:.3e}  ({elapsed:.2f}s)")
            else:
                err(f"  {cfg.label:>24s}  NMAE={val:.4e}  ({elapsed:.2f}s)")
            rows.append(row)

    print()
    print(f"{'config':>24s}  {'mean NMAE':>12s}  {'median NMAE':>12s}  n")
    by_config: dict[str, list[float]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row["nmae"])
    for label, vals in by_config.items():
        arr = np.array(vals)
        print(f"{label:>24s}  {arr.mean():.4e}   {np.median(arr):.4e}   {len(arr)}")

    if output_csv is not None:
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["mp_id", "category", "config", "nmae", "seconds", "grid", "mass_captured", "effective_threshold"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        err(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
