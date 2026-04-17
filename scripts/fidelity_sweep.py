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

from tomat.data.classify import classify_elements
from tomat.data.mp import load_chgcar, list_mp_ids
from tomat.tokenizers import (
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
    for frac in (0.0025, 0.005, 0.01, 0.05, 0.25, 1.0):
        cfgs.append(SweepConfig(f"fourier-lowg-{frac * 100:g}pct", FourierTokenizer(coefficient_fraction=frac)))
    # Δρ-Fourier variants — scheme 4 + scheme 5 composed.
    # PADS is crude Gaussian (see `src/tomat/pads.py` caveat); results
    # bound the "Δρ with a very approximate promolecule density" scenario.
    for frac in (0.0025, 0.005, 0.01, 0.05, 0.25):
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


def compute_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Return a dict of reconstruction-error metrics.

    Addresses Yael's Jan 2026 observation that NMAE is dominated by
    high-density (near-nucleus) regions, under-weighting the chemically
    interesting low-density bonding signal. Reporting multiple metrics
    with different ρ-weighting behaviors gives a fuller picture.

    All metrics are scalar; lower is better; 0 = perfect reconstruction.

    * ``nmae``: standard — sum|Δ| / sum|ρ|. High-ρ dominated.
    * ``chi_sq``: Σ Δ² / max(|ρ|, ε). Amplifies errors at low-ρ voxels.
    * ``hellinger``: sqrt(½ Σ (√|ρ| − √|ρ̂|)²) / sqrt(sum|ρ|). Mid-weighted.
    * ``jsd``: Jensen-Shannon divergence between normalized |ρ|, |ρ̂|
      treated as probability distributions.
    * ``weighted_mae``: Σ |Δ|/max(|ρ|,ε) / N. Low-ρ emphasized.
    """
    ref = np.abs(reference).astype(np.float64).ravel()
    rec = np.abs(reconstruction).astype(np.float64).ravel()
    diff = ref - rec
    eps = max(float(ref.max()) * 1e-8, 1e-12)

    total = ref.sum()
    metrics: dict[str, float] = {
        "nmae": float(np.abs(reference - reconstruction).sum() / total),
        "chi_sq": float((diff * diff / np.maximum(ref, eps)).sum() / total),
        "hellinger": float(
            np.sqrt(0.5 * ((np.sqrt(ref) - np.sqrt(rec)) ** 2).sum()) / np.sqrt(total)
        ),
        "weighted_mae": float(
            (np.abs(diff) / np.maximum(ref, eps)).sum() / ref.size
        ),
    }

    # Jensen-Shannon over normalized distributions.
    from scipy.spatial.distance import jensenshannon
    if total > 0 and rec.sum() > 0:
        # scipy returns sqrt(JSD); square to get JSD itself, range [0, ln(2)].
        metrics["jsd"] = float(jensenshannon(ref / total, rec / rec.sum()) ** 2)
    else:
        metrics["jsd"] = 0.0

    return metrics


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
            metrics = compute_metrics(density, recon)
            row = dict(
                mp_id=mp_id,
                category=category,
                config=cfg.label,
                seconds=elapsed,
                grid=str(density.shape),
                **metrics,
            )
            if isinstance(encoded, CutoffEncoded):
                row["mass_captured"] = encoded.mass_captured
                row["effective_threshold"] = encoded.effective_threshold
                err(
                    f"  {cfg.label:>26s}  NMAE={metrics['nmae']:.3e}  χ²={metrics['chi_sq']:.3e}  "
                    f"H={metrics['hellinger']:.3e}  mass={encoded.mass_captured:.3f}  ({elapsed:.2f}s)"
                )
            else:
                err(
                    f"  {cfg.label:>26s}  NMAE={metrics['nmae']:.3e}  χ²={metrics['chi_sq']:.3e}  "
                    f"H={metrics['hellinger']:.3e}  ({elapsed:.2f}s)"
                )
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
        fieldnames = [
            "mp_id", "category", "config",
            "nmae", "chi_sq", "hellinger", "jsd", "weighted_mae",
            "seconds", "grid", "mass_captured", "effective_threshold",
        ]
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        err(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
