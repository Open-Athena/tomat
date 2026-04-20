"""Shared helpers for the fidelity sweep scripts.

Extracted from ``scripts/fidelity_sweep.py`` so both the local CLI and the
Modal-based per-sample parallel runner (``scripts/fidelity_sweep_modal.py``)
can reuse the same tokenizer configuration + metric definitions.

Canonical CSV column order is exposed as :data:`CSV_FIELDNAMES`.
"""

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from tomat.float_codec import FP16Codec
from tomat.tokenizers import (
    CutoffTokenizer,
    DeltaDensityTokenizer,
    DensityTokenizer,
    DirectCodedTokenizer,
    DirectTokenizer,
    FourierCodedTokenizer,
    FourierTokenizer,
)

DEFAULT_CODEC_CONFIG = Path("configs/fp16-channels.json")

CSV_FIELDNAMES = [
    "mp_id", "category", "config", "tokens",
    "nmae", "chi_sq", "hellinger", "jsd", "weighted_mae",
    "seconds", "grid", "mass_captured", "effective_threshold",
]

err = partial(print, file=sys.stderr)


@dataclass
class SweepConfig:
    label: str
    tokenizer: DensityTokenizer


def default_configs(codec_config: Path = DEFAULT_CODEC_CONFIG) -> list[SweepConfig]:
    """Each cutoff/fourier entry keeps a fraction of the representation.

    Label convention reflects what's kept:

    * ``cutoff-top-Npct`` = top N% of voxels by density value.
    * ``fourier-lowg-Npct`` = lowest N% of FFT coefficients by |G| (smoothest modes).

    Both sort by the scheme's natural ranking, but the ranking criterion differs.
    """
    cfgs: list[SweepConfig] = [SweepConfig("direct", DirectTokenizer())]
    density_codec: FP16Codec | None = None
    fourier_codec: FP16Codec | None = None
    if codec_config.exists():
        density_codec = FP16Codec.from_json(codec_config, channel="density")
        fourier_codec = FP16Codec.from_json(codec_config, channel="fourier")
        cfgs.append(SweepConfig("direct-coded", DirectCodedTokenizer(codec=density_codec)))
    else:
        err(f"[warn] {codec_config} missing; skipping coded variants. "
            "Run scripts/fit_density_codec.py to generate it.")
    for frac in (0.01, 0.05, 0.25, 1.0):
        cfgs.append(SweepConfig(f"cutoff-top-{frac * 100:g}pct", CutoffTokenizer(top_fraction=frac)))
    for frac in (0.0025, 0.005, 0.01, 0.05, 0.25, 1.0):
        cfgs.append(SweepConfig(f"fourier-lowg-{frac * 100:g}pct", FourierTokenizer(coefficient_fraction=frac)))
        if fourier_codec is not None:
            cfgs.append(SweepConfig(
                f"fourier-coded-lowg-{frac * 100:g}pct",
                FourierCodedTokenizer(FourierTokenizer(coefficient_fraction=frac), codec=fourier_codec),
            ))
    # Δρ-Fourier variants — scheme 4 + scheme 5 composed.
    for frac in (0.0025, 0.005, 0.01, 0.05, 0.25):
        cfgs.append(
            SweepConfig(
                f"delta-fourier-lowg-{frac * 100:g}pct",
                DeltaDensityTokenizer(FourierTokenizer(coefficient_fraction=frac)),
            )
        )
        if fourier_codec is not None:
            cfgs.append(SweepConfig(
                f"delta-fourier-coded-lowg-{frac * 100:g}pct",
                DeltaDensityTokenizer(
                    FourierCodedTokenizer(FourierTokenizer(coefficient_fraction=frac), codec=fourier_codec),
                ),
            ))
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
