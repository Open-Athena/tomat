#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib", "fsspec", "gcsfs"]
# ///
"""Visualize LMQ codec partitions vs the empirical density PDF.

Loads {16k, 32k, 65k} LMQ codecs from GCS + their `per_bin_count` metadata
(the train-full empirical distribution is implicit in those counts). Produces:

  Panel A: log-x density histogram with bin boundaries overlaid for each codec.
  Panel B: per-bin quantization error (MAE assumption) for each codec.
  Panel C: mean-vs-median recon point comparison (per-bin).

Output: site/public/lmq-codecs.png (4-panel composite).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

CODECS = [
    ("16k", "gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz", "#1f77b4"),
    ("32k", "gs://marin-eu-west4/tomat/codecs/lmq-v2-32k.npz", "#2ca02c"),
    ("65k", "gs://marin-eu-west4/tomat/codecs/lmq-v2-65k.npz", "#d62728"),
]


def load_codec(path):
    with fsspec.open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        return {
            "boundaries": np.asarray(data["boundaries"], dtype=np.float64),
            "recon": np.asarray(data["recon_points"], dtype=np.float64),
            "per_bin": np.asarray(data["per_bin_count"], dtype=np.int64),
            "clip_max": float(data["clip_max"]),
            "lin_lo": float(data["lin_lo"]),
            "lin_hi": float(data["lin_hi"]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("site/public/lmq-codecs.png"))
    args = ap.parse_args()

    codecs = []
    for label, path, color in CODECS:
        c = load_codec(path)
        codecs.append((label, c, color))
        print(f"{label}: n_bins={len(c['recon']):,}, clip_max={c['clip_max']:.0f}, "
              f"recon range=[{c['recon'].min():.3e}, {c['recon'].max():.3e}], "
              f"total counts={int(c['per_bin'].sum()):,}")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # Panel 1: log-density distribution from per_bin_count of the 65k codec
    # (highest fidelity — counts naturally smooth out)
    ax = axes[0]
    label, c65, color = "65k", codecs[2][1], codecs[2][2]
    log_recon = np.log10(np.maximum(c65["recon"], 1e-12))
    ax.hist(log_recon, bins=200, weights=c65["per_bin"],
            color=color, alpha=0.6,
            label=f"weighted by 65k LMQ recon points (proxy for empirical PDF)")
    ax.set_xlabel("log10 density (e/bohr³)")
    ax.set_ylabel("voxel count")
    ax.set_yscale("log")
    ax.set_title("Empirical density distribution (train-full, from LMQ-v2-65k bin counts)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    # Mark log_min/log_max from old codec
    ax.axvline(-4.13, ls="--", color="black", alpha=0.4, label="old fp16 codec range [-4.13, 4.97]")
    ax.axvline(4.97, ls="--", color="black", alpha=0.4)

    # Panel 2: bin-boundary density (cumulative density of bin edges, log-x)
    ax = axes[1]
    for label, c, color in codecs:
        # Sort recon points and plot CDF of bin-boundary placements
        rec = c["recon"]
        cum_rank = np.linspace(0, 1, len(rec))
        ax.plot(rec, cum_rank, color=color, lw=1.5, label=f"{label} bins ({len(rec):,})")
    # Reference: ideal 1:1 mapping of "density quantile" → bin index
    ax.set_xscale("log")
    ax.set_xlabel("density (e/bohr³, log scale)")
    ax.set_ylabel("normalized bin rank (0..1)")
    ax.set_title("Bin-boundary placement vs density (log scale): "
                 "Lloyd-Max concentrates bins where the data is")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Panel 3: per-bin quantization error (in MAE units)
    ax = axes[2]
    for label, c, color in codecs:
        # Estimate per-bin MAE: half the bin width (approximation for uniform-mass bins)
        bounds = c["boundaries"]  # (n_bins-1,)
        recon = c["recon"]
        # bin widths: distance between adjacent recon points
        widths = np.empty_like(recon)
        widths[:-1] = recon[1:] - recon[:-1]
        widths[-1] = widths[-2]  # rough
        # MAE within a bin, assuming uniform distribution within: width/4 (L1) or width/(2*sqrt(3)) (L2)
        # Use width/4 for L1.
        per_bin_mae = widths / 4
        ax.plot(recon, per_bin_mae, color=color, lw=1.0, alpha=0.8, label=f"{label}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("density (e/bohr³)")
    ax.set_ylabel("per-bin MAE (≈ width/4)")
    ax.set_title("Quantization error per bin vs density value")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
