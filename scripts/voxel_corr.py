#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "matplotlib", "numpy", "utz"]
# ///
"""Voxel-position correlation analysis from `--dump-predictions` .npz files.

Stacks predictions across mats into a (N_patches_total, P³) matrix, then
computes correlations between voxel-positions to see whether the model has
learned spatial structure at the patch's raster offsets.

Two outputs:
  1. **1D collapse**: avg correlation between positions at row-major offset k,
     plotted over k ∈ [0, P³). Peaks at k=1 (primary axis), P (secondary
     axis), P² (tertiary axis) = the model is using spatial neighbors.
     Flat or noisy = it isn't.
  2. **2D heatmap**: full P³×P³ correlation matrix, downsampled to a viewable
     image. Diagonals at multiples of P, P² would be the same signal.

Usage:
    voxel_corr.py tmp/predictions -P 19 -o tmp/voxel_corr.png
"""
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from utz.cli import arg, cmd, flag, opt


def _load_pred_stack(d: Path, kind: str = "pred") -> tuple[np.ndarray, list[str]]:
    """Return (n_patches_total, P³) stacked predictions + mp_ids contributing."""
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files under {d}")
    chunks = []
    mp_ids = []
    P3 = None
    for f in files:
        with np.load(f) as z:
            arr = z[kind].astype(np.float32)  # (n_patches, P³)
        if P3 is None:
            P3 = arr.shape[1]
        elif arr.shape[1] != P3:
            raise ValueError(f"P³ mismatch: {f.name} has {arr.shape[1]}, expected {P3}")
        chunks.append(arr)
        mp_ids.append(f.stem)
    stacked = np.concatenate(chunks, axis=0)
    return stacked, mp_ids


def _corr_offset_curve(X: np.ndarray) -> np.ndarray:
    """1D collapse: c[k] = avg corr(ρ_i, ρ_{i+k}) over all i."""
    # Standardize along the patches axis to get unit-variance per voxel-position.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd                       # (N, P³)
    N, P3 = Z.shape
    # corr(i, j) = (1/N) · Z[:, i] · Z[:, j]. The full Z.T @ Z / N is P³×P³.
    # For 6859² ≈ 47M entries this is fine on float32 (~190 MB).
    C = (Z.T @ Z) / N                       # (P³, P³)
    # Reduce to a 1D curve over offset k = j - i (k ≥ 0): average the values
    # of C along each anti-diagonal-of-positive-offset.
    out = np.zeros(P3, dtype=np.float32)
    for k in range(P3):
        out[k] = np.mean(np.diagonal(C, offset=k))
    return out


def _downsample_corr(X: np.ndarray, target: int = 600) -> np.ndarray:
    """Compute corr matrix then downsample to (target, target) for heatmap."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    N, P3 = Z.shape
    C = (Z.T @ Z) / N                       # (P³, P³)
    # Block-average to (target, target).
    if P3 <= target:
        return C
    bin_size = P3 // target
    n_bins = P3 // bin_size
    trim = n_bins * bin_size
    C = C[:trim, :trim].reshape(n_bins, bin_size, n_bins, bin_size).mean(axis=(1, 3))
    return C


@cmd()
@opt("-P", "--patch-size", type=int, default=19, help="patch side P; P³ must match the .npz pred-axis dim")
@opt("-o", "--out", default="tmp/voxel_corr.png", help="output PNG path")
@opt("-k", "--kind", type=click.Choice(["pred", "true", "residual"]), default="pred",
     help="which series to correlate: model predictions, true rho, or residual (pred-true)")
@opt("-t", "--top-k-markers", type=int, default=12, help="how many top-corr-offset peaks to annotate")
@arg("preds_dir")
def main(
    preds_dir: str,
    patch_size: int,
    out: str,
    kind: str,
    top_k_markers: int,
) -> None:
    """Stack `.npz` predictions then render corr-vs-offset curve + heatmap."""
    P = patch_size
    P3 = P ** 3
    src = Path(preds_dir)

    if kind == "residual":
        X_pred, _ = _load_pred_stack(src, "pred")
        X_true, _ = _load_pred_stack(src, "true")
        X = X_pred - X_true
        label = "residual"
    else:
        X, _ = _load_pred_stack(src, kind)
        label = kind

    if X.shape[1] != P3:
        raise ValueError(f"P³={P3} but data has {X.shape[1]} voxel positions")

    print(f"[voxel-corr] loaded {X.shape[0]} patches × {X.shape[1]} positions "
          f"({label}, P={P})", file=sys.stderr)
    print(f"[voxel-corr] computing 1D offset curve…", file=sys.stderr)
    curve = _corr_offset_curve(X)
    print(f"[voxel-corr] downsampling 2D heatmap…", file=sys.stderr)
    C_small = _downsample_corr(X, target=600)

    # Markers at k=1, P, P² (the spatially-adjacent offsets in raster order).
    canonical = [1, P, P ** 2]

    # ── Render ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[1, 2.5])
    fig.suptitle(
        f"Voxel-position correlations — {label} ({X.shape[0]} patches, P={P})",
        fontsize=12,
    )

    # (1) 1D curve over offset k.
    ax = axes[0]
    ax.plot(curve, lw=0.8, color="#1976d2")
    ax.set_xlim(0, P3)
    ax.set_xlabel("row-major position offset k")
    ax.set_ylabel("avg corr(ρ_i, ρ_{i+k})")
    ax.set_title("1D: avg correlation at offset k")
    ax.grid(True, alpha=0.3)
    for k in canonical:
        if k < P3:
            ax.axvline(k, ls="--", color="#e91e63", alpha=0.6, lw=0.8)
            ax.text(k, ax.get_ylim()[1] * 0.95, f"k={k}", color="#e91e63",
                    fontsize=8, rotation=90, va="top", ha="right")
    # Top-k local-max markers (anything above the local average).
    if top_k_markers > 0:
        local = curve.copy()
        local[:1] = -1  # ignore k=0 self-corr
        peaks = np.argsort(local)[-top_k_markers:]
        peaks = sorted(peaks)
        for k in peaks:
            if k in canonical:
                continue
            ax.scatter([k], [curve[k]], s=12, color="#ff9800", zorder=5)

    # (2) Downsampled 2D heatmap.
    ax = axes[1]
    im = ax.imshow(C_small, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[0, P3, 0, P3], aspect="equal")
    plt.colorbar(im, ax=ax, label="correlation")
    ax.set_xlabel("position j")
    ax.set_ylabel("position i")
    ax.set_title(f"2D: corr(ρ_i, ρ_j), downsampled to {C_small.shape[0]}×{C_small.shape[0]}")
    # Diagonal lines at offsets P and P².
    for k in canonical[1:]:
        if k < P3:
            ax.plot([0, P3 - k], [k, P3], color="#e91e63", alpha=0.4, lw=0.6, ls="--")
            ax.plot([k, P3], [0, P3 - k], color="#e91e63", alpha=0.4, lw=0.6, ls="--")

    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"[voxel-corr] wrote {out_path}", file=sys.stderr)

    # Also print a quick numerical summary.
    print()
    print("offset k  |  avg corr  |  note")
    print("-" * 40)
    for k in canonical:
        if k < P3:
            print(f"{k:>8d}  |  {curve[k]:>8.4f}  |  "
                  f"{'primary axis' if k == 1 else 'secondary axis (+P)' if k == P else 'tertiary axis (+P²)'}")


if __name__ == "__main__":
    main()
