#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plot voxel-shell count r₃(n) and cumulative V(n) vs R², compared to
cube P³ references and the continuous (4/3)πR³ approximation.

Outputs site/public/ball-voxel-counts.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


def compute_counts(max_r2: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (shell, cum) arrays of length max_r2+1."""
    R = int(np.ceil(np.sqrt(max_r2))) + 1
    shell = np.zeros(max_r2 + 1, dtype=np.int64)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= max_r2:
                    shell[r2] += 1
    return shell, shell.cumsum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-r2", type=int, default=200)
    ap.add_argument("--output", type=Path,
                    default=Path("site/public/ball-voxel-counts.png"))
    args = ap.parse_args()

    shell, cum = compute_counts(args.max_r2)

    r2 = np.arange(args.max_r2 + 1)
    R = np.sqrt(r2)
    continuous = (4 / 3) * np.pi * R ** 3

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]},
    )

    # ---- top panel: shell counts r₃(R²) ----
    ax1.bar(r2, shell, width=0.9, color="#4a90e2", alpha=0.85,
            label="$r_3(R^2)$ — shell count (integer (i,j,k) with $i^2+j^2+k^2 = R^2$)")
    # Mark r3(n)=0 spots on the x-axis.
    zeros = np.where(shell == 0)[0]
    ax1.scatter(zeros, np.zeros_like(zeros), color="red", s=18, zorder=5,
                label=f"$r_3=0$ (gap; n of form $4^a(8b{{+}}7)$) — {len(zeros)} gaps")
    ax1.set_ylabel("shell count  $r_3(R^2)$")
    ax1.set_title("Integer lattice shell counts vs $R^2$  —  \"sum-of-three-squares\" (OEIS A005875)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, shell[1:].max() * 1.15)

    # ---- bottom panel: cumulative + continuous + cube refs ----
    ax2.plot(r2, cum, color="#2b5f9c", lw=2,
             label="$V(R^2)$ — cumulative voxel count in ball (OEIS A117609)")
    ax2.plot(r2, continuous, "--", color="#d98f22", lw=1.8,
             label=r"$\frac{4}{3}\pi R^3$  (continuous-sphere volume)")
    # Cube reference dots
    cubes = [(P, P ** 3) for P in range(4, 21)]
    for P, V in cubes:
        r2_match = np.searchsorted(cum, V)
        if r2_match > args.max_r2:
            continue
        ax2.scatter([r2_match], [V], color="#d9471f", s=32, zorder=6)
        if P in (6, 8, 10, 12, 14, 15, 16, 18, 19, 20):
            ax2.annotate(f"P={P}", (r2_match, V), xytext=(5, 4),
                         textcoords="offset points", fontsize=9, color="#d9471f")
    # Vertical lines at the key ablation-R² values
    for r2v, lbl in [(75, "R²=75\n(≈P=14)"), (86, "R²=86\n(≈P=15)"),
                     (138, "R²=138\n(P=19 exact)"), (153, "R²=153\n(≈P=20)")]:
        if r2v <= args.max_r2:
            ax2.axvline(r2v, color="#2b8a3e", lw=0.8, ls=":", alpha=0.6)
            ax2.text(r2v, cum[r2v] * 0.03 + 300, lbl, rotation=90,
                     fontsize=8, color="#2b8a3e", ha="right", va="bottom")

    ax2.set_xlabel("$R^2$ (integer squared-radius threshold)")
    ax2.set_ylabel("cumulative voxel count")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_xlim(0, args.max_r2)
    ax2.set_ylim(0, cum.max() * 1.05)

    # Dashed x-axis labels showing corresponding R = sqrt(R²)
    r2_ticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    r2_ticks = [r for r in r2_ticks if r <= args.max_r2]
    ax2.set_xticks(r2_ticks)
    ax2.set_xticklabels([f"{r}\n(R={np.sqrt(r):.2f})" for r in r2_ticks], fontsize=8)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
