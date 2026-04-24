#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plot preamble-size + sequence-length distributions across train-full
materials, showing (P, R, CL) budgets.

Inputs: CSV with (mp_id, nx, ny, nz, n_atoms) from pull_preamble_stats_modal.py.

Output: site/public/preamble-dist.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


PER_ATOM = 10
CUBE_SHARED = 28
BALL_SHARED = 29


def ball_cum_counts(max_r2: int) -> np.ndarray:
    R = int(np.ceil(np.sqrt(max_r2))) + 1
    shell = np.zeros(max_r2 + 1, dtype=np.int64)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= max_r2:
                    shell[r2] += 1
    return shell.cumsum()


def load(csv_path: Path) -> np.ndarray:
    atoms = []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        if header[:5] != ["mp_id", "nx", "ny", "nz", "n_atoms"]:
            raise ValueError(f"bad header: {header}")
        for row in rdr:
            if len(row) != 5:
                continue
            atoms.append(int(row[4]))
    return np.array(atoms, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path("site/public/preamble-dist.png"))
    args = ap.parse_args()

    n_atoms = load(args.csv)
    n_mats = len(n_atoms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})

    # ---- panel 1: atom-count histogram -------------------------------------
    ax1.hist(n_atoms, bins=np.arange(0, n_atoms.max() + 2),
             color="#3b6a9e", edgecolor="white", linewidth=0.2, alpha=0.9)
    p50 = int(np.median(n_atoms))
    p99 = int(np.percentile(n_atoms, 99))
    for x, label, color in [(p50, f"p50={p50}", "#2b8a3e"),
                            (p99, f"p99={p99}", "#a63d40"),
                            (n_atoms.max(), f"max={n_atoms.max()}", "#7d3c98")]:
        ax1.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax1.text(x, ax1.get_ylim()[1] * 0.85, f" {label}", color=color,
                 fontsize=9, rotation=90, va="top")

    ax1.set_xlabel("atoms per material")
    ax1.set_ylabel("material count")
    ax1.set_title(f"Atom-count distribution across {n_mats:,} train-full materials")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, n_atoms.max() + 5)
    ax1.set_yscale("log")

    # ---- panel 2: preamble + seq-length bar per P/R at 8k/16k ----------------
    targets = [
        ("cube P=14", CUBE_SHARED + 2 * 14 ** 3, "#1f77b4"),
        ("cube P=15", CUBE_SHARED + 2 * 15 ** 3, "#1f77b4"),
        ("cube P=16", CUBE_SHARED + 2 * 16 ** 3, "#1f77b4"),
        ("cube P=18", CUBE_SHARED + 2 * 18 ** 3, "#1f77b4"),
        ("cube P=19", CUBE_SHARED + 2 * 19 ** 3, "#1f77b4"),
        ("cube P=20", CUBE_SHARED + 2 * 20 ** 3, "#1f77b4"),
    ]
    V = ball_cum_counts(160)
    ball_cfgs = [
        ("ball R²=75", 75),
        ("ball R²=86", 86),
        ("ball R²=138", 138),
        ("ball R²=153", 153),
    ]
    for lbl, r2 in ball_cfgs:
        targets.append((lbl, BALL_SHARED + 2 * int(V[r2]), "#d9471f"))

    labels, base_sizes, colors = zip(*targets)
    base = np.array(base_sizes)

    # Per-mat seq-len at p50 and p99 atom counts.
    N_p50 = int(np.median(n_atoms))
    N_p99 = int(np.percentile(n_atoms, 99))
    seq_p50 = base + PER_ATOM * N_p50
    seq_p99 = base + PER_ATOM * N_p99
    seq_max = base + PER_ATOM * n_atoms.max()

    x = np.arange(len(labels))
    ax2.bar(x, base, color=colors, alpha=0.45,
            label="fixed (shape + overhead, no atoms)")
    ax2.bar(x, seq_p50 - base, bottom=base, color="#cccccc", alpha=0.85,
            label=f"+ atoms (median N={N_p50})")
    ax2.bar(x, seq_p99 - seq_p50, bottom=seq_p50, color="#999999", alpha=0.85,
            label=f"+ atoms (p99 N={N_p99})")
    ax2.bar(x, seq_max - seq_p99, bottom=seq_p99, color="#555555", alpha=0.85,
            label=f"+ atoms (max N={n_atoms.max()})")

    for cl, color, lbl in [(8192, "#2b8a3e", "CL = 8k"),
                           (16384, "#d9a60b", "CL = 16k")]:
        ax2.axhline(cl, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        ax2.text(len(labels) - 0.5, cl * 1.02, lbl, color=color, fontsize=9,
                 ha="right", fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("total tokens (preamble + density + EOS + PAD-free)")
    ax2.set_title("Sequence length per patch-config, by atom count")
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(0, max(seq_max) * 1.05)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
