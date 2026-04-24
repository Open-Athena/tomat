#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "tabulate"]
# ///
"""Compute voxel-coverage bounds for the tomat training set.

Input: a CSV of `mp_id,nx,ny,nz` (from pull_grid_shapes.py).

For each material with grid shape (nx, ny, nz) and patch size P, define:

    V_mat   = nx·ny·nz                                      # total voxels
    V_patch = P³                                            # voxels per patch
    D_max   = ⌊nx/P⌋·⌊ny/P⌋·⌊nz/P⌋                          # disjoint-tile max
    A_max   = (nx-P+1)·(ny-P+1)·(nz-P+1)                    # all valid offsets

Coverage ratio for M random patches per material:

    c_mat(M) = M · V_patch / V_mat
    E[coverage](M) ≈ 1 − exp(−c_mat)                        # expected voxel
                                                             # fraction hit
                                                             # (large V_mat)

Aggregate coverage across the corpus of N materials:

    C_total(M) = Σ M · V_patch / Σ V_mat
               = M · V_patch / mean(V_mat)   (when M uniform)

Prints a table of (M, mean coverage, corpus coverage, disjoint-max-needed)
for a range of M and P values.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from tabulate import tabulate


def load_shapes(path: str) -> np.ndarray:
    """Load (N, 3) array of grid shapes from CSV."""
    data = []
    with open(path) as f:
        header = f.readline()
        if "mp_id" not in header:
            raise ValueError(f"expected header with mp_id, got {header!r}")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            data.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.array(data)


def summarize_shapes(shapes: np.ndarray) -> None:
    nx, ny, nz = shapes.T
    V = nx.astype(np.int64) * ny * nz
    print(f"n_materials: {len(shapes):,}")
    print(f"grid-dim distribution (nx=ny dims pooled across axes):")
    all_dims = shapes.flatten()
    print(f"  min={all_dims.min()}, p1={np.percentile(all_dims, 1):.0f}, "
          f"p25={np.percentile(all_dims, 25):.0f}, "
          f"median={np.median(all_dims):.0f}, "
          f"p75={np.percentile(all_dims, 75):.0f}, "
          f"p99={np.percentile(all_dims, 99):.0f}, max={all_dims.max()}")
    print(f"total voxels: min={V.min():,}, median={int(np.median(V)):,}, "
          f"mean={int(V.mean()):,}, max={V.max():,}")
    print(f"corpus volume (sum): {V.sum():,} voxels "
          f"({V.sum()/1e9:.2f} G voxels)")
    print()


def coverage_table(shapes: np.ndarray, P: int, Ms: list[int]) -> None:
    nx, ny, nz = shapes.T
    V = nx.astype(np.int64) * ny * nz
    V_patch = P ** 3
    # disjoint max per mat
    D_max = (nx // P) * (ny // P) * (nz // P)
    # all-offsets max per mat (zero if P > any dim)
    A_max = np.maximum(nx - P + 1, 0) * np.maximum(ny - P + 1, 0) * np.maximum(nz - P + 1, 0)
    frac_small = (V_patch >= V).mean()  # patch bigger than whole mat

    print(f"P = {P}, V_patch = {V_patch:,} voxels")
    print(f"  disjoint-tile max per mat: median={int(np.median(D_max))}, "
          f"mean={D_max.mean():.0f}, max={D_max.max()}, "
          f"sum across corpus={D_max.sum():,}")
    print(f"  all-offsets max per mat:   median={int(np.median(A_max)):,}, "
          f"mean={A_max.mean():.0f}, max={A_max.max():,}, "
          f"sum across corpus={A_max.sum():,}")
    if frac_small > 0:
        print(f"  (warning: patch exceeds full mat for {frac_small:.1%} of mats)")

    corpus_V = V.sum()
    rows = []
    for M in Ms:
        # per-mat coverage ratio
        c_mat = M * V_patch / V
        # expected fraction of voxels hit (1 - (1-p)^M approximation for uniform
        # random offset sampling; equivalent to 1-exp(-c) for small p)
        p_hit = V_patch / V
        E_hit = 1 - (1 - p_hit) ** M
        # corpus-wide ratio: fraction of "voxel exposures" per training set voxel
        C_corpus = M * V_patch * len(shapes) / corpus_V
        rows.append([
            M,
            f"{c_mat.mean():.3f}",
            f"{np.median(c_mat):.3f}",
            f"{E_hit.mean():.1%}",
            f"{np.median(E_hit):.1%}",
            f"{C_corpus:.3f}",
            f"{(M >= D_max).mean():.1%}",  # % of mats where M≥disjoint-max
        ])
    print()
    print(tabulate(
        rows,
        headers=[
            "M (patches/mat)",
            "c̄ (mean)",
            "c̃ (median)",
            "E[cov] mean",
            "E[cov] median",
            "C_corpus",
            "% mats where\nM≥D_max",
        ],
        tablefmt="simple",
    ))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Output of pull_grid_shapes.py")
    ap.add_argument("--P", type=int, nargs="+", default=[14, 15, 19, 20],
                    help="Patch sizes to analyze")
    ap.add_argument("--M", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024],
                    help="Patches-per-material values")
    ap.add_argument("--total-N", type=int, default=77498,
                    help="Total materials to extrapolate corpus math to")
    args = ap.parse_args()

    shapes = load_shapes(args.csv)
    summarize_shapes(shapes)

    for P in args.P:
        coverage_table(shapes, P, args.M)

    # Extrapolation to full train-full corpus (assume same distribution)
    print(f"=== Extrapolation to train-full (N={args.total_N:,}) ===")
    nx, ny, nz = shapes.T
    V = nx.astype(np.int64) * ny * nz
    V_corpus_est = V.mean() * args.total_N
    print(f"Est. corpus volume: {V_corpus_est/1e9:.2f} G voxels "
          f"(assumes same shape distribution as sample)")
    print()
    for P in args.P:
        V_patch = P ** 3
        rows = []
        for M in args.M:
            total_patches = M * args.total_N
            C = total_patches * V_patch / V_corpus_est
            rows.append([M, total_patches, f"{C:.2f}"])
        print(f"P = {P} (V_patch = {V_patch:,}):")
        print(tabulate(
            rows,
            headers=["M", "total patches", "C_corpus (voxel-multiplicity)"],
            tablefmt="simple",
        ))
        print()


if __name__ == "__main__":
    main()
