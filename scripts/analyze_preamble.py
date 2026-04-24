#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib", "tabulate"]
# ///
"""Preamble-size distribution analysis.

Given a CSV of `mp_id,nx,ny,nz,n_atoms`, compute the tokenized-sequence
length for each (material × patch-shape) combo and report:

1. Preamble size distribution (tokens).
2. For each (shape, P or R) × context-length pair, the fraction of
   materials whose sequences fit (i.e. are not dropped by the
   pad-overflow skip).

Token accounting (matches src/tomat/tokenizers/{patch,ball}.py):

    cube (P):   28 + 10·n_atoms + 2·P³
    ball (R²):  29 + 10·n_atoms + 2·V_ball(R²)

where V_ball(R²) = cumulative integer lattice points with i²+j²+k² ≤ R².

28/29 token overhead = BOS + 7 START/END block pairs (14) + 3 grid +
3 shape/offset (cube) or 1 radius + 3 center + 6 bounds (ball) + 3 hi
(cube only) + 2 DENS_START/END + EOS. See preamble_overhead() below.

Usage:
    scripts/analyze_preamble.py /tmp/train-full-preamble.csv --P 14 15 16 18 20 --ball-R2 75 86 100 138 153 --CL 8192 16384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate


# Per-atom preamble cost: 1 atom token + 3·3 = 9 position-codec tokens = 10.
# Shared overhead (cube & ball both include): BOS, GRID(+/-), ATOMS(+/-),
# POS(+/-), DENS(+/-), EOS = 1 + 2*4 + 1 = 10 specials. Plus 3 grid ints.
# = 13. Shape-specific:
#   cube:   SHAPE(+/-) + 3 ints + OFFSET(+/-) + 3 ints + HI(+/-) + 3 ints = 15 more → 28 total.
#   ball:   RADIUS(+/-) + 1 int + CENTER(+/-) + 3 ints + BOUNDS(+/-) + 6 ints = 16 more → 29 total.
PER_ATOM = 10
CUBE_SHARED = 28
BALL_SHARED = 29


def ball_cum_counts(max_r2: int) -> np.ndarray:
    """V(n) for n in 0..max_r2."""
    R = int(np.ceil(np.sqrt(max_r2))) + 1
    shell = np.zeros(max_r2 + 1, dtype=np.int64)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= max_r2:
                    shell[r2] += 1
    return shell.cumsum()


def load_preamble(csv_path: Path) -> np.ndarray:
    """Return structured array with (mp_id, nx, ny, nz, n_atoms)."""
    import csv

    rows = []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        if header[:5] != ["mp_id", "nx", "ny", "nz", "n_atoms"]:
            raise ValueError(f"bad header: {header}")
        for row in rdr:
            if len(row) != 5:
                continue
            rows.append((row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])))
    return rows


def seq_len_cube(n_atoms: np.ndarray, P: int) -> np.ndarray:
    return CUBE_SHARED + PER_ATOM * n_atoms + 2 * (P ** 3)


def seq_len_ball(n_atoms: np.ndarray, r2: int, V_cum: np.ndarray) -> np.ndarray:
    return BALL_SHARED + PER_ATOM * n_atoms + 2 * V_cum[r2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="preamble CSV from pull_preamble_stats_modal.py")
    ap.add_argument("--P", type=int, nargs="+", default=[14, 15, 16, 18, 20])
    ap.add_argument("--ball-R2", type=int, nargs="+", default=[75, 86, 138, 153],
                    help="Ball r² thresholds to evaluate.")
    ap.add_argument("--CL", type=int, nargs="+", default=[8192, 16384, 32768],
                    help="Context lengths (pad_to values) to evaluate.")
    args = ap.parse_args()

    rows = load_preamble(args.csv)
    n_atoms = np.array([r[4] for r in rows], dtype=np.int64)
    n_mats = len(rows)
    err = lambda *a: print(*a, file=sys.stderr)
    err(f"loaded {n_mats:,} materials")
    err(f"n_atoms: min={n_atoms.min()} p1={np.percentile(n_atoms,1):.0f} "
        f"p50={int(np.median(n_atoms))} p75={int(np.percentile(n_atoms,75))} "
        f"p99={int(np.percentile(n_atoms,99))} max={n_atoms.max()}")
    err(f"preamble-only size (cube, 28+10N): p50={CUBE_SHARED+10*int(np.median(n_atoms))} "
        f"p99={CUBE_SHARED+10*int(np.percentile(n_atoms,99))} "
        f"max={CUBE_SHARED+10*n_atoms.max()}")

    # Precompute ball cumulative voxel counts up to the max r2 threshold we'll need.
    max_r2 = max(args.ball_R2) if args.ball_R2 else 0
    V_cum = ball_cum_counts(max_r2) if max_r2 > 0 else np.array([0])

    # Table: fraction of mats that fit, per (shape, P or R²) × CL.
    rows_out = []
    for P in args.P:
        row = [f"cube P={P}", P ** 3]
        seq = seq_len_cube(n_atoms, P)
        for CL in args.CL:
            fit = (seq <= CL).mean()
            row.append(f"{fit*100:5.2f}%")
        rows_out.append(row)
    for r2 in args.ball_R2:
        row = [f"ball R²={r2}", int(V_cum[r2])]
        seq = seq_len_ball(n_atoms, r2, V_cum)
        for CL in args.CL:
            fit = (seq <= CL).mean()
            row.append(f"{fit*100:5.2f}%")
        rows_out.append(row)

    headers = ["shape", "V_patch"] + [f"fit% at CL={c}" for c in args.CL]
    print()
    print(f"Materials fitting in each (shape × CL) — N={n_mats:,}")
    print(tabulate(rows_out, headers=headers, tablefmt="simple"))

    # Find max P that keeps >= 99% / 95% / 100% fit for each CL.
    print()
    print("Largest cube P that keeps 100% / 99% / 95% of mats per CL")
    rows_out2 = []
    for CL in args.CL:
        thresholds = []
        for target in (1.0, 0.99, 0.95):
            P_max = 0
            for P in range(4, 35):
                seq = seq_len_cube(n_atoms, P)
                if (seq <= CL).mean() >= target:
                    P_max = P
                else:
                    break
            thresholds.append(P_max)
        rows_out2.append([f"CL={CL}", *thresholds])
    print(tabulate(rows_out2, headers=["context_len", "P (100%)", "P (99%)", "P (95%)"],
                   tablefmt="simple"))

    # Same for ball R² (smaller per-voxel cost means larger R² possible).
    V_cum_big = ball_cum_counts(400)
    print()
    print("Largest ball R² that keeps 100% / 99% / 95% of mats per CL")
    rows_out3 = []
    for CL in args.CL:
        thresholds = []
        for target in (1.0, 0.99, 0.95):
            r2_max_ok = 0
            for r2 in range(1, 401):
                seq = BALL_SHARED + PER_ATOM * n_atoms + 2 * V_cum_big[r2]
                if (seq <= CL).mean() >= target:
                    r2_max_ok = r2
                else:
                    break
            thresholds.append(r2_max_ok)
        rows_out3.append([f"CL={CL}", *thresholds])
    print(tabulate(rows_out3, headers=["context_len", "R² (100%)", "R² (99%)", "R² (95%)"],
                   tablefmt="simple"))


if __name__ == "__main__":
    main()
