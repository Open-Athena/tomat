#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "utz", "numpy"]
# ///
"""Empirical sampling-distribution preview for the v3 tokenizer.

For each candidate weighting strategy ({uniform, electrons}), compute
the per-material patch budget M_i under fixed total budget
(M̄ × n_mats), and report the distribution.

Sanity-check for the per-electron-vs-per-mat trade-off discussion in
issue #3. Pulls n_electrons (and n_atoms) from MPDB.

Usage:
    scripts/sampling_distribution_preview.py            # default split=train, M=64
    scripts/sampling_distribution_preview.py -s val
    scripts/sampling_distribution_preview.py -M 32
"""

from __future__ import annotations

import sqlite3
import sys
from functools import partial
from pathlib import Path

import numpy as np

from utz.cli import cmd, opt

err = partial(print, file=sys.stderr)


def stats(arr: np.ndarray, label: str) -> None:
    p = np.percentile(arr, [0, 1, 10, 50, 90, 99, 100])
    print(
        f"  {label:12s} min={p[0]:>7.2f}  p1={p[1]:>7.2f}  p10={p[2]:>7.2f}  "
        f"p50={p[3]:>7.2f}  p90={p[4]:>7.2f}  p99={p[5]:>7.2f}  max={p[6]:>7.2f}  "
        f"max:min={p[6]/max(p[0], 1e-9):>6.1f}x"
    )


@cmd
@opt('-d', '--db', type=Path, default=Path("data/mpdb.sqlite"))
@opt('-M', '--mean-budget', type=int, default=64, help="Target mean M (= total_budget / n_mats)")
@opt('-s', '--split', default="train")
def main(db: Path, mean_budget: int, split: str):
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT n_atoms, n_electrons FROM mats WHERE split = ? AND n_electrons IS NOT NULL",
        (split,),
    ).fetchall()
    conn.close()

    if not rows:
        err(f"no rows for split={split!r}")
        sys.exit(1)

    n_atoms = np.array([r[0] for r in rows], dtype=np.int64)
    n_electrons = np.array([r[1] for r in rows], dtype=np.int64)
    n_mats = len(rows)
    total_budget = mean_budget * n_mats

    print(f"split={split}, n_mats={n_mats:,}, mean_M={mean_budget}, total_M={total_budget:,}")
    print()
    print("Per-mat M distribution under each weighting:")
    print()

    # uniform: M_i = mean_budget for every mat
    uniform_M = np.full(n_mats, mean_budget, dtype=np.float64)
    stats(uniform_M, "uniform")

    # electrons: M_i ∝ n_electrons, normalized so sum = total_budget
    weights_e = n_electrons / n_electrons.sum()
    electrons_M = weights_e * total_budget
    stats(electrons_M, "electrons")

    # atoms: M_i ∝ n_atoms (alternative knob; cheap to also report)
    weights_a = n_atoms / n_atoms.sum()
    atoms_M = weights_a * total_budget
    stats(atoms_M, "atoms")

    print()
    print("Tail behavior (electrons weighting):")
    n_starved = int((electrons_M < 1).sum())
    print(f"  mats with M_i < 1.0  (effectively never sampled): {n_starved:,} "
          f"({100*n_starved/n_mats:.2f}%)")
    n_drowned = int((electrons_M > 10 * mean_budget).sum())
    print(f"  mats with M_i > 10 × mean: {n_drowned:,} ({100*n_drowned/n_mats:.2f}%)")

    print()
    print(f"Per-mat NMAE comparability: electrons-weighted training will give ~"
          f"{100*(electrons_M < 1).sum()/n_mats:.1f}% of mats near-zero gradient share.")


if __name__ == "__main__":
    main()
