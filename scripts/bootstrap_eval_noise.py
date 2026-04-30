#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["fsspec", "gcsfs", "numpy", "click"]
# ///
"""Bootstrap-based noise estimator for mat-NMAE evals.

Given one or more per-mat result JSONs (written to GCS by eval_mat_nmae.py),
compute bootstrap stderr for mean / median / p99 NMAE, and report whether
the difference between any pair of evals is significant relative to that
stderr.

Typical usage:
    bootstrap_eval_noise.py \\
        gs://marin-eu-west4/tomat/eval/results/<run-label>/val_200/step-7000.json \\
        gs://marin-eu-west4/tomat/eval/results/<run-label>/val_200/step-7999.json

Or just:
    bootstrap_eval_noise.py 'gs://.../val_200/*.json'   # globs the directory
"""
from __future__ import annotations

import json
import sys
from functools import partial

import fsspec
import numpy as np
from click import argument, command, option

err = partial(print, file=sys.stderr)


def load_per_mat(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        b = json.load(f)
    nmaes = np.array([r["nmae"] for r in b["per_mat"]])
    return {
        "path": path,
        "checkpoint": b["checkpoint"],
        "mat_set": b.get("mat_set"),
        "n": len(nmaes),
        "nmaes": nmaes,
    }


def bootstrap_stats(nmaes: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(nmaes)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = nmaes[idx]
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    p99s = np.percentile(samples, 99, axis=1)
    return {
        "mean": (nmaes.mean(), means.std()),
        "median": (np.median(nmaes), medians.std()),
        "p99": (np.percentile(nmaes, 99), p99s.std()),
    }


def expand(paths: list[str]) -> list[str]:
    fs = fsspec.filesystem("gs")
    out = []
    for p in paths:
        if "*" in p or "?" in p:
            matches = fs.glob(p)
            out.extend("gs://" + m if not m.startswith("gs://") else m for m in matches)
        else:
            out.append(p)
    return sorted(out)


@command()
@option('-b', '--n-boot', type=int, default=2000, help='Bootstrap resamples.')
@option('-s', '--seed', type=int, default=0)
@argument('paths', nargs=-1, required=True)
def main(n_boot, seed, paths):
    expanded = expand(list(paths))
    err(f"Loading {len(expanded)} per-mat result files...")
    runs = [load_per_mat(p) for p in expanded]

    print(f"{'CKPT':<50}{'N':>5}  {'MEAN':>14}{'MEDIAN':>14}{'P99':>14}")
    print("-" * 100)
    stats_by_path = {}
    for r in runs:
        s = bootstrap_stats(r["nmaes"], n_boot, seed)
        stats_by_path[r["path"]] = s
        ckpt_tail = r["checkpoint"].rstrip("/").split("/")[-1]
        ms = r["mat_set"] or "?"
        label = f"{ms}/{ckpt_tail}"
        print(f"{label:<50}{r['n']:>5}  "
              f"{s['mean'][0]*100:>7.3f}±{s['mean'][1]*100:.2f}  "
              f"{s['median'][0]*100:>7.3f}±{s['median'][1]*100:.2f}  "
              f"{s['p99'][0]*100:>7.3f}±{s['p99'][1]*100:.2f}")

    # Pairwise diff if exactly 2 runs
    if len(runs) == 2:
        a, b = runs
        sa, sb = stats_by_path[a["path"]], stats_by_path[b["path"]]
        print()
        print(f"{'STATISTIC':<10}{'DIFF (B-A)':>15}{'STDERR(diff)':>18}{'Z (diff/SE)':>15}{'P95 RANGE':>20}")
        print("-" * 80)
        for stat in ["mean", "median", "p99"]:
            va, ea = sa[stat]
            vb, eb = sb[stat]
            d = vb - va
            se = float(np.hypot(ea, eb))
            z = d / se if se > 0 else float('nan')
            ci = 1.96 * se
            print(f"{stat:<10}{d*100:>10.3f}%   {se*100:>10.3f}%   "
                  f"{z:>10.2f} σ   ±{ci*100:>5.2f}% (95%)")
        print()
        sig = abs(z) >= 1.96
        if sig:
            print(f"=> The {stat} difference is significant at 95% (|Z|>=1.96).")
        else:
            print(f"=> The differences are within 95% noise. Increase n_mats if needed.")


if __name__ == "__main__":
    main()
