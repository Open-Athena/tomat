#!/usr/bin/env python
r"""Plot fidelity-sweep CSV output.

Produces three PNGs:

1. ``nmae-vs-fraction.png`` — log-log NMAE vs retained-fraction, overlay of
   cutoff (by voxel) and fourier (by \|G\|), median across samples with
   the min/max band shaded.
2. ``nmae-by-category.png`` — grouped bar of median NMAE per material
   category, for each scheme at a single representative fraction
   (default 5%).
3. ``mass-captured-cutoff.png`` — per-sample scatter of NMAE vs
   (1 − mass_captured) for cutoff configs, demonstrating the identity.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from click import argument, command, option

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

CUTOFF_RE = re.compile(r"cutoff-top-(\d+(?:\.\d+)?)pct")
FOURIER_RE = re.compile(r"fourier-lowg-(\d+(?:\.\d+)?)pct")


def load(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        r["nmae"] = float(r["nmae"])
        if r.get("mass_captured"):
            r["mass_captured"] = float(r["mass_captured"])
        if r.get("effective_threshold"):
            r["effective_threshold"] = float(r["effective_threshold"])
    return rows


def plot_nmae_vs_fraction(rows: list[dict], out: Path):
    """Log-log NMAE vs fraction kept, cutoff vs fourier overlay."""
    # (scheme_label, fraction) → list of nmae
    curves: dict[str, dict[float, list[float]]] = {"cutoff": defaultdict(list), "fourier": defaultdict(list)}
    for r in rows:
        for scheme, regex in (("cutoff", CUTOFF_RE), ("fourier", FOURIER_RE)):
            if m := regex.match(r["config"]):
                curves[scheme][float(m.group(1)) / 100].append(r["nmae"])
                break

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"cutoff": "#c44", "fourier": "#4480e0"}
    for scheme, data in curves.items():
        fractions = sorted(data.keys())
        medians = [np.median(data[f]) for f in fractions]
        mins = [min(data[f]) for f in fractions]
        maxs = [max(data[f]) for f in fractions]
        ax.plot(fractions, medians, marker="o", label=f"{scheme} (median)", color=colors[scheme], lw=2)
        ax.fill_between(fractions, mins, maxs, color=colors[scheme], alpha=0.15, label=f"{scheme} (min/max range)")

    ax.axhline(0.026, ls="--", color="gray", lw=1, label="electrAI achieved 2.6%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of representation kept")
    ax.set_ylabel("Reconstruction NMAE floor")
    ax.set_title(f"Tokenizer reconstruction floor vs retained fraction (n={len({r['mp_id'] for r in rows})} MP structures)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(which="both", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def plot_by_category(rows: list[dict], out: Path, fraction: float = 0.05):
    """Grouped bar of median NMAE per category, at a single fraction for each scheme."""
    target_cutoff = f"cutoff-top-{fraction * 100:g}pct"
    target_fourier = f"fourier-lowg-{fraction * 100:g}pct"
    cats_nmae: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"cutoff": [], "fourier": []})
    for r in rows:
        if r["config"] == target_cutoff:
            cats_nmae[r["category"]]["cutoff"].append(r["nmae"])
        elif r["config"] == target_fourier:
            cats_nmae[r["category"]]["fourier"].append(r["nmae"])

    categories = sorted(cats_nmae, key=lambda c: -len(cats_nmae[c]["cutoff"]))
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cutoff_vals = [np.median(cats_nmae[c]["cutoff"]) for c in categories]
    fourier_vals = [np.median(cats_nmae[c]["fourier"]) for c in categories]
    counts = [len(cats_nmae[c]["cutoff"]) for c in categories]

    ax.bar(x - width / 2, cutoff_vals, width, color="#c44", label=f"cutoff-top-{fraction * 100:g}%")
    ax.bar(x + width / 2, fourier_vals, width, color="#4480e0", label=f"fourier-lowg-{fraction * 100:g}%")
    ax.axhline(0.026, ls="--", color="gray", lw=1, label="electrAI achieved 2.6%")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(categories, counts)], fontsize=9)
    ax.set_ylabel("Median NMAE")
    ax.set_title(f"Reconstruction NMAE floor by material category (kept={fraction * 100:g}%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def plot_mass_vs_nmae(rows: list[dict], out: Path):
    """Scatter NMAE vs (1 - mass_captured) for cutoff — should lie on y=x."""
    xs: list[float] = []
    ys: list[float] = []
    cats: list[str] = []
    for r in rows:
        if not CUTOFF_RE.match(r["config"]):
            continue
        if not r.get("mass_captured") and r["mass_captured"] != 0:
            continue
        xs.append(1.0 - r["mass_captured"])
        ys.append(r["nmae"])
        cats.append(r["category"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cat_colors = {cat: plt.cm.tab10(i) for i, cat in enumerate(sorted(set(cats)))}
    for cat in sorted(set(cats)):
        mask = [c == cat for c in cats]
        ax.scatter([xs[i] for i, m in enumerate(mask) if m], [ys[i] for i, m in enumerate(mask) if m],
                   color=cat_colors[cat], label=cat, alpha=0.7, s=25)
    lim = [1e-10, 1.0]
    ax.plot(lim, lim, ls="--", color="gray", lw=1, label="y = x (identity)")
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("1 − mass_captured  (fraction of ρ dropped)")
    ax.set_ylabel("NMAE")
    ax.set_title("Cutoff: NMAE = (1 − mass captured) by construction")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(which="both", alpha=0.3)
    ax.set_aspect("equal")
    fig.savefig(out)
    plt.close(fig)


@command()
@argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@option("-o", "--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("results/plots"))
@option("-f", "--category-fraction", type=float, default=0.05, help="Fraction to use in the by-category bar plot")
def main(csv_path: Path, out_dir: Path, category_fraction: float):
    rows = load(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_nmae_vs_fraction(rows, out_dir / "nmae-vs-fraction.png")
    plot_by_category(rows, out_dir / "nmae-by-category.png", fraction=category_fraction)
    plot_mass_vs_nmae(rows, out_dir / "mass-captured-cutoff.png")
    print(f"Wrote 3 PNGs to {out_dir}")


if __name__ == "__main__":
    main()
