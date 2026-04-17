#!/usr/bin/env python
"""Summarize a fidelity-sweep CSV into Markdown tables.

Produces an overall table (mean/median/min/max per config) and a
per-material-category table. By default all available metrics are
emitted (NMAE, χ², Hellinger, JSD, Weighted MAE); pass ``-m NAME`` to
restrict.

Usage: ``uv run scripts/summarize_sweep.py results/sweep-n50.csv``
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from click import argument, command, option

METRIC_LABELS = {
    "nmae": "NMAE",
    "chi_sq": "χ²",
    "hellinger": "Hellinger",
    "jsd": "JSD",
    "weighted_mae": "Weighted MAE",
}


def fmt(x: float) -> str:
    return f"{x:.2e}" if x != 0 else "0"


@command()
@argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@option("-H", "--html", is_flag=True, help="Wrap output in a <details> block (for mdcmd embedding in README)")
@option("-m", "--metric", multiple=True, type=click.Choice(list(METRIC_LABELS)), help="Restrict to a subset of metrics (default: all available in CSV)")
@option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Write markdown to this path instead of stdout")
def main(csv_path: Path, html: bool, metric: tuple[str, ...], output: Path | None):
    rows = list(csv.DictReader(csv_path.open()))
    available = [m for m in METRIC_LABELS if m in rows[0]]
    metrics = list(metric) if metric else available

    configs: list[str] = []
    seen = set()
    for r in rows:
        if r["config"] not in seen:
            seen.add(r["config"])
            configs.append(r["config"])

    by_config: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_cat_config: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    samples_per_cat: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        for m in metrics:
            v = float(r[m])
            by_config[(r["config"], m)].append(v)
            by_cat_config[(r["category"], r["config"], m)].append(v)
        samples_per_cat[r["category"]].add(r["mp_id"])

    categories = sorted(samples_per_cat, key=lambda c: (-len(samples_per_cat[c]), c))
    total_samples = len({r["mp_id"] for r in rows})
    lines: list[str] = []

    lines.append(f"### Overall (n={total_samples}, 128³ grid)\n")
    mean_header = ["config"] + [f"mean {METRIC_LABELS[m]}" for m in metrics] + ["mean mass captured"]
    lines.append("| " + " | ".join(mean_header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(mean_header) - 1)) + "|")
    for cfg in configs:
        cells = [f"`{cfg}`"]
        for m in metrics:
            cells.append(fmt(float(np.mean(by_config[(cfg, m)]))))
        mass_vals = [float(r["mass_captured"]) for r in rows if r["config"] == cfg and r.get("mass_captured")]
        cells.append(f"{np.mean(mass_vals):.3f}" if mass_vals else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    for m in metrics:
        label = METRIC_LABELS[m]
        lines.append(f"### {label} by material category (mean)\n")
        header = ["config"] + [f"{cat} (n={len(samples_per_cat[cat])})" for cat in categories]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] + [":---:"] * len(categories)) + "|")
        for cfg in configs:
            row_cells = [f"`{cfg}`"]
            for cat in categories:
                vals = by_cat_config.get((cat, cfg, m), [])
                row_cells.append(fmt(float(np.mean(vals))) if vals else "—")
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    text = "\n".join(lines)
    if html:
        text = "<details open><summary>Fidelity-sweep tables</summary>\n\n" + text + "\n</details>"
    if output is not None:
        output.write_text(text)
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
