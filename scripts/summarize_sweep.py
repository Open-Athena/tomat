#!/usr/bin/env python
"""Summarize a fidelity-sweep CSV into Markdown tables.

Produces two tables:

1. overall NMAE per (config) — mean / median / std over all samples
2. NMAE per (category × config) — mean, alongside sample counts per category

Usage: ``uv run scripts/summarize_sweep.py tmp/sweep-n50.csv``
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from click import argument, command, option


@command()
@argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@option("-H", "--html", is_flag=True, help="Wrap output in a <details> block (for mdcmd embedding in README)")
@option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Write markdown to this path instead of stdout")
def main(csv_path: Path, html: bool, output: Path | None):
    rows = list(csv.DictReader(csv_path.open()))
    configs: list[str] = []
    seen = set()
    for r in rows:
        if r["config"] not in seen:
            seen.add(r["config"])
            configs.append(r["config"])

    by_config: dict[str, list[float]] = defaultdict(list)
    by_cat_config: dict[tuple[str, str], list[float]] = defaultdict(list)
    samples_per_cat: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        n = float(r["nmae"])
        by_config[r["config"]].append(n)
        by_cat_config[(r["category"], r["config"])].append(n)
        samples_per_cat[r["category"]].add(r["mp_id"])

    categories = sorted(samples_per_cat, key=lambda c: (-len(samples_per_cat[c]), c))
    lines: list[str] = []

    lines.append("### Overall NMAE over all samples\n")
    total_samples = len({r["mp_id"] for r in rows})
    lines.append(f"n={total_samples} MP structures, grid=128³.\n")
    lines.append("| config | mean NMAE | median | min | max | mean mass captured |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cfg in configs:
        vals = np.array(by_config[cfg])
        mass_vals = [float(r["mass_captured"]) for r in rows if r["config"] == cfg and r.get("mass_captured")]
        mass_cell = f"{np.mean(mass_vals):.3f}" if mass_vals else "—"
        lines.append(
            f"| `{cfg}` | {vals.mean():.2e} | {np.median(vals):.2e} | {vals.min():.2e} | {vals.max():.2e} | {mass_cell} |"
        )
    lines.append("")

    lines.append("### NMAE by material category (mean)\n")
    header = "| config | " + " | ".join(f"{cat} (n={len(samples_per_cat[cat])})" for cat in categories) + " |"
    sep = "|---|" + "|".join([":---:"] * len(categories)) + "|"
    lines.append(header)
    lines.append(sep)
    for cfg in configs:
        row_cells = [f"`{cfg}`"]
        for cat in categories:
            vals = by_cat_config.get((cat, cfg), [])
            if not vals:
                row_cells.append("—")
            else:
                row_cells.append(f"{np.mean(vals):.2e}")
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
