#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["fsspec", "gcsfs", "matplotlib", "numpy", "click"]
# ///
"""Plot mat-NMAE / NEMD trajectories across one or more tomat training runs.

Pulls per-mat-result JSONs from GCS (gs://marin-eu-west4/tomat/eval/results/
<run_label>/<split>/step-N.json) and overlays val_200 + train_200 curves
with NMAE on the left axis and NEMD on the right.

Usage:
  plot_nmae_nemd_trajectory.py shuf1k cont7k cont7k-ext -o plots/cont7k-ext.png
"""
import json
from pathlib import Path

import click
import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "marin-eu-west4/tomat/eval/results"
RUN_PREFIX = "train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k"

# Reference markers
CHINCHILLA_STEP = 3300  # ~4B tokens at BS=128, ctx=8192 → D ≈ 20N for 200M
EPOCH_STEP = 19375      # one epoch through 2.48M seqs at BS=128


def load_run(short_label: str) -> dict:
    full = RUN_PREFIX if short_label == "shuf1k" else f"{RUN_PREFIX}-{short_label}"
    out: dict = {"label": short_label, "full": full, "val": {}, "train": {}}
    fs = fsspec.filesystem("gcs")
    for split in ("val_200", "train_200"):
        base = f"{BASE}/{full}/{split}"
        if not fs.exists(base):
            continue
        for path in fs.ls(base):
            name = path.rsplit("/", 1)[-1]
            if not (name.startswith("step-") and name.endswith(".json")):
                continue
            step = int(name[len("step-") : -len(".json")])
            try:
                with fs.open(path, "r") as f:
                    d = json.load(f)
            except Exception:
                continue
            out[split.replace("_200", "")][step] = {
                "nmae": d.get("nmae_mean"),
                "nemd": d.get("nemd_mean"),
                "nmae_p99": d.get("nmae_p99"),
            }
    return out


def _best_step(d: dict, key: str) -> tuple[int, float] | None:
    """(step, value%) of min `key` across steps in `d`, or None."""
    pts = [(s, d[s][key] * 100) for s in d if d[s][key] is not None]
    if not pts:
        return None
    return min(pts, key=lambda p: p[1])


@click.command()
@click.option("-o", "--out", default=None, type=click.Path(), help="output PNG path")
@click.option("-j", "--json-out", default=None, type=click.Path(), help="also dump aggregated trajectory data as JSON (for the React TrajectoryPlot component).")
@click.option("--title", default=None)
@click.option("--annotate-best/--no-annotate-best", default=True,
              help="mark min-val-NEMD ckpt of each run with a star.")
@click.option("--reference-lines/--no-reference-lines", default=True,
              help="draw vertical lines at Chinchilla-optimal step + 1-epoch boundary.")
@click.argument("short_labels", nargs=-1, required=True)
def main(out: str | None, json_out: str | None, title: str | None, annotate_best: bool, reference_lines: bool,
         short_labels: tuple[str, ...]):
    runs = [load_run(lab) for lab in short_labels]
    if json_out:
        # Emit a compact, web-friendly shape: one entry per (run, split) with parallel arrays.
        out_data: dict = {
            "schema_version": 1,
            "chinchilla_step": CHINCHILLA_STEP,
            "epoch_step": EPOCH_STEP,
            "runs": [],
        }
        for r in runs:
            entry: dict = {"label": r["label"], "full": r["full"], "splits": {}}
            for split in ("val", "train"):
                d = r[split]
                steps = sorted(d.keys())
                entry["splits"][split] = {
                    "steps": steps,
                    "nmae": [d[s]["nmae"] for s in steps],
                    "nemd": [d[s]["nemd"] for s in steps],
                    "nmae_p99": [d[s].get("nmae_p99") for s in steps],
                }
            out_data["runs"].append(entry)
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(out_data, f, indent=1)
        print(f"wrote {json_out}")
        if not out:
            return  # JSON-only mode

    fig, (ax_nmae, ax_nemd) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for r, color in zip(runs, colors):
        for split, ls, marker_size in (("val", "-", 5), ("train", "--", 3)):
            d = r[split]
            steps = sorted(d.keys())
            nmae = [d[s]["nmae"] * 100 if d[s]["nmae"] else np.nan for s in steps]
            nemd = [d[s]["nemd"] * 100 if d[s]["nemd"] else np.nan for s in steps]
            if not all(np.isnan(nmae)):
                ax_nmae.plot(steps, nmae, ls=ls, color=color, marker="o", ms=marker_size,
                             label=f"{r['label']} {split}", alpha=0.85)
            if not all(np.isnan(nemd)):
                ax_nemd.plot(steps, nemd, ls=ls, color=color, marker="o", ms=marker_size,
                             label=f"{r['label']} {split}", alpha=0.85)

        if annotate_best and r["val"]:
            best_nemd = _best_step(r["val"], "nemd")
            if best_nemd is not None:
                s, v = best_nemd
                ax_nemd.plot(s, v, marker="*", color=color, ms=18,
                             markeredgecolor="black", markeredgewidth=0.8, zorder=10)
                ax_nemd.annotate(f"step-{s}\n{v:.2f}%", xy=(s, v),
                                 xytext=(6, 6), textcoords="offset points",
                                 fontsize=8, color=color, fontweight="bold")
            best_nmae = _best_step(r["val"], "nmae")
            if best_nmae is not None:
                s, v = best_nmae
                ax_nmae.plot(s, v, marker="*", color=color, ms=18,
                             markeredgecolor="black", markeredgewidth=0.8, zorder=10)
                ax_nmae.annotate(f"step-{s}\n{v:.2f}%", xy=(s, v),
                                 xytext=(6, 6), textcoords="offset points",
                                 fontsize=8, color=color, fontweight="bold")

    if reference_lines:
        for ax in (ax_nmae, ax_nemd):
            ax.axvline(CHINCHILLA_STEP, color="gray", ls=":", lw=1, alpha=0.6)
            ax.axvline(EPOCH_STEP, color="gray", ls=":", lw=1, alpha=0.6)
        ax_nemd.text(CHINCHILLA_STEP, ax_nemd.get_ylim()[1] * 0.98 if False else 0,
                     "  Chinchilla-opt\n  (~3.3k steps)", fontsize=7, color="gray",
                     verticalalignment="bottom")
        ax_nemd.text(EPOCH_STEP, 0, "  1 epoch\n  (~19.4k steps)", fontsize=7,
                     color="gray", verticalalignment="bottom")

    ax_nmae.set_ylabel("mat-NMAE mean (%)")
    ax_nmae.grid(alpha=0.3)
    ax_nmae.legend(fontsize=8, ncol=2, loc="upper right")
    ax_nmae.set_title(title or "mat-NMAE trajectory (★ = min val per run)")

    ax_nemd.set_xlabel("training step")
    ax_nemd.set_ylabel("mat-NEMD mean (%)")
    ax_nemd.grid(alpha=0.3)
    ax_nemd.legend(fontsize=8, ncol=2, loc="upper right")
    ax_nemd.set_title("mat-NEMD trajectory (val solid, train dashed)")

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
