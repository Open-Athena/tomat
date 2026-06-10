#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["fsspec", "gcsfs", "numpy", "matplotlib", "click"]
# ///
"""Analyze + visualize the LMQ codec.

Produces three plots that motivate the bin-idx σ ablation:

  1. ``bin-centers-vs-log-density.png`` — bin index vs log10(recon_points[i]).
     Strongly concave-up, showing equal-probability quantile binning concentrates
     bins in the low-density region.

  2. ``bin-spacing-dist.png`` — histogram of dρ = diff(recon_points) on a log
     y-axis. Shows the ~6-OOM range of bin widths in density-value units; p5,
     p50, p95, max are annotated.

  3. ``h-q-vs-density.png`` — per-ρ_true entropy H(Q_gauss(ρ_true; σ=0.5)) (nats)
     vs log10(ρ_true). Formula matches ``marin/qwen3_density.py:475-484``.
     Shows the dramatic monotone decrease from ~6.6 nats at low density to ~0
     nats at high density. The mean (≈ 4.56 nats with σ=0.5) is overlaid.

Usage:
    analyze_lmq_codec.py                                  # all 3 plots
    analyze_lmq_codec.py -p bin-centers                   # one plot
    analyze_lmq_codec.py --codec gs://...lmq-v2-16k.npz   # remote
    analyze_lmq_codec.py --sigma 0.5                      # H(Q) σ value
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Iterable

import click
import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CODEC_REMOTE = "gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz"
DEFAULT_CODEC_LOCAL = "tmp/lmq-v2-16k.npz"
DEFAULT_OUT = "posts/img/01-lmq"
DEFAULT_SIGMA = 0.5

err = partial(print, file=sys.stderr)


# ---------- styling ----------

BG = "#0a0a0a"
FG = "#e0e0e0"
GRID = "#2a2a2a"
ACCENT_PRIMARY = "#5eb1ff"   # blue
ACCENT_SECONDARY = "#ffb86b"  # orange
ACCENT_TERTIARY = "#a78bfa"   # purple
ACCENT_MEAN = "#f87171"       # red for mean line


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(FG)
    ax.tick_params(colors=FG, which="both")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, alpha=0.25, color=GRID, which="both")


def new_fig() -> tuple[plt.Figure, plt.Axes]:
    # 1200x800 @ 150 dpi → figsize (8, 5.33)
    fig, ax = plt.subplots(figsize=(8, 5.33), dpi=150, facecolor=BG)
    style_axes(ax)
    return fig, ax


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    err(f"wrote {out_path}")


# ---------- codec loading ----------

def load_codec(path: str) -> dict:
    """Load `.npz` codec from local path or fsspec URI."""
    if path.startswith(("gs://", "s3://", "http://", "https://")):
        with fsspec.open(path, "rb") as f:
            data = f.read()
        import io
        npz = np.load(io.BytesIO(data), allow_pickle=True)
    else:
        npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def resolve_codec_path(codec: str) -> str:
    """Prefer a local copy if one exists; otherwise fall back to the gs:// URI."""
    if codec != DEFAULT_CODEC_REMOTE:
        return codec
    if Path(DEFAULT_CODEC_LOCAL).exists():
        err(f"using local codec {DEFAULT_CODEC_LOCAL}")
        return DEFAULT_CODEC_LOCAL
    err(f"using remote codec {codec}")
    return codec


# ---------- plot 1: bin centers vs log density ----------

def plot_bin_centers(recon: np.ndarray, out_path: Path) -> dict:
    n = len(recon)
    idx = np.arange(n)
    log_rho = np.log10(np.maximum(recon, 1e-12))

    fig, ax = new_fig()
    ax.plot(idx, log_rho, color=ACCENT_PRIMARY, lw=1.5)

    # Annotate i = n/10, n/2, 9n/10.
    annots = {
        "i = n/10": n // 10,
        "i = n/2": n // 2,
        "i = 9n/10": 9 * n // 10,
    }
    for label, i in annots.items():
        ax.scatter([i], [log_rho[i]], color=ACCENT_SECONDARY, s=40, zorder=5,
                   edgecolor=FG, linewidth=0.5)
        ax.annotate(
            f"{label}\nρ = {recon[i]:.3g}",
            xy=(i, log_rho[i]),
            xytext=(10, -28),
            textcoords="offset points",
            color=FG,
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color=FG, lw=0.5, alpha=0.6),
        )

    ax.set_xlabel("bin index")
    ax.set_ylabel(r"$\log_{10}\,\rho$  (recon_points[i])")
    ax.set_title(
        f"LMQ bin centers: equal-probability quantile binning  (n = {n})"
    )
    save_fig(fig, out_path)

    return {
        "n_bins": n,
        "rho_min": float(recon.min()),
        "rho_max": float(recon.max()),
        "rho_at_n_over_10": float(recon[n // 10]),
        "rho_at_n_over_2": float(recon[n // 2]),
        "rho_at_9n_over_10": float(recon[9 * n // 10]),
    }


# ---------- plot 2: bin-spacing distribution ----------

def plot_bin_spacing(recon: np.ndarray, out_path: Path) -> dict:
    drho = np.diff(recon)
    drho_pos = drho[drho > 0]

    pcts = {p: float(np.percentile(drho, p)) for p in (5, 25, 50, 75, 95, 99)}
    stats = {
        "min": float(drho.min()),
        "max": float(drho.max()),
        "mean": float(drho.mean()),
        **{f"p{p}": v for p, v in pcts.items()},
    }

    # Log-binned histogram of dρ.
    lo = max(drho_pos.min(), 1e-9)
    hi = drho_pos.max()
    bins = np.geomspace(lo, hi * 1.05, 80)

    fig, ax = new_fig()
    ax.hist(drho_pos, bins=bins, color=ACCENT_PRIMARY, alpha=0.85,
            edgecolor=FG, linewidth=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\rho$ = recon_points[i+1] $-$ recon_points[i]")
    ax.set_ylabel("count (bins)")
    ax.set_title(
        r"LMQ bin spacing $\Delta\rho$ spans ~6 orders of magnitude in density-value units"
    )

    # Overlay vertical lines for p5, p50, p95, max.
    # Stagger label heights so adjacent marks don't overlap.
    marks = [
        ("p5",  pcts[5],  ACCENT_SECONDARY, 0.30),
        ("p50", pcts[50], ACCENT_TERTIARY,  0.50),
        ("p95", pcts[95], ACCENT_SECONDARY, 0.30),
        ("max", stats["max"], ACCENT_MEAN,  0.50),
    ]
    ymin, ymax = ax.get_ylim()
    log_y = np.log10(np.array([max(ymin, 0.5), ymax]))
    for label, x, color, frac in marks:
        ax.axvline(x, color=color, ls="--", lw=1.0, alpha=0.85)
        y_log = log_y[0] + frac * (log_y[1] - log_y[0])
        y = 10 ** y_log
        ax.text(
            x, y, f" {label} = {x:.3g}",
            color=color, fontsize=9, rotation=90,
            ha="left", va="center",
        )
    ax.set_ylim(ymin, ymax)

    save_fig(fig, out_path)
    return stats


# ---------- plot 3: H(Q_gauss) vs density ----------

def compute_h_q_gauss(recon: np.ndarray, sigma: float) -> np.ndarray:
    """Per-row entropy of Q(v) ∝ exp(-(ρ_v − ρ_true)² / 2σ²) over density bins.

    Mirrors ``marin/qwen3_density.py:475-484``: log_q_unnorm → log_softmax → H.
    Returns an array of shape (n_bins,) of H(Q) in nats.
    """
    # (n, n): row i = ρ_true index, col j = ρ_v index.
    log_q_unnorm = -((recon[None, :] - recon[:, None]) ** 2) / (2.0 * sigma ** 2)
    log_q = log_q_unnorm - log_q_unnorm.max(axis=1, keepdims=True)
    q = np.exp(log_q)
    q /= q.sum(axis=1, keepdims=True)
    # H(Q) = -Σ q log q (per row).
    log_q_safe = np.log(np.clip(q, 1e-300, None))
    return -np.einsum("ij,ij->i", q, log_q_safe)


def plot_h_q(recon: np.ndarray, sigma: float, out_path: Path) -> dict:
    err(f"computing H(Q_gauss) with σ={sigma} over {len(recon)} bins …")
    h_q = compute_h_q_gauss(recon, sigma)
    mean_h_q = float(h_q.mean())

    log_rho = np.log10(np.maximum(recon, 1e-12))

    fig, ax = new_fig()
    ax.plot(log_rho, h_q, color=ACCENT_PRIMARY, lw=1.5,
            label=f"H(Q_gauss(ρ_true; σ={sigma}))")
    ax.axhline(
        mean_h_q, color=ACCENT_MEAN, ls="--", lw=1.0,
        label=f"mean = {mean_h_q:.3f} nats",
    )

    # Annotate the i = n/10, n/2, 9n/10 reference points.
    n = len(recon)
    annots = [
        ("i = n/10",   n // 10),
        ("i = n/2",    n // 2),
        ("i = 9n/10",  9 * n // 10),
    ]
    for label, i in annots:
        ax.scatter([log_rho[i]], [h_q[i]], color=ACCENT_SECONDARY,
                   s=40, zorder=5, edgecolor=FG, linewidth=0.5)
        ax.annotate(
            f"{label}\nH = {h_q[i]:.2f}",
            xy=(log_rho[i], h_q[i]),
            xytext=(10, 14),
            textcoords="offset points",
            color=FG,
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color=FG, lw=0.5, alpha=0.6),
        )

    ax.set_xlabel(r"$\log_{10}\,\rho_\mathrm{true}$")
    ax.set_ylabel("H(Q_gauss) [nats]")
    ax.set_title(
        f"Gaussian-target entropy collapses with ρ_true  (σ = {sigma}, value-space)"
    )
    leg = ax.legend(loc="lower left", facecolor=BG, edgecolor=FG,
                    labelcolor=FG)
    for text in leg.get_texts():
        text.set_color(FG)
    save_fig(fig, out_path)

    return {
        "sigma": sigma,
        "h_q_min": float(h_q.min()),
        "h_q_max": float(h_q.max()),
        "h_q_mean": mean_h_q,
        "h_q_p5": float(np.percentile(h_q, 5)),
        "h_q_p50": float(np.percentile(h_q, 50)),
        "h_q_p95": float(np.percentile(h_q, 95)),
        "h_q_at_n_over_10": float(h_q[n // 10]),
        "h_q_at_n_over_2": float(h_q[n // 2]),
        "h_q_at_9n_over_10": float(h_q[9 * n // 10]),
    }


# ---------- CLI ----------

ALL_PLOTS = ("bin-centers", "bin-spacing", "h-q")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("-c", "--codec", default=DEFAULT_CODEC_REMOTE,
              help=f"path or fsspec URI to LMQ codec .npz  (default: {DEFAULT_CODEC_LOCAL} if present, else {DEFAULT_CODEC_REMOTE})")
@click.option("-o", "--out", default=DEFAULT_OUT,
              type=click.Path(),
              help=f"output directory  (default: {DEFAULT_OUT})")
@click.option("-p", "--plot", "plots", multiple=True,
              type=click.Choice(ALL_PLOTS),
              help="plot to produce; pass multiple times. default: all 3.")
@click.option("-s", "--sigma", default=DEFAULT_SIGMA, type=float,
              help=f"σ for the H(Q_gauss) plot (default: {DEFAULT_SIGMA})")
def main(codec: str, out: str, plots: tuple[str, ...], sigma: float) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec_path = resolve_codec_path(codec)
    cdc = load_codec(codec_path)
    recon = np.asarray(cdc["recon_points"], dtype=np.float64)
    err(f"loaded codec: n_bins={len(recon)}  range=[{recon.min():.4g}, {recon.max():.4g}]")

    selected = plots or ALL_PLOTS

    results: dict[str, dict] = {}
    if "bin-centers" in selected:
        results["bin-centers"] = plot_bin_centers(
            recon, out_dir / "bin-centers-vs-log-density.png"
        )
    if "bin-spacing" in selected:
        results["bin-spacing"] = plot_bin_spacing(
            recon, out_dir / "bin-spacing-dist.png"
        )
    if "h-q" in selected:
        results["h-q"] = plot_h_q(
            recon, sigma, out_dir / "h-q-vs-density.png"
        )

    # Stdout: a compact summary other scripts can parse.
    print("=== analyze_lmq_codec summary ===")
    for k, v in results.items():
        print(f"[{k}]")
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"  {kk} = {vv:.6g}")
            else:
                print(f"  {kk} = {vv}")


if __name__ == "__main__":
    main()
