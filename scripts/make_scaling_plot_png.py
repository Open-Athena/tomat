#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["plotly>=6", "kaleido", "pandas>=2"]
# ///
"""Render the site's scaling loss plot to a static PNG for slides.

Reads the per-run CSVs from `site/public/run-histories/*.csv` and writes
a composite loss-vs-step plot to `site/public/scaling-loss.png`. Same
colors and labels as the live ScalingLossPlot React component.
"""

from __future__ import annotations

from pathlib import Path
import math

import plotly.graph_objects as go
import pandas as pd

RUNS = [
    ("val-full-5k-bs32-bs32-seed42",        "A100:1 bs=32 (val-full, 30M)",      "#1f77b4"),
    ("val-full-5k-bs32-2gpu-bs32-seed42",   "A100:2 bs=32 (val-full, 30M)",      "#2ca02c"),
    ("val-full-5k-bs64-4gpu-bs64-seed42",   "A100:4 bs=64 (val-full, 30M)",      "#ff7f0e"),
    ("val-full-5k-bs128-8gpu-bs128-seed42", "A100:8 bs=128 (val-full, 30M)",     "#d62728"),
    ("val-full-tpu-bs128-seed42",           "TPU v6e-4 bs=128 (val-full, 30M)",  "#9467bd"),
    ("train-full-tpu8-bs256-seed42",        "TPU v6e-8 bs=256 (train-full, 30M)", "#8c564b"),
    ("train-full-tpu16-30M-bs512-seed42",   "TPU v6e-16 bs=512 (train-full, 30M)", "#17becf"),
    ("train-full-tpu8-200M-bs128-val-bf16-seed42", "TPU v6e-8 bs=128 (train-full, 200M)", "#ffd400"),
]

base = Path("site/public/run-histories")
out = Path("site/public/scaling-loss.png")

fig = go.Figure()
for run_id, name, color in RUNS:
    csv_path = base / f"{run_id}.csv"
    if not csv_path.exists():
        print(f"skip {run_id}: no csv")
        continue
    df = pd.read_csv(csv_path).dropna(subset=["train_loss"])
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["train_loss"],
        mode="lines", name=name,
        line=dict(color=color, width=2),
    ))

baseline = math.log(6792)
fig.add_trace(go.Scatter(
    x=[0, 5000], y=[baseline, baseline],
    mode="lines", name="uniform baseline (ln 6792)",
    line=dict(color="#888", width=1, dash="dash"),
))

fig.update_layout(
    title=dict(text="tomat scale runs — train/loss vs step", x=0.5),
    width=1600, height=900,
    margin=dict(t=60, r=40, b=60, l=70),
    xaxis=dict(title="step"),
    yaxis=dict(title="train loss (nats/token)", rangemode="tozero"),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"),
    template="plotly_white",
    font=dict(size=14),
)
fig.write_image(out, engine="kaleido")
print(f"wrote {out}")
