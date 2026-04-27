#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Educational static viz of how one P=14 patch becomes a token sequence.

Top: 3D-rendered density grid with a P=14 cube patch highlighted.
Bottom: the resulting token sequence with each block colour-coded.

Purpose: walk a non-tomat audience through the tokenization scheme. For
slides + the homepage's "Patch tokenization" section.

Output: site/public/patch-tokenization.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("site/public/patch-tokenization.png"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1.0], width_ratios=[1, 1.4])

    # === Panel 1: 3D grid + patch (left) ===========================
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    nx, ny, nz = 64, 108, 108
    P = 14
    patch_offset = (5, 9, 44)

    # Draw unit-cell wireframe
    corners = [
        (0, 0, 0), (nx, 0, 0), (nx, ny, 0), (0, ny, 0),
        (0, 0, nz), (nx, 0, nz), (nx, ny, nz), (0, ny, nz),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        xs, ys, zs = zip(corners[a], corners[b])
        ax3d.plot(xs, ys, zs, color="#888", lw=1, alpha=0.6)

    # Synthetic density blobs (atomic positions) — 9 atoms (Y3Si3Ag3 from mp-2282417)
    rng = np.random.default_rng(42)
    atom_positions = rng.uniform(0, 1, size=(9, 3)) * np.array([nx, ny, nz])
    atom_colors = ["#3C7CB6"] * 3 + ["#A0522D"] * 3 + ["#9D9D9D"] * 3  # Y, Si, Ag
    for (x, y, z), c in zip(atom_positions, atom_colors):
        ax3d.scatter(x, y, z, color=c, s=60, edgecolors="black", lw=0.5, alpha=0.9)

    # Draw the highlighted patch as a wireframe cube
    ox, oy, oz = patch_offset
    px, py, pz = ox + P, oy + P, oz + P
    patch_corners = [
        (ox, oy, oz), (px, oy, oz), (px, py, oz), (ox, py, oz),
        (ox, oy, pz), (px, oy, pz), (px, py, pz), (ox, py, pz),
    ]
    for a, b in edges:
        xs, ys, zs = zip(patch_corners[a], patch_corners[b])
        ax3d.plot(xs, ys, zs, color="#d62728", lw=2.5, alpha=0.95)
    # Light fill of the patch volume
    ax3d.scatter(*np.array(patch_corners).T.tolist(), color="#d62728", s=20, alpha=0.5)

    ax3d.set_xlabel("x voxels")
    ax3d.set_ylabel("y voxels")
    ax3d.set_zlabel("z voxels")
    ax3d.set_title(
        f"$ρ(\\mathbf{{r}})$ on a {nx}×{ny}×{nz} grid (mp-2282417, Y₃Si₃Ag₃)\n"
        f"with one $P={P}$ patch highlighted (offset {patch_offset})",
        fontsize=11,
    )
    ax3d.view_init(elev=18, azim=-50)

    # === Panel 2: token-block legend (right top) ===========================
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.text(0.5, 0.96, "Tokens emitted (preamble + density)", ha="center",
                fontsize=12, weight="bold")

    # Block list: (label, color, count, description)
    blocks = [
        ("[BOS]", "#444", "1", "sequence start"),
        ("[GRID_*]", "#3b82c4", "5", f"int tokens for grid ({nx}, {ny}, {nz})"),
        ("[ATOMS_*]", "#3C7CB6", f"{2 + 9}", "9 atom Z tokens (Y, Si, Ag)"),
        ("[POS_*]", "#f59e0b", f"{2 + 9 * 9}", "atomic frac coords (3 tokens × 3 coords × 9 atoms)"),
        ("[SHAPE_*]", "#888", "5", "patch dims (P, P, P)"),
        ("[OFFSET_*]", "#888", "5", "patch low corner (offset)"),
        ("[HI_*]", "#888", "5", "patch wrapped high corner"),
        ("[DENS_*]", "#d62728", f"{2 + 2*P**3}", f"{P**3} voxels × 2 codec tokens (= {2*P**3:,})"),
        ("[EOS]", "#444", "1", "sequence end"),
        ("[PAD] ×", "#ddd", "?", "right-pad to 8192"),
    ]
    y = 0.88
    for label, color, count, desc in blocks:
        ax_leg.add_patch(Rectangle((0.04, y - 0.04), 0.025, 0.05, color=color, alpha=0.9))
        ax_leg.text(0.085, y - 0.012, f"{label}", fontsize=10, family="monospace", weight="bold")
        ax_leg.text(0.30, y - 0.012, f"{count} tokens", fontsize=9, family="monospace")
        ax_leg.text(0.46, y - 0.012, desc, fontsize=9)
        y -= 0.075

    # Total
    total = 1 + 5 + (2 + 9) + (2 + 9 * 9) + 5 + 5 + 5 + (2 + 2 * P ** 3) + 1
    ax_leg.text(0.04, 0.10,
                f"Σ = {total:,} tokens (preamble {1+5+11+83+5+5+5+1+1+1} + density "
                f"{2*P**3:,}); padded to 8192 with [PAD]",
                fontsize=9, weight="bold")
    ax_leg.text(0.04, 0.04,
                "Total vocab = 6,792 (18 specials + 118 atomic Z + 1024 ints + "
                "1024 position-codec + 4608 density-codec)",
                fontsize=8, color="#555", style="italic")

    # === Panel 3: tape of actual tokens (bottom, full width) ============
    ax_tape = fig.add_subplot(gs[1, :])
    ax_tape.set_xlim(0, 1)
    ax_tape.set_ylim(0, 1)
    ax_tape.axis("off")
    ax_tape.set_title("Token sequence (linear) — sample from real mp-2282417 row", fontsize=11)

    # Synthesize a mock layout — drawn proportional to the actual block sizes
    blocks_to_draw = [
        ("[BOS]", "#444", 1),
        ("GRID", "#3b82c4", 5),
        ("ATOMS", "#3C7CB6", 11),
        ("POS  9 atoms × 9 toks", "#f59e0b", 83),
        ("SHAPE", "#888", 5),
        ("OFF", "#888", 5),
        ("HI", "#888", 5),
        ("DENSITY  2,744 voxels × 2 toks = 5,488", "#d62728", 5490),
        ("[EOS]", "#444", 1),
        ("[PAD] × 2,586", "#ddd", 2586),
    ]
    total_w = sum(b[2] for b in blocks_to_draw)
    x_left = 0.02
    x_right = 0.98
    span = x_right - x_left
    cur = x_left
    for lbl, color, n in blocks_to_draw:
        w = span * n / total_w
        ax_tape.add_patch(Rectangle((cur, 0.40), w, 0.30, color=color, alpha=0.85, ec="black", lw=0.4))
        # label inside / above
        if w > 0.02:
            ax_tape.text(cur + w / 2, 0.55, lbl, ha="center", va="center", fontsize=8,
                         color="white" if color in ("#444", "#d62728", "#3C7CB6", "#3b82c4") else "black",
                         weight="bold" if "DENSITY" in lbl or "PAD" in lbl else "normal")
        cur += w

    # Position markers
    ax_tape.text(x_left, 0.32, "0", fontsize=8, ha="center")
    ax_tape.text(x_right, 0.32, "8,191", fontsize=8, ha="center")
    ax_tape.text(0.5, 0.18,
                 "Each row of the tokenized parquet = exactly 8,192 tokens — "
                 "the input the LM trains on (predict-next-token over the whole stream).",
                 ha="center", fontsize=9, style="italic", color="#444")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
