#!/usr/bin/env python
# /// script
# dependencies = ["numpy", "matplotlib", "tabulate"]
# ///
"""Tokenization design-space analysis for post 08.

Compute, from `data/mpdb.sqlite`:

1. Atom-count distribution per split.
2. Per-encoding preamble-length distribution:
     F0 (current discrete xyz):  21 + 10*N    tokens
     F1 (continuous sinusoidal):  9 + N       tokens
     F2 (RoPE-3D atom emb):       9 + N       tokens
   Plot all three on one axis.
3. Packing-density table for ctx=8192 and the (P, F) cells:
     P in {19,14,10,7}, F in {F0,F1,F2}.
     density_block = P^3 tokens (LMQ).
     seqs/row = floor(8192 / (preamble_p99 + P^3 + 3))   # +DENS_START, +DENS_END, +EOS
4. Storage / token-count table for {M=64, M=M_max/2, M=M_max}
     where M_max is computed per-material from the median train grid as
     floor(nx/P)*floor(ny/P)*floor(nz/P).

Writes:
  plots/posts/08/atoms_hist.png
  plots/posts/08/preamble_hist.png
  tmp/t10n_design_space.json   # raw stats for the post
  prints markdown tables to stdout.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

DB = Path("data/mpdb.sqlite")
OUT_PLOTS = Path("plots/posts/08")
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path("tmp/t10n_design_space.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

CTX = 8192

# Encoding labels + per-mat fixed cost / per-atom cost.
# F0 = current discrete xyz tokenizer (v3, LMQ density, default-shape).
#       BOS + GRID(5) + LATTICE(8) + ATOMS(2+N) + POS(2+9N) + DENS(2+P^3) + EOS
#       = 21 + 10N + P^3
#   "preamble" (the part not in the density block) = 21 + 10N - 2 - P^3 ...
#   For comparison we treat "preamble" = everything before DENS_START
#   contents = 1 + 5 + 8 + (2+N) + (2+9N) + 1 = 19 + 10N
#   plus DENS_END + EOS = 2 tail tokens
#   plus DENS_START itself folded into the density block accounting:
#         density-side = 1 (DENS_START) + P^3 + 1 (DENS_END) + 1 (EOS) = P^3 + 3
# F1 = continuous coordinates: drop the 9-token position encoding per atom;
#      one "atom embedding" position per atom (the model projects (Z, x, y, z)
#      into the embed dim at the input layer). Same headers as F0.
#       preamble = 19 + N (BOS+GRID+LATTICE+ATOMS+1-per-atom-pos+POS_END) ... but
#       since "1 token per atom carries both atom-type and coords", we can also
#       drop the explicit ATOMS block. Conservative accounting that keeps
#       ATOMS and POS separate, 1 tok each: 19 + 2N tokens. We model the
#       "fused" accounting (1 tok per atom carrying everything) as 9 + N.
# F2 = RoPE-3D atom emb. Identical sequence layout to F1 (1 token per atom);
#      the RoPE is applied at attention-time, not as extra tokens.
SHARED_HEAD = 9   # BOS + GRID(5) + LATTICE(8) - 5 (we move LATTICE into per-atom for F1/F2?)
# Cleanest: keep these explicit so the post is unambiguous.

# F0 components (LMQ):
F0_FIXED = 19          # 1(BOS)+5(GRID)+8(LATTICE)+2(ATOMS)+2(POS)+1(DENS_START)
F0_PER_ATOM = 10       # 1(atom-type) + 9(3 coords × 3 codec-tokens)
F0_TAIL = 2            # DENS_END + EOS
# So total = F0_FIXED + F0_PER_ATOM * N + P^3 + F0_TAIL  (=21+10N+P^3, matches)

# F1 components (continuous sinusoidal; 1 fused atom-emb token per atom):
F1_FIXED = 17          # 1(BOS)+5(GRID)+8(LATTICE)+2(ATOMS_START/END framing? we drop those)
# Be explicit: BOS(1) + GRID_START+GRID_END(2) + 3 grid ints(3) +
# LATTICE_START+LATTICE_END(2) + 6 lattice ints(6) + ATOMS_START(1) +
# ATOMS_END(1) + DENS_START(1) = 17 tokens
# Per-atom: 1 fused atom-embedding token.
F1_PER_ATOM = 1
F1_TAIL = 2            # DENS_END + EOS

# F2 = same sequence layout as F1.
F2_FIXED = F1_FIXED
F2_PER_ATOM = F1_PER_ATOM
F2_TAIL = F1_TAIL


def fetch_arr(conn: sqlite3.Connection, sql: str) -> np.ndarray:
    return np.asarray([row[0] for row in conn.execute(sql)], dtype=np.int64)


def percentiles(arr: np.ndarray, qs=(1, 10, 50, 90, 99, 99.9)) -> dict:
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def preamble_len(n_atoms: np.ndarray, F: str) -> np.ndarray:
    if F == "F0":
        return F0_FIXED - 1 + F0_PER_ATOM * n_atoms   # exclude DENS_START
    if F == "F1":
        return F1_FIXED - 1 + F1_PER_ATOM * n_atoms
    if F == "F2":
        return F2_FIXED - 1 + F2_PER_ATOM * n_atoms
    raise ValueError(F)


def total_seq_len(n_atoms: np.ndarray, F: str, P: int) -> np.ndarray:
    """Total tokens incl. density block."""
    if F == "F0":
        return F0_FIXED + F0_PER_ATOM * n_atoms + P ** 3 + F0_TAIL
    if F == "F1":
        return F1_FIXED + F1_PER_ATOM * n_atoms + P ** 3 + F1_TAIL
    if F == "F2":
        return F2_FIXED + F2_PER_ATOM * n_atoms + P ** 3 + F2_TAIL
    raise ValueError(F)


def main():
    conn = sqlite3.connect(str(DB))
    by_split = {}
    for split in ("train", "val"):
        arr = fetch_arr(
            conn,
            f"SELECT n_atoms FROM mats WHERE split='{split}' AND n_atoms IS NOT NULL",
        )
        by_split[split] = arr
        print(f"{split}: n={len(arr):,}  mean={arr.mean():.1f}  median={int(np.median(arr))}  "
              f"p99={int(np.percentile(arr, 99))}  max={int(arr.max())}")

    # Grid shapes from train (median per-axis used for M_max).
    nx = fetch_arr(conn, "SELECT nx FROM mats WHERE split='train' AND n_atoms IS NOT NULL")
    ny = fetch_arr(conn, "SELECT ny FROM mats WHERE split='train' AND n_atoms IS NOT NULL")
    nz = fetch_arr(conn, "SELECT nz FROM mats WHERE split='train' AND n_atoms IS NOT NULL")
    grid_med = tuple(int(np.median(a)) for a in (nx, ny, nz))
    grid_p10 = tuple(int(np.percentile(a, 10)) for a in (nx, ny, nz))
    grid_p1 = tuple(int(np.percentile(a, 1)) for a in (nx, ny, nz))
    print(f"\nTrain grid: median={grid_med} p10={grid_p10} p1={grid_p1}")

    # ---- Plot atom-count histograms --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.arange(0, 160, 2)
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        a = by_split[split]
        ax.hist(
            a.clip(0, 158), bins=bins,
            histtype="step", linewidth=2,
            label=f"{split} (n={len(a):,}, p50={int(np.median(a))}, p99={int(np.percentile(a, 99))})",
            color=color,
        )
    ax.set_xlabel("n_atoms per material (clipped at 158)")
    ax.set_ylabel("count")
    ax.set_title("Atom counts across the MPDB splits")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "atoms_hist.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT_PLOTS / 'atoms_hist.png'}")

    # ---- Plot preamble-length histograms (train only — same as before) ---
    train_n = by_split["train"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for F, color in (("F0", "tab:red"), ("F1", "tab:green"), ("F2", "tab:purple")):
        pre = preamble_len(train_n, F)
        bins = np.linspace(0, max(pre.max(), 1000), 80)
        ax.hist(
            pre, bins=bins,
            histtype="step", linewidth=2,
            label=f"{F} (mean={pre.mean():.0f}, p50={int(np.median(pre))}, "
                  f"p99={int(np.percentile(pre, 99))}, max={int(pre.max())})",
            color=color,
        )
    ax.set_xlabel("preamble length (tokens, everything before density block)")
    ax.set_ylabel("count (log scale)")
    ax.set_yscale("log")
    ax.set_title("Preamble length across train-full (per-mat fixed cost; not multiplied by M)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "preamble_hist.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT_PLOTS / 'preamble_hist.png'}")

    # ---- Cross-split panel: n_atoms, F0 preamble, F1 preamble per split ---
    splits = [s for s in ("train", "val") if s in by_split]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    split_colors = {"train": "tab:blue", "val": "tab:orange"}

    # (a) n_atoms
    ax = axes[0]
    bins_n = np.arange(0, 160, 2)
    for sp in splits:
        a = by_split[sp]
        ax.hist(
            a.clip(0, 158), bins=bins_n,
            histtype="step", linewidth=2,
            color=split_colors[sp],
            label=f"{sp} (n={len(a):,}, p50={int(np.median(a))}, p99={int(np.percentile(a, 99))})",
        )
    ax.set_xlabel("n_atoms")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("n_atoms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) F0 preamble: 19 + 10*N
    ax = axes[1]
    max_pre_f0 = max(preamble_len(by_split[sp], "F0").max() for sp in splits)
    bins_f0 = np.linspace(0, max(max_pre_f0, 1000), 80)
    for sp in splits:
        pre = preamble_len(by_split[sp], "F0")
        ax.hist(
            pre, bins=bins_f0,
            histtype="step", linewidth=2,
            color=split_colors[sp],
            label=f"{sp} (mean={pre.mean():.0f}, p50={int(np.median(pre))}, p99={int(np.percentile(pre, 99))})",
        )
    ax.set_xlabel("F0 preamble (tokens)")
    ax.set_yscale("log")
    ax.set_title("F0 preamble (19 + 10·N)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (c) F1 preamble: 16 + N (matches preamble_len("F1") definition)
    ax = axes[2]
    max_pre_f1 = max(preamble_len(by_split[sp], "F1").max() for sp in splits)
    bins_f1 = np.linspace(0, max(max_pre_f1, 150), 80)
    for sp in splits:
        pre = preamble_len(by_split[sp], "F1")
        ax.hist(
            pre, bins=bins_f1,
            histtype="step", linewidth=2,
            color=split_colors[sp],
            label=f"{sp} (mean={pre.mean():.0f}, p50={int(np.median(pre))}, p99={int(np.percentile(pre, 99))})",
        )
    ax.set_xlabel("F1 preamble (tokens)")
    ax.set_yscale("log")
    ax.set_title("F1 preamble (16 + N)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Cross-split distributions: train vs val", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "cross_split_hist.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PLOTS / 'cross_split_hist.png'}")

    # ---- Stats per-F preamble (train; kept for back-compat with old JSON) -
    preamble_stats = {}
    for F in ("F0", "F1", "F2"):
        pre = preamble_len(train_n, F)
        preamble_stats[F] = dict(
            mean=float(pre.mean()),
            median=float(np.median(pre)),
            p90=float(np.percentile(pre, 90)),
            p99=float(np.percentile(pre, 99)),
            p99_9=float(np.percentile(pre, 99.9)),
            max=int(pre.max()),
        )

    # ---- Per-split preamble stats (new) ---------------------------------
    preamble_by_split = {}
    for sp in splits:
        n = by_split[sp]
        for F in ("F0", "F1", "F2"):
            pre = preamble_len(n, F)
            preamble_by_split[f"{sp}.{F}"] = dict(
                mean=float(pre.mean()),
                median=float(np.median(pre)),
                p90=float(np.percentile(pre, 90)),
                p99=float(np.percentile(pre, 99)),
                p99_9=float(np.percentile(pre, 99.9)),
                max=int(pre.max()),
            )

    # ---- Packing table (ctx=8192) ----------------------------------------
    Ps = [19, 14, 10, 7]
    Fs = ["F0", "F1", "F2"]
    pack_rows = []
    for P in Ps:
        for F in Fs:
            pre_p99 = preamble_stats[F]["p99"]
            # Per-seq cost in a packed buffer = preamble + DENS_START + P^3 + DENS_END + EOS
            seq_p99 = pre_p99 + 1 + P ** 3 + 1 + 1
            seqs_per_row = max(1, int(CTX // seq_p99))
            density_per_row = seqs_per_row * (P ** 3)
            efficiency = density_per_row / CTX
            pack_rows.append(dict(
                P=P, F=F, pre_p99=int(pre_p99),
                seq_p99=int(seq_p99),
                seqs_per_row=seqs_per_row,
                density_per_row=density_per_row,
                efficiency=efficiency,
            ))

    # ---- M_max + token / storage table ----------------------------------
    # M_max per-mat = floor(nx/P)*floor(ny/P)*floor(nz/P).  Median-mat M_max:
    n_mats_train = len(train_n)
    storage_tables = {}
    for F in Fs:
        rows = []
        pre_mean = np.mean(preamble_len(train_n, F))
        for P in Ps:
            Mmax_per_mat = (nx // P) * (ny // P) * (nz // P)
            Mmax_med = int(np.median(Mmax_per_mat))
            Mmax_p1 = int(np.percentile(Mmax_per_mat, 1))
            Mmax_p99 = int(np.percentile(Mmax_per_mat, 99))
            # Three M choices
            for label, M in (
                ("M=64", 64),
                (f"M=Mmax/2={Mmax_med//2}", Mmax_med // 2),
                (f"M=Mmax={Mmax_med}", Mmax_med),
            ):
                n_seqs = M * n_mats_train
                tot_tokens_unpacked = n_seqs * CTX        # padded layout
                # Effective real (non-pad) tokens:
                seq_real = pre_mean + 1 + P ** 3 + 2  # +DENS_START +DENS_END +EOS
                tot_tokens_real = n_seqs * seq_real
                rows_count = n_seqs                       # 1 seq / row (current scheme)
                disk_bytes = rows_count * CTX * 4         # int32 padded
                rows.append((
                    P, label, M, Mmax_med, Mmax_p1, Mmax_p99,
                    n_seqs, tot_tokens_real, disk_bytes,
                ))
        storage_tables[F] = rows

    # ---- Emit JSON ----
    out = dict(
        ctx=CTX,
        by_split={k: percentiles(v) | {"n": int(len(v))} for k, v in by_split.items()},
        train_grid=dict(median=list(grid_med), p10=list(grid_p10), p1=list(grid_p1)),
        n_atoms_train_p99=int(np.percentile(by_split["train"], 99)),
        n_atoms_train_max=int(by_split["train"].max()),
        preamble_stats=preamble_stats,
        preamble_by_split=preamble_by_split,
        packing=pack_rows,
        storage=storage_tables,
        Mmax_per_P_med=[
            int(np.median((nx // P) * (ny // P) * (nz // P))) for P in Ps
        ],
        n_mats_train=n_mats_train,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT_JSON}")

    # ---- Pretty markdown tables -----------------------------------------
    print("\n## Atom-count percentiles\n")
    print(tabulate(
        [(s, len(v), int(np.percentile(v, 1)), int(np.percentile(v, 10)),
          int(np.median(v)), int(np.percentile(v, 90)),
          int(np.percentile(v, 99)), int(v.max())) for s, v in by_split.items()],
        headers=["split", "n", "p1", "p10", "p50", "p90", "p99", "max"],
        tablefmt="github",
    ))

    print("\n## Preamble-length stats (tokens, per-mat, **not** × M)\n")
    print(tabulate(
        [(F, f"{s['mean']:.1f}", int(s['median']), int(s['p90']),
          int(s['p99']), int(s['p99_9']), int(s['max']))
         for F, s in preamble_stats.items()],
        headers=["F", "mean", "p50", "p90", "p99", "p99.9", "max"],
        tablefmt="github",
    ))

    print("\n## Per-split preamble stats (F0, F1) — train vs val\n")
    rows = []
    for sp in splits:
        for F in ("F0", "F1"):
            s = preamble_by_split[f"{sp}.{F}"]
            rows.append((
                sp, F, f"{s['mean']:.1f}", int(s['median']), int(s['p90']),
                int(s['p99']), int(s['p99_9']), int(s['max']),
            ))
    print(tabulate(
        rows,
        headers=["split", "F", "mean", "p50", "p90", "p99", "p99.9", "max"],
        tablefmt="github",
    ))
    if "test" not in by_split:
        print("\n(no `test` split in MPDB — only train + val are populated)")

    print(f"\n## Packing @ ctx={CTX} (using preamble_p99)\n")
    print(tabulate(
        [(r["P"], r["F"], r["pre_p99"], r["seq_p99"], r["seqs_per_row"],
          r["density_per_row"], f"{r['efficiency']*100:.1f}%") for r in pack_rows],
        headers=["P", "F", "pre_p99", "seq_p99", "seqs/row", "ρ-tok/row", "ρ-efficiency"],
        tablefmt="github",
    ))

    print("\n## M_max per P (median train mat, M_max = floor(nx/P)·floor(ny/P)·floor(nz/P))\n")
    for P in Ps:
        Mmax = (nx // P) * (ny // P) * (nz // P)
        print(f"- P={P}: median M_max={int(np.median(Mmax))}, "
              f"p1={int(np.percentile(Mmax, 1))}, p99={int(np.percentile(Mmax, 99))}, "
              f"max={int(Mmax.max())}")

    print("\n## Token + storage table (F0) — current encoding\n")
    print(tabulate(
        [(P, label, M,
          f"{tot:,}",
          f"{tot_tok / 1e9:.1f}B",
          f"{disk_b / 1e9:.1f} GB")
         for (P, label, M, _Mmax, _p1, _p99, tot, tot_tok, disk_b) in storage_tables["F0"]],
        headers=["P", "label", "M", "n_seqs", "real-tokens", "disk (padded)"],
        tablefmt="github",
    ))

    print("\n## Token + storage table (F1) — fused atom-emb (1 tok/atom)\n")
    print(tabulate(
        [(P, label, M,
          f"{tot:,}",
          f"{tot_tok / 1e9:.1f}B",
          f"{disk_b / 1e9:.1f} GB")
         for (P, label, M, _Mmax, _p1, _p99, tot, tot_tok, disk_b) in storage_tables["F1"]],
        headers=["P", "label", "M", "n_seqs", "real-tokens", "disk (padded)"],
        tablefmt="github",
    ))


if __name__ == "__main__":
    main()
