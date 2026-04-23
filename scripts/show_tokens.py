#!/usr/bin/env python
# Reads `tomat` from the repo; run from the tomat venv (`direnv` auto-activates).
"""Render a human-readable excerpt of a tokenized training example.

Reads one `input_ids` row from a parquet shard, decodes token IDs back
into their semantic names (special tokens, element symbols, raw ints,
position-codec digits, density-codec digits), and prints a compact
formatted view suitable for docs / slides.

Usage:
    scripts/show_tokens.py \\
        --parquet tmp/train-full-pull/train-full/worker-00/shard-00000.parquet \\
        --row 0
"""

from __future__ import annotations

from functools import partial
import sys

import click
import pyarrow.parquet as pq

from tomat.tokenizers.patch import PatchTokenizer, SPECIAL_TOKENS

err = partial(print, file=sys.stderr)

# Layout: 18 specials | 118 atomic Zs | 1024 ints | 1024 position-codec | 4608 density-codec
N_SPECIALS = len(SPECIAL_TOKENS)
ATOM_OFFSET = N_SPECIALS
INT_OFFSET = N_SPECIALS + 118  # 136
# position + density offsets come from the tokenizer instance


def _specials_by_id() -> dict[int, str]:
    return {v: k for k, v in SPECIAL_TOKENS.items()}


def _element_symbol(z: int) -> str:
    # Minimal periodic table — covers all Z in the MP dataset.
    PT = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
        "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
        "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
        "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
        "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
        "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    ).split()
    return PT[z - 1] if 1 <= z <= len(PT) else f"Z{z}"


def decode_token(tid: int, tokenizer: PatchTokenizer) -> str:
    """Convert a single numeric ID back to a short human-readable label."""
    v = tokenizer.vocab
    if tid < N_SPECIALS:
        return _specials_by_id()[tid]
    if tid < INT_OFFSET:
        return _element_symbol(tid - ATOM_OFFSET + 1)
    if tid < v.position_offset:
        return str(tid - INT_OFFSET)  # raw int
    if tid < v.density_offset:
        # Position codec: 3 tokens per coord (byte0, byte1, byte2).
        # Just render as `p<offset>` so it's compact.
        return f"p{tid - v.position_offset}"
    # Density codec: 2 tokens per voxel (hi, lo).
    return f"d{tid - v.density_offset}"


@click.command()
@click.option("-p", "--parquet", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("-r", "--row", type=int, default=0, help="which row of the parquet to render")
@click.option("-P", "--patch-size", type=int, default=14)
@click.option("--dens-head", type=int, default=6, help="density tokens to show at start")
@click.option("--dens-tail", type=int, default=6, help="density tokens to show at end")
def main(parquet: str, row: int, patch_size: int, dens_head: int, dens_tail: int) -> None:
    t = PatchTokenizer()
    table = pq.read_table(parquet)
    ids = table.column("input_ids").to_pylist()[row]
    mp_id = table.column("task_id")[row].as_py() if "task_id" in table.column_names else None
    if mp_id:
        print(f"# {mp_id}  https://elvis.oa.dev/?m={mp_id}")

    # Walk block-by-block so we can group like-kinded tokens per line.
    S = SPECIAL_TOKENS
    BOS, EOS = S["[BOS]"], S["[EOS]"]
    blocks = [
        ("GRID", S["[GRID_START]"], S["[GRID_END]"]),
        ("ATOMS", S["[ATOMS_START]"], S["[ATOMS_END]"]),
        ("POS", S["[POS_START]"], S["[POS_END]"]),
        ("SHAPE", S["[SHAPE_START]"], S["[SHAPE_END]"]),
        ("OFFSET", S["[OFFSET_START]"], S["[OFFSET_END]"]),
        ("HI", S["[HI_START]"], S["[HI_END]"]),
        ("DENS", S["[DENS_START]"], S["[DENS_END]"]),
    ]

    def fmt(tid: int) -> str:
        return decode_token(tid, t)

    def find(tok: int, start: int = 0) -> int:
        return ids.index(tok, start)

    print(f"{fmt(ids[0])}")
    assert ids[0] == BOS, "expected BOS as first token"

    cur = 1
    for name, o, c in blocks:
        o_i = find(o, cur)
        c_i = find(c, o_i + 1)
        inner = ids[o_i + 1: c_i]
        if name == "DENS":
            head = " ".join(fmt(x) for x in inner[:dens_head])
            tail = " ".join(fmt(x) for x in inner[-dens_tail:])
            n = len(inner)
            print(f"{fmt(o)}   {head}  …  {tail}  "
                  f"# {n:,} density tokens (= 2 × {patch_size}³)")
        elif name == "POS":
            # Group per-atom: 9 tokens = 3 coords × 3 codec-tokens.
            per_atom = 9
            atoms_in = [inner[i:i + per_atom] for i in range(0, len(inner), per_atom)]
            toks_preview = "  ".join(
                "(" + " ".join(fmt(x) for x in a) + ")"
                for a in atoms_in[:2]
            )
            n_atoms = len(atoms_in)
            if n_atoms > 2:
                toks_preview += f"  …  (+{n_atoms - 2} more atoms)"
            print(f"{fmt(o)}    {toks_preview}")
        else:
            toks_str = " ".join(fmt(x) for x in inner)
            print(f"{fmt(o)}  {toks_str}")
        print(f"{fmt(c)}")
        cur = c_i + 1

    eos_i = find(EOS, cur)
    print(f"{fmt(ids[eos_i])}")
    n_pad = len(ids) - eos_i - 1
    if n_pad > 0:
        print(f"[PAD] × {n_pad}   # right-padded to {len(ids)}")


if __name__ == "__main__":
    main()
