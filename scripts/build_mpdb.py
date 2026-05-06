#!/usr/bin/env python
# /// script
# dependencies = ["click"]
# ///
"""Build or update the tomat materials DB (MPDB).

SQLite at data/mpdb.sqlite. Schema:

  mats(mp_id TEXT PRIMARY KEY,
       split TEXT,              -- "train" | "val" | "test" | NULL
       nx, ny, nz INTEGER,
       n_atoms INTEGER,
       n_electrons INTEGER,     -- sum of Z across atoms; v2+
       n_voxels INTEGER GENERATED ALWAYS AS (nx*ny*nz),
       cube_seq_pN INTEGER GENERATED ALWAYS AS (28 + 10*n_atoms + 2*N^3) for N in 14..20,
       ball_seq_rN INTEGER GENERATED ALWAYS AS (29 + 10*n_atoms + 2*V_ball(N))     for R² in {75,86,100,138,153}
  )

Schema versions (PRAGMA user_version):
  0/unset — original schema (no n_electrons)
  2       — adds n_electrons; val rows backfilled with n_atoms

Usage:
  scripts/build_mpdb.py ingest --csv /tmp/train-full-preamble-clean.csv --split train
  scripts/build_mpdb.py ingest --csv /tmp/val-full-shapes-clean.csv    --split val
  scripts/build_mpdb.py query   --filter 'n_atoms > 100'                 # sample query
"""

from __future__ import annotations

import csv as csv_mod
import sqlite3
import sys
from functools import partial
from pathlib import Path

import click

err = partial(print, file=sys.stderr)

DEFAULT_DB = Path("data/mpdb.sqlite")
SCHEMA_VERSION = 2  # bumped to 2 when n_electrons was added

# V_ball(R²) — precomputed cumulative voxel counts for the R² values we care about.
BALL_VOXELS = {75: 2777, 86: 3407, 100: 4169, 138: 6859, 153: 8025}


def ball_cum_counts(max_r2: int) -> list[int]:
    """Return V(n) for n = 0..max_r2."""
    import numpy as np  # optional, only used for ingestion
    R = int(np.ceil(np.sqrt(max_r2))) + 1
    shell = np.zeros(max_r2 + 1, dtype=np.int64)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= max_r2:
                    shell[r2] += 1
    return shell.cumsum().tolist()


# ---- schema -------------------------------------------------------------


def _schema(ball_r2s: list[int]) -> str:
    ball_cols = ",\n  ".join(
        f"ball_seq_r{r2} INTEGER GENERATED ALWAYS AS "
        f"(CASE WHEN n_atoms IS NOT NULL THEN 29 + 10*n_atoms + 2*{BALL_VOXELS[r2]} ELSE NULL END) VIRTUAL"
        for r2 in ball_r2s
    )
    cube_cols = ",\n  ".join(
        f"cube_seq_p{P} INTEGER GENERATED ALWAYS AS "
        f"(CASE WHEN n_atoms IS NOT NULL THEN 28 + 10*n_atoms + 2*{P**3} ELSE NULL END) VIRTUAL"
        for P in (14, 15, 16, 17, 18, 19, 20)
    )
    return f"""
CREATE TABLE IF NOT EXISTS mats (
  mp_id        TEXT    PRIMARY KEY,
  split        TEXT,
  nx           INTEGER,
  ny           INTEGER,
  nz           INTEGER,
  n_atoms      INTEGER,
  n_electrons  INTEGER,
  n_voxels     INTEGER GENERATED ALWAYS AS (nx * ny * nz) VIRTUAL,
  max_dim      INTEGER GENERATED ALWAYS AS (MAX(nx, ny, nz)) VIRTUAL,
  {cube_cols},
  {ball_cols}
);
CREATE INDEX IF NOT EXISTS idx_mats_split  ON mats(split);
CREATE INDEX IF NOT EXISTS idx_mats_atoms  ON mats(n_atoms);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_schema(conn: sqlite3.Connection, ball_r2s: list[int]) -> None:
    conn.executescript(_schema(ball_r2s))
    # ALTER TABLE for upgrade from v1 (no n_electrons) → v2.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(mats)").fetchall()}
    if "n_electrons" not in cols:
        conn.execute("ALTER TABLE mats ADD COLUMN n_electrons INTEGER")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


# ---- CLI ----------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db", type=click.Path(path_type=Path), default=DEFAULT_DB, help=f"DB path (default {DEFAULT_DB})")
@click.option("--csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Input CSV with mp_id,nx,ny,nz[,n_atoms].")
@click.option("--split", type=click.Choice(["train", "val", "test"]), required=True)
@click.option("--replace/--no-replace", default=False, help="Delete existing rows for this split first.")
def ingest(db: Path, csv: Path, split: str, replace: bool):
    """Ingest a CSV of preamble stats into MPDB."""
    conn = connect(db)
    ensure_schema(conn, sorted(BALL_VOXELS))

    if replace:
        n_del = conn.execute("DELETE FROM mats WHERE split = ?", (split,)).rowcount
        err(f"[mpdb] deleted {n_del:,} existing {split!r} rows")

    with open(csv) as f:
        rdr = csv_mod.DictReader(f)
        rows = []
        for r in rdr:
            mp_id = r["mp_id"]
            try:
                nx, ny, nz = int(r["nx"]), int(r["ny"]), int(r["nz"])
            except (KeyError, ValueError):
                continue
            n_atoms = int(r["n_atoms"]) if r.get("n_atoms") else None
            n_electrons = int(r["n_electrons"]) if r.get("n_electrons") else None
            rows.append((mp_id, split, nx, ny, nz, n_atoms, n_electrons))

    cur = conn.executemany(
        "INSERT OR REPLACE INTO mats (mp_id, split, nx, ny, nz, n_atoms, n_electrons) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    err(f"[mpdb] ingested {cur.rowcount:,} rows from {csv} (split={split})")

    n = conn.execute("SELECT COUNT(*) FROM mats").fetchone()[0]
    n_by_split = dict(conn.execute("SELECT split, COUNT(*) FROM mats GROUP BY split").fetchall())
    err(f"[mpdb] total: {n:,}  per-split: {n_by_split}")


@cli.command()
@click.option("--db", type=click.Path(path_type=Path), default=DEFAULT_DB)
@click.option("--filter", "where", default="1=1", help="Extra WHERE clause (SQL).")
@click.option("--cols", default="mp_id,split,nx,ny,nz,n_atoms,n_voxels,cube_seq_p15", help="SELECT cols")
@click.option("--limit", type=int, default=20)
def query(db: Path, where: str, cols: str, limit: int):
    """Run an ad-hoc SELECT against MPDB."""
    conn = connect(db)
    sql = f"SELECT {cols} FROM mats WHERE {where} LIMIT {limit}"
    err(f"[mpdb] {sql}")
    rows = list(conn.execute(sql))
    if not rows:
        print("(no rows)")
        return
    col_names = [d[0] for d in conn.execute(sql).description]
    print(" | ".join(col_names))
    print("-" * 80)
    for r in rows:
        print(" | ".join(str(x) for x in r))


@cli.command()
@click.option("--db", type=click.Path(path_type=Path), default=DEFAULT_DB)
def summary(db: Path):
    """Print per-split counts, atom-count distribution, fits-in-CL counts."""
    conn = connect(db)
    n_total = conn.execute("SELECT COUNT(*) FROM mats").fetchone()[0]
    err(f"total mats: {n_total:,}")

    print("\nPer-split counts:")
    for split, n in conn.execute("SELECT split, COUNT(*) FROM mats GROUP BY split"):
        print(f"  {split or '(null)':10s} {n:>7,}")

    print("\nAtom-count percentiles (across all splits with n_atoms set):")
    pcts = [0.0, 0.5, 0.75, 0.99, 1.0]
    vals = [
        conn.execute(
            f"SELECT n_atoms FROM mats WHERE n_atoms IS NOT NULL "
            f"ORDER BY n_atoms LIMIT 1 OFFSET (SELECT COUNT(*) FROM mats WHERE n_atoms IS NOT NULL) * ? / 100"
        , (int(p*100),)).fetchone()
        for p in pcts
    ]
    for p, v in zip(pcts, vals):
        print(f"  p{int(p*100):3d}: {v[0] if v else '—'}")

    print("\nFits-in-CL counts (% of mats with n_atoms known):")
    n_with = conn.execute("SELECT COUNT(*) FROM mats WHERE n_atoms IS NOT NULL").fetchone()[0]
    if n_with == 0:
        print("  (no rows with n_atoms)")
        return
    for CL in (8192, 16384, 32768):
        row = []
        for P in (14, 15, 16, 19, 20):
            n_fit = conn.execute(
                f"SELECT COUNT(*) FROM mats WHERE n_atoms IS NOT NULL AND cube_seq_p{P} <= ?",
                (CL,),
            ).fetchone()[0]
            row.append(f"P={P}:{100*n_fit/n_with:5.2f}%")
        for R2 in sorted(BALL_VOXELS):
            n_fit = conn.execute(
                f"SELECT COUNT(*) FROM mats WHERE n_atoms IS NOT NULL AND ball_seq_r{R2} <= ?",
                (CL,),
            ).fetchone()[0]
            row.append(f"R²={R2}:{100*n_fit/n_with:5.2f}%")
        print(f"  CL={CL:>5}: " + " ".join(row))


if __name__ == "__main__":
    cli()
