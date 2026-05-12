# MPDB — materials metadata DB

Status: **v1 live** (81,769 mats ingested). Written 2026-04-23.

## Purpose

Single source of truth for per-material metadata. Replaces the ad-hoc
CSV scans we've been running on Modal to answer questions like:

- "What fraction of train-full fits P=15 at 8k context?"
- "Which materials have > 100 atoms?"
- "Show me the max-voxel material in val-full."
- "Pre-filter materials for a specific (shape, P, R, CL) combo."

## Schema (SQLite)

Path: `data/mpdb.sqlite`. Generated/updated by `scripts/build_mpdb.py`.

```sql
CREATE TABLE mats (
  mp_id    TEXT PRIMARY KEY,
  split    TEXT,                       -- "train" | "val" | "test" | NULL
  nx       INTEGER,
  ny       INTEGER,
  nz       INTEGER,
  n_atoms  INTEGER,                    -- NULL for val (no atom counts pulled yet)
  n_voxels INTEGER GENERATED ALWAYS AS (nx * ny * nz) VIRTUAL,
  max_dim  INTEGER GENERATED ALWAYS AS (MAX(nx, ny, nz)) VIRTUAL,
  cube_seq_p14 INTEGER GENERATED AS (28 + 10*n_atoms + 2*14³) VIRTUAL,
  cube_seq_p15 INTEGER GENERATED AS (28 + 10*n_atoms + 2*15³) VIRTUAL,
  cube_seq_p16 INTEGER GENERATED AS (28 + 10*n_atoms + 2*16³) VIRTUAL,
  cube_seq_p17 INTEGER GENERATED AS (28 + 10*n_atoms + 2*17³) VIRTUAL,
  cube_seq_p18 INTEGER GENERATED AS (28 + 10*n_atoms + 2*18³) VIRTUAL,
  cube_seq_p19 INTEGER GENERATED AS (28 + 10*n_atoms + 2*19³) VIRTUAL,
  cube_seq_p20 INTEGER GENERATED AS (28 + 10*n_atoms + 2*20³) VIRTUAL,
  ball_seq_r75  INTEGER GENERATED AS (29 + 10*n_atoms + 2*2777) VIRTUAL,
  ball_seq_r86  INTEGER GENERATED AS (29 + 10*n_atoms + 2*3407) VIRTUAL,
  ball_seq_r100 INTEGER GENERATED AS (29 + 10*n_atoms + 2*4169) VIRTUAL,
  ball_seq_r138 INTEGER GENERATED AS (29 + 10*n_atoms + 2*6859) VIRTUAL,
  ball_seq_r153 INTEGER GENERATED AS (29 + 10*n_atoms + 2*8025) VIRTUAL
);

CREATE INDEX idx_mats_split  ON mats(split);
CREATE INDEX idx_mats_atoms  ON mats(n_atoms);
```

Derived cols are computed inline by SQLite each query (VIRTUAL, no
storage). Only source cols (mp_id, split, nx, ny, nz, n_atoms) live
on disk.

## CLI

```bash
# Ingest a CSV (with mp_id,nx,ny,nz,n_atoms — atoms column optional):
scripts/build_mpdb.py ingest --csv path.csv --split train

# Ad-hoc query:
scripts/build_mpdb.py query --filter 'n_atoms > 100' --cols mp_id,n_atoms,cube_seq_p15

# Whole-DB summary (per-split counts, atom distribution, fits-in-CL):
scripts/build_mpdb.py summary
```

## Current state

- **77,466** train mats (with full `nx, ny, nz, n_atoms`)
- **4,303** val mats (shape only; atom counts not pulled — follow-up
  script needed; `scripts/pull_preamble_stats_modal.py --label val-full`
  will close the gap)

Total: **81,769** rows. Disk: ~6 MB SQLite file (vs 2 MB CSV — small
index overhead).

## Planned additions

### Phase 2: MP API enrichment
Pull per-material metadata from the Materials Project REST API
(composition, formula, reduced formula, space group, lattice
parameters, energy per atom, etc.). Adds ~50 MB. Enables richer
filtering and stratified analyses.

```python
# sketch
from mp_api.client import MPRester
with MPRester() as mpr:
    for chunk in batched(mp_ids, 500):
        docs = mpr.summary.search(material_ids=chunk,
                                  fields=["material_id", "formula_pretty",
                                          "reduced_formula", "spacegroup",
                                          "lattice", "energy_per_atom"])
        ...
```

### Phase 3: Full MP universe
Fetch metadata for *all* MP materials (~150k, as of 2026). Flag
`in_della=0` for the ones without rho_gga coverage. Enables "what
fraction of MP do we train on?" analyses.

### Phase 4: D1 mirror
Export to Cloudflare D1 for low-latency queries from the tomat site.
Enables live filtering UIs (e.g. `/mats?n_atoms_min=100&fits_cl=8192`).

```bash
sqlite3 data/mpdb.sqlite .dump > data/mpdb.sql
wrangler d1 create tomat-mpdb
wrangler d1 execute tomat-mpdb --file data/mpdb.sql
```

### Phase 5: Training-variant results
Add a companion table `variant_results(variant, mp_id, in_corpus,
skip_reason, n_patches, patch_nmae)` — joins with `mats` to track per-
variant, per-material stats (which mats were skipped, which had high
NMAE at eval, etc.). Populated at the end of each tokenize + eval
cycle.

## Consumers

- `scripts/analyze_preamble.py` — can migrate to MPDB queries (from
  the current CSV-load-and-compute path).
- `scripts/analyze_voxel_coverage.py` — same.
- site dashboards: `/mats` page with filtered-table + histograms
  driven from D1 once Phase 4 lands.

## DVX tracking

`data/mpdb.sqlite` could be DVX-tracked (per spec 09 pattern); the
`.dvc` file encodes the git-state + scripts/commands that produced
it, making the DB reproducible from scratch. Deferred until we
have multiple tokenized variants each contributing rows.

## Related

- `scripts/pull_preamble_stats_modal.py` — Modal-parallel pull of
  (grid dims, atom counts) from tokenized parquets. Populates MPDB.
- `specs/09-dvx-track-val-full.md` — DVX tracking infra.
