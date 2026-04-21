# Seed `tomat-rho-gga` Modal Volume from della

This spec is for the tomat session running on della. The companion
session (on laptop) is implementing `PatchTokenizer` + a Zarr loader
against the Modal-volume-mounted rho_gga data; both hinge on the volume
existing and being seeded with a useful subset.

## Goal

Stand up a new Modal Volume named **`tomat-rho-gga`** and populate it
with the **validation split** of the native-resolution MP charge-density
dataset (rho_gga Zarrs). The val split is 4,305 structures × ~5 MB =
~22 GB — small enough to commit ($11/month) while the training pipeline
is being built; full train split (~390 GB) comes later.

Source on della:
```
/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/
  label/                         # 86 192 .zarr dirs, one per task
  data/                          # (not used by tomat; skip)
  mp_filelist.txt                # all task IDs (mp-XXXXX, one per line)
  split_limit_22M.json           # canonical train/val/test split
  split_limit_40M.json           # ditto, larger cap
  README.md
```

**Note on the split file format**: `split_limit_22M.json` has keys
`train`, `validation`, `test`, and values are lists of **ints** — 0-based
indices into `mp_filelist.txt` lines, not mp-IDs directly. The initial
spec assumed strings; the stage script resolves indices → mp-IDs via
the filelist.

## Prereqs

- ssh access to della (already set up for `ssh della`)
- A Modal account under the same org the laptop session uses (i.e., the
  one that currently owns volumes `electrai-data`, `electrai-checkpoints`,
  `helico-bench-data`) — `open-athena` workspace

## Plan

### 1. Install + auth Modal CLI on della

```bash
cd /scratch/gpfs/ROSENGROUP/$USER/tomat       # or ~/tomat if symlink exists
vsw 3.12                                      # 3.12 venv (project: >=3.11,<3.13)
uv pip install modal
modal token new                               # headless: copy printed URL
                                              # to a local browser
```

Verify:
```bash
modal profile current               # -> default (or workspace name)
modal volume list                   # should show electrai-data, etc.
```

### 2. Create the `tomat-rho-gga` volume

```bash
modal volume create tomat-rho-gga
modal volume list | grep tomat-rho-gga     # sanity
```

### 3. Stage the val-split subset

Build a flat staging directory of just the val-split Zarr dirs, plus
the filelists, so a single `modal volume put` copies the lot:

```bash
RHO_GGA=/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga
STAGE=/scratch/gpfs/ROSENGROUP/${USER}/tomat-rho-gga-stage

mkdir -p "$STAGE/label"

# Resolve val indices → mp-IDs and write val_ids.txt (4,305 entries):
python3 -c "
import json, pathlib
rho, stage = '$RHO_GGA', '$STAGE'
split = json.load(open(f'{rho}/split_limit_22M.json'))
with open(f'{rho}/mp_filelist.txt') as f:
    filelist = [l.strip() for l in f]
val_ids = [filelist[i] for i in split['validation']]
pathlib.Path(f'{stage}/val_ids.txt').write_text('\n'.join(val_ids) + '\n')
print('val task IDs:', len(val_ids))
"

# Hard-link each val Zarr dir into the staging area (no copy cost; the
# .zarr dir is itself a directory of small files).
cd "$STAGE/label"
while read id; do
    if [ -d "$RHO_GGA/label/$id.zarr" ]; then
        cp -al "$RHO_GGA/label/$id.zarr" .
    else
        echo "MISSING: $id" >&2
    fi
done < "$STAGE/val_ids.txt"

# Also copy the split + filelists for reproducibility.
cp "$RHO_GGA/split_limit_22M.json" "$RHO_GGA/mp_filelist.txt" "$RHO_GGA/README.md" "$STAGE/"

# Sanity: total size + file count
du -sh "$STAGE"                                     # -> 22G
ls "$STAGE/label" | wc -l                           # -> 4305
```

### 4. Upload to the Modal volume

```bash
# One big put — volume-put handles recursive dirs. The REMOTE arg must
# NOT end with a trailing slash, or modal will nest the source dir
# inside it (landed as /label/label/ on the first attempt).
modal volume put tomat-rho-gga "$STAGE/label" /label
modal volume put tomat-rho-gga "$STAGE/split_limit_22M.json" split_limit_22M.json
modal volume put tomat-rho-gga "$STAGE/mp_filelist.txt" mp_filelist.txt
modal volume put tomat-rho-gga "$STAGE/README.md" README.md
modal volume put tomat-rho-gga "$STAGE/val_ids.txt" val_ids.txt
```

**Gotcha encountered**: on the first run we used `label/` as the remote
destination, which put the source dir *inside* that remote dir →
`/label/label/mp-*.zarr`. Fixed server-side via a small Modal function
that mounts the volume and `os.rename`s every child up one level; see
`scripts/fix_modal_volume_layout.py` (one-shot, kept in repo as a
reference — recursive server-side `modal volume cp -r` is not
supported on V1 volumes, which is the default). The correct
destination form for subsequent uploads is just `/label` (no trailing
slash). Upload took ~20 min for 22 GB over della's network.

### 5. Verify

```bash
modal volume ls tomat-rho-gga /
modal volume ls tomat-rho-gga /label | wc -l         # -> 4305

# End-to-end read test: mount the volume in a Modal function,
# open a Zarr group, read charge_density_total + structure attr.
modal run scripts/verify_modal_volume.py
```

### 6. Clean up staging

```bash
rm -rf "$STAGE"    # hard-links, so this won't touch the original data
```

## Outcome (2026-04-21)

- Volume `tomat-rho-gga` created under the `open-athena` workspace.
- 4,305 val-split `mp-*.zarr` dirs under `/label/` (~22 GB).
- Metadata at root: `mp_filelist.txt`, `split_limit_22M.json`,
  `val_ids.txt`, `README.md`.
- `scripts/verify_modal_volume.py` successfully reads a sample Zarr:
  `mp-1774446` (shape 56×56×252, float32, 12 sites, elements
  {As, S, Tm}).

## Follow-ups (separate PR, after the val subset is validated)

Once `PatchTokenizer` + training loop are working against val:

- Upload the **train split** (77,498 structures, ~390 GB) the same way
  but pointing at `split["train"]` instead of `split["validation"]`.
  Expected cost: ~$195/month Modal-storage at current pricing.

- Decide whether to also upload `data/` (SAD inputs). tomat doesn't use
  them for tokenization; they'd only be needed if we wanted to model
  the SAD → DFT trajectory (which isn't on the current roadmap).

- Decide whether to upload the `split_limit_40M.json` split too (fits
  bigger grids on bigger accelerators; relevant if we move off A100).

- Consider creating future volumes as V2 (`modal volume create
  --version 2`) — V1 has no recursive server-side copy, which made the
  `/label/label` → `/label` rescue more awkward than it needed to be
  (had to write a Modal function instead of a one-liner).

## Out of scope

- Converting additional MP structures to Zarr (rho_gga already covers
  86 k of MP's ~155 k; the gap is mostly non-GGA tasks; chase later).
- Changing the Zarr format (already v3, float32, zstd-compressed —
  good-enough for now).

## Done criteria

- [x] `modal volume ls tomat-rho-gga /label` shows ~4,305 entries.
- [x] A script can open a Zarr via Modal-volume-mount and read
  `charge_density_total` + the `structure` JSON from the attrs
  (`scripts/verify_modal_volume.py`).
- [x] Move this spec to `specs/done/`.
