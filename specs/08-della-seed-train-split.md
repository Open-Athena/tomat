# Seed train split onto `tomat-rho-gga` Modal volume (from della)

Follow-up to spec 03 (val-only seed). Spec 07 plans parallel tokenization
of the full train split at Modal-side; this spec is the **della-side**
prerequisite: get the train-split Zarrs onto the Modal volume so spec 07
has something to tokenize.

This spec is for **the della session** (the laptop session can't reach
della / `/scratch/gpfs/...`). Laptop session is standing by to run parallel
tokenize on Modal the moment the upload finishes.

## Goal

Upload the **train split** (77,498 structures, ~390 GB) of the native-
resolution rho_gga Zarrs to a new Modal volume named **`tomat-rho-gga-train`**
(V2 format so recursive `cp -r` works server-side; per spec 03's follow-up).
Keep `tomat-rho-gga` as the val-only volume.

## Prereqs

- della session with `modal` installed + authenticated to the
  `open-athena` workspace (spec 03's `modal token new` etc. — ran Apr 21).
  Sanity: `modal profile current` should report `open-athena`.
- rho_gga on disk at
  `/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/`
  (confirmed present during spec 03).
- ~400 GB free on `/scratch/gpfs/ROSENGROUP/<user>` for hard-link staging.

## Plan

### 1. Create the volume (V2 format)

```bash
modal volume create --version 2 tomat-rho-gga-train
modal volume list | grep tomat-rho-gga-train     # sanity
```

Spec 03 used V1 for `tomat-rho-gga`; V2 supports server-side recursive
copy which avoids the `/label/label/` rescue we hit before. Keep V1
for val; use V2 for this new volume.

### 2. Resolve train-split task IDs

`split_limit_22M.json`'s `train` key is a list of **int indices** into
`mp_filelist.txt`, per spec 03's format note. Resolve to mp-IDs:

```bash
RHO_GGA=/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga
STAGE=/scratch/gpfs/ROSENGROUP/${USER}/tomat-rho-gga-train-stage

mkdir -p "$STAGE/label"
python3 -c "
import json, pathlib
split = json.load(open('$RHO_GGA/split_limit_22M.json'))
with open('$RHO_GGA/mp_filelist.txt') as f: filelist = [l.strip() for l in f]
train_ids = [filelist[i] for i in split['train']]
pathlib.Path('$STAGE/train_ids.txt').write_text('\\n'.join(train_ids) + '\\n')
print(f'train task IDs: {len(train_ids):,}')
"
# Expect ~77,498
```

### 3. Hard-link stage the Zarr dirs

```bash
cd "$STAGE/label"
missing=0
while read id; do
    if [ -d "$RHO_GGA/label/$id.zarr" ]; then
        cp -al "$RHO_GGA/label/$id.zarr" .
    else
        missing=$((missing+1))
        echo "MISSING: $id" >&2
    fi
done < "$STAGE/train_ids.txt"
echo "missing: $missing"

du -sh "$STAGE"                           # expect ~370 GB
ls "$STAGE/label" | wc -l                 # expect ~77,498 minus missing
```

### 4. Upload to the Modal volume

`tomat-rho-gga-train` is V2, so a single `modal volume put` works
without the trailing-slash nesting bug we hit on V1.

```bash
modal volume put tomat-rho-gga-train "$STAGE/label"            /label
modal volume put tomat-rho-gga-train "$STAGE/train_ids.txt"    train_ids.txt
modal volume put tomat-rho-gga-train "$RHO_GGA/split_limit_22M.json" split_limit_22M.json
modal volume put tomat-rho-gga-train "$RHO_GGA/mp_filelist.txt"      mp_filelist.txt
```

Upload time: **~hours** over della's network for 370 GB. Run overnight
if possible. `modal volume put` is resumable-ish (re-running skips files
that match on size; re-running on interruption should work).

### 5. Verify

```bash
modal volume ls tomat-rho-gga-train /               # root listing
modal volume ls tomat-rho-gga-train /label | wc -l  # expect ~77,498

# End-to-end read test — spec 03's scripts/verify_modal_volume.py is
# val-volume-specific but easy to adapt: change VOLUME_NAME to
# "tomat-rho-gga-train" and pass any mp-id from train_ids.txt.
```

### 6. Clean up staging

```bash
rm -rf "$STAGE"   # hard-links, so this doesn't touch the source data
```

## Blocker on laptop side while this runs

Nothing — the laptop session is training on val-full and tokenizing
val-full-m128 in the meantime. As soon as this spec's done criteria
are green, laptop session fires:

```bash
modal run scripts/tokenize_patches_modal.py::parallel \
    --label train-full \
    --split train \
    --n-workers 64 \
    --pad-to 8192
```

(Note: the `parallel` entrypoint currently hardcodes the volume name
`tomat-rho-gga`. Needs a small edit to accept `--volume-name
tomat-rho-gga-train` before this runs. Laptop session will do that
once the volume is live.)

## Out of scope

- Tokenizing the train split on Modal (spec 07 / laptop session).
- Hashing input Zarrs into a content manifest (spec 07, stage 3).
- Downloading the `split_limit_40M.json` split (relevant only for
  TPU/H100, deferred).

## Done criteria

- [ ] `modal volume list` shows `tomat-rho-gga-train` (V2).
- [ ] `modal volume ls tomat-rho-gga-train /label | wc -l` ≈ 77,498
      (a few hundred missing is fine — flag the count).
- [ ] Spec moved to `specs/done/08-della-seed-train-split.md` after
      the verification read succeeds + laptop session's tokenize kicks off.

## Outcome (fill in when done)

- `tomat-rho-gga-train` created at `YYYY-MM-DD HH:MM TZ`.
- Train task IDs: ___; hard-link stage size ~___ GB; missing count ___.
- Upload wall time: ___.
- Verification: read `mp-____.zarr` shape ___, n_sites ___.
