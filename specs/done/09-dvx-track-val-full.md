# DVX-track `val-full` parquet + raw zarrs (drift detection, not backup)

Follow-up to a silent GCS-upload corruption of `worker-01/shard-00002.parquet`
(1 file of 80 in `val-full`) that blew up a TPU training run on Marin with
`OSError: ZSTD decompression failed: Data corruption detected`.

All 80 parquets on the Modal volume (`tomat-rho-gga`) were fine — the
corruption happened silently in the `gcloud storage cp` upload from my laptop
to `gs://marin-eu-west4/tomat/tokenized/val-full/`. Only 1-in-80 failed the
post-upload md5 check; I only noticed because Levanter hit a bad row group
at shard idx 7.

This spec: make that class of failure detectable-by-default, across raw zarrs
(della) and tokenized parquets (Modal + GCS).

## Model: verify-not-restore

DVX adds two things: (a) md5 manifests checked into the repo, (b) an
`.dvc/cache/` of the tracked data. For our case **we only want (a)** —
every tracked artifact already lives in an authoritative place:

| artifact | canonical locations | size | drift risk |
|---|---|---:|---|
| raw `rho_gga` zarrs | della `/scratch/gpfs/ROSENGROUP/.../rho_gga/label/*.zarr` | ~22 GB val / ~390 GB train | filesystem rot, accidental chmod, upload truncation |
| `val-full` parquets | Modal volume `tomat-rho-gga:/tokenized/val-full/` + `gs://marin-eu-west4/tomat/tokenized/val-full/` | ~1.3 GB | silent cp corruption (demonstrated) |
| `val-full-m128` | same two locations | ~5 GB | same |
| `train-full` (planned, spec 07) | same two locations | ~50 GB | same |

So: **never `dvx push`**, and shape the caches so they cost ~0 storage.

- **Zarrs on della**: `dvx add` with `cache.type=hardlink` in `.dvc/config`.
  Same filesystem → hardlinks are free. The zarr files stay in place; DVX
  just records md5s in `.dvc` files committed to git.
- **Parquets (GCS-primary)**: build the `.dvc` manifest directly from GCS
  md5s — GCS stores MD5 as object metadata, no download needed. No
  `.dvc/cache/` entry at all (or an empty one).
- **Modal volume parquets**: covered transitively by the GCS `.dvc` —
  Modal → GCS upload is the drift-prone edge; the Modal volume itself is
  not something we currently suspect. If we want explicit coverage, the
  Modal-side scan script ([`scripts/verify_val_full_parquet.py`](../scripts/verify_val_full_parquet.py))
  can run under CI and compare Modal md5s to the `.dvc` manifest.

`.dvc` files are tiny (~150 B per tracked file) and committed to git — this
is the "drift detection" feature, not a second copy of the data.

## Plan

### 1. Della: DVX-track raw rho_gga zarrs (hardlink cache)

On della, in the tomat clone at `/scratch/gpfs/ROSENGROUP/rw5240/tomat`:

```bash
# Must be *same filesystem* as the zarrs for hardlink to work.
# /scratch/gpfs/ROSENGROUP is all one FS; verify with `stat -f` if unsure.
dvx init --no-scm    # if not yet
dvx config cache.type hardlink
dvx config cache.dir .dvx/cache    # local; never push

# Point DVX at the real zarr tree.
cd data/                           # or wherever feels right in the repo
ln -s /scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/ rho_gga
# (or use a real path under data/ — just don't copy. Link keeps it 1 FS.)

dvx add rho_gga/label             # 86,192 zarrs; hardlinks into .dvx/cache
```

Expected: `rho_gga/label.dvc` is ~a few KB (single `.dir` hash entry). No
extra storage beyond hardlinks. `dvx status` reports drift on any file.

Commit `rho_gga/label.dvc` (and `.gitignore` entries for the data itself).

### 2. Parquets: `.dvc` manifests from GCS md5s (no download)

`gcloud storage ls -L <glob>` returns `Hash (MD5): <base64>` for every
object. Pattern:

```python
# scripts/make_gcs_dvc.py
# Usage: python scripts/make_gcs_dvc.py val-full > data/tokenized/val-full.dvc
#
# - gcloud storage ls -L gs://.../val-full/worker-*/*.parquet
# - parse object paths + md5s
# - write a DVC-compatible .dir manifest with those md5s
# - include meta.computation (cmd + source md5s, mirroring val-smoke.dvc)
```

Shape matches [`data/tokenized/val-smoke.dvc`](../data/tokenized/val-smoke.dvc).
The `.dir` contents list 80 parquets × (relpath, md5); DVX treats that as
the canonical manifest. `dvx status` runs by re-listing GCS, comparing
md5s — zero-download.

Repeat for `val-full-m128`. Later, `train-full`.

### 3. Sync-to-GCS wrapper (spec 07 follow-up)

The corruption we hit was upload-side. `gcloud storage cp` already does
CRC32C on the wire; the failure mode is *after* the cp (client-side md5
drift against the local source). Make every parquet push go through:

```python
# scripts/sync_parquets_to_gcs.py
# 1. Compute md5 of every local parquet → manifest.json
# 2. gcloud storage cp --recursive local gs://...
# 3. Fetch GCS md5 per object, compare to manifest
# 4. Exit nonzero on any mismatch; no "mostly succeeded"
```

This runs at upload time. DVX gives us the same check at rest
(`dvx status`).

## Non-goals

- `.dvc/cache/` as a backup. The original files (on della, Modal, GCS)
  are authoritative. DVX here is a manifest.
- `dvx push` or any DVX remote. We'd just be mirroring GCS to another GCS.
- DVX-driven `dvx repro` of the tokenize pipeline. Worth doing later; not
  the point of this spec.

## Success criteria

- [ ] On della: `data/rho_gga/label.dvc` committed; `dvx status` clean;
      `.dvx/cache/` is all hardlinks (verify with `stat` — same inode as
      corresponding `label/*.zarr/**` files).
- [ ] `data/tokenized/val-full.dvc` committed, built from GCS md5s.
- [ ] `dvx status` or equivalent surfaces future silent drift on either.
- [ ] Same for `val-full-m128`.
- [ ] Follow-up for `train-full` once spec 07 completes tokenize.
