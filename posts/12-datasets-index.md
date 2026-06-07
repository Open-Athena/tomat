# Datasets: an index

**Status**: living index — update as new shards / variants are produced.

This page documents the various tokenized parquet datasets we feed
training and eval jobs from. The source of truth for every number below
is the per-worker `meta.json` written at tokenize time
(`gs://marin-eu-west4/tomat/tokenized/<label>/worker-NN/meta.json`).

## At a glance

Current "v3" tokenization family — used by all live training runs as of
2026-06-06:

| Label                  | Role                | Materials | Sequences (rows) | On-disk (GCS) | Seed | Modal volume          | GCS bucket                                  |
| ---------------------- | ------------------- | --------- | ---------------- | ------------- | ---- | --------------------- | ------------------------------------------- |
| `train-full-v3`        | training (TS0)      | 77,568    | 4,964,352        | 55.83 GiB     | 42   | `tomat-rho-gga-train` | `gs://marin-eu-west4/tomat/tokenized/…`     |
| `train-full-v3-shard1` | training (TS1)      | 77,568    | 4,964,352        | 55.83 GiB     | 43   | `tomat-rho-gga-train` | mirrored                                    |
| `train-full-v3-shard2` | training (TS2)      | 77,568    | 4,964,352        | 55.83 GiB     | 44   | `tomat-rho-gga-train` | mirrored                                    |
| `train-full-v3-shard3` | training (TS3)      | 77,568    | 4,964,352        | 55.82 GiB     | 45   | `tomat-rho-gga-train` | mirrored                                    |
| `val-full-v3`          | held-out validation | ~4,300    | ~278,500         | 3.09 GiB      | 42   | `tomat-rho-gga`       | mirrored                                    |

Totals if we run on all 4 training shards: **310,272 materials,
19,857,408 sequences, ~223 GiB compressed parquet**. Held-out validation
is **~5.6%** the size of one training shard (~278k rows in 64 workers,
not 256 — val was tokenized with fewer worker shards from the start).

## Format

Every shard above uses the same v3 tokenization config:

- `tokenizer_version: v3` (per-patch translation, 19³ voxel patches,
  no SHAPE/OFFSET/HI special tokens — see post 02 for the v2→v3 redesign)
- `patch_size: 19` (P=19, so each patch is 19³ = 6,859 voxels)
- `patches_per_material: 64` (M=64 patches sampled per material)
- `density_codec_name: lmq` (LMQ v2, 16k-bin codec; see post 01)
- `atom_encoding: f0` (the new shards record this explicitly;
  `train-full-v3` was tokenized before the key was added but used the
  same F0 path — see spec 34)
- `pad_to: 8192` (each sequence is padded to 8192 tokens)
- Layout: 256 worker subdirs (`worker-NN/`), each with ~10 parquet
  shards of 2,048 rows. Row groups inside the parquet are
  **mat-aligned at 64 rows/row-group** (= M) so a row-group read =
  one material's worth of consecutive patches.

Tokens-per-sequence varies (it depends on per-material preamble length).
Padded length is 8,192, so worst-case the 4-shard dataset is
**~163 B padded tokens**. Real (non-pad) token count is lower; left as
a TODO to compute precisely.

## What lives where

### Modal volumes

- **`tomat-rho-gga-train`** — train splits (`tokenized/train-full-v3*`,
  `tokenized/v3-p1*-pack/…`, codecs, results dirs for Modal-trained
  runs). Read by every Modal training fire.
- **`tomat-rho-gga`** — val/test splits + smaller derivatives
  (`tokenized/val-full-v3/`, `tokenized/val-smoke/`, MPDB copies).
  Mounted by eval scripts.
- **Levanter cache tarballs** — `tomat-rho-gga-train:cache_tarballs/`.
  Produced by `scripts/build_cache_modal.py` (see
  `specs/31-cache-build-modal-cpu.md`). Each tarball is the Zarr-format
  cache for one source label, ready for `_train_bakeoff_impl` to
  extract to container-local SSD at startup.

### GCS

All parquet datasets above are mirrored to `gs://marin-eu-west4/tomat/`
for TPU consumption (Modal reads from volume; iris/TPU jobs read from
GCS). The cache tarballs are NOT mirrored to GCS — only the source
parquets are.

## How the new shards (TS1–TS3) differ from TS0

- **Same tokenization config**, same Python codepath
  (`scripts/tokenize_patches_modal.py`).
- **Different random seeds** (42 vs 43/44/45) → entirely different
  per-material patch coordinates. The 64 patches per material are
  sampled independently per shard, so shard1's 4.96M rows are
  effectively disjoint from TS0's 4.96M rows even though they cover
  the same 77,568 materials.
- **Why disjoint patches matters**: training on TS0 to ~76k steps
  caused per-material memorization (train loss ≪ held-out val loss
  plateau). Switching to TS1 gives the model "fresh" patches from
  the same materials, which tests whether the plateau is data
  starvation (more patches → keeps dropping) or model capacity
  (same plateau on any new patches).

## Split semantics

There are three material-level splits established at tokenize time
(see `~/.claude/projects/-Users-ryan-c-oa-tomat/memory/datasets-val-test-train.md`):

- **train** — ~77.6k materials (this is what `train-full-v3` and
  shards 1/2/3 sample patches from)
- **val** — ~17.4k materials, **never seen by training**
  (`val-full-v3`)
- **test** — exists in the original MP split file but **has never
  been touched** in tomat's history; reserved for the final
  cross-paper / cross-codec comparison. Don't fire evals on it
  without explicit ask.

The val_200 / train_200 sets used for mat-NMAE evals are 200-mat
samples from `val-full-v3` and `train-full-v3` respectively.

## Older / archived tokenizations (FFR)

Still on the volume but not used by current runs:

- `train-full-lmq-v2` (~) — v2 lattice-aware (P=14 / M=32). All
  pre-2026-05-11 results are from this. Don't re-fire from this label;
  use v3 going forward.
- `train-full-v3-p15` — v3 tokenized with P=15 / ctx=4608, used for
  the P=15 ablation arm; no live runs.
- `v3-p10-pack`, `v3-p14-pack`, `v3-p19-f1-pack` — packed (sequence
  packing scheme 2a, see spec 36); not yet used in training.
- `train-full-lmq-v2-lat`, `train-full-m256`, etc. — legacy. See
  `gs://marin-eu-west4/tomat/tokenized/` directory listing for the
  complete set.

## Source

Raw DFT charge densities come from Materials Project — see post 06
for the physical-scale notes (uniform ~0.065 Å/voxel grid, adaptive
ENCUT) and post 01 for the codec design.
