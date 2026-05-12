# Scale to full rho_gga train split (~86 k materials)

Follow-up to specs 03 (val-only Modal volume), 04 (patch scheme), 05
(Modal smoke). Goal: get tomat's data pipeline sized for Hananeh's
~86 k-material training regime and keep loss curves apples-to-apples
with electrAI's ResUNet results.

## Motivation

- **Scale gap**: the current `tomat-rho-gga` volume has the **val split
  only** (4,305 materials, ~22 GB). Hananeh's April 2026 ResUNet runs
  train on **86,106 GGA samples** (Meeting 26) — 20× more data. At our
  current 4.3 k-mat val the model may saturate on tokenizer-noise
  before the data exhausts its capacity; impossible to tell until we
  scale the data pipeline.
- **Parallel tokenize**: serial tokenize is ~1 mat/sec on a single
  Modal worker (bottleneck = Zarr-read-and-decompress per material).
  At 86 k mats that's ~24 hours in one process. Shardable across N
  Modal function invocations trivially — each processes `task_ids[i::N]`.
- **Provenance**: the val-full `.dvc` treats `/label/` on the Modal
  volume as a named external dep. That's fine for a small shared
  workspace, but scaling the volume means more opportunities for silent
  state drift — a content hash of the input Zarrs is cheap insurance.
- **Non-patch baseline**: Hananeh's first patchified run was 1.5× worse
  NMAE than non-patchified (Meeting 25). She's a 3D ResUNet, not an
  LLM over tokens, so the failure mode may not transfer — but it's
  worth running our own non-patch baseline (downsampled-voxel direct
  tokenization, the old spec 02 approach) alongside the patch track.

## Scope

In scope:

1. **Seed the train split** onto a new Modal volume (or extend
   `tomat-rho-gga`).
2. **Parallel tokenize** across N Modal workers.
3. **Hash input Zarrs** into a committed manifest file; `.dvc` for the
   parquet references the manifest file's md5 as its source-data dep.
4. **Storage cost accounting** — rough budget for the train-scale
   volume(s) + parquet outputs.
5. **Non-patch ablation track** — reuse the downsampled/direct
   tokenizers from spec 02 at a matched material-count to separate
   "tokenization strategy" from "model capacity."

Out of scope:

- Marin + GCP TPU cluster setup (see spec 05's Follow-up section).
  Training at 86 k × 32 patches × many epochs will want TPU or H100 —
  that's a parallel track, landed in its own spec.
- Converting additional MP structures to Zarr. rho_gga already covers
  ~86 k of MP's ~155 k; the gap is mostly non-GGA tasks, deferred.
- A third split (`split_limit_40M.json`) — the current `22M.json` cap
  matches A100/H100 GPU memory at bf16 per Hananeh's testing. Revisit
  only if a 40 M target becomes operational.

## Plan

### 1. Seed the train split

On della (where the raw rho_gga lives):

```bash
RHO_GGA=/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga
STAGE=/scratch/gpfs/ROSENGROUP/${USER}/tomat-rho-gga-train-stage

mkdir -p "$STAGE/label"
python3 -c "
import json, pathlib
split = json.load(open('$RHO_GGA/split_limit_22M.json'))
with open('$RHO_GGA/mp_filelist.txt') as f: filelist = [l.strip() for l in f]
train_ids = [filelist[i] for i in split['train']]
pathlib.Path('$STAGE/train_ids.txt').write_text('\n'.join(train_ids) + '\n')
print(f'train task IDs: {len(train_ids):,}')
"
# ~77,500 IDs expected (val=4,305 + test=4,305 → train ≈ 86k - 8.6k)

# Hard-link the train Zarrs into the staging area — same pattern as
# spec 03's val stage.
cd "$STAGE/label"
while read id; do cp -al "$RHO_GGA/label/$id.zarr" .; done < "$STAGE/train_ids.txt"
du -sh "$STAGE"    # ~370 GB expected
```

Decision point: new volume vs extend existing?

- **New volume `tomat-rho-gga-train`**: cleaner separation, independent
  storage cost, easy to drop if we change our mind. Recommended.
- **Extend `tomat-rho-gga`**: one volume, but `/label/` would hold ~82 k
  Zarrs mixing train + val, which complicates downstream sharding and
  storage-cost accounting.

Going with a **new volume** (`tomat-rho-gga-train`), V2 format per
spec 03's follow-up note — V2 supports recursive server-side copy so
the `/label/label/` rescue is a `cp -r` instead of a Modal function.

```bash
modal volume create --version 2 tomat-rho-gga-train
modal volume put tomat-rho-gga-train "$STAGE/label" /label
modal volume put tomat-rho-gga-train "$STAGE/train_ids.txt" train_ids.txt
modal volume put tomat-rho-gga-train "$RHO_GGA/split_limit_22M.json" split_limit_22M.json
modal volume put tomat-rho-gga-train "$RHO_GGA/mp_filelist.txt" mp_filelist.txt
```

Upload time at della's network: ~hours for 370 GB. Run overnight.

### 2. Parallel tokenize

Shard the tokenize work across `N` Modal workers. Each worker processes
`task_ids[i::N]`, writes parquet shards to its own output subpath
(`/tokenized/<label>/worker-<i>/shard-*.parquet`), and commits the
volume. After all workers finish, a merge step concatenates the
per-worker shards into the canonical `/tokenized/<label>/` (flat
`shard-NNNNN.parquet` layout).

New script: `scripts/tokenize_patches_modal_parallel.py`. Invocation:

```bash
modal run scripts/tokenize_patches_modal_parallel.py \
    --label train-full \
    --split train \
    --n-workers 16 \
    --pad-to 8192
```

Sketch:

```python
@app.function(volumes={MOUNT: volume}, timeout=7200)
def tokenize_shard(worker_idx: int, n_workers: int, label: str, ...):
    # task_ids for this worker only
    all_ids = resolve_split_ids(...)
    my_ids = all_ids[worker_idx::n_workers]
    run_tokenize_patches(task_ids=my_ids,
                         output_dir=f"{MOUNT}/tokenized/{label}/worker-{worker_idx}",
                         ...)
    volume.commit()

@app.local_entrypoint()
def main(n_workers: int = 16, ...):
    tokenize_shard.map(range(n_workers), ...)
    merge_shards.remote(label=label)  # renames worker-*/shard-*.parquet → flat shard-NNNNN.parquet
```

Modal's `.map()` or `.starmap()` dispatches the shard calls in
parallel. At 16× parallelism and 1 mat/sec per worker, 77 k mats →
~80 min wall. Total parquet ~20 GB for (codec=two_token_9_12, P=14,
M=32).

**Merge step** (a second Modal function, CPU-only, seconds): walks
each worker's dir, reads shard metadata, rewrites into the flat layout
with contiguous shard indexes. Writes a combined `meta.json` that
sums row counts + merges `missing_task_ids` lists. Original per-worker
subdirs deleted after successful merge.

### 3. Hash input Zarrs into a manifest

Currently `.dvc`'s `meta.computation.deps` references only code-file
md5s; the input rho_gga is a named-by-path external dep. Scaling to
86 k Zarrs means more opportunities for silent state drift — a
content-manifest hash turns that into a verifiable dep.

New Modal function: `scripts/hash_rho_gga_manifest.py`. Walks the
mounted volume's `/label/*.zarr`, reads `zarr.json` + every chunk-
data file per Zarr, md5s the concatenation in a deterministic order,
writes a JSONL manifest to the volume + pulls it locally:

```
task_id,zarr_content_md5,n_chunks,bytes
mp-1234,5e4f...,8,4_823_104
mp-5678,a1b2...,16,9_645_312
...
```

The manifest file is small (~86 k lines × ~80 chars ≈ 7 MB) — commits
cleanly. The `.dvc` for the parquet output lists the manifest as a
dep with its own md5:

```yaml
meta:
  computation:
    cmd: modal run scripts/tokenize_patches_modal_parallel.py --label train-full --split train --n-workers 16 --pad-to 8192
    deps:
      scripts/tokenize_patches_modal_parallel.py: <md5>
      scripts/tokenize_patches.py: <md5>
      src/tomat/tokenizers/patch.py: <md5>
      src/tomat/data/zarr_io.py: <md5>
      src/tomat/float_codec.py: <md5>
      data/manifests/rho-gga-train.jsonl: <md5>  # ← the new input-data hash
```

If someone re-uploads the volume with different Zarrs, the manifest's
md5 changes → `dvx status` flags the parquet as stale. The manifest
itself invalidates on any rho_gga content change but not on
mtime-only touches.

Cheap side benefit: the manifest is a diff-able artifact, so "what
changed between the Apr 2026 volume and the Nov 2026 volume" becomes a
`comm`-style question.

### 4. Storage cost

Back-of-envelope at current Modal pricing ($0.50 / GB / month):

| artifact | size | $/mo |
|---|---:|---:|
| `tomat-rho-gga` (val, existing) | 22 GB | $11 |
| `tomat-rho-gga-train` (train, new) | 370 GB | $185 |
| parquet `/tokenized/val-full/` (current) | 1 GB | $0.50 |
| parquet `/tokenized/train-full/` (new) | 20 GB | $10 |
| parquet per extra (codec, P) combo at train-scale | 20 GB | $10 |

Full sweep (6 combos) at train-scale: ~$60/mo extra. Drop when done
iterating and re-tokenize from the Zarrs if needed later.

### 5. Non-patch ablation track

From Hananeh's Meeting 25: patchified training was **1.5× worse** than
non-patchified on the ResUNet. Our architecture is different
(sequence-model vs convolutional), but the same hypothesis warrants a
direct test: **is the patch scheme throwing away signal our full-grid
scheme wouldn't?**

The spec 02 (`done`) tokenizers — `DirectCodedTokenizer`,
`FourierCodedTokenizer`, `DeltaDensityTokenizer` — are all still wired
up. A non-patch run looks like:

- Downsample `(nx, ny, nz)` → `(32, 32, 32)` via avg-pool
  (`tomat.tokenizers.downsampled`).
- Tokenize the full downsampled grid into one sequence of ~32³ × 2 =
  65 k tokens; still fits a 64 k context transformer or a 128 k-context
  one.
- Train at the same data scale and compare NMAE to the patch track.

Cost: similar tokenize budget (~1 mat/sec on the same volume). Budget
~$10/mo parquet + training compute comparable to patch at matched
step count.

Not a blocker for train-scale; run in parallel once the patch pipeline
is training on train-full.

## Open questions

- **Parallel tokenize and `random_offsets`**: the current
  `PatchTokenizer.random_offsets` uses a single RNG. Sharded workers
  each seeding with `rng(seed + worker_idx)` keeps per-material offset
  reproducibility and avoids cross-worker correlation. Requires a
  small change to `scripts/tokenize_patches.py` that's benign for the
  single-worker case.
- **Volume format V1 vs V2**: V2 is newer and supports server-side
  recursive copy (pain avoided from spec 03's rescue). Default to V2
  for all new volumes. V1's `tomat-rho-gga` stays as-is.
- **Deleting train_ids.txt's hard-links**: after the modal volume put,
  removing the staging area drops the hard links but not the original
  data. Safe.
- **Incremental tokenize**: if we add new Zarrs later (unlikely for
  rho_gga but possible), parallel tokenize should detect existing
  per-worker output dirs and skip already-processed task IDs. MVP
  overwrites; add idempotency if needed.

## Done criteria

- [ ] `tomat-rho-gga-train` Modal volume exists with ~77 k train-split
      `mp-*.zarr` dirs under `/label/`.
- [ ] `scripts/tokenize_patches_modal_parallel.py` (new) dispatches
      N workers + a merge function; default `N=16`.
- [ ] `data/manifests/rho-gga-train.jsonl` generated + committed;
      `scripts/hash_rho_gga_manifest.py` (new) produces it reproducibly.
- [ ] `data/tokenized/train-full.dvc` exists with the manifest as a
      dep; `dvx status` green.
- [ ] At least one training run against `train-full` launched on Modal
      (duration and step count per operator choice — ultimate success
      is "scales to TPU+Marin via spec 05's follow-up").
- [ ] Non-patch baseline track: one `DownsampledDirectCoded` run at
      matched n_materials, comparable NMAE metric logged to W&B under
      a separate project (`tomat-direct-32`). Data point to compare
      against Hananeh's patchified-vs-non result.
- [ ] Move spec to `specs/done/07-full-train-scale.md` when all of the
      above are checked.
