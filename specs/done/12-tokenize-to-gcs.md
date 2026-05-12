# Tokenize-to-GCS: skip the Modal-volume parquet step

Status: **proposed**. Written 2026-04-23.

## Motivation

Current tokenize flow (spec 05, 07):

```
raw Zarrs (Modal volume)
  └──▶ parallel tokenize (64 Modal CPUs) ──▶ parquets on Modal volume
          └──▶ sync_parquets_to_gcs.py (md5 verify) ──▶ GCS
```

Two issues:
1. Two-step write (volume → GCS) is where we hit a silent bitflip
   during `modal volume get` on train-full — took md5 verification to
   catch.
2. Tokenize variants coming up (P=15, balls, M=256 bumps, different
   codecs) each require a full tokenize pass. Each variant is ~60 min
   at 64 workers on the current infra, end-to-end.

## Proposed flow

```
raw Zarrs (Modal volume, unchanged)
  └──▶ tokenize (1000 Modal CPUs) ──▶ parquets direct to GCS
```

Eliminate the local-to-volume parquet write + the verify/sync step.

- Each Modal function reads one material's Zarr from the mounted
  volume, tokenizes it, and writes the resulting parquet slice directly
  to `gs://marin-eu-west4/tomat/tokenized/<label>/` using the new
  `tomat-gcp-sa` Modal secret.
- Fan out to ~1000 workers (one function per ~80 materials). Modal
  supports this via `@app.function(concurrency_limit=1000)`.
- Per-variant turnaround: laptop → kick off Modal run → ~10 min →
  parquets ready on GCS.

## Implementation

### 1. GCS-write helper

Module-level util: `tomat.data.gcs_parquet` with `write_parquet_gcs(path,
table, schema)` — same signature as a local `pq.ParquetWriter` but
writes to `gs://...`. gcsfs + pyarrow handles this cleanly; no
buffering needed since each function writes one small parquet.

### 2. Modal function

```python
@app.function(
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name("tomat-gcp-sa")],
    timeout=600,
    concurrency_limit=1000,
)
def tokenize_mat(task_id: str, args: TokenizeArgs) -> TokenizeResult:
    # Read: zarr_path = f"{MOUNT}/label/{task_id}.zarr"
    # Tokenize: same pathway as current tokenize_patches.py
    # Write: parquet bytes → gs://marin-eu-west4/tomat/tokenized/<label>/shard-<task_id>.parquet
    ...
```

One parquet file per material (not per worker stripe). At 77k materials,
that's 77k files — but each is tiny (~100–400 KB). GCS object-count
budget: fine. Simpler than the current per-worker shard layout; downstream
Levanter glob `worker-*/*.parquet` widens to just `*.parquet`.

If per-material parquets feels too granular, group N=100 mats per call
via `chunk_size=100` in `.map()` — each call still writes 1 parquet, but
contains ~N materials, giving 770 shards total.

### 3. Driver

```python
@app.local_entrypoint()
def run(label: str, shape: str = "cube", patch_size: int = 14,
        r2_max: int = 75, patches_per_material: int = 32, ...):
    task_ids = resolve_task_ids(split, split_file)      # as before
    for rows_per_task in tokenize_mat.map(task_ids, ...):
        ...  # accumulate stats, emit aggregated meta.json to GCS at the end
    write_meta_to_gcs(label, total_rows, shape, ...)
```

### 4. meta.json

Write once at the end of the driver (on laptop or via a final Modal
function). Same schema as today but `n_shards` is now
`len(task_ids)` (or `ceil(len/chunk_size)`).

### 5. Remove sync step

`sync_parquets_to_gcs.py` becomes obsolete for new tokenize labels.
Keep it around for any legacy val-full / train-full parquets we want
to re-verify, but it's no longer part of the main pipeline.

## Cost

Modal CPU: ~1000 × 1 core × 10 min × $0.04/core-hr ≈ **$7/variant**.

GCS writes: negligible within same region (us-east or eu-west, same
as the Marin TPU pods that consume the data).

## Migration path

1. Keep existing `tokenize_patches_modal.py` entry points for
   backward-compatibility.
2. Add new `tokenize_patches_gcs.py` (or `tokenize_gcs` entry in the
   same module) implementing the per-material flow above.
3. Use the new path for all future tokenize variants.
4. Only convert val-full / train-full to the new layout if we
   retokenize them (probably for codec / P changes — natural trigger).

## Open questions

- **Per-material parquets vs chunked**: per-material is simpler but
  creates 77k tiny objects on GCS. Levanter's parquet loader is fine
  with this (tested), but it may be slower to list at the start of a
  training run. Chunked (77k/100 ≈ 770 parquets) matches current
  worker-64 layout scale.
- **Concurrency limit vs Modal's cluster-fairness**: 1000-wide may get
  throttled during busy cluster periods. Fallback: spin up over 10 min
  instead of all-at-once.
- **GCS write retries**: Modal function retries are free; re-running a
  specific `tokenize_mat(task_id)` call is idempotent (same seed →
  same output). Make sure writes use `generation=0` (atomic) or check
  existence first.

## Adjacent specs

- `specs/done/08-della-seed-train-split.md` — the della-side upload.
- `specs/done/07-full-train-scale.md` — current parallel-tokenize flow.
- `specs/09-dvx-track-val-full.md` — manifest/checksum tracking, relates
  to the end-of-pipeline verification that we're removing here.
- `specs/10-ball-patches.md` — ball variant that'll be the first
  customer of this new flow.
- `specs/11-per-mat-validation.md` — per-material NMAE infra; natural
  next user of per-material parquet sharding.
