# Modal CPU cache builder for Levanter

## What

`scripts/build_cache_modal.py` is a CPU-only Modal function that converts
tokenized parquets (under `/vol/tokenized/<label>/`) into the Levanter
TreeCache (Zarr) format that the training data loader reads at every step.

The output is a single **tarball** on the Modal volume
(`/vol/cache_tarballs/<name>.tar`). At train time,
`train_smoke_modal.py::_train_bakeoff_impl` accepts
`cache_tarball_path=<relpath>` and extracts the tarball to `/tmp/cache_<run>`
on container-local SSD. Training reads the cache from local SSD — never from
the volume — so the training container is decoupled from the slow cache
build.

## Why this exists

Levanter's data loader doesn't read parquet directly at train time.
Internally, `LmDataConfig` requires a `TreeCache` (chunked-array Zarr
format) for efficient row-level random access — which is what BlockShuffle
needs. The conversion is a one-time prep step, but for a 3-shard config
(~15M rows) it produces **~54,000 small zarr files** (each source parquet →
1 cache-shard dir with ~7 files inside: `zarr.json`, data chunk, offsets,
ledger, `.success` marker).

### The two failure modes we hit

1. **Naïve approach: write zarr cache directly to Modal volume.**
   Per-file metadata operations are slow on Modal volume (each new file
   needs a network round-trip + commit). 54k small file writes throttled
   to ~0.15 files/sec — projected **~12 hours on H200×8 GPU container**,
   most of it idle waiting on volume IO. Cost ~$440 of GPU time mostly
   not used for compute.

2. **Read-from-volume + write-to-local-SSD.**
   Bypassed the destination-write problem but the source-read also has
   per-row metadata overhead. Initial 32-CPU CPU-only container did 11
   files in 10 min — same throughput as GPU.

### The actual fix

**Pre-stage source parquets to local SSD via bulk `cp -r`, then build cache
to local SSD.** Bulk-copy gives ~62 MB/s sustained from volume (16 min/shard
≈ acceptable). After staging, both reads and writes are local-SSD — no
network IO. Then tar the cache (one big seq write) and copy the tarball
back to volume.

After staging: still CPU-bound (per-row Python decode/encode). Sustained
~0.15 task/sec with 32 CPUs / 128 worker threads. Modal's per-function CPU
ceiling is **64 cores** (not 96 — Modal's actual error message:
`"Function CPU request out of bounds. Must be between 0.125 and 64 cores."`).
At 64 CPUs ETA roughly halves.

## Three-stage pipeline

```
1. Stage (network IO):
   /vol/tokenized/<label>/  → cp -r → /tmp/parquets/<label>/
   ~16 min per shard at ~62 MB/s

2. Build cache (CPU):
   /tmp/parquets/  → Levanter LmDataConfig.build_caches("train")
                   → /tmp/cache/{train,validation}/
   ~0.15 task/sec/32cpu = ~14 hr for 7680 tasks
   ~0.30 task/sec/64cpu = ~7 hr for 7680 tasks  (theoretical 2x)

3. Tar + upload (network IO):
   /tmp/cache/  → tar -cf  → /tmp/cache.tar
   /tmp/cache.tar  → cp  → /vol/<output_tarball>
   ~minutes for 150-200 GB at sequential volume-write speeds
```

## Sizing rules of thumb

| Shards | Source rows | Cache entries | Cache size on disk | ETA @ 64 CPU |
|---|---|---|---|---|
| 1 | ~5M | 2560 | ~50-70 GB | ~2.5 hr |
| 3 | ~15M | 7680 | ~150-200 GB | ~7 hr |
| 6 | ~30M | 15360 | ~300-400 GB | ~14 hr |

Disk allocation: Modal `ephemeral_disk` minimum is **512 GB**.
For 3+ shards budget 600 GB (cache + source parquets + tarball).

## CLI

```bash
TOMAT_VOLUME=tomat-rho-gga-train modal run --detach \
  scripts/build_cache_modal.py::run \
  --labels train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \
  --output-tarball cache_tarballs/v4-epochwin-ts123.tar
```

Then in the training spawn script:

```python
call = train_bakeoff_h200x8.spawn(
    ...
    cache_tarball_path="cache_tarballs/v4-epochwin-ts123.tar",
    ...
)
```

`_train_bakeoff_impl` will detect the param, copy the tarball to
`/tmp/cache_<results_label>.tar`, extract to `/tmp/cache_<results_label>/`,
and pass that path as `cache_dir` to Levanter. No code change at train
time besides the new optional kwarg.

## Future-work simplifications

### Push offsets into per-shard `build_cache` jobs ✓ (landed 2026-06-06)

`build_cache(source_idx, global_idx_offset, ...)` renames cache subdirs
right after Levanter writes them, so the per-shard tarball has
globally-correct names. `merge_caches` becomes a dumb concat: extract
each tarball into the same `/tmp/merged/train/` (no collisions because
the names already disambiguate by source), then write a stitched
top-level ledger. ~20 lines for merge instead of ~80.

`orchestrate` computes the offsets up-front by counting parquet files
per label (each source PQT shard contributes its file count to the
running `global_idx_offset` for the next shard's builder). It also
passes 1-indexed `source_idx` per shard.

### Make PQT shards == Zarr cache shards

Today's mismatch:
- Tokenizer produces N "physical" PQT shards (each ~5M rows, 256 workers).
- Cache builder runs M containers; was implicitly M = N but doesn't have to.

For 6-way cache parallelism with current 3 PQT shards, we'd need to
either (a) split each PQT shard into two glob-ranges at cache-build time,
or (b) physically split the PQT shards. Both add a layer.

Going forward, make the tokenize job take a target rows-per-shard or
explicit shard-count param, so PQT shard count == cache parallelism.
Then `build_cache --label shardN` is the natural unit. Smaller shards
= more parallelism options + smaller failure blast radius if one
container dies mid-build.

Apply both of these the next time tokenization changes (new patch_size,
codec, atom_encoding, etc.) — and at that point land the
`per-shard-checkpoint after each build's tarball lands on volume`
recoverability too, so kills don't lose all progress.

## Limits + future work

- **Cache build is per-image-config.** A different `marin/qwen3_density.py`
  or a different `OA_MARIN_SHA` may produce a subtly different cache layout.
  If a tarball was built under SHA A but consumed under SHA B, Levanter
  may rebuild from scratch (slow). Keep `build_cache_modal.py`'s
  `OA_MARIN_SHA` in lockstep with `train_smoke_modal.py`'s.

- **CPU ceiling is the bottleneck.** Per-row Python decode/encode is
  pure GIL-bound work. Real wins require either (a) multi-container
  parallelism (one builder per shard, then merge — `cache/train/`
  subdirs are independently sharded so this is mostly safe), or
  (b) rewriting the inner loop in C/Rust upstream in Levanter.

- **Tarball format.** Using plain `tar` (no compression) — the cache is
  mostly already-compressed zarr chunks (lz4 codec). Compression would
  burn CPU for ≤5% savings.

- **Cache reuse.** The same tarball serves multiple training runs that
  use the same source data. Cost is amortized.

## File-by-file

- `scripts/build_cache_modal.py` — Modal CPU function
- `scripts/train_smoke_modal.py::_train_bakeoff_impl` — accepts
  `cache_tarball_path` arg and handles extract-on-startup
- `tmp/spawn_v4_ts123_bs256.py` — example spawn that uses the prebuilt
  cache via `cache_tarball_path="cache_tarballs/v4-epochwin-ts123.tar"`
