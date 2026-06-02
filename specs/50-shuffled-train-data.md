# 50 — Globally pre-shuffled `train-full-v3-shuffled`

## Why

Spec 44's update (2026-06-02) traced the v4 MaskGIT TL sawtooth to
Levanter's `BlockShuffleConfig`: it only shuffles WITHIN sequence
windows, and adjacent windows are physically adjacent slices of the
parquet. Concretely, with `io_block_size=256, window_blocks=512` the
period is `256 × 512 / 128 = 1024` steps, and ACF at lag 1024 in
cont-2 is 0.81 (3.3× stronger than pre-cont-2). The model "learns"
each window's material distribution over 1024 steps, then jumps to a
fresh window → step-up at the boundary.

The clean fix is to globally pre-shuffle the source parquet once,
then any windowed shuffle the trainer applies on top is a no-op on an
already-uniform dataset. Costs one-shot CPU time + ~56 GiB disk
duplication; saves continuous training compute waste from window-
boundary forgetting.

## What

A new tokenized label `train-full-v3-shuffled` with the same shard
structure as `train-full-v3`:

- 256 worker dirs (`worker-00/` … `worker-255/`)
- 10 shards/worker (`shard-00000.parquet` … `shard-00009.parquet`)
- Same per-shard row count as the corresponding input shard
- Same parquet schema (`task_id`, `offset_x/y/z`, `input_ids`) and the
  same `pq` knobs (`zstd` compression, `row_group_size=64`).

The label is written to both regions, matching `train-full-v3`'s
layout:

- `gs://marin-eu-west4/tomat/tokenized/train-full-v3-shuffled/` (used
  by Modal training)
- `gs://marin-us-east5/tomat/tokenized/train-full-v3-shuffled/` (used
  by TPU runs in us-east1)

Plus three sidecar files (one per region):

- `_perm.npy` — numpy `int64[N]` saved permutation; `perm[i]` is the
  global input row index that ended up at global output row `i`.
- `_perm.json` — metadata: seed, total rows, sample-mode flag, etc.
- `worker-00/meta.json` — copy of `train-full-v3/worker-00/meta.json`
  (vocab layout, codec paths, tokenizer version); unchanged by the
  shuffle.

### Permutation seed

```
seed = 20260602
```

`numpy.random.default_rng(20260602).permutation(N)` → bit-identical
permutation across reproductions. `N = total_rows` is read from
`train-full-v3`'s parquet metadata at shuffle time. As of 2026-06-02
the count is 4,964,352 (256 workers × ~19392 rows; aligns with
`EPOCH_SEQUENCES['train-full-v3'] = 4954176` to within a small
last-worker variance — exact N is logged + stored in `_perm.json`).

## How to use

Set `TOMAT_LABEL=train-full-v3-shuffled` (or pass `--label
train-full-v3-shuffled`) to any consumer:

- `marin/train_tomat_tpu.py` (TPU trainer): `TOMAT_LABEL` env var
  picks up `parquet_glob = "<bucket>/tokenized/<label>/worker-*/*.parquet"`
  and `meta_url = "<bucket>/tokenized/<label>/worker-00/meta.json"`.
  Vocab layout comes from the copied `meta.json`.
- `scripts/train_smoke_modal.py` (Modal trainer): `--label` flag on
  the train entrypoints, or `TOMAT_LABEL` env var.

Once switched over, `BlockShuffleConfig` becomes effectively cosmetic
— any window size yields a uniform sample. Keeping
`BlockShuffleConfig(io_block_size=32, window_blocks=8192)` (or any
similar large window) costs nothing and preserves robustness against
the unlikely case that this shuffled label gets accidentally
overwritten with sorted data later.

## How

Single Modal CPU container (`scripts/shuffle_tokenized_modal.py`):

1. Discover every input shard + its row count → in-memory manifest.
2. Build deterministic permutation `perm` of length N (numpy RNG,
   seeded).
3. Parallel-download every input shard into RAM (zero-copy Arrow).
4. `pa.concat_tables` → single global Arrow `Table` view.
5. For each output shard `o`: `out_tbl = global_table.take(perm[o.start:o.end])`,
   write parquet bytes, upload to all configured buckets.
6. Sidecar JSON + permutation `.npy` + `meta.json` copy.

Output shard structure mirrors input 1-to-1 (same worker count, same
shards per worker, same row count per shard) so downstream consumers
that infer `n_shards` etc. from glob-counts don't surprise.

## Hardware + cost estimate

Single container, CPU-only (no GPU needed — this is bandwidth-bound).
Provisioned:

- 16 vCPU, 512 GiB RAM (Arrow concat of all ~5M rows ≈ 150 GiB int32
  list buffers + `take()` transient; 512 GiB leaves safe headroom)
- 512 GiB ephemeral disk (Modal floor; unused in the RAM-only path)
- 4h timeout

Cost estimate (Modal CPU pricing as of 2026):

- 16 vCPU × 0.5 h × ~$0.02/vCPU-h ≈ $0.16
- 512 GiB RAM × 0.5 h × ~$0.0036/GiB-h ≈ $0.92
- Egress to GCS: free between Modal/GCP same-region (~$0)
- **Modal compute total: ~$1**

Cross-region GCS traffic (Modal default region is us-east):

- Reads of `gs://marin-eu-west4` from us-east Modal container:
  inter-region egress ~$0.08/GB × 56 GiB ≈ **$4.50**.
- Writes to `gs://marin-us-east5` from us-east Modal: same-region or
  cheap ~$0.02/GB ≈ $1.
- Writes to `gs://marin-eu-west4` from us-east Modal: $0.08/GB × 56 GiB
  ≈ **$4.50**.

So end-to-end ~**$10-15** for the full 2-region run, dominated by GCP
inter-region egress. If we pin Modal to `region="europe-west4"`, the
read pass goes free and only the `gs://marin-us-east5` write incurs
inter-region egress (~$4.50 total). The egress estimate is the rough
upper bound — actual GCS pricing in 2026 may be lower; check before
firing.

Extrapolated wallclock from the smoke (52 input shards / 22s
download → ~18 min linear; 52 output shards / 36s write × 2 buckets →
~58 min linear). Actual will likely be less due to GCS parallel-read
saturation, but worst case is bounded by the 4h Modal timeout.

## Smoke-test result (2026-06-02)

Ran with `--sample-rows 100000 --eu-only` (single region, first 52
input shards covering 100,000 rows):

```
modal run scripts/shuffle_tokenized_modal.py::run \
    --label train-full-v3 --sample-rows 100000 --eu-only
```

Outcome (Modal app `ap-vn4Z6auFcGO5j9ypdVusI8`):

- Discovery (row-count metadata of all 2560 shards, parallel): **28.5s**
- Download 52 input shards (~3 GiB, 64 threads): **22.0s**
- Concat + permute + write 52 output shards (1.39 GB out, eu-west4): **35.9s**
- Total wallclock: **~90s** (excl. ~30s image build / cold start)
- Total rows confirmed: **4,954,176** (matches `EPOCH_SEQUENCES['train-full-v3']`)

Verifications:

- Output shard structure mirrors input 1-to-1: 52 output shards, 2048
  rows each in `worker-00/shard-0000{0..9}.parquet`.
- Output schema preserved: `task_id`, `offset_x/y/z`, `input_ids`
  (LIST<INT32>), same as input.
- Output is genuinely shuffled across materials: `worker-00/shard-00000.parquet`
  has 1149 unique `task_id`s across its 2048 rows (vs ~32 expected if
  unshuffled — each input shard holds ~32 mats with 64 patches each).
- `_perm.npy` (0.8 MB) and `_perm.json` (sidecar) successfully written.
- `worker-00/meta.json` copied from input (vocab layout preserved).

Smoke output location:
`gs://marin-eu-west4/tomat/tokenized/train-full-v3-shuffled-smoke/`

## To run the full shuffle

```bash
modal run scripts/shuffle_tokenized_modal.py::run --label train-full-v3
```

Without `--sample-rows`, the entrypoint emits to both regions and
covers all ~5M rows.

## Caveat: mapping output → original

If we ever need to map `(output_row_index → original_row_id)` (e.g.
to associate a wandb log line "batch starts at step 12345" with the
original input parquet row), download `_perm.npy`:

```python
import numpy as np
import fsspec
with fsspec.open(
    "gs://marin-eu-west4/tomat/tokenized/train-full-v3-shuffled/_perm.npy",
    "rb",
) as f:
    perm = np.load(f)
# perm[output_global_row_index] == original input global row index.
```

Original input row indexing is by `(worker_idx, shard_idx, row_in_shard)`
in sorted order; `_perm.json` records the exact shard ordering used.

## Linked work

- Spec 44 (sawtooth investigation) — root-cause analysis that
  motivates this.
- `site/src/runs/runMeta.ts` — `EPOCH_SEQUENCES` entry added for the
  new label so dashboards' `nEpochs` math handles
  `train-full-v3-shuffled`.
