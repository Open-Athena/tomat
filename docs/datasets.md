# tomat datasets

Raw DFT densities live as Zarr v3 on Princeton's della HPC. We stage subsets
onto Modal volumes (for immediate compute) and tokenize them into parquet
shards for Levanter. Eventual destination for training at scale is GCS.

## Raw rho_gga (della)

Native-resolution DFT electron densities for the Materials Project subset
that has GGA calculations. Base path:

```
/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/
  label/*.zarr                    # 86,192 .zarr dirs, one per mp-id
  mp_filelist.txt                 # canonical mp-id order (indices referenced by split files)
  split_limit_22M.json            # train / validation / test split (int indices into mp_filelist.txt)
  split_limit_40M.json            # larger-grid cap (for bigger accelerators; not in use yet)
```

Each Zarr: float32, zstd-compressed, shape varies from ~40³ to ~448³ voxels
(2 slab-like outliers seen at 48 × 48 × 1120-1372). Per-structure size
~5 MB average, ~22 GB aggregate for the val split, ~390 GB for train.

## Modal volumes

| volume | created | contents | size |
|---|---|---|---:|
| `tomat-rho-gga` | 2026-04-20 | val-split raw Zarrs (4,305 `mp-*.zarr`) + tokenized parquet dirs | ~22 GB raw + ~100 MB parquet |
| `tomat-rho-gga-train` | 2026-04-22 | train-split raw Zarrs (77,498 `mp-*.zarr`) + `tokenized/` output from spec 07 parallel tokenize | ~370 GB |

Spec references: [03-modal-seed](../specs/done/03-modal-seed.md) for the val
volume; [08-della-seed-train-split](../specs/08-della-seed-train-split.md)
for the train volume.

## Tokenized parquet sets (on `tomat-rho-gga`)

All: val split, codec `two_token_9_12`, patch_size P=14, pad_to 8192, seed 42.

Token counts are `rows × seq_len` at `pad_to=8192` (training counts padded
positions — actual non-pad tokens per row vary by material size).

| label | mats | patches/mat (M) | rows | tokens (pad) | on-disk (GCS) | notes |
|---|---:|---:|---:|---:|---:|---|
| `val-smoke-n2` | 2 | 32 | 64 | 524 K | — (local only) | throwaway |
| `val-smoke` | 128 | 32 | 4,096 | 34 M | ~33 MB | earliest smoke-training target (`rosy-durian-1`) |
| `val-full` | 4,305 | 32 | 137,696 | 1.13 B | 1.59 GB | primary val scale (2 oversized skipped → 137,664 effective) |
| `val-full-m128` | 4,305 | 128 | 550,784 | 4.51 B | 1.55 GB | 4× more unique patches/mat; first scale run 2026-04-23 (v6e-16 target) |

**Oversized materials skipped**: 2 materials (`mp-1884050`, `mp-1849033`) have
grid dims > 1024 on one axis (48 × 48 × 1120/1372 — slab-like structures).
The `INT_VOCAB_SIZE=1024` cap in [`src/tomat/tokenizers/patch.py`](../src/tomat/tokenizers/patch.py)
can't represent them; preprocessing logs them in each shard's `meta.json`.

## Tokenized parquet sets — train-split (on `tomat-rho-gga-train`)

della → `tomat-rho-gga-train` upload complete (77,498 structures).
Parallel tokenize kicked off 2026-04-23 (64 Modal workers):

```
TOMAT_VOLUME=tomat-rho-gga-train modal run \
    scripts/tokenize_patches_modal.py::parallel \
    --label train-full --split train --n-workers 64
```

| label | mats | patches/mat (M) | rows (est) | tokens (est) | on-disk (est) | notes |
|---|---:|---:|---:|---:|---:|---|
| `train-full` | ~77,498 | 32 | ~2.48 M | ~20 B | ~28 GB | first pass; per-material sizing matches `val-full` |

## W&B projects

Runs across different `(codec, patch_size)` combos land in separate projects
so loss curves are always apples-to-apples within a project.

| project | codec | P | notes |
|---|---|---:|---|
| [`tomat-two_token_9_12-P14`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14) | `two_token_9_12` | 14 | primary project (all current runs) |

Within a project, **groups** split runs by the training-side sampling axes
(`M<N>-N<N>`); **tags** carry filterable dimensions (`smoke`/`scale`,
`bs32`/`bs128`, `mats128`/`mats4305`, `seed42`, ...).

### Scale runs (2026-04-22 / 23, all complete)

Qwen3-30M (hidden=512, 6 layers, 4 heads, seq=8192), seed 42. Project:
[`tomat-two_token_9_12-P14`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14).

| run | data | compute | batch | per-dev | steps | MFU | tok/s | final loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| [`val-full-5k-bs32-bs32-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-bs32-seed42) | val-full | Modal A100:1 | 32 | 32 | 2,560 (OOM) | 12.4% | 80 k | 2.235 |
| [`val-full-5k-bs32-2gpu-bs32-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-2gpu-bs32-seed42) | val-full | Modal A100:2 | 32 | 16 | 5,000 | 12.0% | 157 k | **1.962** |
| [`val-full-5k-bs64-4gpu-bs64-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs64-4gpu-bs64-seed42) | val-full | Modal A100:4 | 64 | 16 | 5,000 | 11.96% | 313 k | 1.975 |
| [`val-full-5k-bs128-8gpu-bs128-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs128-8gpu-bs128-seed42) | val-full | Modal A100:8 | 128 | 16 | 5,000 | 11.86% | 624 k | 2.022 |
| [`val-full-tpu-bs128-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-tpu-bs128-seed42) | val-full | Marin TPU v6e-4 | 128 | 32 | 1,000 | 10.25% | 792 k | 2.620 |
| [`train-full-tpu8-bs256-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-bs256-seed42) | **train-full** | Marin TPU v6e-8 | 256 | 32 | 2,000 | 8.38% | **1,297 k** | **2.214** |

Headlines:
- **A100 scaling is linear**: 157 k → 313 k → 624 k tok/s at per-dev bs=16,
  MFU flat ~12% across 2/4/8 chips.
- **TPU v6e-4 ≈ 10× A100:1** tok/s at same per-device bs (matches the 12×
  hardware-FLOPs ratio minus a ~17% MFU gap).
- **train-full run**: 4.19 B tokens trained (18× more data than val-full),
  loss dropped 2.62 → 2.21 (0.41 nats). MFU is *lower* at 8.38% because
  30 M is too small to saturate v6e-8 — 4 B tokens through a 30 M model is
  ~7× past Chinchilla-optimal, so we're parameter-bound. Bigger model next.

## DVX provenance

Every parquet shard dir that we care about has a `.dvc` file recording the
full `modal run ...` cmd + md5s of all tokenizer source files it depended
on. [`data/tokenized/val-smoke.dvc`](../data/tokenized/val-smoke.dvc) is the
canonical example; new tokenize runs should follow the same shape.
