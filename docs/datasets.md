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

## Tokenized parquet sets

All: codec `two_token_9_12`, patch_size P=14, pad_to 8192, seed 42.

### Example training sequence

A real row from `train-full` (Y₃Si₃Ag₃, grid 64 × 108 × 108, P=14 patch at offset (5, 9, 44)):

```
[BOS]
[GRID_START]  64 108 108                      [GRID_END]
[ATOMS_START] Y Y Y Si Si Si Ag Ag Ag         [ATOMS_END]
[POS_START]   (p236 p699 p1003 p240 p767 p1005 p0 p512 p768)  …  (+7 more atoms)  [POS_END]
[SHAPE_START] 14 14 14                        [SHAPE_END]
[OFFSET_START] 5 9 44                         [OFFSET_END]
[HI_START]    18 22 57                        [HI_END]
[DENS_START]  d172 d909 d169 d4175 …  d158 d2204    # 5,488 density tokens = 2 × 14³
[DENS_END]
[EOS]
[PAD] × 2,586                                 # right-padded to 8,192
```

- Atom Zs render as element symbols (`Y`, `Si`, `Ag`).
- Position tokens are a 3-byte fixed-point codec (512 + 256 + 256 vocabs); one
  coord → 3 tokens; one atom (3 coords) → 9 tokens → `(p… p… p…   p… p… p…   p… p… p…)`.
- Density tokens are a 2-token 9/12-bit codec → `2 × P³` per patch (5,488 at P=14).
- Preamble (BOS + grid + atoms + positions + shape + offset + hi + DENS_START)
  is ~30 + 10·n_atoms tokens; padded tail absorbs the rest. For a 100-atom
  structure the preamble is ~1,030 tokens, leaving ~1,670 of pad.

The helper script [`scripts/show_tokens.py`](../scripts/show_tokens.py)
renders any parquet row into this layout.

### A note on "tokens (pad)"

Each parquet row is one training sequence, right-padded to `pad_to=8192`
tokens so the model's fixed-length context is always full. The **tokens
(pad)** column is `rows × pad_to` — the number of token positions the
model trains *on*, including the padded positions. This is the right
number for compute accounting (FLOPs/token × tokens trained), Chinchilla
ratios, and comparing runs.

The actual *non-pad* content per row is smaller and varies by material:

- **Density block**: `2 × P³` tokens (density codec emits 2 tokens/voxel).
  At P=14, that's `2 × 14³ = 5,488` tokens.
- **Preamble**: grid shape + atomic inventory (Z + fractional coords) +
  patch anchor. Scales with atom count; typically ~200 tokens for a
  100-atom structure, less for smaller.

So for a 100-atom material at P=14: ~5,688 non-pad tokens + ~2,504 pad
= 8,192. Padded positions contribute ~0 to train loss (they're masked in
the cross-entropy) but the model still runs forward/backward through them.

| label | split | mats | patches/mat (M) | rows | tokens (pad) | on-disk (GCS) | notes |
|---|---|---:|---:|---:|---:|---:|---|
| `val-smoke-n2` | val | 2 | 32 | 64 | 524 K | — (local only) | throwaway |
| `val-smoke` | val | 128 | 32 | 4,096 | 34 M | ~33 MB | earliest smoke target (`rosy-durian-1`) |
| `val-full` | val | 4,305 | 32 | 137,696 | 1.13 B | 1.49 GB | primary val scale (2 oversized skipped → 137,664 effective) |
| `val-full-m128` | val | 4,305 | 128 | 549,664 | 4.50 B | 1.44 GB | 4× more unique patches/mat |
| `train-full` | train | 77,498 | 32 | 2,478,912 | 20.31 B | 21.1 GB | first run on this 2026-04-23 |

Val-split labels live on Modal volume `tomat-rho-gga`; train-split lives on
`tomat-rho-gga-train`. Production tokenize command (env-var-swappable volume):

```
TOMAT_VOLUME=tomat-rho-gga-train modal run \
    scripts/tokenize_patches_modal.py::parallel \
    --label train-full --split train --n-workers 64
```

**Oversized materials skipped**: 2 materials (`mp-1884050`, `mp-1849033`) have
grid dims > 1024 on one axis (48 × 48 × 1120/1372 — slab-like structures).
The `INT_VOCAB_SIZE=1024` cap in [`src/tomat/tokenizers/patch.py`](../src/tomat/tokenizers/patch.py)
can't represent them; preprocessing logs them in each shard's `meta.json`.

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

| run | model | data | compute | batch | per-dev | steps | tokens | total FLOPs | MFU | tok/s | final loss |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`val-full-5k-bs32-bs32-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-bs32-seed42) | 30M | val-full | Modal A100:1 | 32 | 32 | 2,560 (OOM) | 0.67 B | 0.32 × 10¹⁸ | 12.4% | 80 k | 2.235 |
| [`val-full-5k-bs32-2gpu-bs32-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-2gpu-bs32-seed42) | 30M | val-full | Modal A100:2 | 32 | 16 | 5,000 | 1.31 B | 0.62 × 10¹⁸ | 12.0% | 157 k | **1.962** |
| [`val-full-5k-bs64-4gpu-bs64-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs64-4gpu-bs64-seed42) | 30M | val-full | Modal A100:4 | 64 | 16 | 5,000 | 2.62 B | 1.25 × 10¹⁸ | 11.96% | 313 k | 1.975 |
| [`val-full-5k-bs128-8gpu-bs128-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs128-8gpu-bs128-seed42) | 30M | val-full | Modal A100:8 | 128 | 16 | 5,000 | 5.24 B | 2.49 × 10¹⁸ | 11.86% | 624 k | 2.022 |
| [`val-full-tpu-bs128-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-tpu-bs128-seed42) | 30M | val-full | Marin TPU v6e-4 | 128 | 32 | 1,000 | 1.05 B | 0.50 × 10¹⁸ | 10.25% | 792 k | 2.620 |
| [`train-full-tpu8-bs256-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-bs256-seed42) | 30M | **train-full** | Marin TPU v6e-8 | 256 | 32 | 2,000 | **4.19 B** | **2.00 × 10¹⁸** | 8.38% | 1,297 k | **2.214** |
| [`train-full-tpu16-30M-bs512-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu16-30M-bs512-seed42) | 30M | train-full | Marin TPU v6e-16 (multihost, 4 VMs) | 512 | 32 | in flight | — | — | — | **2,042 k** | — |
| [`train-full-tpu8-200M-bs128-val-bf16-seed42`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-200M-bs128-val-bf16-seed42) | **208M** | train-full | Marin TPU v6e-8 + bf16 | 128 | 16 | in flight | — | — | — | 294 k | — |

Headlines:
- **A100 scaling is linear**: 157 k → 313 k → 624 k tok/s at per-dev bs=16,
  MFU flat ~12% across 2/4/8 chips.
- **TPU v6e-4 ≈ 10× A100:1** tok/s at same per-device bs (matches the 12×
  hardware-FLOPs ratio minus a ~17% MFU gap).
- **train-full**: 4.19 B tokens through the 30 M model, loss 2.62 → 2.21
  (0.41 nats on 18× more data). MFU drops to 8.4% on v6e-8 because 30 M
  is too small to saturate the chip — parameter-bound, not data-bound.
- **Multihost TPU (v6e-16) works**: 4 VMs × 4 chips, 2.04 M tok/s
  (1.57× v6e-8, ~78% scaling efficiency). Unblocked by adding
  `jax.distributed.initialize()` at script entry — Levanter's `WandbConfig.init`
  tries a multihost broadcast before the trainer's own distributed setup fires.
- **200 M model** (hidden=1024, 12 layers, 16 heads) running on v6e-8
  with bf16 compute and a 256-seq held-out validation split — first real
  generalization number coming.

## DVX provenance

Every parquet shard dir that we care about has a `.dvc` file recording the
full `modal run ...` cmd + md5s of all tokenizer source files it depended
on. [`data/tokenized/val-smoke.dvc`](../data/tokenized/val-smoke.dvc) is the
canonical example; new tokenize runs should follow the same shape.
