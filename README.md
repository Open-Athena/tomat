# tomat 🍅

**to**kenized **mat**erials — an LLM/transformer approach to predicting
DFT-converged electron density for periodic crystals. Sibling to
[tomol] (tokenized molecules). Positioned against [electrAI]/RHOAR-Net,
the 3D ResUNet over voxel grids.

**Interactive dashboard**: [tomat.oa.dev](https://tomat.oa.dev) ([source](./site/)).

[electrAI]: https://github.com/Quantum-Accelerators/electrai
[tomol]: https://huggingface.co/ihxds/ToMol-marin-1B

## Patch tokenization (v3, current)

Each training example is one P × P × P sub-cube of a material's
native-resolution density, prefixed with:

- The full grid shape `(nx, ny, nz)`.
- The material's lattice `(a, b, c, α, β, γ)` (added 2026-04-30; v3-lat).
- The material's atomic inventory (Z + per-atom *patch-translated*
  fractional coordinates — v3 wraps atoms relative to the patch's
  anchor so the model never has to learn PBC modular arithmetic).

At `P = 19` with the LMQ-v2 1-token-per-voxel density codec, each
sequence is `19³ = 6,859` density tokens plus a small preamble — fits
8k context. Vocab is **~18.5k tokens** (20 specials + 118 atomic Z +
ints for grid/positions/lattice + 16,384 LMQ density bins). Each
material gets **M = 64** randomly-sampled patches (one patch per
sequence).

Earlier eras: v2 (P=14, 2-token density codec, vocab ~6.8k, SHAPE/OFFSET/HI
preamble blocks) is archived in
[`docs/2026-04-30-overview-snapshot.md`](docs/2026-04-30-overview-snapshot.md).

## Example training sequence (v3)

Schematic of one v3 row (real rows live in `data/tokenized/train-full-v3/`):

```
[BOS]
[GRID_START]    nx ny nz                              [GRID_END]
[LATTICE_START] a b c α β γ                           [LATTICE_END]
[ATOMS_START]   Z₁ Z₂ … Z_N                           [ATOMS_END]
[POS_START]     (x₁ y₁ z₁  x₂ y₂ z₂  …) (patch-translated coords)  [POS_END]
[DENS_START]    d₁ d₂ … d_{19³}    # 6,859 density tokens = 1 × 19³
[DENS_END]
[EOS]
[PAD] × …                                             # right-padded to 8,192
```

Atom Zs render as element symbols. LMQ density codec emits one token
per voxel (Lloyd-Max quantized — see
[`docs/lmq-vs-equal-mass.md`](docs/lmq-vs-equal-mass.md)).
[`scripts/show_tokens.py`](./scripts/show_tokens.py) renders any parquet
row in this form.

For the current state of experiments, ckpts, and NMAE numbers, see
[`OVERVIEW.md`](./OVERVIEW.md).

## Tokenized datasets

Current default: **`train-full-v3` / `val-full-v3`** — LMQ-v2 1-token
density codec, P=19, M=64 patches/mat, lattice-aware preamble, pad_to=8192,
seed 42. ~77 k train mats × 64 = ~2.5 M sequences; ~4.3 k val mats × 64 =
~277 k sequences. Stored at `gs://marin-eu-west4/tomat/tokenized/{train,val}-full-v3/`.

Just-tokenized (2026-05-10): `train-full-v3-m128` / `val-full-v3-m128`
(same recipe, M=128) — enables 2-epoch training without repeating tokens.
Awaiting GCS sync from the Modal volumes.

Raw Zarrs live on Princeton della (`/scratch/gpfs/…/rho_gga/`, ~412 GB
total); staged onto two Modal volumes (`tomat-rho-gga` val, 22 GB;
`tomat-rho-gga-train` train, 370 GB) where tokenize runs and emits
parquet, which syncs to `gs://marin-eu-west4/tomat/tokenized/`.

Full historical table + v2-era datasets (P=14, 2-token codec) lives in
[`docs/datasets.md`](./docs/datasets.md) (partially out-of-date pending
v3 refresh).

## Scale training runs

Current runs (200M / 1B v3): see [`OVERVIEW.md`](./OVERVIEW.md) for the
live table + best NMAE/NEMD numbers. `./tomat runs links` regenerates a
slack-paste-ready markdown table.

The original v2-era 30M/208M/1B scaling-study (P=14, two-token codec)
is archived below for reference.

![scaling loss curves](./site/public/scaling-loss.png)

Seed 42, 8k context, P=14. A100 runs on val-full ("4 k mats"); TPU
runs on the full train split. Project
[`tomat-two_token_9_12-P14`](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14).

| run | model | data | compute | batch (per-dev) | steps | tokens | FLOPs (×10¹⁸) | MFU | tok/s | final loss |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [A100:1 bs=32](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-bs32-seed42) | 30M | val-full | Modal A100:1 | 32 (32) | 2,560 / 5k (OOM) | 0.67 B | 0.32 | 12.4% | 80 k | 2.235 |
| [A100:2 bs=32](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-2gpu-bs32-seed42) | 30M | val-full | Modal A100:2 | 32 (16) | 5,000 | 1.31 B | 0.62 | 12.0% | 157 k | **1.962** |
| [A100:4 bs=64](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs64-4gpu-bs64-seed42) | 30M | val-full | Modal A100:4 | 64 (16) | 5,000 | 2.62 B | 1.25 | 11.96% | 313 k | 1.975 |
| [A100:8 bs=128](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs128-8gpu-bs128-seed42) | 30M | val-full | Modal A100:8 | 128 (16) | 5,000 | 5.24 B | 2.49 | 11.86% | 624 k | 2.022 |
| [TPU v6e-4 bs=128](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-tpu-bs128-seed42) | 30M | val-full | Marin TPU v6e-4 | 128 (32) | 1,000 | 1.05 B | 0.50 | 10.25% | 792 k | 2.620 |
| [**TPU v6e-8 bs=256**](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-bs256-seed42) | 30M | **train-full** | Marin TPU v6e-8 | 256 (32) | 2,000 | **4.19 B** | **2.00** | 8.38% | 1,297 k | **2.214** |
| [**TPU v6e-16 bs=512** (multihost)](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu16-30M-bs512-seed42) | 30M | train-full | Marin TPU v6e-16 (4 VMs) | 512 (32) | 2,000 | **8.39 B** | **4.00** | 6.6% | **1,983 k** | **2.212** |
| [**TPU v6e-8 bs=128** (+ val, bf16)](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-200M-bs128-val-bf16-seed42) | **208M** | train-full | Marin TPU v6e-8 | 128 (16) | 6,000 | 6.29 B | **15.55** | 9.86% | 293 k | **1.661** (eval 1.683) |
| [**TPU v6e-16 bs=128** (1B)](https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu16-1B-bs128-val-bf16-seed42) | **1B** | train-full | Marin TPU v6e-16 (4 hosts) | 128 (8) | 4,000 | 4.19 B | **43.20** | **17.53%** | 250 k | **1.524** (eval 1.537) |

Headlines:

- **A100 scaling is linear**: 157 k → 313 k → 624 k tok/s across A100:2/4/8
  at per-device bs=16 (2× per doubling, MFU flat ~12%).
- **TPU v6e-4 ≈ 10× A100:1 tok/s** at the same per-device batch — matching
  the 12× hardware-FLOPs ratio minus a ~17% MFU gap.
- **train-full** (18× more data): loss drops 2.62 → 2.21, but the 30M
  model is now **~7× past Chinchilla-optimal** so it's parameter-bound,
  not data-bound. Bigger model is the next axis.
- **Multihost TPU (v6e-16) works**: 4 VMs × 4 chips, 1.98 M tok/s at
  ~78% scaling efficiency vs v6e-8. Required adding
  `jax.distributed.initialize()` at script entry because Levanter's
  `WandbConfig.init` calls a multihost broadcast before the trainer's
  own distributed setup fires.
- **208M Qwen3** (hidden=1024, 12 layers, 16 heads, bf16, with real
  val split) on train-full, extended to **loss 1.661 on 6.29 B tokens**
  (eval 1.683 / BPB 0.595) — 0.55 nats below the 30 M baseline. 15.55 EF
  compute.
- **1B Qwen3** (hidden=2048, 20 layers, 16 heads, 5632 ffn) on v6e-16
  multihost (4 hosts × 4 chips), 4 B tokens at bs=128 → **loss 1.524
  (eval 1.537)**. 0.137 nats better than 208 M on half the tokens.
  **MFU jumps to 17.5 %** (vs 9.9 % at 208 M, 8–10 % at 30 M),
  confirming the small-model-under-saturates-chip hypothesis. 1 B at
  4 tok/param is still ~5× under Chinchilla — clean "more tokens" headroom.

> **v2-era caveat (still archived in
> [`docs/2026-04-30-overview-snapshot.md`](./docs/2026-04-30-overview-snapshot.md)):**
> these train/eval *losses* and *BPB* are token-space cross-entropy,
> not directly comparable to electrAI / ChargE3Net's voxel-space NMAE.
> Mat-NMAE + mat-NEMD eval is now in place and reported in OVERVIEW;
> current best is 1.73% / 1.76% (200M cont7k-ext).

## Running

Setup:

```bash
spd                                 # direnv + versioned venv
uv sync                             # install deps
uv run pytest tests/                # tokenizer roundtrip tests
```

Tokenize on Modal:

```bash
TOMAT_VOLUME=tomat-rho-gga-train modal run \
  scripts/tokenize_patches_modal.py::parallel \
  --label train-full-v3 --split train \
  --patches-per-material 64 --patch-size 19 --tokenizer-version v3 \
  --lmq-path gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz \
  --n-workers 256 --seed 42 --pad-to 8192
```

Train on Marin TPU (via the `tomat` CLI; wraps iris):

```bash
./tomat train -m 200M -T v6e-16 -b 128 -s 8000 -D train-full-v3 \
    --shuffle-window-blocks 1024 --share-cache \
    -e MARIN_I_WILL_PAY_FOR_ALL_FEES=1 \
    train-full-v3-200M-bs128-emd-do-8k-tpu16
```

See `./tomat --help` for subcommands (`runs`, `iris`, `evals`, `train`,
`runs links`).

## Layout

```
src/tomat/
  float_codec.py                 # FP16-like log-uniform codec (3 tokens per signed float)
  promolecule.py                 # analytic atomic-density models (Δρ subtraction; scheme 4)
  tokenizers/
    patch.py                     # patch tokenizer (the one used for training)
    base.py, direct.py,          # earlier fidelity-sweep tokenizers (schemes 1/3/5)
    cutoff.py, fourier*.py, delta.py
  data/
    mp.py                        # S3 → pymatgen Chgcar, local caching
    zarr_io.py                   # Zarr → density array (from della/Modal volume)
    classify.py                  # material-type classifier
scripts/
  tokenize_patches*.py           # patch tokenizer + Modal parallel wrapper
  train_smoke_modal.py           # Modal A100 training (A100:{1,2,4,8} variants)
  fidelity_sweep*.py, fit_*.py   # earlier fidelity-sweep entry points
  show_tokens.py                 # decode a parquet row to human-readable form
  sync_parquets_to_gcs.py        # Modal-vol → GCS upload with md5 verify
  pull_wandb_runs.py             # W&B → CSV dump for plots
  make_scaling_plot_png.py       # static scaling-loss PNG for README / slides
  verify_val_full_parquet.py     # Modal-side row-group integrity scan
marin/
  train_tomat_tpu.py             # TPU training script (v6e-4/8/16, 30M/200M, bf16, val)
  pyproject.toml, uv.lock        # marin-community find-links + TPU-gated jax
docs/
  datasets.md                    # raw Zarr layout, tokenized sets, scale-runs table
site/                            # React + Plotly interactive dashboard (tomat.oa.dev)
specs/
  00..09-*.md                    # design / project / spec documents
```

## Follow-ups

- Scale model further (600M–1B) now that 208M cleared the 30M baseline.
- DVX-track raw Zarrs + parquet manifests (spec 09, della-side).
- TransformerEngine on Modal A100:4 for the bs=128 apples-to-apples
  GPU point (currently limited to per-device bs=16 due to attention
  OOM; spec / post-meeting).

## Earlier work

For the pre-training fidelity sweep (NMAE / χ² reconstruction floors
across cutoff/Fourier/Δρ tokenizers), see
[`specs/done/02-fidelity-sweep.md`](./specs/done/02-fidelity-sweep.md)
and [`results/sweep-n50.csv`](./results/sweep-n50.csv). Headline from
that phase: Fourier lowpass beats voxel cutoff by ~2 orders of magnitude
on NMAE at every sparsity; direct-float Fourier encoding needs ≥64 k
context to get in budget → patches (what we train on today) were the
right answer.
