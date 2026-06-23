# tomat 🍅

**to**kenized **mat**erials — an LLM/transformer approach to predicting
DFT-converged electron density for periodic crystals. Inspired by
[tomol] (tokenized molecules), [electrAI] (the 3D ResUNet over voxel grids).

**Interactive dashboard**: [tomat.oa.dev](https://tomat.oa.dev) ([source](./site/)).

[electrAI]: https://github.com/Quantum-Accelerators/electrai
[tomol]: https://huggingface.co/ihxds/ToMol-marin-1B

## Current state

Best NMAE so far: `train-mg-kl-bin5-fs-tpu`
(200M Qwen3, KL-Gauss σ=5, absorbing-prior MaskGIT, 100k steps on v6e-16):
**val mean 7.46% / median 5.35%**, train mean 6.37% / median 5.26%
(200 mats each).

For reference: ChargE3Net SOTA is **0.196 %**, electrAI ResUNet ~**0.18 %**.

In flight: 10k from-scratch ablations sweeping the bin-width σ ∈ {3, 5, 10, 20}
and the mask schedule (cos-r vs absorbing); a +10k extension of bin5 to test
whether it keeps improving past 100k.

For the live per-run table, ckpt list, per-mat [ELVis] diffs, and NMAE trajectories, see [`tomat.oa.dev`](https://tomat.oa.dev).

[ELVis]: https://elvis.oa.dev

## How it works

**Patch tokenization (v3).** Each training example is one *P* × *P* × *P*
sub-cube of a material's native-resolution density, prefixed with:

- the full grid shape `(nx, ny, nz)`,
- the material's lattice `(a, b, c, α, β, γ)` (added 2026-04-30; v3-lat),
- the material's atomic inventory (atomic numbers `Z` + per-atom
  *patch-translated* fractional coordinates — v3 wraps atoms relative
  to the patch's anchor so the model never has to learn PBC modular
  arithmetic).

At `P = 19` with the LMQ-v2 1-token-per-voxel density codec, each sequence
is `19³ = 6,859` density tokens plus a small preamble — fits an 8k context.
Vocab is **~18.5k tokens** (20 specials + 118 atomic Zs + ints for grid /
positions / lattice + 16,384 LMQ density bins). Each material gets
**M = 64** randomly-sampled patches (one patch per sequence).

Schematic of one row (real rows live in `data/tokenized/train-full-v3/`):

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

**Density head.** The headline runs replace LM-head next-token CE with a
[MaskGIT]-style sampler over the LMQ density bins, trained with a
KL-Gauss loss that targets a Gaussian-blurred quantile distribution
(width σ in bin-units). The training mask prior `absorbing` masks every
density token in a sequence (the model sees only the preamble); inference
runs K MaskGIT iterations conditioning on already-decoded bins.

**Honest K=1 eval.** Mat-NMAE / mat-NEMD are computed on
`recon[filled_bins]` — the decoded density at positions the model
actually predicted — not on a re-forward pass over the fully-filled grid
(which is OOD for a causally-trained model). See
[`marin/eval_mat_nmae.py`](./marin/eval_mat_nmae.py) and
[`specs/60-m-eval-drilldown.md`](./specs/60-m-eval-drilldown.md) (the
upcoming per-mat dashboard view).

[MaskGIT]: https://arxiv.org/abs/2202.04200

## Tokenized datasets

Current default: **`train-full-v3` / `val-full-v3`** — LMQ-v2 1-token
density codec, P=19, M=64 patches/mat, lattice-aware preamble, pad_to=8192,
seed 42. ~77 k train mats × 64 = ~2.5 M sequences; ~4.3 k val mats × 64 =
~277 k sequences. Stored at `gs://marin-eu-west4/tomat/tokenized/{train,val}-full-v3/`.

Also tokenized: `train-full-v3-m128` / `val-full-v3-m128` (M=128, enables
2-epoch training without repeating tokens).

Raw Zarrs live on Princeton della (`/scratch/gpfs/…/rho_gga/`, ~412 GB
total); staged onto two Modal volumes (`tomat-rho-gga` val, 22 GB;
`tomat-rho-gga-train` train, 370 GB) where tokenize runs and emits parquet,
which syncs to `gs://marin-eu-west4/tomat/tokenized/`.

Full historical table + v2-era datasets (P=14, 2-token codec) lives in
[`docs/datasets.md`](./docs/datasets.md).

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

Train on Marin TPU (via the `tomat` CLI; wraps iris). The bin5, cos-r, and
σ-ablation fires are checked in under [`scripts/fires/`](./scripts/fires/);
clone one to start a new arm.

```bash
./scripts/fires/cos-r-fs-tpu.sh    # one of the in-flight ablations
```

See `./tomat --help` for the full CLI surface (`runs`, `iris`, `evals`,
`train`, `runs links`).

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
  fires/                         # checked-in iris fire scripts (bin5, cos-r, σ-ablations, …)
  tokenize_patches*.py           # patch tokenizer + Modal parallel wrapper
  train_smoke_modal.py           # Modal A100 training (A100:{1,2,4,8} variants)
  fidelity_sweep*.py, fit_*.py   # earlier fidelity-sweep entry points
  show_tokens.py                 # decode a parquet row to human-readable form
  sync_parquets_to_gcs.py        # Modal-vol → GCS upload with md5 verify
  pull_wandb_runs.py             # W&B → CSV dump for plots
  verify_val_full_parquet.py     # Modal-side row-group integrity scan
marin/
  train_tomat_tpu.py             # TPU training script
  eval_mat_nmae.py               # mat-NMAE / mat-NEMD eval (K=1 / K=12 MaskGIT)
  qwen3_density.py               # density-head + MaskGIT loss subclasses
  pyproject.toml, uv.lock        # marin-community find-links + TPU-gated jax
docs/                            # design docs, dataset inventory, snapshots
site/                            # React + Plotly interactive dashboard (tomat.oa.dev)
specs/                           # specs (in-progress) + specs/done/ (completed)
```

## Earlier work

- **v2-era scaling study** (30M / 208M / 1B, P=14, two-token codec) — first
  multihost TPU + MFU footing. Archived at
  [`docs/v2-scaling-study.md`](docs/v2-scaling-study.md).
- **Pre-training fidelity sweep** (NMAE / χ² reconstruction floors across
  cutoff / Fourier / Δρ tokenizers) — see
  [`specs/done/02-fidelity-sweep.md`](./specs/done/02-fidelity-sweep.md) and
  [`results/sweep-n50.csv`](./results/sweep-n50.csv). Fourier lowpass beats
  voxel cutoff by ~2 orders of magnitude on NMAE; direct-float Fourier
  encoding needs ≥64 k context to fit budget → patches (the v3 design) were
  the right answer.
