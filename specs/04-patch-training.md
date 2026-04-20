# Patch-based training (30M hello-world)

Pivot from scheme-level fidelity sweeps to an actual training run. Driver:
the fidelity-sweep Pareto (specs/02) said no candidate scheme fits useful
NMAE in 16k context; meanwhile the team's priority shifted from
"characterize tokenizer floors" to "have a working model end-to-end this
week." Patch tokenization resolves the context-length constraint without
downsampling the underlying density grid.

## What's a patch?

One training example = a ``P × P × P`` sub-cube of a material's full-grid
density, prefixed with:

1. The full grid shape ``(nx, ny, nz)``.
2. The material's atomic inventory (atomic numbers + fractional
   coordinates) — the same for every patch of the same material.
3. The patch's anchor offset ``(ix, iy, iz)`` and shape ``(P, P, P)``.

Token layout emitted by [`src/tomat/tokenizers/patch.py`](../src/tomat/tokenizers/patch.py):

```
[BOS] [GRID_START] nx ny nz [GRID_END]
[ATOMS_START] Z₁ … Zₙ [ATOMS_END]
[POS_START] x₁ y₁ z₁ … [POS_END]
[SHAPE_START] P P P [SHAPE_END]
[OFFSET_START] ix iy iz [OFFSET_END]
[DENS_START] d₀ d₁ … d_{P³−1} [EOS]
```

* **Position codec** — tomol 3-byte (SE + M0 + M1, 1 024 vocab).
* **Density codec** — 2-token 9+12 (SE + M, 4 608 vocab).
* **Total vocab** — 6 790 tokens.

## Patch sampling

Any voxel in any material is a valid anchor — the loader picks ``M``
random offsets per material per pass, extracts the ``P³`` sub-cube with
PBC wrapping, and emits that as a training row. Anchoring is uniform
over ``[0, N_axis)`` per axis; wrap-around is cheap because crystals are
periodic.

This gives three nice properties:

1. **No downsampling.** The model sees native-resolution data.
2. **Material-level coverage is the first-order variance.** At ``M=32``
   patches × 4 303 val structures = ~138 k rows, each row is a distinct
   local context — more examples than a non-patch approach at the same
   structure count.
3. **Offsets are free data augmentation.** Different runs can use
   different anchor RNG seeds to expand the effective dataset.

## Budget

Target: a **30 M-parameter Qwen3** training on **8k context**, one patch
per sequence. At the defaults above:

* P = 14 → density payload 2 × 14³ = 5 488 tokens; structure preamble
  ~200 tokens; total ~5.7 k. Fits 8 k with ~2 k buffer for large
  structures (up to ~100 atoms before overflow).
* Embeddings at hidden=512 tied: 6 790 × 512 ≈ 3.5 M params — ~12 % of
  the 30 M budget.

## Pipeline

```
rho_gga Zarr (della)
   │
   │  scripts/tokenize_patches.py -r <rho_gga> -s split.json -k validation
   │                              -m 32 -p 14 -o <parquet-dir>
   ▼
parquet shards (task_id, offset_x/y/z, input_ids)
   │
   │  GCS / Modal volume mirror (TBD)
   ▼
Levanter: PrebuiltLmDatasetFormat(input_ids_key="input_ids")
   │
   ▼
experiments/tomat_patch_30m.py  →  Qwen3 training on v5p-8
```

## Code that exists

* [`src/tomat/tokenizers/patch.py`](../src/tomat/tokenizers/patch.py)
  — `PatchTokenizer.tokenize/detokenize/extract_patch/random_offsets/
  export_hf_tokenizer_json`.
* [`src/tomat/data/zarr_io.py`](../src/tomat/data/zarr_io.py)
  — `load_rho_gga` returns a CHGCAR-shim for any tokenizer.
* [`scripts/tokenize_patches.py`](../scripts/tokenize_patches.py) — CLI
  that writes zstd-parquet shards from a Zarr directory + a split JSON.
* [`experiments/tomat_patch_30m.py`](../experiments/tomat_patch_30m.py)
  — Marin launch config scaffold (dataset URL still TODO).
* Tests: `tests/test_patch.py`, `tests/test_zarr_io.py` — 11 tests
  covering roundtrip, PBC wrap, vocab layout, HF-JSON export.

## Code that's still missing

1. **Dataset destination** — tokenize_patches writes parquet locally;
   need a GCS bucket (or Modal-volume-mounted-in-TPU setup) the training
   job can read. Wire the URL into `tomat_train_source` /
   `tomat_val_source` in the experiment file.
2. **Codec config fit on rho_gga** — current defaults (log_min=-4.13,
   log_max=4.97) were fit on the 128³ dataset_4 data. Rerun
   `scripts/fit_density_codec.py` against a rho_gga subset to confirm
   the range still covers native-resolution densities. Likely identical
   since values are per-voxel, not per-structure; worth verifying.
3. **Training launch** — Marin deps not yet installed in this repo;
   `experiments/tomat_patch_30m.py` has import-time dependencies on
   `levanter`, `fray`, `marin.execution`. Add those via
   `pyproject.toml` when ready to run, or run from a marin-experiments
   checkout that already has them.

## Open design questions

* **Patch size.** 14³ chosen because it fits 8k with buffer. 12³ fits
  more easily (smaller density payload); 16³ requires 12 k context.
  Worth a small ablation later.
* **Multi-patch per sequence.** A single sequence currently = one patch.
  Packing multiple non-overlapping patches into one sequence would
  amortize the structure preamble and improve FLOPs/patch; defer until
  the single-patch pipeline trains.
* **Cross-patch consistency.** Independent-patch training might produce
  seams at patch boundaries on whole-structure reconstruction. If
  stitching shows artefacts, consider overlapping patches + averaging
  or an autoregressive inter-patch objective.
* **Loss masking.** Right now every token contributes equally to the
  language-modelling loss. Masking the structure-preamble tokens (so
  only density tokens count) would be more aligned with "predict
  density from crystal context." Low priority for the hello-world.

## Done criteria

Move this spec to `specs/done/` once:

1. A parquet shard set for the val split exists at a training-accessible
   URL (GCS or Modal mount).
2. `experiments/tomat_patch_30m.py` launches without TODOs.
3. A 30 M Qwen3 trains for at least 100 steps with monotonically
   decreasing train loss — not a science claim, just "it trains."
