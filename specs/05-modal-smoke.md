# Modal-GPU smoke training

Shortest path to spec 04's done-criterion #3 — "a 30 M Qwen3 trains
for at least 100 steps with monotonically decreasing train loss" — on
a Modal GPU, since that's our immediate-access compute surface. Marin
+ GCP TPU Research Cluster will land for real training runs in a later
spec (see **Follow-up: Marin+TPU** at the bottom for the handoff).

The `tomat-rho-gga` Modal volume (spec 03) is already seeded with the
val split of native-resolution rho_gga Zarrs; smoke tokenizes + trains
inside Modal with the volume mounted, then pulls outputs back locally
for DVX tracking.

## Goal

≥100 Qwen3 steps on a Modal A100, monotonically-decreasing train loss,
reading patch-tokenized parquet from the `tomat-rho-gga` volume.
Artifacts (parquet dir + run output) end up tracked by `.dvc` files
in the tomat repo.

## Non-goals (deferred)

- Marin orchestrator / `iris` launcher, GCP TPU Research Cluster setup
  — see **Follow-up: Marin+TPU** below.
- GCS-hosted parquet — Modal volume is sufficient for the smoke.
- Full train split tokenization — smoke uses val only; train==val is
  fine for "does the loop close."
- Hyperparameter tuning, eval harness, wandb polish — inherit the
  scaffold's config; nothing fancy.
- Multi-codec / multi-patch-size sweeps — pick one point:
  `(codec=two_token_9_12, P=14)`. Sweep infra (spec 04) takes over
  once the single-point pipeline works.

## Prereqs

- Local `modal` CLI authenticated to the `open-athena` workspace
  (already true — the laptop session used it for spec 03).
- `tomat-rho-gga` volume exists with `/label/mp-*.zarr` populated.
  Verify with `scripts/verify_modal_volume.py`.
- Project venv with `modal`, `pyarrow`, `zarr`, `numpy`, and the tomat
  package already installed locally (for `dvx` driver + reading the
  pulled-back parquet). No need for levanter locally — it only runs
  inside the Modal image.

## DVX: data-artifact tracking

Every produced artifact gets a `.dvc` in the local repo so the exact
command + code version that made it are recorded inline, mirroring the
existing `results/sweep-n50.csv.dvc`. Two artifacts:

| artifact path | produced by | `.dvc` at |
|---|---|---|
| `data/tokenized/val-smoke/` (parquet shards + `meta.json`) | `scripts/tokenize_patches_modal.py` | `data/tokenized/val-smoke.dvc` |
| `results/smoke/` (loss csv + final ckpt + log) | `scripts/train_smoke_modal.py` | `results/smoke.dvc` |

Volume roundtrip: each Modal function writes to a subdir on
`tomat-rho-gga` (e.g. `/tokenized/val-smoke/` or `/results/smoke/`),
then the local-entrypoint does `modal volume get ...` into the repo
working tree. DVX then hashes the local copy like any other artifact.

External dep (not hashed): the `/label/` tree on the volume — record
by the volume name + split-file filename (`split_limit_22M.json`) in
the `.dvc`'s `meta.computation.deps` section.

Levanter is a pip dep installed into the Modal image — record its
version string in the smoke `.dvc`'s `meta.levanter_version` field so
the run is reproducible at the package level even though we don't hash
the installed library.

## Plan

### 1. Modal tokenize function → parquet on the volume

Write `scripts/tokenize_patches_modal.py` — a thin Modal wrapper over
`scripts/tokenize_patches.py`. Model it on
`scripts/verify_modal_volume.py` for the volume-mount idiom:

- `image = modal.Image.debian_slim().pip_install("zarr>=3", "numpy", "pyarrow>=15", "pymatgen", "click")` + copy the tomat package into the image (either `add_local_python_source("tomat")` or pip-install from the repo root).
- Mount `tomat-rho-gga` at `/vol`.
- `@app.function(volumes={"/vol": volume}, timeout=3600)` runs the
  existing tokenize logic with `--rho-gga-dir /vol` and writes to
  `/vol/tokenized/val-smoke/`.
- Local entrypoint: `modal run scripts/tokenize_patches_modal.py`,
  then `modal volume get tomat-rho-gga /tokenized/val-smoke data/tokenized/`.

Cmd the stage wraps (what runs inside the Modal function):

```bash
scripts/tokenize_patches.py \
    -r /vol \
    -s /vol/split_limit_22M.json \
    -k validation \
    -m 32 \
    -p 14 \
    -c two_token_9_12 \
    -n 128 \
    -o /vol/tokenized/val-smoke \
    -S 42
```

128 materials × 32 patches = 4,096 rows ≈ 22 MB zstd.

**Seeding**: `-S 42` controls the patch-offset RNG at preprocessing —
determines *which* 32 patches per material exist in the shards. This
is a separate stream from the training-side block-shuffle seed (step
2), so reshuffling the same shards doesn't require re-tokenizing.

DVX-wrap:

```bash
dvx repro --single data/tokenized/val-smoke.dvc
```

where the `.dvc`'s `meta.computation.cmd` is the `modal run …`
invocation + volume pull, and `deps` lists `scripts/tokenize_patches.py`,
`scripts/tokenize_patches_modal.py`, `src/tomat/{float_codec,
tokenizers/patch,data/zarr_io}.py` plus `__init__`s — same style as
`results/sweep-n50.csv.dvc`.

**Sanity**: pulled-back `meta.json` should show `total_rows: 4096`,
`n_materials: 128`, `vocab.total_size: 6792`,
`density_codec_name: two_token_9_12`.

### 2. Modal train function with Levanter

Write `scripts/train_smoke_modal.py`. **Levanter is not on PyPI** —
Marin ships `marin`, `marin-levanter`, `marin-iris`, `marin-zephyr`,
`marin-rigging` via custom wheel indexes at
`https://github.com/marin-community/marin/releases/expanded_assets/
<pkg>-latest`. See `marin-experiments/tiny-stories/pyproject.toml` for
the full incantation:

- `dependencies = ["marin-levanter", ...]`
- `[tool.uv] prerelease = "allow"` + `find-links = [...]`
- `override-dependencies = [...]` (pins for `omegaconf`,
  `antlr4-python3-runtime`, etc.)
- `[[tool.uv.index]] name = "marin-resiliparse"` for the fork of
  resiliparse that marin needs.

The commented block in tomat's `pyproject.toml` is an older version of
this — refresh it against the tiny-stories file and uncomment.

Image build: have Modal `uv sync` the refreshed pyproject inside the
image. Reference `marin-experiments/tiny-stories/launch.py` for
`ResourceConfig.with_gpu("a100", ...)` usage — it's the closest-scale
prior art (~30 M Grug transformer).

Resource: `@app.function(gpu="A100", volumes={"/vol": volume},
timeout=7200)`.

Inside the function, build a minimal Levanter training entry:

1. `Qwen3Config`: 6 layers, hidden=512, 4 heads, max_seq_len=8192,
   vocab=**6792** — identical to the Marin scaffold
   (`experiments/tomat_patch_30m.py`). Read the vocab size dynamically
   from the tokenized shards' `meta.json` rather than hardcoding, so a
   codec swap at preprocessing doesn't silently misalign.
2. `LmDataConfig` with `UrlDatasetSourceConfig(urls=[
   "/vol/tokenized/val-smoke/*.parquet"], format=PrebuiltLmDatasetFormat(
   input_ids_key="input_ids"))` for both train and validation. Local
   file:// URLs work because the volume is mounted in-function.
3. Override: `num_train_steps=100`, `train_batch_size=8` (fits A100
   with 8k context and 30M params easily; bump if headroom obvious),
   `steps_per_eval=50`.
4. `TrainerConfig.seed` + `BlockShuffleConfig.seed` both settable via
   a `--seed` CLI flag; default `None` (non-deterministic) for
   open-ended experimentation, explicit ints when reproducibility
   matters. These are the training-side streams; the preprocessing
   seed (step 1's `-S/--seed`) is already separate and independently
   controllable.
5. Call `levanter.main.train_lm.main(TrainLmConfig(...))` (or the
   current equivalent — locate via `grep -R "def main"` inside the
   installed `marin-levanter` package).
6. Write loss csv, final ckpt, and stdout tee to `/vol/results/smoke/`.

Local entrypoint: `modal run scripts/train_smoke_modal.py`, then
`modal volume get tomat-rho-gga /results/smoke results/`.

DVX-wrap:

```bash
dvx repro --single results/smoke.dvc
```

Code deps: `scripts/train_smoke_modal.py` + any config file it reads.
Data dep: `data/tokenized/val-smoke/` (DVX picks up the prior
`.dvc` linkage automatically). `meta.levanter_version` gets the
installed version string (pull it from the Modal function's output —
e.g. print `levanter.__version__` at startup and grep).

### 3. Sanity + acceptance

Want to see:
- No tokenization / vocab-size mismatch at startup (`meta.json.vocab.
  total_size == Qwen3Config.vocab_size`).
- `input_ids` min ≥ 0, max < 6792 — catches codec mis-config early.
- Loss printed each step. Success = `loss[100] < loss[0]`, no NaNs,
  no flat-line.

### 4. Commit + move this spec

On success:

- Commit `scripts/tokenize_patches_modal.py`,
  `scripts/train_smoke_modal.py`, the two `.dvc` files, and any
  config file used.
- Append an **Outcome** section to this spec with loss at step 0/50/
  100, wall-clock (including cold-start), A100 utilization, and any
  debugging surprises. Move to `specs/done/05-modal-smoke.md`.
- File a follow-up note in `specs/04-patch-training.md`'s "Code that's
  still missing" list crossing off #3 for the Modal-GPU path, leaving
  the TPU branch open.

## Known gotchas

- **Codec log range fit.** `tokenize_patches.py`'s defaults
  (`log_min=-4.127, log_max=4.967`) were fit on 128³ dataset_4 data.
  Probably still fine for native rho_gga (per-voxel densities have
  similar dynamic range regardless of grid resolution), but if smoke
  loss is suspiciously high or vocab utilization looks skewed, rerun
  `scripts/fit_density_codec.py` against an rho_gga subset and update
  the defaults.
- **PrebuiltLmDatasetFormat expects `input_ids`.** That's what
  `tokenize_patches.py` already writes; no renaming needed. If
  Levanter complains about missing columns, the parquet schema is the
  first thing to check (`input_ids: list<int32>`).
- **Modal image size.** Levanter + jax[cuda12] is big (~2 GB). First
  cold start will be slow (5–10 min); cache the image build.
- **Volume-get overwrite.** `modal volume get` won't clobber local
  files by default. If re-running, `rm -rf data/tokenized/val-smoke
  results/smoke` before the pull, or use a fresh output subdir name.
- **Single-patch-per-sequence.** Sequences are padded/truncated to 8k
  context. One patch ≈ 5.7k tokens; the rest is padding. Wasteful but
  correct for smoke.

## Done criteria

- [ ] `data/tokenized/val-smoke.dvc` exists, tracks a 4,096-row
      parquet dir with `vocab.total_size == 6792` in `meta.json`, and
      records `meta.computation.{cmd,deps}`.
- [ ] `scripts/tokenize_patches_modal.py` and
      `scripts/train_smoke_modal.py` committed.
- [ ] Modal training function runs ≥100 steps without crashing.
- [ ] Train loss at step 100 strictly less than step 0 (no NaNs, no
      flat-line).
- [ ] `results/smoke.dvc` exists tracking the run's output directory,
      with the tokenized parquet dir as a dep and levanter version
      recorded in `meta`.
- [ ] Spec moved to `specs/done/05-modal-smoke.md` with an **Outcome**
      section.

## Follow-up: Marin+TPU (separate spec)

Once smoke is green, move to Marin on GCP TPU Research Cluster for
real training. Preparatory info already in hand:

- `/Users/ryan/c/oa/marin-experiments/tiny-stories/` — ~30 M Grug
  transformer, closest-scale reference. `launch.py` uses `iris
  --cluster=marin` with `ACCELERATOR` env var (`tpu` default; `gpu`
  and `cpu` also wired). Pipeline: HF download → tokenize → train.
  Our equivalent: rho_gga on `tomat-rho-gga` volume (or GCS) →
  `tokenize_patches.py` → Qwen3 train.
- `/Users/ryan/c/oa/marin/docs/tutorials/`:
  - `first-experiment.md` — basic launch pattern
  - `tpu-cluster-setup.md` — GCP project + quota + cluster template
  - `train-an-lm.md` — LM-training-specific guidance
  - `local-gpu.md` — local-GPU iris cluster (useful for dev before
    dispatching to TPU)
- `/Users/ryan/c/oa/marin/lib/iris/examples/` — YAML cluster configs
  (`local.yaml`, `marin.yaml`, `smoke-gcp.yaml`, `coreweave.yaml`).
- Existing `experiments/tomat_patch_30m.py` in tomat is already
  Marin-shaped; just needs deps installed + a real dataset URL.

Open items for that spec:
1. Marin deps in tomat's `pyproject.toml` (currently deferred; see the
   commented block there).
2. GCS bucket + dataset upload (or train from `tomat-rho-gga` volume
   if TPU workers can mount Modal volumes — unlikely; GCS probably
   simpler).
3. `iris` cluster choice: `marin.yaml` (shared cluster, europe-west4
   v6e) vs our own GCP project.
4. Launch + monitoring (wandb integration, checkpointing cadence).
