"""Modal-based VL (`eval/loss`) backfill for tomat MaskGIT training runs.

Many of tomat's pre-2026-06 runs trained *without* a validation carve-out
(`num_validation_sequences` never plumbed through), so `eval/loss` was
never logged to wandb. This module retroactively computes the same
`eval/loss` for every saved ckpt by:

1. Loading the ckpt as `Qwen3MaskGITLMHeadModel` (+1 vocab for MASK).
2. Configuring the MaskGIT loss args matching the trainer (loss_type, prior,
   optional kl_sigma).
3. Building an `LmDataConfig` with `num_validation_sequences={"tomat": N}`
   against the same parquet on the `tomat-rho-gga-train` Modal volume —
   carves out a fixed `val_seqs`-sequence val split (`PRNGKey(0)` feistel
   shuffle, last N after shuffle).
4. Running `TaggedEvaluator.evaluate(model)` once per ckpt and writing
   `EvalResult.micro_avg_loss` to
   `gs://<bucket>/tomat/eval/vl-backfill/<ckpt_leaf>/step-<N>.json`.

`tomat runs sync` already merges `eval/vl-backfill/<ckpt_leaf>/step-*.json`
files into the wandb history → R2 parquet pipeline (see `_vl_backfill_rows`
in the `tomat` CLI), so backfilled steps land in the runs dashboard
automatically.

The CLI surface is `tomat evals vl-backfill <run>`; this module owns the
Modal app + function body and a few discovery helpers consumed by the CLI.
"""
from __future__ import annotations

import re
import subprocess
import sys
from functools import partial

import modal


err = partial(print, file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

VOL_NAME = "tomat-rho-gga-train"
MOUNT = "/vol"

# Match scripts/eval_modal.py + scripts/train_smoke_modal.py exactly so the
# Modal image layer is shared across calls. Bump in lockstep when bumping
# those scripts.
OA_MARIN_SHA = "97eea237598bfe0d0af1143dce92c0c00526a8f0"
OA_MARIN_GIT = f"git+https://github.com/Open-Athena/marin.git@{OA_MARIN_SHA}"
_marin_git_pkgs = " ".join(
    f"'marin-{p} @ {OA_MARIN_GIT}#subdirectory=lib/{p}'"
    for p in ("haliax", "levanter", "fray")
)
EXTRA_FIND_LINKS = [
    "https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799",
    "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4",
]
_extra_find_links_args = " ".join(f"--find-links {u}" for u in EXTRA_FIND_LINKS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --pre {_extra_find_links_args} "
        f"{_marin_git_pkgs} dupekit "
        "'jax[cuda12]' 'pyarrow>=15' fsspec gcsfs "
        "zarr 'pymatgen<2025' boto3",
    )
    # `add_local_python_source("marin")` is necessary but not sufficient:
    # marin-haliax/levanter transitively install marin-core
    # (`lib/marin/src/marin/`), which lands at `/root/marin/` (Modal places
    # non-installed-package mounts under `/root/`) and resolves to the OA
    # marin contents (`marin.__file__ → /root/marin/__init__.py` with only
    # `__init__.py` + `utils.py` + cluster/core/datakit/... subdirs). Our
    # tomat-local `qwen3_density.py` + `eval_mat_nmae.py` (never upstreamed)
    # are silently dropped, so `from marin.qwen3_density import ...` reports
    # ModuleNotFoundError. Force-copy the tomat-local files into
    # `/root/marin/` so they live alongside the OA modules. `copy=True` so
    # they become part of the image layer.
    .add_local_file(
        "marin/qwen3_density.py",
        "/root/marin/qwen3_density.py",
        copy=True,
    )
    .add_local_file(
        "marin/eval_mat_nmae.py",
        "/root/marin/eval_mat_nmae.py",
        copy=True,
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-vl-backfill", image=image)
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
train_volume = modal.Volume.from_name(VOL_NAME)


# ──────────────────────────────────────────────────────────────────────
# Run-shape preset autodetect
# ──────────────────────────────────────────────────────────────────────


CANONICAL_BUCKETS = (
    "gs://marin-eu-west4/tomat",
    "gs://marin-us-east5/tomat",
    "gs://marin-us-central1/tomat",
    "gs://marin-us-east1/tomat",
)


def resolve_run_bucket(run_label: str, ckpt_leaf: str | None = None) -> str | None:
    """Find which canonical GCS bucket a run's ckpts live in. Mirrors the
    `_resolve_run_bucket` helper in the `tomat` CLI; duplicated here so the
    library is importable without depending on the top-level script.
    """
    leaf = ckpt_leaf or run_label
    for bkt in CANONICAL_BUCKETS:
        path = f"{bkt}/results/{run_label}/checkpoints/{leaf}/"
        r = subprocess.run(
            ["gcloud", "storage", "ls", path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return bkt
    return None


def discover_ckpts(
    run_label: str,
    bucket: str,
    ckpt_leaf: str | None = None,
) -> list[int]:
    """List `step-N` values present under `<bucket>/results/<run>/checkpoints/<leaf>/`."""
    leaf = ckpt_leaf or run_label
    base = f"{bucket}/results/{run_label}/checkpoints/{leaf}/"
    r = subprocess.run(
        ["gcloud", "storage", "ls", base],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        err(f"[vl-backfill] gcloud ls {base} failed: {r.stderr.strip()}")
        return []
    steps = sorted({int(m.group(1)) for m in re.finditer(r"step-(\d+)/?", r.stdout)})
    return steps


def existing_results(
    ckpt_leaf: str,
    bucket: str,
) -> set[int]:
    """Return the set of `step-N` JSONs already present under
    `<bucket>/eval/vl-backfill/<ckpt_leaf>/`.
    """
    base = f"{bucket}/eval/vl-backfill/{ckpt_leaf}/"
    r = subprocess.run(
        ["gcloud", "storage", "ls", base],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return set()  # missing dir => no existing results
    return {
        int(m.group(1))
        for m in re.finditer(r"step-(\d+)\.json", r.stdout)
    }


def select_steps(
    run_label: str,
    bucket: str,
    *,
    explicit: str | None,
    every: int | None,
    ckpt_leaf: str | None = None,
) -> list[int]:
    """Resolve `--steps` / `--every` against the GCS ckpt set.

    `explicit` (comma-sep ints) wins. Otherwise `every` filters available
    steps to multiples of `every`. Both None → all available steps.
    """
    avail = discover_ckpts(run_label, bucket, ckpt_leaf=ckpt_leaf)
    if not avail:
        return []
    if explicit:
        wanted = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        avail_set = set(avail)
        chosen: list[int] = []
        for n in wanted:
            if n in avail_set:
                chosen.append(n)
            elif (n - 1) in avail_set:
                err(f"[vl-backfill] WARN: step-{n} missing on GCS; using step-{n-1} (Levanter OBO)")
                chosen.append(n - 1)
            else:
                err(f"[vl-backfill] WARN: step-{n} not on GCS; skipping")
        return chosen
    if every:
        return [s for s in avail if s % every == 0]
    return list(avail)


# ──────────────────────────────────────────────────────────────────────
# Modal function body — shared between H100×8 and H200×8 variants
# ──────────────────────────────────────────────────────────────────────


def _run_backfill(
    *,
    run_label: str,
    ckpt_leaf: str,
    step: int,
    bucket: str,
    parquet_label: str,
    lmq_path: str,
    model_preset: str,
    val_seqs: int,
    eval_batch: int,
    seq_len: int,
    loss_type: str,
    kl_sigma: float | None,
    kl_sigma_unit: str,
    mask_prior: str,
    cache_dir_override: str | None,
) -> dict:
    """Eval ONE ckpt's training-time `eval/loss` against `val_seqs` held-out
    sequences and write a per-step JSON to `<bucket>/eval/vl-backfill/<ckpt_leaf>/step-<N>.json`.
    Imports kept inside the function so module import on the local side stays cheap.
    """
    import json
    import os
    import time

    import equinox as eqx
    import fsspec
    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np

    import haliax as hax
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning

    import levanter
    from levanter.checkpoint import load_checkpoint
    from levanter.data.text import (
        DatasetComponent,
        LmDataConfig,
        PrebuiltLmDatasetFormat,
        UrlDatasetSourceConfig,
    )
    from levanter.eval import TaggedEvaluator, _default_lm_eval_loss_fn
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode

    # Materialize the GCP SA creds from the secret's env var → a file +
    # ADC, so gcsfs/tensorstore can read the codec / write the JSON.
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        with open("/tmp/gcp-sa.json", "w") as f:
            f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-sa.json"

    # MG mode + loss-args installation must happen before tracing the
    # model's compute_next_token_loss — same gating the trainer does.
    os.environ["TOMAT_MG_MODE"] = "1"
    os.environ["TOMAT_MG_MASK_PRIOR"] = mask_prior
    os.environ["TOMAT_MG_LOSS_TYPE"] = loss_type
    if loss_type == "kl_gauss":
        if kl_sigma is None:
            raise ValueError("kl_sigma required when loss_type=kl_gauss")
        os.environ["TOMAT_MG_KL_SIGMA"] = str(kl_sigma)
        os.environ["TOMAT_MG_KL_SIGMA_UNIT"] = kl_sigma_unit

    train_volume.reload()

    import pathlib as _pathlib
    parquet_dir = f"{MOUNT}/tokenized/{parquet_label}"
    worker_dirs = sorted(_pathlib.Path(parquet_dir).glob("worker-*"))
    if not worker_dirs:
        raise FileNotFoundError(f"no worker-*/ shards under {parquet_dir}")
    parquet_glob = f"{parquet_dir}/worker-*/*.parquet"
    meta_path = f"{worker_dirs[0]}/meta.json"
    meta = json.loads(_pathlib.Path(meta_path).read_text())
    vocab_size_raw = meta["vocab"]["total_size"]
    err(f"[vl-backfill] vocab_raw={vocab_size_raw}, patch={meta['patch_size']}, "
        f"codec={meta['density_codec_name']}, model={model_preset}, step={step}, "
        f"loss={loss_type}")

    # Codec read → density offset math (mirrors train_smoke_modal.py's
    # _train_bakeoff_impl MG branch).
    with fsspec.open(lmq_path, "rb") as f:
        codec_data = np.load(f, allow_pickle=True)
        codec_recon = np.asarray(codec_data["recon_points"], dtype=np.float32)

    specials = meta["vocab"].get("specials", {})
    has_lattice = "[LATTICE_START]" in specials or "LATTICE_START" in specials
    n_specials = 20 if has_lattice else 18
    n_atoms = 118
    n_ints = 1024
    pc = meta["vocab"]["position_codec"]
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    density_offset = n_specials + n_atoms + n_ints + pos_total
    n_density_bins = len(codec_recon)
    penalty = 10.0 * float(codec_recon.max())

    # MaskGIT model + loss config (same as training-side).
    from marin.qwen3_density import (
        Qwen3MaskGITConfig,
        build_maskgit_loss_args,
        configure_maskgit_loss,
    )

    mask_id = vocab_size_raw
    vocab_size = vocab_size_raw + 1  # +1 for MASK

    Vocab_mg = hax.Axis("vocab", vocab_size)
    loss_kwargs: dict = dict(
        Vocab=Vocab_mg,
        density_offset=density_offset,
        n_density_bins=n_density_bins,
        codec_recon=codec_recon,
        penalty=penalty,
        mask_id=mask_id,
        prior=mask_prior,
        weight=1.0,
        loss_type=loss_type,
    )
    if loss_type == "kl_gauss":
        loss_kwargs["kl_sigma"] = float(kl_sigma)
        loss_kwargs["kl_sigma_unit"] = kl_sigma_unit
    mg_loss_args = build_maskgit_loss_args(**loss_kwargs)
    configure_maskgit_loss(mg_loss_args)
    err(f"[vl-backfill] MaskGIT loss configured: penalty={penalty:.3f} "
        f"offset={density_offset} bins={n_density_bins} mask_id={mask_id}")

    # Eval data: `num_validation_sequences={"tomat": val_seqs}` + default
    # `shuffle_before_trainval_split=True` carves the last `val_seqs`
    # sequences (after a PRNGKey(0) feistel shuffle) into the val slice.
    # Persist the levanter cache on the Modal Volume so spawns past the
    # first share the (~hour-long) build cost. Per-parquet-label subdir so
    # different vocabs get separate caches.
    cache_dir = cache_dir_override or f"{MOUNT}/vl-backfill-cache/{parquet_label}"
    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt_fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source, cache_dir=cache_dir, format=prebuilt_fmt,
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        components={"tomat": component},
        block_cross_document_attention=False,
        shuffle=False,  # eval doesn't need full shuffle
        num_validation_sequences={"tomat": val_seqs},
        shuffle_before_trainval_split=True,
    )

    # Trainer config: only used here to get a device mesh + axis mappings;
    # we never call .train() — we hand-build the evaluator and invoke it
    # directly.
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32,
    )
    trainer_cfg = TrainerConfig(
        id="vl-backfill",
        seed=42,
        num_train_steps=1,
        train_batch_size=eval_batch,
        per_device_eval_parallelism=max(1, eval_batch // 8),
        tracker=(),
        mp=mp_policy,
    )
    levanter.initialize(trainer_cfg)
    compute_mapping = trainer_cfg.compute_axis_mapping
    param_mapping = trainer_cfg.parameter_axis_mapping

    model_cfg = Qwen3MaskGITConfig(
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        gradient_checkpointing=False,  # inference only
        hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16,
        intermediate_dim=4096,
    )
    key = jax.random.PRNGKey(0)

    checkpoint = f"{bucket}/results/{run_label}/checkpoints/{ckpt_leaf}/step-{step}"
    err(f"[vl-backfill] checkpoint={checkpoint}")

    with trainer_cfg.use_device_mesh():
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), compute_mapping,
        )
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[vl-backfill] loading checkpoint")
            model = load_checkpoint(model, checkpoint, subpath="model")
        model = hax.shard_with_axis_mapping(model, param_mapping)
        model = inference_mode(model, True)
        model = mp_policy.cast_to_compute(model)

        Pos = Axis("position", seq_len)
        tagged_eval_sets = data.tagged_eval_sets(Pos)
        if len(tagged_eval_sets) == 0:
            raise RuntimeError(
                "tagged_eval_sets empty after val carve-out — check "
                "num_validation_sequences / cache build."
            )
        EvalBatch = trainer_cfg.EvalBatch

        def loss_fn(m, batch):
            return _default_lm_eval_loss_fn(
                m, batch, EvalBatch=EvalBatch, mp=mp_policy,
            )

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=tagged_eval_sets,
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=trainer_cfg.device_mesh,
            axis_mapping=compute_mapping,
        )
        err(f"[vl-backfill] evaluator built; running evaluate()")
        t0 = time.time()
        # Wrap in BOTH `set_mesh` and `axis_mapping` so the dataloader's
        # `jax.jit`-ed `stack_tree` and haliax's named-jit fns alike see the
        # same GPU mesh + compute-mapping that `accum_for_batch` expects.
        with hax.partitioning.set_mesh(trainer_cfg.device_mesh):
            with hax.axis_mapping(compute_mapping):
                result = evaluator.evaluate(model)
        elapsed = time.time() - t0
        err(f"[vl-backfill] eval done: micro_avg_loss={result.micro_avg_loss:.4f} "
            f"(elapsed {elapsed:.1f}s)")

    out = {
        "step": int(step),
        "eval_loss": float(result.micro_avg_loss),
        "n_seqs": int(val_seqs),
        "model_preset": model_preset,
        "run_label": run_label,
        "ckpt_leaf": ckpt_leaf,
        "checkpoint": checkpoint,
        "elapsed_seconds": float(elapsed),
        "loss_type": loss_type,
        "kl_sigma": float(kl_sigma) if kl_sigma is not None else None,
        "mask_prior": mask_prior,
    }
    dst = f"{bucket}/eval/vl-backfill/{ckpt_leaf}/step-{step}.json"
    with fsspec.open(dst, "w") as f:
        json.dump(out, f, indent=2)
    err(f"[vl-backfill] wrote {dst}")

    # Commit the volume so the (~hour-long) levanter cache build under
    # `{MOUNT}/vl-backfill-cache/{parquet_label}/` is visible to the next
    # spawn — without this each step pays the full build cost again. Idempotent
    # if cache already exists.
    try:
        train_volume.commit()
        err(f"[vl-backfill] volume committed (cache persisted for next spawn)")
    except Exception as e:
        err(f"[vl-backfill] WARN: volume.commit() failed: {type(e).__name__}: {e}")
    return out


# ──────────────────────────────────────────────────────────────────────
# Modal function variants — registered separately because Modal's `gpu=`
# decorator argument is set at decoration time, not call time.
# ──────────────────────────────────────────────────────────────────────


# Per-call cost cap. Modal's `timeout=` is set at decoration time (no
# per-call override), so we register a short and a long variant per GPU.
# - `*_short` (600s): safe cap for sweeps — at H100×8 list ~$30/h that's
#   ~$5/fire if it hangs.
# - `*_long` (3600s): for first-time fires on a fresh parquet, where
#   building the levanter cache takes up to ~1h.
TIMEOUT_SHORT = 600
TIMEOUT_LONG = 3600


@app.function(
    gpu="H100:8", secrets=[gcp_secret], volumes={MOUNT: train_volume},
    timeout=TIMEOUT_SHORT,
)
def backfill_one_h100x8(**kwargs) -> dict:
    return _run_backfill(**kwargs)


@app.function(
    gpu="H100:8", secrets=[gcp_secret], volumes={MOUNT: train_volume},
    timeout=TIMEOUT_LONG,
)
def backfill_one_h100x8_long(**kwargs) -> dict:
    return _run_backfill(**kwargs)


@app.function(
    gpu="H200:8", secrets=[gcp_secret], volumes={MOUNT: train_volume},
    timeout=TIMEOUT_SHORT,
)
def backfill_one_h200x8(**kwargs) -> dict:
    return _run_backfill(**kwargs)


@app.function(
    gpu="H200:8", secrets=[gcp_secret], volumes={MOUNT: train_volume},
    timeout=TIMEOUT_LONG,
)
def backfill_one_h200x8_long(**kwargs) -> dict:
    return _run_backfill(**kwargs)


def get_fn(gpu: str, *, long: bool = False):
    """Return the registered Modal function matching (`--gpu`, `--long`).

    `long=True` selects the 3600s timeout variant for first-fire cache builds;
    otherwise the 600s variant caps cost-per-call.
    """
    if gpu in ("H100:8", "h100x8", "h100:8"):
        return backfill_one_h100x8_long if long else backfill_one_h100x8
    if gpu in ("H200:8", "h200x8", "h200:8"):
        return backfill_one_h200x8_long if long else backfill_one_h200x8
    raise ValueError(f"unsupported gpu={gpu!r}; expected H100:8 or H200:8")
