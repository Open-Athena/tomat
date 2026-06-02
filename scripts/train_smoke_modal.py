#!/usr/bin/env python
"""Modal smoke training: 100 Qwen3 steps on the val-smoke parquet shards.

Reads `/vol/tokenized/val-smoke/*.parquet` (produced by
`scripts/tokenize_patches_modal.py`) and runs a 30M-param Qwen3
training loop directly via Levanter's `train_lm.main` — no Marin
orchestration, no iris cluster, just a single A100 Modal function.

Satisfies spec 04's done-criterion #3 ("100 steps, monotonic loss").

Usage::

    modal run scripts/train_smoke_modal.py

Outputs land on the volume at `/vol/results/smoke/`; pull back with::

    modal volume get --force tomat-rho-gga /results/smoke results/
"""

from functools import partial
from pathlib import Path
import sys

import modal

err = partial(print, file=sys.stderr)

VOLUME_NAME = "tomat-rho-gga"
MOUNT = "/vol"
# Train volume holds train-full-* labels; small `tomat-rho-gga` only has val.
TRAIN_VOLUME_NAME = "tomat-rho-gga-train"

# Model presets — mirror marin/train_tomat_tpu.py for matched a2a comparison.
MODEL_PRESETS = {
    "30M":  dict(hidden_dim=512,  num_layers=6,  num_heads=4,  num_kv_heads=4,  intermediate_dim=2048),
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    "1B":   dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}

# marin-levanter + friends from Open-Athena/marin (our fork) — installs
# from git so the iris-jax-init + flash-None patches reach the Modal image.
# Upstream marin-community releases (find-links) shipped a Levanter that
# crashes MaskGIT's bidir mask=None case on GPU. Pin keeps the v4 fire's
# image deterministic; bump with each marin SHA bump in marin/pyproject.toml.
OA_MARIN_SHA = "97eea237598bfe0d0af1143dce92c0c00526a8f0"
OA_MARIN_GIT = f"git+https://github.com/Open-Athena/marin.git@{OA_MARIN_SHA}"
_marin_git_pkgs = " ".join(
    f"'marin-{p} @ {OA_MARIN_GIT}#subdirectory=lib/{p}'"
    for p in ("haliax", "levanter", "fray")
)

# dupekit + kitoken still come from upstream (no fork divergence there).
EXTRA_FIND_LINKS = [
    "https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799",
    "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4",
]
_extra_find_links_args = " ".join(f"--find-links {u}" for u in EXTRA_FIND_LINKS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # levanter.tracker pulls in GitPython which needs the CLI
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --pre {_extra_find_links_args} "
        f"{_marin_git_pkgs} dupekit "
        "'jax[cuda12]' 'pyarrow>=15' fsspec",
    )
    .add_local_python_source("tomat")
    # marin/ contains qwen3_density (density-loss config used for real
    # tomat training); iris bundles it flat on TPU, here we mount it as a
    # proper package so `from marin.qwen3_density import ...` works.
    .add_local_python_source("marin")
)

app = modal.App("tomat-train-smoke", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)
wandb_secret = modal.Secret.from_name("wandb-credentials")
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")  # for reading LMQ codec from gs://


@app.function(gpu="A100", volumes={MOUNT: volume}, secrets=[wandb_secret], timeout=28800)  # 8h (5k bs=32 at 3.2 s/step ≈ 4.4 h; leaves slack)
def train_smoke(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
) -> dict:
    """Run `levanter.main.train_lm.main` with a prebuilt-parquet config."""
    return _train_smoke_impl(steps, batch_size, seed, label, results_label)


@app.function(gpu="A100:2", volumes={MOUNT: volume}, secrets=[wandb_secret], timeout=28800)
def train_smoke_2gpu(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
) -> dict:
    """A100:2 — same body as train_smoke_4gpu. Use bs=32 (per-device bs=16)
    to stay under the attention-matrix OOM threshold without TE."""
    return _train_smoke_impl(steps, batch_size, seed, label, results_label)


@app.function(gpu="A100:8", volumes={MOUNT: volume}, secrets=[wandb_secret], timeout=28800)
def train_smoke_8gpu(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
) -> dict:
    """A100:8 — same body as train_smoke_4gpu. Use bs=128 (per-device bs=16)."""
    return _train_smoke_impl(steps, batch_size, seed, label, results_label)


@app.function(gpu="A100:4", volumes={MOUNT: volume}, secrets=[wandb_secret], timeout=28800)
def train_smoke_4gpu(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
) -> dict:
    """Same as train_smoke but on 4× A100 (intra-node). Levanter auto-shards
    the batch across JAX's default 'data' mesh axis; no code change needed.
    Use a 4× bigger `--batch-size` than the single-GPU run to keep per-device
    batch constant — that way MFU/GPU stays comparable and wall-time drops
    ~3-3.5× (NCCL tax ≈ 10-15%)."""
    return _train_smoke_impl(steps, batch_size, seed, label, results_label)


def _train_smoke_impl(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
) -> dict:
    """Shared body — identical Levanter config under both A100:1 and A100:4."""
    import json
    from levanter.data.text import (
        DatasetComponent,
        LmDataConfig,
        PrebuiltLmDatasetFormat,
        UrlDatasetSourceConfig,
    )
    from levanter.data.text.datasets import BlockShuffleConfig
    from datetime import timedelta
    from levanter.checkpoint import CheckpointerConfig
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.main.train_lm import TrainLmConfig, main as train_lm_main
    from levanter.models.qwen import Qwen3Config
    from levanter.optim import AdamConfig
    from levanter.tracker.json_logger import JsonLoggerConfig
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig

    # Ensure we see the latest volume state (in case a separate tokenize
    # function just wrote new parquet; without this the container can keep a
    # stale view from function-start time).
    volume.reload()

    parquet_dir = f"{MOUNT}/tokenized/{label}"
    results_dir = f"{MOUNT}/results/{results_label}"
    cache_dir = f"{results_dir}/cache"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Parallel tokenize writes per-worker subdirs (worker-00/, worker-01/, …).
    # Detect and glob all their shards; pick any one worker's meta.json since
    # vocab + patch_size + codec are shared across workers.
    worker_dirs = sorted(Path(parquet_dir).glob("worker-*"))
    is_parallel = bool(worker_dirs)
    if is_parallel:
        parquet_glob = f"{parquet_dir}/worker-*/*.parquet"
        meta_path = f"{worker_dirs[0]}/meta.json"
    else:
        parquet_glob = f"{parquet_dir}/*.parquet"
        meta_path = f"{parquet_dir}/meta.json"

    # Keep any existing dataset cache from a prior run — Levanter's cache is
    # content-hash-keyed, so if the parquet or hyperparams change the
    # effective path changes and it rebuilds automatically. Removed the
    # earlier unconditional nuke because a resume-run would otherwise
    # rebuild the ~1 min cache every time we pick up a checkpoint.

    # Read vocab size from the tokenizer's meta.json so a codec swap at
    # preprocessing doesn't silently misalign model and data.
    meta = json.loads(Path(meta_path).read_text())
    # For parallel runs, worker-0's meta.n_materials is its slice only.
    # Sum across all workers to report the true dataset size.
    if is_parallel:
        total_mats = 0
        total_rows = 0
        for wd in worker_dirs:
            m = json.loads((wd / "meta.json").read_text())
            total_mats += m["n_materials"]
            total_rows += m["total_rows"]
        meta = {**meta, "n_materials": total_mats, "total_rows": total_rows}
    vocab_size = meta["vocab"]["total_size"]
    err(f"[smoke] vocab_size={vocab_size}, patch_size={meta['patch_size']}, "
        f"codec={meta['density_codec_name']}, rows={meta['total_rows']}")

    # Data: single component, train-only for smoke. No validation_urls so
    # Levanter doesn't build a val cache and wire TaggedEvaluator (which calls
    # PassthroughTokenizer.encode(".") and crashes — BPB computation expects
    # a real tokenizer).
    source = UrlDatasetSourceConfig(
        train_urls=[parquet_glob],
    )
    prebuilt_fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=prebuilt_fmt,
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        components={"tomat": component},
        # No validation set for smoke — Levanter's TaggedEvaluator init calls
        # tokenizer.encode(".") to compute bytes/token for BPB, which
        # PassthroughTokenizer can't service (non-integer input).
        block_cross_document_attention=False,
    )

    model = Qwen3Config(
        max_seq_len=8192,
        hidden_dim=512,
        intermediate_dim=2048,
        num_heads=4,
        num_kv_heads=4,
        num_layers=6,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
    )

    # Project per preprocessing axis: loss curves of two runs in the same W&B
    # project are guaranteed to share codec + patch_size (so they share the
    # same parquet shards, vocab, sequence length). Mixing (codec, P) combos
    # would put runs with different train-loss meanings on the same plot.
    # Group captures the training-side sampling axes (M = patches per material,
    # N = shuffle buffer) — relevant for the sweep in spec 04.
    M = meta["patches_per_material"]
    # Shuffle buffer N isn't plumbed into the smoke config yet; emit a literal
    # placeholder so the group name composes cleanly once we do. Update when
    # `BlockShuffleConfig.window_blocks` becomes a real knob.
    N = "default"
    project = f"tomat-{meta['density_codec_name']}-P{meta['patch_size']}"
    group = f"M{M}-N{N}"

    # Deterministic run_id so a re-invocation after a Modal function timeout
    # lands in the same W&B run (appends to the curve) AND finds the existing
    # Levanter checkpoints under results_label/checkpoints/<run_id>/.
    # Excludes `steps` deliberately — extending a 1k-step run into a 5k-step
    # run should be the same run, not a new one.
    run_id = f"{results_label}-bs{batch_size}-seed{seed}"

    # Dual-tracker: W&B for browsable runs + JSON-to-stdout as a fallback that
    # makes the loss curve visible in the Modal log even if W&B init hiccups.
    is_smoke = meta["n_materials"] <= 200
    trackers = (
        WandbConfig(
            id=run_id,
            resume="allow",
            project=project,
            group=group,
            tags=[
                "smoke" if is_smoke else "scale",
                f"mats{meta['n_materials']}",
                f"bs{batch_size}",
                f"seed{seed}",
            ],
            save_code=False,  # Modal image doesn't carry the repo tree; skip wandb's code-snapshot
        ),
        JsonLoggerConfig(),
    )

    # Checkpoint to volume so timeout/preempt → rerun resumes cleanly.
    # `append_run_id_to_base_path=True` means actual dir is
    # f"{results_dir}/checkpoints/{run_id}/"; Levanter auto-resumes from the
    # latest step when `trainer.initial_state` finds one there.
    checkpointer = CheckpointerConfig(
        base_path=f"{results_dir}/checkpoints",
        save_interval=timedelta(minutes=10),
        # Keep permanent snapshots every 1000 steps (overrides 15-min rolling).
        keep=[{"every": 1000}],
    )

    trainer = TrainerConfig(
        id=run_id,
        seed=seed,
        num_train_steps=steps,
        train_batch_size=batch_size,
        steps_per_eval=max(steps // 2, 1),
        tracker=trackers,
        checkpointer=checkpointer,
    )

    optimizer = AdamConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup=0.1,
        min_lr_ratio=0.0,
        beta1=0.9,
        beta2=0.95,
    )

    config = TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        train_seq_len=8192,
    )

    err(f"[smoke] calling levanter.main.train_lm.main …")
    train_lm_main(config)
    err(f"[smoke] done")

    volume.commit()
    return {"results_dir": results_dir, "vocab_size": vocab_size}


@app.local_entrypoint()
def main(
    steps: int = 100,
    batch_size: int = 8,
    seed: int = 42,
    label: str = "val-smoke",
    results_label: str = "smoke",
) -> None:
    err(f"[modal] smoke train: {steps} steps, batch={batch_size}, seed={seed}")
    result = train_smoke.remote(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        label=label,
        results_label=results_label,
    )
    err(f"[modal] done: results at {result['results_dir']} on volume {VOLUME_NAME}")
    err(f"[modal] pull with: modal volume get --force {VOLUME_NAME} /results/{results_label} results/")


@app.local_entrypoint()
def main_4gpu(
    steps: int = 5000,
    batch_size: int = 128,  # 4× single-GPU default to keep per-device bs constant
    seed: int = 42,
    label: str = "val-full",
    results_label: str = "val-full-5k-bs128-4gpu",
) -> None:
    err(f"[modal] A100:4 smoke train: {steps} steps, nominal batch={batch_size} "
        f"(per-device {batch_size // 4}), seed={seed}")
    result = train_smoke_4gpu.remote(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        label=label,
        results_label=results_label,
    )
    err(f"[modal] done: results at {result['results_dir']} on volume {VOLUME_NAME}")


@app.local_entrypoint()
def main_2gpu(
    steps: int = 5000,
    batch_size: int = 32,  # per-device bs=16 to stay OOM-safe without TE
    seed: int = 42,
    label: str = "val-full",
    results_label: str = "val-full-5k-bs32-2gpu",
) -> None:
    err(f"[modal] A100:2 train: {steps} steps, nominal batch={batch_size} "
        f"(per-device {batch_size // 2}), seed={seed}")
    result = train_smoke_2gpu.remote(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        label=label,
        results_label=results_label,
    )
    err(f"[modal] done: results at {result['results_dir']} on volume {VOLUME_NAME}")


@app.local_entrypoint()
def main_8gpu(
    steps: int = 5000,
    batch_size: int = 128,  # per-device bs=16
    seed: int = 42,
    label: str = "val-full",
    results_label: str = "val-full-5k-bs128-8gpu",
) -> None:
    err(f"[modal] A100:8 train: {steps} steps, nominal batch={batch_size} "
        f"(per-device {batch_size // 8}), seed={seed}")
    result = train_smoke_8gpu.remote(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        label=label,
        results_label=results_label,
    )
    err(f"[modal] done: results at {result['results_dir']} on volume {VOLUME_NAME}")


# ============================================================================
# Modal-vs-TPU MFU bakeoff (probe runs on train-full volume; H100 GPUs)
# ============================================================================

train_volume = modal.Volume.from_name(TRAIN_VOLUME_NAME)


def _train_bakeoff_impl(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
    model_preset: str,
    parquet_root: str,
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",  # match TPU recipe; FP32 was the v1 trap
    lmq_path: str | None = None,  # gs://...codec.npz; enables EMD-density loss
    density_loss_type: str = "emd",  # "emd" (W₁) or "l1" (legacy)
    density_only: bool = True,
    maskgit_mode: bool = False,
    maskgit_prior: str = "absorbing",
    maskgit_loss_type: str = "ce",
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,  # None = disable mid-training eval (old default)
    # Shuffle params. TPU reads these from `meta.patches_per_material` (default
    # 64 for train-full-v3) + the `TOMAT_SHUFFLE_*` env vars (see
    # `marin/train_tomat_tpu.py:651-664`); the Modal side takes them as kwargs.
    # The right `io_block_size` is `patches_per_material` so one io_block =
    # exactly one material's-worth of consecutive patches, and intra-window
    # BlockShuffle interleaves materials rather than slicing them.
    # `window_blocks=0` disables BlockShuffle (Levanter then uses its built-in
    # default — the source of the 1024-step sawtooth; see spec 44).
    shuffle_io_block_size: int = 64,
    shuffle_window_blocks: int = 1024,
) -> dict:
    """200M-or-other preset training body for the bakeoff probe.

    Diverges from `_train_smoke_impl` in three ways: (1) reads a
    parameterized model preset, (2) targets a parameterized parquet root
    (so the train-full volume can be used), (3) keeps eval and BPB
    machinery off — we want clean MFU numbers without TaggedEvaluator
    side effects. Also exposes ``gradient_checkpointing`` and
    ``compute_dtype`` knobs for the v2 sweep.

    When ``lmq_path`` is provided, mirrors train_tomat_tpu.py's
    density-loss wiring: loads the LMQ codec, builds density_loss_args,
    runs configure_density_loss, and swaps Qwen3Config →
    Qwen3DensityConfig. This is what real (non-MFU-probe) training
    needs.

    When ``maskgit_mode=True`` (mutually exclusive with the density-EMD
    branch via ``density_loss_type``/``density_only`` — we forbid both
    set together), mirrors the TPU trainer's MG wiring: bumps vocab by 1
    for [MASK], builds MaskGITLossArgs, runs configure_maskgit_loss, and
    swaps Qwen3Config → Qwen3MaskGITConfig. Requires ``lmq_path``.
    """
    import json
    import os
    import jax.numpy as jnp
    import jmp
    import numpy as np
    from levanter.data.text import (
        DatasetComponent,
        LmDataConfig,
        PrebuiltLmDatasetFormat,
        UrlDatasetSourceConfig,
    )
    from levanter.data.text.datasets import BlockShuffleConfig
    from datetime import timedelta
    from levanter.checkpoint import CheckpointerConfig
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.main.train_lm import TrainLmConfig, main as train_lm_main
    from levanter.models.qwen import Qwen3Config
    from levanter.optim import AdamConfig
    from levanter.tracker.json_logger import JsonLoggerConfig
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig

    # Write GCP service-account JSON from env to a file + point ADC at it.
    # No-op if the secret isn't in env (e.g. tests).
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        sa_path = "/tmp/gcp-sa.json"
        with open(sa_path, "w") as f:
            f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    # MaskGIT / density-EMD mutual exclusion. MG sets a different loss path +
    # different model subclass; combining is undefined.
    if maskgit_mode:
        if maskgit_prior not in ("cosine", "uniform", "high", "absorbing"):
            raise ValueError(
                f"maskgit_prior must be cosine/uniform/high/absorbing, got "
                f"{maskgit_prior!r}"
            )
        if maskgit_loss_type not in ("ce", "ce_emd", "emd"):
            raise ValueError(
                f"maskgit_loss_type must be ce/ce_emd/emd, got "
                f"{maskgit_loss_type!r}"
            )
        if not lmq_path:
            raise ValueError(
                "maskgit_mode=True requires lmq_path (codec needed to identify "
                "density positions for MASK_ID + EMD term)"
            )
        # The MG loss path replaces the density-EMD path entirely; reject
        # configurations that ask for both at once.
        if density_only is False:
            raise ValueError(
                "maskgit_mode=True is mutually exclusive with density_only=False "
                "(density-EMD additive recipe). MG defines its own loss."
            )
        # Set the env var the MG model checks at trace time (the runtime guard
        # in Qwen3MaskGITLMHeadModel.compute_next_token_loss errors if
        # TOMAT_MG_MODE=1 but _MASKGIT_LOSS_ARGS is None — keeps the
        # configure_maskgit_loss() call honest).
        os.environ["TOMAT_MG_MODE"] = "1"
        os.environ["TOMAT_MG_MASK_PRIOR"] = maskgit_prior
        os.environ["TOMAT_MG_LOSS_TYPE"] = maskgit_loss_type

    # Install gcsfs at runtime (skipped in image build because adding it to the
    # image-level pip line busts the layer cache and re-resolves the marin-*
    # wheels — currently broken on the latest tags with a version cycle).
    # ~5-10s; needed for the gs:// codec read *and* GCS checkpoint writes.
    import subprocess
    subprocess.run(["pip", "install", "--quiet", "gcsfs"], check=True)

    train_volume.reload()

    parquet_dir = parquet_root
    results_dir = f"{MOUNT}/results/{results_label}"
    cache_dir = f"{results_dir}/cache"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    worker_dirs = sorted(Path(parquet_dir).glob("worker-*"))
    if not worker_dirs:
        raise FileNotFoundError(f"no worker-*/ shards under {parquet_dir}")
    parquet_glob = f"{parquet_dir}/worker-*/*.parquet"
    meta_path = f"{worker_dirs[0]}/meta.json"
    meta = json.loads(Path(meta_path).read_text())
    vocab_size = meta["vocab"]["total_size"]
    err(f"[bakeoff] vocab_size={vocab_size}, patch_size={meta['patch_size']}, "
        f"codec={meta['density_codec_name']}, model={model_preset}")

    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt_fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source, cache_dir=cache_dir, format=prebuilt_fmt,
    )
    # See `train_bakeoff_h200x8` signature for the shuffle param semantics.
    # `window_blocks=0` → disable BlockShuffle (use Levanter's noisy default).
    if shuffle_window_blocks > 0:
        shuffle_cfg: BlockShuffleConfig | bool = BlockShuffleConfig(
            io_block_size=shuffle_io_block_size,
            window_blocks=shuffle_window_blocks,
        )
        err(f"[bakeoff] shuffle: BlockShuffle(io_block_size={shuffle_io_block_size}, "
            f"window_blocks={shuffle_window_blocks}) — window covers "
            f"{shuffle_io_block_size * shuffle_window_blocks} seqs "
            f"= {(shuffle_io_block_size * shuffle_window_blocks) // batch_size} steps "
            f"at BS={batch_size}")
    else:
        shuffle_cfg = False
        err(f"[bakeoff] shuffle: OFF (window_blocks=0)")
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        components={"tomat": component},
        block_cross_document_attention=False,
        shuffle=shuffle_cfg,
    )

    preset = MODEL_PRESETS[model_preset]
    if lmq_path:
        # Codec load + density-offset compute is shared between the density
        # and MG branches; pull it out.
        import fsspec
        with fsspec.open(lmq_path, "rb") as f:
            codec_data = np.load(f, allow_pickle=True)
            codec_boundaries = np.asarray(codec_data["boundaries"], dtype=np.float32)
            codec_recon = np.asarray(codec_data["recon_points"], dtype=np.float32)
            codec_clip = float(codec_data["clip_max"])

        # Compute density vocab offset per PatchVocab layout. Match the TPU
        # trainer's `_compute_density_offset`: specials are 18 by default but
        # bump to 20 if `[LATTICE_START]` is in the vocab specials.
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

        if maskgit_mode:
            # MaskGIT path: bump vocab by 1 for [MASK], build MG loss args,
            # swap to Qwen3MaskGITConfig. Mirrors train_tomat_tpu.py:578-817.
            from marin.qwen3_density import (
                Qwen3MaskGITConfig,
                build_maskgit_loss_args,
                configure_maskgit_loss,
            )
            mask_id = vocab_size
            vocab_size = vocab_size + 1
            err(f"[bakeoff] MaskGIT mode: MASK_ID={mask_id}, "
                f"new vocab_size={vocab_size}, prior={maskgit_prior}, "
                f"loss_type={maskgit_loss_type}")

            import haliax as hax
            Vocab_mg = hax.Axis("vocab", vocab_size)
            mg_loss_args = build_maskgit_loss_args(
                Vocab=Vocab_mg,
                density_offset=density_offset,
                n_density_bins=n_density_bins,
                codec_recon=codec_recon,
                penalty=penalty,
                mask_id=mask_id,
                prior=maskgit_prior,
                weight=1.0,
                loss_type=maskgit_loss_type,
            )
            configure_maskgit_loss(mg_loss_args)
            err(f"[bakeoff] MaskGIT loss configured: penalty={penalty:.3f} "
                f"offset={density_offset} bins={n_density_bins}")
            model_config_cls = Qwen3MaskGITConfig
            # Re-bind data config: vocab_size grew by 1.
            data = LmDataConfig(
                tokenizer="passthrough",
                vocab_size=vocab_size,
                cache_dir=cache_dir,
                components={"tomat": component},
                block_cross_document_attention=False,
                # Re-use the shuffle_cfg computed for the main `data` above.
                shuffle=shuffle_cfg,
            )
        else:
            # Mirror train_tomat_tpu.py density-loss setup.
            from marin.qwen3_density import (
                Qwen3DensityConfig,
                build_density_loss_args,
                configure_density_loss,
            )

            import haliax as hax
            Vocab = hax.Axis("vocab", vocab_size)
            density_loss_args = build_density_loss_args(
                Vocab=Vocab,
                density_offset=density_offset,
                n_density_bins=n_density_bins,
                codec_recon=codec_recon,
                penalty=penalty,
                weight=1.0,
                mode="add",
                loss_type=density_loss_type,
                density_only=density_only,
            )
            configure_density_loss(density_loss_args)
            err(f"[bakeoff] density loss: type={density_loss_type} only={density_only} "
                f"offset={density_offset} bins={n_density_bins} penalty={penalty:.3f}")
            model_config_cls = Qwen3DensityConfig
    else:
        if maskgit_mode:
            raise ValueError(
                "maskgit_mode=True requires lmq_path (already enforced above; "
                "guard duplicated for clarity)"
            )
        model_config_cls = Qwen3Config
    model = model_config_cls(
        max_seq_len=8192,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        gradient_checkpointing=gradient_checkpointing,
        **preset,
    )

    # Allow caller to override the W&B project (e.g. land MG-bakeoff runs in
    # the same project as their TPU counterpart for direct comparison —
    # `tomat-lmq-P19` is the canonical P19/v3 project).
    project = wandb_project or f"tomat-{meta['density_codec_name']}-P{meta['patch_size']}"
    group = f"bakeoff-modal-{model_preset}"
    run_id = f"{results_label}-bs{batch_size}-seed{seed}"
    tags = ["bakeoff", "modal", f"model{model_preset}",
            f"bs{batch_size}", f"seed{seed}"]
    if maskgit_mode:
        tags += ["maskgit", f"mg-prior-{maskgit_prior}", f"mg-loss-{maskgit_loss_type}"]
    trackers = (
        WandbConfig(
            entity="open-athena",  # corporate team; new tomat runs go here.
            id=run_id, resume="allow",
            project=project, group=group,
            tags=tags,
            save_code=False,
        ),
        JsonLoggerConfig(),
    )
    # Checkpoints → GCS, not the Modal volume: durable across abrupt container
    # death (a volume only persists writes on a clean exit / explicit commit),
    # and readable by the NMAE eval watchdog without Modal-volume access. Same
    # `<results>/<RL>/checkpoints/<run_id>/step-N` layout the TPU runs use.
    ckpt_base = f"gs://marin-eu-west4/tomat/results/{results_label}/checkpoints"
    checkpointer = CheckpointerConfig(
        base_path=ckpt_base,
        save_interval=timedelta(minutes=10),
        keep=[{"every": 1000}],
    )
    compute_jdtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[compute_dtype]
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=compute_jdtype,
        output_dtype=jnp.float32,
    )
    # `steps_per_eval=None` keeps the historical behavior (eval only at the
    # final step). Pass an explicit value to surface VL mid-training — cheap
    # for MG-CE on the val cache (one bidi forward per eval period).
    spe = steps_per_eval if steps_per_eval is not None else max(steps, 1)
    trainer = TrainerConfig(
        id=run_id, seed=seed,
        num_train_steps=steps, train_batch_size=batch_size,
        steps_per_eval=spe,
        tracker=trackers, checkpointer=checkpointer,
        mp=mp_policy,
    )
    optimizer = AdamConfig(
        learning_rate=3e-4, weight_decay=0.0,
        warmup=0.05, min_lr_ratio=0.0,
        beta1=0.9, beta2=0.95,
    )
    config = TrainLmConfig(
        data=data, trainer=trainer, model=model, optimizer=optimizer,
        train_seq_len=8192,
    )
    err(f"[bakeoff] calling levanter.main.train_lm.main …")
    train_lm_main(config)
    err(f"[bakeoff] done")

    train_volume.commit()
    return {"results_dir": results_dir, "vocab_size": vocab_size}


@app.function(
    gpu="H100:8",
    volumes={MOUNT: train_volume},
    secrets=[wandb_secret, gcp_secret],
    timeout=72000,  # 20h — 10k bs=256 at ~6 s/step ≈ 16.7h (+ slack). Don't exceed Modal's 24h cap.
)
def train_bakeoff_h100x8(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
    model_preset: str,
    parquet_root: str,
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",
    lmq_path: str | None = None,
    density_loss_type: str = "emd",
    density_only: bool = True,
) -> dict:
    """8× H100 SXM5 bakeoff probe; per-GPU BS = batch_size / 8."""
    return _train_bakeoff_impl(
        steps, batch_size, seed, label, results_label, model_preset, parquet_root,
        gradient_checkpointing=gradient_checkpointing,
        compute_dtype=compute_dtype,
        lmq_path=lmq_path,
        density_loss_type=density_loss_type,
        density_only=density_only,
    )


@app.local_entrypoint()
def main_bakeoff_h100x8(
    steps: int = 1000,
    batch_size: int = 128,        # matches TPU recipe; per-GPU = 16
    seed: int = 42,
    label: str = "train-full-v3",
    results_label: str = "bakeoff-200M-h100x8-1k-v3",
    model_preset: str = "200M",
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",
    lmq_path: str = "",  # empty = vanilla CE; otherwise gs://.../lmq-v2-16k.npz
    density_loss_type: str = "emd",
    density_only: bool = True,
) -> None:
    parquet_root = f"{MOUNT}/tokenized/{label}"
    err(f"[modal] H100:8 bakeoff: {model_preset} × {steps} steps, "
        f"bs={batch_size} (per-GPU {batch_size // 8}), "
        f"ckpt={int(gradient_checkpointing)}, dtype={compute_dtype}, label={label}")
    err(f"[modal] reading parquets from {parquet_root} on volume {TRAIN_VOLUME_NAME}")
    # Use .spawn() so the function survives local disconnects under `modal run
    # --detach`. With .remote() the function is canceled when the local
    # subprocess exits (Modal warns about this). .spawn() returns a
    # FunctionCall handle and detaches immediately.
    call = train_bakeoff_h100x8.spawn(
        steps=steps, batch_size=batch_size, seed=seed,
        label=label, results_label=results_label,
        model_preset=model_preset, parquet_root=parquet_root,
        gradient_checkpointing=gradient_checkpointing,
        compute_dtype=compute_dtype,
        lmq_path=lmq_path or None,
        density_loss_type=density_loss_type,
        density_only=density_only,
    )
    err(f"[modal] spawned call_id={call.object_id} — function runs independently of this process")


@app.function(
    gpu="H200:8",
    volumes={MOUNT: train_volume},
    secrets=[wandb_secret, gcp_secret],
    timeout=72000,  # 20h — matches h100x8; MG 10k @ bs=128 should fit comfortably.
)
def train_bakeoff_h200x8(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
    model_preset: str,
    parquet_root: str,
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",
    lmq_path: str | None = None,
    density_loss_type: str = "emd",
    density_only: bool = True,
    maskgit_mode: bool = False,
    maskgit_prior: str = "absorbing",
    maskgit_loss_type: str = "ce",
    wandb_project: str | None = None,
    steps_per_eval: int | None = None,
    shuffle_io_block_size: int = 64,
    shuffle_window_blocks: int = 1024,
) -> dict:
    """8× H200 bakeoff probe; same SM as H100, 1.4× HBM bandwidth."""
    return _train_bakeoff_impl(
        steps, batch_size, seed, label, results_label, model_preset, parquet_root,
        gradient_checkpointing=gradient_checkpointing,
        compute_dtype=compute_dtype,
        lmq_path=lmq_path,
        density_loss_type=density_loss_type,
        density_only=density_only,
        maskgit_mode=maskgit_mode,
        maskgit_prior=maskgit_prior,
        maskgit_loss_type=maskgit_loss_type,
        wandb_project=wandb_project,
        steps_per_eval=steps_per_eval,
        shuffle_io_block_size=shuffle_io_block_size,
        shuffle_window_blocks=shuffle_window_blocks,
    )


@app.local_entrypoint()
def main_bakeoff_h200x8(
    steps: int = 500,
    batch_size: int = 128,
    seed: int = 42,
    label: str = "train-full-v3",
    results_label: str = "bakeoff-200M-h200x8-500-v3",
    model_preset: str = "200M",
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",
    lmq_path: str = "",  # empty = vanilla CE; otherwise gs://.../lmq-v2-16k.npz
    density_loss_type: str = "emd",
    density_only: bool = True,
    maskgit_mode: bool = False,
    maskgit_prior: str = "absorbing",
    maskgit_loss_type: str = "ce",
    wandb_project: str = "",
) -> None:
    parquet_root = f"{MOUNT}/tokenized/{label}"
    # MG mode requires an LMQ codec; default to the canonical lmq-v2-16k.npz
    # if the caller didn't specify one (matches what `tomat ... train ...`
    # uses on TPU — see LMQ_PATH constant in the tomat CLI).
    if maskgit_mode and not lmq_path:
        lmq_path = "gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz"
    err(f"[modal] H200:8 bakeoff: {model_preset} × {steps} steps, "
        f"bs={batch_size} (per-GPU {batch_size // 8}), "
        f"ckpt={int(gradient_checkpointing)}, dtype={compute_dtype}, label={label}")
    if maskgit_mode:
        err(f"[modal] MaskGIT mode: prior={maskgit_prior}, "
            f"loss_type={maskgit_loss_type}, lmq_path={lmq_path}")
    # Use .spawn() under `modal run --detach`; falls back to a blocking
    # .remote() if invoked without --detach so a backgrounded `modal run`
    # call survives at least as long as the local process. Toggle via
    # TOMAT_MODAL_SPAWN=0 to force the blocking path.
    import os as _os
    use_spawn = _os.environ.get("TOMAT_MODAL_SPAWN", "1") == "1"
    if use_spawn:
        call = train_bakeoff_h200x8.spawn(
            steps=steps, batch_size=batch_size, seed=seed,
            label=label, results_label=results_label,
            model_preset=model_preset, parquet_root=parquet_root,
            gradient_checkpointing=gradient_checkpointing,
            compute_dtype=compute_dtype,
            lmq_path=lmq_path or None,
            density_loss_type=density_loss_type,
            density_only=density_only,
            maskgit_mode=maskgit_mode,
            maskgit_prior=maskgit_prior,
            maskgit_loss_type=maskgit_loss_type,
            wandb_project=wandb_project or None,
        )
        err(f"[modal] spawned call_id={call.object_id} — function runs independently of this process")
    else:
        err(f"[modal] using blocking .remote() (TOMAT_MODAL_SPAWN=0); local process must stay alive")
        result = train_bakeoff_h200x8.remote(
            steps=steps, batch_size=batch_size, seed=seed,
            label=label, results_label=results_label,
            model_preset=model_preset, parquet_root=parquet_root,
            gradient_checkpointing=gradient_checkpointing,
            compute_dtype=compute_dtype,
            lmq_path=lmq_path or None,
            density_loss_type=density_loss_type,
            density_only=density_only,
            maskgit_mode=maskgit_mode,
            maskgit_prior=maskgit_prior,
            maskgit_loss_type=maskgit_loss_type,
            wandb_project=wandb_project or None,
        )
        err(f"[modal] done: {result['results_dir']} (vocab={result['vocab_size']})")


@app.function(
    gpu="B200:8",
    volumes={MOUNT: train_volume},
    secrets=[wandb_secret, gcp_secret],
    timeout=14400,
)
def train_bakeoff_b200x8(
    steps: int,
    batch_size: int,
    seed: int,
    label: str,
    results_label: str,
    model_preset: str,
    parquet_root: str,
    gradient_checkpointing: bool = True,
    compute_dtype: str = "bfloat16",
    lmq_path: str | None = None,
    density_loss_type: str = "emd",
    density_only: bool = True,
) -> dict:
    """8× B200 bakeoff probe; Blackwell, ~2.3× H100 BF16 FLOPS + larger HBM."""
    return _train_bakeoff_impl(
        steps, batch_size, seed, label, results_label, model_preset, parquet_root,
        gradient_checkpointing=gradient_checkpointing,
        compute_dtype=compute_dtype,
        lmq_path=lmq_path,
        density_loss_type=density_loss_type,
        density_only=density_only,
    )


@app.local_entrypoint()
def main_bakeoff_b200x8(
    steps: int = 500,
    batch_size: int = 256,
    seed: int = 42,
    label: str = "train-full-v3",
    results_label: str = "bakeoff-200M-b200x8-500-v3",
    model_preset: str = "200M",
    gradient_checkpointing: bool = False,
    compute_dtype: str = "bfloat16",
) -> None:
    parquet_root = f"{MOUNT}/tokenized/{label}"
    err(f"[modal] B200:8 bakeoff: {model_preset} × {steps} steps, "
        f"bs={batch_size} (per-GPU {batch_size // 8}), "
        f"ckpt={int(gradient_checkpointing)}, dtype={compute_dtype}, label={label}")
    result = train_bakeoff_b200x8.remote(
        steps=steps, batch_size=batch_size, seed=seed,
        label=label, results_label=results_label,
        model_preset=model_preset, parquet_root=parquet_root,
        gradient_checkpointing=gradient_checkpointing,
        compute_dtype=compute_dtype,
        lmq_path=lmq_path or None,
        density_loss_type=density_loss_type,
        density_only=density_only,
    )
    err(f"[modal] done: {result['results_dir']} (vocab={result['vocab_size']})")


