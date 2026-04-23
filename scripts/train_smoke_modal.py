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

# marin-levanter + friends are NOT on PyPI — they live on GitHub Releases'
# expanded_assets pages. See `marin-experiments/tiny-stories/pyproject.toml`
# for the exact list. jax[cuda12] needed for A100 training.
MARIN_FIND_LINKS = [
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-haliax-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-levanter-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-iris-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-zephyr-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-rigging-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-fray-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799",
    "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4",
]

_find_links_args = " ".join(f"--find-links {u}" for u in MARIN_FIND_LINKS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # levanter.tracker pulls in GitPython which needs the CLI
    .pip_install("uv")
    .run_commands(
        # uv pip install with all find-links URLs + --pre (marin wheels are
        # tagged as prereleases). Kept separate from Modal's native pip_install
        # because the latter takes find_links as a single str, not a list.
        f"uv pip install --system --pre {_find_links_args} "
        "marin-levanter marin-haliax marin-fray dupekit "
        "'jax[cuda12]' 'pyarrow>=15' fsspec",
    )
    .add_local_python_source("tomat")
)

app = modal.App("tomat-train-smoke", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)
wandb_secret = modal.Secret.from_name("wandb-credentials")


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
