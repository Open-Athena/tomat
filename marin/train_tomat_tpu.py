#!/usr/bin/env python
"""Train tomat's Qwen3 patch-LM on Marin's shared TPU cluster.

Mirrors `scripts/train_smoke_modal.py` but targets GCS for data +
checkpoints (Marin's standard data path). Run via:

    cd marin
    uv run iris --cluster=marin job run \\
        --tpu v6e-4 \\
        --env-vars WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python train_tomat_tpu.py

Env-var knobs:
    TOMAT_LABEL           data label under gs://.../tokenized/ (default val-full)
    TOMAT_STEPS           num train steps (default 1000)
    TOMAT_BATCH_SIZE      nominal (total) batch size (default 128)
    TOMAT_SEED            seed (default 42)
    TOMAT_RESULTS_LABEL   overrides W&B run id / checkpoint namespace
    TOMAT_MODEL           model size preset: "30M" (default) or "200M"
    TOMAT_VAL_SEQS        num validation sequences held out (default 0 = no val)
    TOMAT_STEPS_PER_EVAL  eval cadence; default steps // 4 when val is on

Prereqs:
- `gs://marin-eu-west4/tomat/tokenized/<label>/worker-*/*.parquet` populated
- ADC refreshed for `ryan.williams@openathena.ai` on hai-gcp-models.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

# Multihost-capable JAX init. On v6e-16+ Marin spins up 4+ VM replicas; Levanter's
# `WandbConfig.init` calls `jax_utils.multihost_broadcast_sync` before
# `jax.distributed.initialize()` fires internally, which crashes single-process-
# view code paths. We initialize up-front so later multihost calls work. On
# single-host v6e-4 / v6e-8 this is a no-op (one process, auto-discovery).
import jax
try:
    jax.distributed.initialize()
    print(f"[tomat-tpu] jax.distributed.initialize() done "
          f"(process_index={jax.process_index()}/{jax.process_count()})")
except Exception as e:
    # Single-host envs may raise if no coordinator can be discovered — that's
    # fine, Levanter's own init path handles single-process. Log and continue.
    print(f"[tomat-tpu] jax.distributed.initialize() skipped: {type(e).__name__}: {e}")

# Monkey-patch PassthroughTokenizer.encode to handle non-numeric input — Levanter's
# BPB-computation path calls `tokenizer.encode(".")` to estimate bytes-per-token,
# which crashes on the default PassthroughTokenizer (tries `int(".")`). Fallback
# to a benign [0] so BPB math runs (values are meaningless for integer-only
# tokenizer; train/eval loss stays correct).
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
_orig_passthrough_encode = PassthroughTokenizer.encode

def _safe_passthrough_encode(self, text, *, add_special_tokens=False):
    try:
        return _orig_passthrough_encode(self, text, add_special_tokens=add_special_tokens)
    except ValueError:
        return [0]

PassthroughTokenizer.encode = _safe_passthrough_encode

import jax.numpy as jnp
import jmp

from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig, main as train_lm_main
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

BUCKET = "gs://marin-eu-west4/tomat"


MODEL_PRESETS = {
    # (hidden, layers, heads, kv_heads, ffn) — head_dim = hidden // heads
    # 30M: what all earlier runs used (hidden=512, head_dim=128, 6 layers).
    "30M": dict(hidden_dim=512, num_layers=6, num_heads=4, num_kv_heads=4, intermediate_dim=2048),
    # 200M: Chinchilla-zone for ~20 B tokens; hidden=1024, head_dim=64, 12 layers.
    # params ≈ embed(7M tied) + 12 × (4·1024² attn + 3·1024·4096 ffn) ≈ 208M.
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
}


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full")
    steps = int(os.environ.get("TOMAT_STEPS", "1000"))
    batch_size = int(os.environ.get("TOMAT_BATCH_SIZE", "128"))
    seed = int(os.environ.get("TOMAT_SEED", "42"))
    results_label_env = os.environ.get("TOMAT_RESULTS_LABEL")
    model_preset = os.environ.get("TOMAT_MODEL", "30M")
    val_seqs = int(os.environ.get("TOMAT_VAL_SEQS", "0"))
    steps_per_eval_env = os.environ.get("TOMAT_STEPS_PER_EVAL")

    parquet_glob = f"{BUCKET}/tokenized/{label}/worker-*/*.parquet"
    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    import fsspec
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    print(f"[tomat-tpu] label={label}, vocab_size={vocab_size}, "
          f"patch={meta['patch_size']}, codec={meta['density_codec_name']}, "
          f"model={model_preset}, val_seqs={val_seqs}")

    results_label = results_label_env or f"{label}-tpu-{model_preset}-bs{batch_size}-seed{seed}"
    run_id = results_label

    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source,
        cache_dir=f"{BUCKET}/results/{results_label}/cache",
        format=prebuilt,
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=f"{BUCKET}/results/{results_label}/cache",
        components={"tomat": component},
        block_cross_document_attention=False,
        # Hold out TOMAT_VAL_SEQS sequences from train for validation. Levanter
        # types this as `dict[str, int]` keyed by component name — one entry per
        # DatasetComponent. We have a single "tomat" component. val_seqs=0 skips.
        num_validation_sequences={"tomat": val_seqs} if val_seqs > 0 else None,
    )

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown TOMAT_MODEL={model_preset!r}; expected one of {list(MODEL_PRESETS)}")
    model = Qwen3Config(
        max_seq_len=8192,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        **MODEL_PRESETS[model_preset],
    )

    # W&B conventions mirror the Modal side so filters/overlays are consistent.
    trackers = (
        WandbConfig(
            id=run_id,
            resume="allow",
            project=f"tomat-{meta['density_codec_name']}-P{meta['patch_size']}",
            group=f"M32-Ntpu-{model_preset}",
            tags=[
                "scale",
                "tpu",
                "marin",
                f"mats{meta['n_materials']}",
                f"bs{batch_size}",
                f"seed{seed}",
                f"model{model_preset}",
                *(["val"] if val_seqs > 0 else []),
            ],
            save_code=False,
        ),
        JsonLoggerConfig(),
    )

    checkpointer = CheckpointerConfig(
        base_path=f"{BUCKET}/results/{results_label}/checkpoints",
        save_interval=timedelta(minutes=10),
        keep=[{"every": 1000}],
    )

    # Eval cadence: if val is on, every steps // 4 by default (so 4 evals in a
    # 2000-step run — useful plot resolution). With no val, default keeps the
    # old behavior (one mid-run eval, effectively a no-op).
    if steps_per_eval_env:
        steps_per_eval = int(steps_per_eval_env)
    elif val_seqs > 0:
        steps_per_eval = max(steps // 4, 1)
    else:
        steps_per_eval = max(steps // 2, 1)

    # bf16 compute + fp32 params/optimizer. TPU v6e has ~31 GB HBM/chip;
    # fp32 activations blow this at 200M/bs=32-per-chip. bf16 compute is also
    # ~2× faster on TPU tensor cores. Standard config for any >30M model.
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )

    trainer = TrainerConfig(
        id=run_id,
        seed=seed,
        num_train_steps=steps,
        train_batch_size=batch_size,
        steps_per_eval=steps_per_eval,
        tracker=trackers,
        checkpointer=checkpointer,
        mp=mp,
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

    print("[tomat-tpu] calling levanter.main.train_lm.main …")
    train_lm_main(config)
    print("[tomat-tpu] done")


if __name__ == "__main__":
    main()
