#!/usr/bin/env python
"""Train tomat's 30M Qwen3 patch-LM on Marin's shared TPU cluster.

Mirrors `scripts/train_smoke_modal.py` but targets GCS for data +
checkpoints (Marin's standard data path). Run via:

    cd marin
    uv run iris --cluster=marin job run \\
        --tpu v6e-4 \\
        --env-vars WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python train_tomat_tpu.py

Prereqs:
- `gs://marin-eu-west4/tomat/tokenized/val-full/worker-*/*.parquet` populated
  (mirror of the same data on the `tomat-rho-gga` Modal volume).
- ADC refreshed for `ryan.williams@openathena.ai` on hai-gcp-models.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

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
PARQUET_GLOB = f"{BUCKET}/tokenized/val-full/worker-*/*.parquet"


def main():
    steps = int(os.environ.get("TOMAT_STEPS", "1000"))
    batch_size = int(os.environ.get("TOMAT_BATCH_SIZE", "128"))  # v6e-4 = 4 chips; per-chip bs=32
    seed = int(os.environ.get("TOMAT_SEED", "42"))

    # Vocab size is tomat-specific (6,792 for two_token_9_12 + P=14 + HI-block).
    # One of the workers has the meta.json with this; read from GCS to avoid
    # hardcoding (and surface mismatches early).
    meta_url = f"{BUCKET}/tokenized/val-full/worker-00/meta.json"
    import fsspec
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    print(f"[tomat-tpu] vocab_size={vocab_size}, patch={meta['patch_size']}, "
          f"codec={meta['density_codec_name']}")

    results_label = f"val-full-tpu-bs{batch_size}-seed{seed}"
    run_id = results_label

    source = UrlDatasetSourceConfig(train_urls=[PARQUET_GLOB])
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

    # Match Modal-side W&B project so curves are directly comparable.
    # Group 'M32-Ntpu' is a new group to distinguish TPU runs from Modal.
    trackers = (
        WandbConfig(
            id=run_id,
            resume="allow",
            project=f"tomat-{meta['density_codec_name']}-P{meta['patch_size']}",
            group="M32-Ntpu",
            tags=[
                "scale",
                "tpu",
                "marin",
                f"mats{meta['n_materials']}",
                f"bs{batch_size}",
                f"seed{seed}",
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

    print("[tomat-tpu] calling levanter.main.train_lm.main …")
    train_lm_main(config)
    print("[tomat-tpu] done")


if __name__ == "__main__":
    main()
