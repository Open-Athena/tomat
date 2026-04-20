"""Tomat 30M hello-world: Qwen3 on patch-tokenized rho_gga CHGCARs.

**SCAFFOLD — NOT YET RUNNABLE.** One concrete hook still needs to be
wired:

  Point the ``UrlDatasetSourceConfig`` below at the actual parquet-shard
  directory produced by ``scripts/tokenize_patches.py``. Either
  - GCS-hosted parquet (easiest for TPU training) — `rsync` the output
    dir into `gs://<bucket>/tomat/rho-gga-val/`, or
  - Modal-volume-hosted parquet if the v5p job has the tomat-rho-gga
    volume mounted.

The nice news: **Levanter ships a ``PrebuiltLmDatasetFormat``** that
takes ``input_ids`` columns directly, so we don't need a HF tokenizer
artifact at all. Our parquet shards are exactly in that format already
(``input_ids: list<int32>``).

Once the dataset path is set, launch with::

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --extra marin:tpu --tpu v5p-8 -- \\
        python experiments/tomat_patch_30m.py

Config mirrors ``will-tomol/experiments/tatt/tomol25_30m.py`` with:

* Same Qwen3 body (6 layers, hidden=512, 4 heads) ≈ 24 M transformer
  params; with tied 6,790-vocab embeddings adds ~3.5 M → ~28 M total.
* Max seq length bumped to **8,192** (tomol-30M was 4,096) to fit
  a 14³ patch (5,488 density tokens) plus a 100-atom structure preamble
  with buffer.
* Batch 64, 1 B training tokens, v5p-8 — same as tomol.
"""

from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

SEQ_LEN = 8192
BATCH_SIZE = 64
TARGET_TOKENS = 1_000_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN)

model = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=4,
    num_kv_heads=4,
    num_layers=6,
    rope=Llama3RotaryEmbeddingsConfig(),
)

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=0.00864,
    train_seq_len=SEQ_LEN,
    z_loss_weight=1.10e-05,
    optimizer_config=AdamHConfig(
        learning_rate=0.00864,
        adam_lr=0.000502,
        min_lr_ratio=0.0,
        warmup=0.1,
        decay=0.2,
        lr_schedule="linear",
        beta1=0.894,
        beta2=0.999,
        epsilon=2.32e-07,
        max_grad_norm=0.1,
        nesterov=False,
    ),
    steps_per_eval=500,
)

# TODO: point at the tomat tokenized parquet. Change to the real path:
#   scripts/tokenize_patches.py writes
#     <output-dir>/shard-NNNNN.parquet  (columns: task_id, offset_{x,y,z}, input_ids)
# A GCS mirror (`gs://...`) is simplest for TPU jobs; local path works
# for CPU / debugging.
TOMAT_TRAIN_URL = "gs://TODO-bucket/tomat/rho-gga-train/*.parquet"
TOMAT_VAL_URL = "gs://TODO-bucket/tomat/rho-gga-val/*.parquet"

_prebuilt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")

tomat_train_source = UrlDatasetSourceConfig(
    urls=[TOMAT_TRAIN_URL],
    cache_dir="gs://TODO-bucket/tomat/cache/train",
    format=_prebuilt,
)

tomat_val_source = UrlDatasetSourceConfig(
    urls=[TOMAT_VAL_URL],
    cache_dir="gs://TODO-bucket/tomat/cache/val",
    format=_prebuilt,
)

tomat_data = LmDataConfig(
    train_components=[
        DatasetComponent(source=tomat_train_source, cache_dir=tomat_train_source.cache_dir,
                         format=_prebuilt),
    ],
    validation_sets={
        "tomat-val": DatasetComponent(source=tomat_val_source, cache_dir=tomat_val_source.cache_dir,
                                      format=_prebuilt),
    },
)

training_step = default_train(
    name="tomat-patch-30m",
    tokenized=tomat_data,
    model_config=model,
    train_config=train_config,
    tags=["tomat", "patch", "30m", "qwen3", "adamh"],
    use_default_validation=False,
    eval_harness_tasks=[],
)


if __name__ == "__main__":
    executor_main(steps=[training_step])
