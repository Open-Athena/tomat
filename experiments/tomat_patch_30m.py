"""Tomat 30M hello-world: Qwen3 on patch-tokenized rho_gga CHGCARs.

**SCAFFOLD — NOT YET RUNNABLE.** Two Marin-integration pieces need to
be filled in before this launches on a TPU:

1. A HuggingFace tokenizer artifact that matches our 6,790-token vocab.
   Levanter's data-pipeline expects an ``AutoTokenizer.from_pretrained``
   handle; for our numeric-only vocab the simplest thing is a
   ``WordLevel`` tokenizer with ``{str(i): i}`` mappings. Upload to
   ``openathena/tomat-patch-tokenizer`` on HF (or a local path) and
   wire the name below.

2. A pre-tokenized dataset artifact referenced by ``default_tokenize``
   or equivalent. Options:
   - Run ``scripts/tokenize_patches.py`` locally/on della to produce
     parquet shards, push to HF as ``openathena/tomat-rho-gga-val``,
     then use ``default_download`` to pull them on the training node.
   - Or mount the parquet from the ``tomat-rho-gga`` Modal volume
     directly into the training job — cleaner, skips the HF round
     trip. Needs a Marin dataset source config that reads a volume
     mount; look at ``marin.processing.tokenize`` for the hook.

Once those land, replace the ``TODO`` markers below and launch with:

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

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_download, default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

# TODO: replace with the HF path for the passthrough WordLevel tokenizer
# matching tomat's 6 790-token vocab (see `src/tomat/tokenizers/patch.py`
# for the layout). A ``PatchVocab.to_hf_tokenizer()`` helper would be
# the cleanest way to generate it.
TOKENIZER = "openathena/tomat-patch-tokenizer"  # TODO

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

# TODO: wire to the actual tomat pre-tokenized dataset. Two-path sketch:
#
# (a) HF-backed (mirror tomol's flow):
#     tomat_download = default_download(
#         name="raw/tomat-rho-gga-val",
#         hf_dataset_id="openathena/tomat-rho-gga-val",
#         revision="<commit-sha>",
#     )
#     tomat_tokenized = default_tokenize(
#         name="tomat-val",
#         dataset=tomat_download / "data/*.parquet",
#         tokenizer=TOKENIZER,
#     )
#
# (b) Modal-volume-backed (skips HF round-trip):
#     tomat_tokenized = <custom dataset source pointing at the
#                       tomat-rho-gga volume's /tokenized/val/*.parquet>
#
# For the first run: split off a tiny "val" subset (even just 32 of 4303
# structures) so we can iterate fast; full val + train are a later
# knob.
tomat_tokenized = ...  # TODO

tomat_data = lm_data_config(tomat_tokenized)  # TODO: add validation_sets

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
