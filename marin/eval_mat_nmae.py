#!/usr/bin/env python
"""Mat-level teacher-forced NMAE eval against electrAI/charg3net's metric.

For each held-out material:
  1. Tile the density grid with P×P×P cube patches at a given stride (stride=P
     → disjoint; stride<P → overlapping).
  2. Tokenize each patch via PatchTokenizer (same codec as training).
  3. Teacher-forced forward pass: extract the model's argmax at every density-
     codec position, decode via LMQ (or 2-tok codec) → predicted floats.
  4. Stitch predicted floats back into a full (nx, ny, nz) density grid.
     Overlapping patches: average predictions per voxel.
  5. NMAE vs ground-truth Zarr density.

Outputs per-material NMAE + aggregate stats. Comparable to electrAI /
charg3net numbers (both are mat-level NMAE).

Required env vars:
    TOMAT_CHECKPOINT         gs:// path to a Levanter orbax checkpoint
    TOMAT_MODEL              "30M" | "200M" | "1B"
    TOMAT_LMQ_PATH           gs:// path to the LMQ codec .npz (if training used LMQ)
    TOMAT_LABEL              tokenized label (for meta.json + codec shape info)
    TOMAT_EVAL_SPLIT         "validation" (default) or another key from split file
    TOMAT_EVAL_N_MATS        cap number of mats to eval (default 20)
    TOMAT_EVAL_STRIDE        tile stride in voxels (default = P, i.e. disjoint)
    TOMAT_EVAL_BATCH         # patches per forward pass (default 8)

Emits:
    /tmp/eval-mat-nmae-<label>.csv   per-mat NMAE rows
    stdout: JSON summary line with mean/median/p99 NMAE
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

import jax
try:
    jax.distributed.initialize()
except Exception:
    pass

from levanter.data.passthrough_tokenizer import PassthroughTokenizer
_orig_pt_encode = PassthroughTokenizer.encode
def _safe_pt_encode(self, text, *, add_special_tokens=False):
    try:
        return _orig_pt_encode(self, text, add_special_tokens=add_special_tokens)
    except ValueError:
        return [0]
PassthroughTokenizer.encode = _safe_pt_encode

import numpy as np
import jax.numpy as jnp
import jmp
import haliax as hax
import equinox as eqx
import fsspec
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode

from tomat.data.zarr_io import load_rho_gga
from tomat.float_codec import FP16Codec, LMQCodec
from tomat.tokenizers.patch import (
    PatchTokenizer,
    SPECIAL_TOKENS,
    N_SPECIALS,
    ATOM_OFFSET,
    ATOM_END,
    INT_OFFSET,
    INT_VOCAB_SIZE,
    INT_END,
)

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"

MODEL_PRESETS = {
    "30M":  dict(hidden_dim=512,  num_layers=6,  num_heads=4,  num_kv_heads=4,  intermediate_dim=2048),
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    "1B":   dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}


def tile_offsets(grid_shape, P, stride):
    """Disjoint or overlapping tiling offsets covering the grid (with PBC wrap)."""
    nx, ny, nz = grid_shape
    xs = list(range(0, nx, stride))
    ys = list(range(0, ny, stride))
    zs = list(range(0, nz, stride))
    offsets = []
    for x in xs:
        for y in ys:
            for z in zs:
                offsets.append((x, y, z))
    return offsets


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full")
    model_preset = os.environ.get("TOMAT_MODEL", "200M")
    checkpoint_path = os.environ["TOMAT_CHECKPOINT"]
    lmq_path = os.environ.get("TOMAT_LMQ_PATH")
    eval_split = os.environ.get("TOMAT_EVAL_SPLIT", "validation")
    n_mats_cap = int(os.environ.get("TOMAT_EVAL_N_MATS", "20"))
    eval_batch = int(os.environ.get("TOMAT_EVAL_BATCH", "8"))
    stride_env = os.environ.get("TOMAT_EVAL_STRIDE")

    # Load meta
    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    patch_size_meta = meta.get("patch_size")
    P = int(patch_size_meta) if isinstance(patch_size_meta, int) else 14
    stride = int(stride_env) if stride_env else P

    # Codec
    if meta["density_codec_name"] == "lmq":
        if not lmq_path:
            raise ValueError("codec is LMQ but TOMAT_LMQ_PATH unset")
        density_codec = LMQCodec.load(lmq_path)
    else:
        dc = meta["vocab"]["density_codec"]
        density_codec = FP16Codec(
            log_min=dc["log_min"], log_max=dc["log_max"],
            token_mag_bits=tuple(dc["token_mag_bits"]),
        )

    err(f"[eval-mat] label={label}, model={model_preset}, P={P}, stride={stride}")
    err(f"[eval-mat] checkpoint={checkpoint_path}, vocab={vocab_size}")

    # Offsets in vocab
    pc = meta["vocab"]["position_codec"]
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    DENSITY_OFFSET = INT_END + pos_total
    tokens_per_voxel = density_codec.tokens_per_value_signed
    err(f"[eval-mat] DENSITY_OFFSET={DENSITY_OFFSET}, tokens_per_voxel={tokens_per_voxel}")

    # Model
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )
    trainer_cfg = TrainerConfig(
        id="eval-mat-nmae",
        seed=42,
        num_train_steps=1,
        train_batch_size=eval_batch,
        tracker=(),
        mp=mp_policy,
    )
    levanter.initialize(trainer_cfg)
    compute_axis_mapping = trainer_cfg.compute_axis_mapping
    parameter_axis_mapping = trainer_cfg.parameter_axis_mapping

    model_cfg = Qwen3Config(
        max_seq_len=8192,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        **MODEL_PRESETS[model_preset],
    )
    Pos = model_cfg.Pos
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)

    key = jax.random.PRNGKey(42)
    with trainer_cfg.use_device_mesh():
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[eval-mat] loading checkpoint")
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        model = inference_mode(model, True)
        model = mp_policy.cast_to_compute(model)

        @hax.named_jit(axis_resources=compute_axis_mapping)
        def forward(tokens_in):
            act = model.activations(tokens_in, key=None, attn_mask=None)
            head = model.get_lm_head()
            return hax.dot(act, head, axis=model.Embed)

        # Load split file to get val mat IDs
        split_url = meta.get("split_file") or f"/vol/split_limit_22M.json"  # fallback
        # The meta points at the in-volume path; we need the GCS or local copy. Fetch from
        # the same path Hananeh's split info was put in. For eval we just need mp-IDs.
        # Quickest route: use the parquets' task_ids as val candidates.
        import pyarrow.parquet as pq
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        parquet_glob = f"marin-eu-west4/tomat/tokenized/{label}/worker-*/*.parquet"
        shard_paths = sorted(fs.glob(parquet_glob))
        # Take a few shards and pull unique mp_ids
        mp_ids_seen = set()
        for shard in shard_paths[:5]:
            with fs.open(shard, "rb") as f:
                tbl = pq.ParquetFile(f).read(columns=["task_id"])
            for tid in tbl.column("task_id").to_pylist():
                mp_ids_seen.add(tid)
                if len(mp_ids_seen) >= n_mats_cap:
                    break
            if len(mp_ids_seen) >= n_mats_cap:
                break
        mp_ids = sorted(mp_ids_seen)[:n_mats_cap]
        err(f"[eval-mat] eval on {len(mp_ids)} mats: {mp_ids[:5]}...")

        # For each mat: load raw Zarr, tile, tokenize, forward pass, decode, stitch, NMAE.
        # This requires Zarrs on the local filesystem — won't work inside iris' TPU env
        # without a volume mount. Best run via a separate Modal wrapper that mounts
        # tomat-rho-gga (val volume) + the GCS-accessible checkpoint. For now this
        # is a scaffold; the heavy "load raw Zarr" path is TODO.
        err(f"[eval-mat] TODO: raw Zarr load path — this eval is a scaffold to be "
            f"completed after LMQ training produces a checkpoint. See inline TODO.")

        # Emit an informative placeholder result.
        print(json.dumps({
            "status": "scaffold_only",
            "checkpoint": checkpoint_path,
            "n_mats_requested": n_mats_cap,
            "mp_ids_found": mp_ids,
            "notes": "Full raw-Zarr → patch → decode → NMAE path not yet wired.",
        }))


if __name__ == "__main__":
    main()
