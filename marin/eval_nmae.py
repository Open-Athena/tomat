#!/usr/bin/env python
"""Teacher-forced patch-level NMAE eval against electrAI/charg3net's metric.

Loads a Levanter checkpoint, runs one forward pass over N val sequences,
extracts predictions at density-token positions, decodes predicted & true
tokens to float densities via the same FP16Codec used during tokenization,
and reports NMAE = mean|ρ_pred − ρ_true| / mean|ρ_true|.

Usage (via Marin iris):

    cd marin
    uv run iris --cluster=marin job run \\
        --tpu v6e-8 --enable-extra-resources --cpu 32 --memory 64GB \\
        --env-vars WANDB_API_KEY "$WANDB_API_KEY" \\
        --env-vars TOMAT_LABEL train-full \\
        --env-vars TOMAT_MODEL 200M \\
        --env-vars TOMAT_CHECKPOINT gs://.../results/.../checkpoints/.../step-5999 \\
        --env-vars TOMAT_EVAL_SEQS 256 \\
        -- python eval_nmae.py

Caveats:
* Teacher-forced (predictions of token t+1 use *actual* tokens ≤ t).
  Argmax (and/or expected-value via softmax) of the logits at each
  density position → predicted token → decoded float. This is a lower
  bound on autoregressive NMAE (which would compound errors).
* Only density voxels that decode to a valid (A, B) codec pair count.
  Predictions outside the density-token range are treated as "bad"
  (included in NMAE with the max possible error for their position).
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

# Multihost-capable JAX init — same as train_tomat_tpu.py.
import jax
try:
    jax.distributed.initialize()
    print(f"[eval-nmae] jax.distributed done ({jax.process_index()}/{jax.process_count()})", file=sys.stderr)
except Exception as e:
    print(f"[eval-nmae] jax.distributed skipped: {type(e).__name__}: {e}", file=sys.stderr)

from levanter.data.passthrough_tokenizer import PassthroughTokenizer
_orig = PassthroughTokenizer.encode
def _safe_pt_encode(self, text, *, add_special_tokens=False):
    try:
        return _orig(self, text, add_special_tokens=add_special_tokens)
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
from levanter.data import DataLoader
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"

MODEL_PRESETS = {
    "30M":  dict(hidden_dim=512,  num_layers=6,  num_heads=4,  num_kv_heads=4,  intermediate_dim=2048),
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    "1B":   dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full")
    model_preset = os.environ.get("TOMAT_MODEL", "200M")
    checkpoint_path = os.environ["TOMAT_CHECKPOINT"]  # required
    # Reuse the training run's cache_dir so we don't re-cache.
    results_label = os.environ.get("TOMAT_RESULTS_LABEL",
                                   f"{label}-tpu-{model_preset}-bs128-seed42")
    n_eval_seqs = int(os.environ.get("TOMAT_EVAL_SEQS", "256"))
    eval_batch = int(os.environ.get("TOMAT_EVAL_BATCH", "16"))
    seed = int(os.environ.get("TOMAT_SEED", "42"))

    # Load meta (need vocab structure)
    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    specials = meta["vocab"]["specials"]
    DENS_START_TOK = specials["[DENS_START]"]
    DENS_END_TOK = specials["[DENS_END]"]

    # Density-codec vocab offsets
    pc = meta["vocab"]["position_codec"]
    dc = meta["vocab"]["density_codec"]
    position_mag_bits = dc["token_mag_bits"]  # codec structure

    # Reconstruct absolute offsets from meta (matches PatchVocab layout).
    # Specials: 0..17 (18). Atoms: 18..135 (118). Ints: 136..1159 (1024).
    N_SPECIALS = 18
    ATOM_END = N_SPECIALS + 118  # 136
    INT_END = ATOM_END + 1024  # 1160
    # Position codec: signed_vocabs = for token_mag_bits_pos, first has sign bit
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    POS_END = INT_END + pos_total
    # Density codec: same formula
    d_mag = dc["token_mag_bits"]
    dens_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(d_mag))
    dens_total = sum(dens_signed_vocabs)
    DENSITY_OFFSET = POS_END
    DENSITY_END = DENSITY_OFFSET + dens_total
    assert DENSITY_END == vocab_size, (DENSITY_END, vocab_size)

    tokens_per_voxel = len(d_mag)
    assert tokens_per_voxel == 2, f"eval_nmae currently assumes 2-token density codec, got {tokens_per_voxel}"

    # Codec for decoding predicted/actual density bins → floats
    from tomat.float_codec import FP16Codec
    density_codec = FP16Codec(
        log_min=dc["log_min"], log_max=dc["log_max"],
        token_mag_bits=tuple(d_mag),
    )

    # Per-component vocab ranges inside DENSITY slot
    # First density-codec sub-vocab: [DENSITY_OFFSET, DENSITY_OFFSET + dens_signed_vocabs[0])
    # Second: [DENSITY_OFFSET + V0, DENSITY_OFFSET + V0 + V1)
    V0 = dens_signed_vocabs[0]  # e.g. 512
    V1 = dens_signed_vocabs[1]  # e.g. 4096
    TOK_A_LO, TOK_A_HI = DENSITY_OFFSET, DENSITY_OFFSET + V0
    TOK_B_LO, TOK_B_HI = DENSITY_OFFSET + V0, DENSITY_OFFSET + V0 + V1

    err(f"[eval-nmae] label={label}, vocab={vocab_size}, model={model_preset}")
    err(f"[eval-nmae] density: offset={DENSITY_OFFSET}, A=[{TOK_A_LO},{TOK_A_HI}), B=[{TOK_B_LO},{TOK_B_HI})")
    err(f"[eval-nmae] checkpoint={checkpoint_path}")
    err(f"[eval-nmae] n_eval_seqs={n_eval_seqs}, eval_batch={eval_batch}")

    # Build data — reuse the training run's cache dir so we don't re-cache 20 GB.
    parquet_glob = f"{BUCKET}/tokenized/{label}/worker-*/*.parquet"
    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    cache_dir = f"{BUCKET}/results/{results_label}/cache"
    component = DatasetComponent(source=source, cache_dir=cache_dir, format=prebuilt)

    # Build model
    model_cfg = Qwen3Config(
        max_seq_len=8192,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        **MODEL_PRESETS[model_preset],
    )
    Pos = model_cfg.Pos
    Vocab = hax.Axis("vocab", vocab_size)

    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )

    # TrainerConfig: need compute_axis_mapping + mesh.
    trainer_cfg = TrainerConfig(
        id="eval-nmae",
        seed=seed,
        num_train_steps=1,  # forced by Levanter even for eval
        train_batch_size=eval_batch,
        tracker=(),
        mp=mp,
    )
    levanter.initialize(trainer_cfg)

    compute_axis_mapping = trainer_cfg.compute_axis_mapping
    parameter_axis_mapping = trainer_cfg.parameter_axis_mapping
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
    if Vocab.size != vocab_size:
        err(f"[eval-nmae] rounded vocab {vocab_size} → {Vocab.size} for partitioning")

    key = jax.random.PRNGKey(seed)
    with trainer_cfg.use_device_mesh():
        # Init shape on CPU, then load checkpoint weights into it.
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[eval-nmae] loading checkpoint from {checkpoint_path}")
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        model = inference_mode(model, True)
        model = mp.cast_to_compute(model)
        err(f"[eval-nmae] checkpoint loaded.")

        # Build the data's validation set. LmDataConfig.validation_sets requires
        # num_validation_sequences to be set — do so with n_eval_seqs.
        data_cfg = LmDataConfig(
            tokenizer="passthrough",
            vocab_size=vocab_size,
            cache_dir=cache_dir,
            components={"tomat": component},
            block_cross_document_attention=False,
            num_validation_sequences={"tomat": n_eval_seqs},
        )
        datasets_dict = data_cfg.validation_sets(Pos, key=jax.random.PRNGKey(seed))
        tomat_val = next(iter(datasets_dict.values()))
        loader = DataLoader(
            tomat_val,
            batch_size=eval_batch,
            axis_resources=compute_axis_mapping,
        )

        # Jittable forward pass: activations → logits.
        @hax.named_jit(axis_resources=compute_axis_mapping)
        def compute_logits(model_in, tokens_in):
            model_c = mp.cast_to_compute(model_in)
            activations = model_c.activations(tokens_in, key=None, attn_mask=None)
            head = model_c.get_lm_head()
            return hax.dot(activations, head, axis=model_c.Embed)

        all_pred_A = []
        all_pred_B = []
        all_true_A = []
        all_true_B = []
        seq_count = 0
        for batch in loader:
            tokens = batch.tokens  # NamedArray (Batch, Pos)
            logits = compute_logits(model, tokens)  # NamedArray (Batch, Pos, Vocab)

            # Shift to CPU / numpy for post-processing
            tokens_np = np.array(tokens.array)  # (B, S)
            logits_np = np.array(logits.array)  # (B, S, V)

            B, S = tokens_np.shape

            # Teacher-forced: logits[t] predicts token at position t+1.
            targets = np.roll(tokens_np, -1, axis=-1)  # target[t] = tokens[t+1]
            targets[:, -1] = 0  # discard last position

            is_target_A = (targets >= TOK_A_LO) & (targets < TOK_A_HI)
            is_target_B = (targets >= TOK_B_LO) & (targets < TOK_B_HI)

            # A voxel spans 2 consecutive density tokens; mark voxel-start positions
            # (t) where target t is A and target t+1 is B.
            is_target_B_shift = np.zeros_like(is_target_B)
            is_target_B_shift[:, :-1] = is_target_B[:, 1:]
            voxel_start_mask = is_target_A & is_target_B_shift  # (B, S)

            pred_all = logits_np.argmax(axis=-1)  # (B, S) argmax at each position
            b_idx, t_idx = np.where(voxel_start_mask)
            if len(b_idx) == 0:
                seq_count += B
                continue

            pred_A = pred_all[b_idx, t_idx]         # logits[t]   predicts target[t]   = A
            pred_B = pred_all[b_idx, t_idx + 1]     # logits[t+1] predicts target[t+1] = B
            true_A = tokens_np[b_idx, t_idx + 1]    # token at t+1 (= target A)
            true_B = tokens_np[b_idx, t_idx + 2]    # token at t+2 (= target B)

            all_pred_A.append(pred_A)
            all_pred_B.append(pred_B)
            all_true_A.append(true_A)
            all_true_B.append(true_B)

            seq_count += B
            err(f"[eval-nmae] processed {seq_count}/{n_eval_seqs} seqs, voxels so far: "
                f"{sum(len(a) for a in all_pred_A):,}")
            if seq_count >= n_eval_seqs:
                break

    pred_A = np.concatenate(all_pred_A)
    pred_B = np.concatenate(all_pred_B)
    true_A = np.concatenate(all_true_A)
    true_B = np.concatenate(all_true_B)
    err(f"[eval-nmae] total voxels: {len(pred_A):,}")

    # Decode. Map absolute token IDs back to component indices.
    def to_components(tok_A: np.ndarray, tok_B: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (components (N,2), valid mask (N,)). Invalid = token outside expected range."""
        valid_A = (tok_A >= TOK_A_LO) & (tok_A < TOK_A_HI)
        valid_B = (tok_B >= TOK_B_LO) & (tok_B < TOK_B_HI)
        valid = valid_A & valid_B
        comp_A = np.where(valid_A, tok_A - TOK_A_LO, 0)
        comp_B = np.where(valid_B, tok_B - TOK_B_LO, 0)
        return np.stack([comp_A, comp_B], axis=1), valid

    true_comps, true_valid = to_components(true_A, true_B)
    pred_comps, pred_valid = to_components(pred_A, pred_B)

    assert true_valid.all(), "ground-truth should always be a valid density pair"
    n_bad = (~pred_valid).sum()
    err(f"[eval-nmae] predicted pairs outside density range: {n_bad:,} ({n_bad / len(pred_A):.4%})")

    true_floats = density_codec.decode_signed(true_comps)
    # For invalid predictions, substitute 0.0 density (worst case for NMAE).
    pred_floats = density_codec.decode_signed(pred_comps)
    pred_floats = np.where(pred_valid, pred_floats, 0.0)

    # NMAE
    abs_err = np.abs(pred_floats - true_floats)
    denom = np.mean(np.abs(true_floats))
    nmae = abs_err.mean() / denom
    median_nmae = np.median(abs_err) / denom
    p99_nmae = np.percentile(abs_err, 99) / denom

    err(f"[eval-nmae] RESULTS")
    err(f"  n_voxels            : {len(pred_A):,}")
    err(f"  mean |ρ_true|       : {denom:.6f}")
    err(f"  NMAE (argmax, mean) : {nmae:.4%}")
    err(f"  NMAE (argmax, med)  : {median_nmae:.4%}")
    err(f"  NMAE (argmax, p99)  : {p99_nmae:.4%}")

    # Print machine-readable line to stdout for scraping.
    print(json.dumps({
        "checkpoint": checkpoint_path,
        "n_voxels": int(len(pred_A)),
        "mean_abs_true": float(denom),
        "nmae_mean": float(nmae),
        "nmae_median": float(median_nmae),
        "nmae_p99": float(p99_nmae),
        "n_invalid_pred_pairs": int(n_bad),
    }))


if __name__ == "__main__":
    main()
