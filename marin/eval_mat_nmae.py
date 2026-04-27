#!/usr/bin/env python
"""Mat-level teacher-forced NMAE eval — comparable to electrAI/charg3net numbers.

For each held-out material:
  1. Load raw Zarr from gs://.../tomat/rho_gga_raw/<split>/<mp_id>.zarr
  2. Disjoint-tile with P×P×P patches (edge skipping; miss at most ~P voxels/dim)
  3. For each patch:
       - Build preamble + true-density input_ids using the same PatchTokenizer
         logic the training corpus used
       - Teacher-forced forward pass: logits[Pos, Vocab]
       - Argmax at density-target positions → predicted density tokens
       - Decode each predicted token → float via LMQ codec
  4. Write predicted floats into the full (nx, ny, nz) grid
  5. NMAE vs raw density

Env vars:
    TOMAT_CHECKPOINT        gs:// path to Levanter orbax checkpoint
    TOMAT_MODEL             "30M" | "200M" | "1B"
    TOMAT_LABEL             tokenized label (for meta.json vocab layout info)
    TOMAT_LMQ_PATH          gs:// path to LMQ codec
    TOMAT_EVAL_N_MATS       cap number of mats to eval (default 10)
    TOMAT_EVAL_BATCH        patches per forward (default 8)
    TOMAT_ZARR_BASE         gs:// prefix for raw Zarrs (default rho_gga_raw/validation)
"""

from __future__ import annotations

import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import jax
try:
    jax.distributed.initialize()
except Exception:
    pass

# PassthroughTokenizer monkey-patch (same as train_tomat_tpu.py).
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


err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"

MODEL_PRESETS = {
    "30M":  dict(hidden_dim=512,  num_layers=6,  num_heads=4,  num_kv_heads=4,  intermediate_dim=2048),
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    "1B":   dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}

# Vocab layout (matches src/tomat/tokenizers/patch.py's PatchVocab).
N_SPECIALS = 18
N_ATOMS = 118
N_INTS = 1024


# Special token IDs (same as patch.py SPECIAL_TOKENS)
TOK = {
    "PAD": 0, "BOS": 1, "EOS": 2,
    "ATOMS_START": 3, "ATOMS_END": 4,
    "POS_START": 5, "POS_END": 6,
    "GRID_START": 7, "GRID_END": 8,
    "SHAPE_START": 9, "SHAPE_END": 10,
    "OFFSET_START": 11, "OFFSET_END": 12,
    "HI_START": 13, "HI_END": 14,
    "DENS_START": 15, "DENS_END": 16,
    "NL": 17,
}


def load_lmq_codec(path):
    """Load LMQ codec .npz; returns (boundaries, recon_points, clip_max)."""
    with fsspec.open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        return (
            np.asarray(data["boundaries"], dtype=np.float32),
            np.asarray(data["recon_points"], dtype=np.float32),
            float(data["clip_max"]),
        )


def lmq_encode(boundaries, clip_max, values):
    v = np.clip(values, 0.0, clip_max)
    return np.searchsorted(boundaries, v, side="right").astype(np.int32)


def load_structure_from_zarr_attrs(zarr_group):
    """Parse pymatgen structure JSON from a zarr group's attrs. Returns
    (atomic_Zs, frac_coords) — we don't need a full pymatgen Structure."""
    from pymatgen.core.structure import Structure
    attrs = dict(zarr_group.attrs)
    struct_json = attrs.get("structure")
    if isinstance(struct_json, str):
        struct_json = json.loads(struct_json)
    s = Structure.from_dict(struct_json)
    Zs = np.array([site.specie.Z for site in s], dtype=np.int32)
    frac = np.array([site.frac_coords for site in s], dtype=np.float64) % 1.0
    return Zs, frac


def tokenize_patch_for_eval(
    density_values,  # (P, P, P) float
    grid_shape,      # (nx, ny, nz)
    offset,          # (ix, iy, iz)
    atomic_Zs,       # (N,) int
    frac_coords,     # (N, 3) float
    lmq_codec,       # (boundaries, recon_points, clip_max)
    vocab_offsets,   # dict with 'density_offset', 'pos_offset', pos_vocabs, pos_log_min, pos_log_max
    P=14,
):
    """Build an input_ids list for one patch — mirrors PatchTokenizer.tokenize()."""
    bounds, recon, clip_max = lmq_codec
    tokens = [TOK["BOS"]]

    # GRID
    tokens.append(TOK["GRID_START"])
    for d in grid_shape:
        tokens.append(N_SPECIALS + N_ATOMS + int(d))  # int_token offset = 136 + d
    tokens.append(TOK["GRID_END"])

    # ATOMS
    tokens.append(TOK["ATOMS_START"])
    for z in atomic_Zs:
        tokens.append(N_SPECIALS + int(z - 1))
    tokens.append(TOK["ATOMS_END"])

    # POS: each coord → tomol_3byte codec (3 tokens/coord × 3 coords per atom)
    # For eval we care about getting SE-like tokens at valid positions; copy logic
    # from PatchVocab.position_tokens by mimicking tomol_3byte with log range.
    # For now use a simpler approach: use the same codec the tokenizer would have.
    # This requires the same `FP16Codec.tomol_3byte(log_min=-4, log_max=0)` logic.
    tokens.append(TOK["POS_START"])
    # Delegate to a helper (below)
    for xyz in frac_coords:
        for c in xyz:
            tokens.extend(_pos_tokens(float(c), vocab_offsets))
    tokens.append(TOK["POS_END"])

    # SHAPE
    tokens.append(TOK["SHAPE_START"])
    for _ in range(3):
        tokens.append(N_SPECIALS + N_ATOMS + int(P))
    tokens.append(TOK["SHAPE_END"])

    # OFFSET
    tokens.append(TOK["OFFSET_START"])
    for o in offset:
        tokens.append(N_SPECIALS + N_ATOMS + int(o))
    tokens.append(TOK["OFFSET_END"])

    # HI
    tokens.append(TOK["HI_START"])
    nx, ny, nz = grid_shape
    hi = tuple((offset[i] + P - 1) % grid_shape[i] for i in range(3))
    for h in hi:
        tokens.append(N_SPECIALS + N_ATOMS + int(h))
    tokens.append(TOK["HI_END"])

    # DENS — LMQ 1-token per voxel, row-major flatten
    tokens.append(TOK["DENS_START"])
    flat = np.ravel(density_values)
    bin_idx = lmq_encode(bounds, clip_max, flat)
    dens_offset = vocab_offsets["density_offset"]
    for b in bin_idx:
        tokens.append(dens_offset + int(b))
    tokens.append(TOK["DENS_END"])

    tokens.append(TOK["EOS"])
    return tokens


def _pos_tokens(coord, vocab_offsets):
    """tomol_3byte codec for position: 3 tokens per coord, log-uniform [1e-4, 1).
    Matches FP16Codec.tomol_3byte(log_min=-4.0, log_max=0.0).encode_signed([coord])[0].
    Returns list of 3 absolute vocab tokens."""
    pos_off = vocab_offsets["pos_offset"]
    pos_vocabs = vocab_offsets["pos_vocabs"]  # e.g. (512, 256, 256)
    log_min = vocab_offsets["pos_log_min"]
    log_max = vocab_offsets["pos_log_max"]
    mag = abs(coord)
    if mag < 1e-15:
        bin_index = 0
    else:
        log_mag = math.log10(mag)
        # 24 mag bits total for tomol_3byte
        total_bins = 1 << 24
        normalized = (log_mag - log_min) / (log_max - log_min)
        normalized = min(max(normalized, 0), 1)
        bin_index = int(round(normalized * (total_bins - 1)))
        bin_index = min(max(bin_index, 0), total_bins - 1)

    signs_positive = coord >= 0
    # First token: 8 mag bits + sign; vocab = 512
    first_half = 1 << 8
    # shift MSB-first: 24 mag bits total, 8 for each of 3 tokens
    shift = 24
    comps = []
    for i, b in enumerate((8, 8, 8)):
        shift -= b
        chunk = (bin_index >> shift) & ((1 << b) - 1)
        if i == 0:
            chunk = chunk if signs_positive else (first_half + chunk)
        comps.append(chunk)
    # Map to absolute token IDs
    out = []
    cum = 0
    for w, c in zip(pos_vocabs, comps):
        out.append(pos_off + cum + c)
        cum += w
    return out


def tile_disjoint_offsets(grid_shape, P):
    """Return (list of offsets, valid covered region shape)."""
    nx, ny, nz = grid_shape
    xs = list(range(0, nx - P + 1, P))
    ys = list(range(0, ny - P + 1, P))
    zs = list(range(0, nz - P + 1, P))
    offsets = [(x, y, z) for x in xs for y in ys for z in zs]
    covered = (len(xs) * P, len(ys) * P, len(zs) * P)
    return offsets, covered


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full-lmq-v2")
    model_preset = os.environ.get("TOMAT_MODEL", "200M")
    checkpoint_path = os.environ["TOMAT_CHECKPOINT"]
    lmq_path = os.environ["TOMAT_LMQ_PATH"]
    n_mats_cap = int(os.environ.get("TOMAT_EVAL_N_MATS", "10"))
    eval_batch = int(os.environ.get("TOMAT_EVAL_BATCH", "8"))
    split = os.environ.get("TOMAT_EVAL_SPLIT", "validation")
    zarr_base = os.environ.get(
        "TOMAT_ZARR_BASE", f"{BUCKET}/rho_gga_raw/{split}"
    )
    seed = int(os.environ.get("TOMAT_SEED", "42"))

    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    patch_size_meta = meta.get("patch_size")
    P = int(patch_size_meta) if isinstance(patch_size_meta, int) else 14
    err(f"[eval-mat] label={label}, model={model_preset}, P={P}, vocab={vocab_size}")

    # LMQ codec
    bounds, recon, clip_max = load_lmq_codec(lmq_path)
    err(f"[eval-mat] LMQ codec: n_bins={len(recon)}, clip_max={clip_max:.2f}, "
        f"recon range=[{recon.min():.3e}, {recon.max():.3e}]")
    lmq_codec = (bounds, recon, clip_max)

    # Vocab offsets
    pc = meta["vocab"]["position_codec"]
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    density_offset = N_SPECIALS + N_ATOMS + N_INTS + pos_total
    vocab_offsets = {
        "density_offset": density_offset,
        "pos_offset": N_SPECIALS + N_ATOMS + N_INTS,
        "pos_vocabs": pos_signed_vocabs,
        "pos_log_min": pc["log_min"],
        "pos_log_max": pc["log_max"],
    }
    err(f"[eval-mat] DENSITY_OFFSET={density_offset}, density_range=[{density_offset}, {vocab_size})")

    # Set up Trainer + model
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32,
    )
    trainer_cfg = TrainerConfig(
        id="eval-mat-nmae",
        seed=seed,
        num_train_steps=1,
        train_batch_size=eval_batch,
        tracker=(),
        mp=mp_policy,
    )
    levanter.initialize(trainer_cfg)
    compute_mapping = trainer_cfg.compute_axis_mapping
    param_mapping = trainer_cfg.parameter_axis_mapping

    model_cfg = Qwen3Config(
        max_seq_len=8192,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        **MODEL_PRESETS[model_preset],
    )
    key = jax.random.PRNGKey(seed)
    with trainer_cfg.use_device_mesh():
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_mapping)
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[eval-mat] loading checkpoint")
            model = load_checkpoint(model, checkpoint_path, subpath="model")
        model = hax.shard_with_axis_mapping(model, param_mapping)
        model = inference_mode(model, True)
        model = mp_policy.cast_to_compute(model)

        @hax.named_jit(axis_resources=compute_mapping)
        def forward(tokens_in):
            act = model.activations(tokens_in, key=None, attn_mask=None)
            head = model.get_lm_head()
            return hax.dot(act, head, axis=model.Embed)

        # Pick mp-IDs from the val parquets for eval.
        import pyarrow.parquet as pq
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        parquet_glob = f"marin-eu-west4/tomat/tokenized/{label}/worker-*/*.parquet"
        shard_paths = sorted(fs.glob(parquet_glob))
        mp_ids = []
        for shard in shard_paths[:3]:
            with fs.open(shard, "rb") as f:
                tbl = pq.ParquetFile(f).read(columns=["task_id"])
            for tid in tbl.column("task_id").to_pylist():
                if tid not in mp_ids:
                    mp_ids.append(tid)
                if len(mp_ids) >= n_mats_cap:
                    break
            if len(mp_ids) >= n_mats_cap:
                break
        err(f"[eval-mat] will eval on {len(mp_ids)} mats")

        per_mat_results = []
        for mp_id in mp_ids:
            # Load raw Zarr — download from GCS to local /tmp first to dodge
            # the gcsfs/aiohttp async-event-loop conflict with JAX's runtime.
            # fsspec.download works synchronously even when the underlying
            # filesystem is async (gcsfs).
            import zarr
            import tempfile
            zarr_url = f"{zarr_base}/{mp_id}.zarr"
            local_dir = tempfile.mkdtemp(prefix=f"tomat-eval-{mp_id}-")
            local_zarr_path = f"{local_dir}/{mp_id}.zarr"
            err(f"[eval-mat] mat={mp_id}, downloading {zarr_url} → {local_zarr_path}")
            try:
                gcs_fs = fsspec.filesystem("gs")
                gcs_fs.get(zarr_url.replace("gs://", "") + "/", local_zarr_path, recursive=True)
            except Exception as e:
                err(f"[eval-mat] download FAIL {mp_id}: {type(e).__name__}: {e}")
                continue
            group = zarr.open_group(local_zarr_path, mode="r")
            density = np.asarray(group["charge_density_total"][:]).astype(np.float32)
            grid_shape = tuple(density.shape)
            Zs, frac = load_structure_from_zarr_attrs(group)
            err(f"[eval-mat] mat={mp_id}: grid={grid_shape}, atoms={len(Zs)}")

            # Tile
            offsets, covered_shape = tile_disjoint_offsets(grid_shape, P)
            n_patches = len(offsets)
            err(f"[eval-mat] tiling: {n_patches} disjoint patches, covered={covered_shape}")

            # Accumulate predicted density (initialize to true density — uncovered
            # regions stay at true density so they don't contribute to the NMAE
            # numerator; or fill with 0 and report NMAE on covered region only).
            rho_pred = np.zeros(covered_shape, dtype=np.float32)

            # Process in batches. Drop trailing partial batch (TPU mesh
            # needs batch divisible by mesh size = eval_batch); we lose at
            # most eval_batch-1 patches per material.
            total = 0
            n_full_batches = n_patches // eval_batch
            for bi in range(n_full_batches):
                batch_start = bi * eval_batch
                batch_offsets = offsets[batch_start : batch_start + eval_batch]
                batch_input_ids = []
                batch_max_len = 0
                for off in batch_offsets:
                    patch_data = density[off[0]:off[0]+P, off[1]:off[1]+P, off[2]:off[2]+P]
                    ids = tokenize_patch_for_eval(
                        patch_data, grid_shape, off, Zs, frac, lmq_codec,
                        vocab_offsets, P=P,
                    )
                    # Pad to seq len
                    pad_to = 8192
                    if len(ids) > pad_to:
                        err(f"[eval-mat] skip {mp_id} offset {off}: seq_len {len(ids)} > pad_to")
                        continue
                    ids = ids + [TOK["PAD"]] * (pad_to - len(ids))
                    batch_input_ids.append(np.array(ids, dtype=np.int32))

                if not batch_input_ids:
                    continue
                tokens_np = np.stack(batch_input_ids, axis=0)  # (B, 8192)
                Batch = hax.Axis("batch", tokens_np.shape[0])
                Pos = hax.Axis("position", tokens_np.shape[1])
                tokens_ha = hax.named(jnp.asarray(tokens_np), (Batch, Pos))

                logits = forward(tokens_ha)  # NamedArray (B, Pos, Vocab)
                logits_np = np.array(logits.array).astype(np.float32)

                # At density positions: pred_token[t] = argmax(logits[t-1])
                # Density tokens in range [density_offset, density_offset + n_bins)
                n_bins = len(recon)
                is_density_input = (tokens_np >= density_offset) & (tokens_np < density_offset + n_bins)
                # is_density_target[t] = is_density_input[t+1]; we predict token at t+1
                # Use shifted mask
                shift_mask = np.zeros_like(is_density_input)
                shift_mask[:, :-1] = is_density_input[:, 1:]

                # At each position t where shift_mask[t]=True, predicted token for t+1 = argmax(logits[t])
                pred_all = logits_np.argmax(axis=-1)  # (B, Pos)

                for bi, off in enumerate(batch_offsets):
                    if bi >= tokens_np.shape[0]:
                        break
                    # density positions in this sequence (in terms of INPUT token index)
                    dens_positions_t = np.where(is_density_input[bi])[0]
                    if len(dens_positions_t) != P ** 3:
                        err(f"[eval-mat] WARN: expected {P**3} density tokens, got {len(dens_positions_t)} at offset {off}")
                        continue
                    # For each density input position t, the prediction came from logits[t-1]
                    pred_dens_tokens = pred_all[bi, dens_positions_t - 1]
                    # Decode: clip predicted tokens to density range, then bin → float
                    pred_bins = np.clip(pred_dens_tokens - density_offset, 0, n_bins - 1)
                    pred_floats = recon[pred_bins]

                    # Reshape to (P, P, P) — row-major C-order (matches tokenizer flatten)
                    rho_block = pred_floats.reshape(P, P, P)
                    rho_pred[off[0]:off[0]+P, off[1]:off[1]+P, off[2]:off[2]+P] = rho_block
                    total += 1

            err(f"[eval-mat] {mp_id}: processed {total} patches")

            # NMAE on covered region
            rho_true = density[:covered_shape[0], :covered_shape[1], :covered_shape[2]]
            mae = np.mean(np.abs(rho_pred - rho_true))
            denom = np.mean(np.abs(rho_true))
            nmae = mae / max(denom, 1e-30)
            err(f"[eval-mat] {mp_id}: MAE={mae:.4e}, mean|ρ_true|={denom:.4e}, NMAE={nmae:.4%}")
            per_mat_results.append({
                "mp_id": mp_id,
                "grid_shape": list(grid_shape),
                "covered_shape": list(covered_shape),
                "n_patches": total,
                "n_atoms": int(len(Zs)),
                "mae": float(mae),
                "mean_abs_true": float(denom),
                "nmae": float(nmae),
            })

        # Aggregate
        if per_mat_results:
            nmaes = np.array([r["nmae"] for r in per_mat_results])
            err(f"[eval-mat] AGGREGATE over {len(nmaes)} mats:")
            err(f"  mean NMAE   : {nmaes.mean():.4%}")
            err(f"  median NMAE : {np.median(nmaes):.4%}")
            err(f"  p99 NMAE    : {np.percentile(nmaes, 99):.4%}")

        # Machine-readable summary
        print(json.dumps({
            "checkpoint": checkpoint_path,
            "label": label,
            "model": model_preset,
            "n_mats": len(per_mat_results),
            "nmae_mean": float(nmaes.mean()) if per_mat_results else None,
            "nmae_median": float(np.median(nmaes)) if per_mat_results else None,
            "nmae_p99": float(np.percentile(nmaes, 99)) if per_mat_results else None,
            "per_mat": per_mat_results,
        }, indent=2))


if __name__ == "__main__":
    main()
