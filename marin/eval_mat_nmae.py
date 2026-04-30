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
    TOMAT_EVAL_SKIP         skip first N mats (for fan-out across jobs; default 0)
    TOMAT_EVAL_BATCH        patches per forward (default 8; bump to 64 on v6e-8 for ~8× thpt)
    TOMAT_EVAL_DECODER      "median" (default; L_1-optimal point estimator),
                            "argmax" (mode), or "mean" (E[ρ], L_2-optimal).
    TOMAT_ZARR_BASE         gs:// prefix for raw Zarrs (default rho_gga_raw/validation)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
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
# Pre-lat datasets: N_SPECIALS=18, INT_OFFSET=136, no LATTICE block.
# Lat-aware datasets: N_SPECIALS=20, INT_OFFSET=138, LATTICE block between GRID
# and ATOMS. Detected from the dataset's meta.json (LATTICE_START key in
# meta['vocab']['specials']). All offsets shift accordingly; helpers receive
# a `vocab_offsets["n_specials"]` so they don't need a global.
N_SPECIALS_PRELAT = 18
N_SPECIALS_LAT = 20
N_ATOMS = 118
N_INTS = 1024

# Lattice quantization (must match src/tomat/tokenizers/patch.py).
LATTICE_LENGTH_RES_A = 0.05
LATTICE_ANGLE_RES_DEG = 0.2

# Special token IDs (subset; lat datasets add LATTICE_{START,END}=18,19).
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
    "LATTICE_START": 18, "LATTICE_END": 19,
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
    (atomic_Zs, frac_coords, lattice_params) — lattice = (a, b, c, α, β, γ)
    in (Å, Å, Å, deg, deg, deg). lat-aware datasets need it; pre-lat code
    paths just ignore."""
    from pymatgen.core.structure import Structure
    attrs = dict(zarr_group.attrs)
    struct_json = attrs.get("structure")
    if isinstance(struct_json, str):
        struct_json = json.loads(struct_json)
    s = Structure.from_dict(struct_json)
    Zs = np.array([site.specie.Z for site in s], dtype=np.int32)
    frac = np.array([site.frac_coords for site in s], dtype=np.float64) % 1.0
    lat = s.lattice
    lattice_params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    return Zs, frac, lattice_params


def tokenize_patch_for_eval(
    density_values,  # (P, P, P) float
    grid_shape,      # (nx, ny, nz)
    offset,          # (ix, iy, iz)
    atomic_Zs,       # (N,) int
    frac_coords,     # (N, 3) float
    lmq_codec,       # (boundaries, recon_points, clip_max)
    vocab_offsets,   # dict with density_offset, pos_offset, pos_vocabs, pos_log_min, pos_log_max,
                     #   n_specials, lat_aware
    P=14,
    lattice_params=None,
):
    """Build an input_ids list for one patch — mirrors PatchTokenizer.tokenize()."""
    bounds, recon, clip_max = lmq_codec
    n_specials = vocab_offsets["n_specials"]
    int_off = n_specials + N_ATOMS
    lat_aware = vocab_offsets.get("lat_aware", False)
    tokens = [TOK["BOS"]]

    # GRID
    tokens.append(TOK["GRID_START"])
    for d in grid_shape:
        tokens.append(int_off + int(d))
    tokens.append(TOK["GRID_END"])

    # LATTICE (lat-aware only)
    if lat_aware:
        if lattice_params is None:
            raise ValueError("lat-aware dataset requires lattice_params")
        a, b, c, alpha, beta, gamma = lattice_params
        lat_ints = [
            int(round(a / LATTICE_LENGTH_RES_A)),
            int(round(b / LATTICE_LENGTH_RES_A)),
            int(round(c / LATTICE_LENGTH_RES_A)),
            int(round(alpha / LATTICE_ANGLE_RES_DEG)),
            int(round(beta  / LATTICE_ANGLE_RES_DEG)),
            int(round(gamma / LATTICE_ANGLE_RES_DEG)),
        ]
        tokens.append(TOK["LATTICE_START"])
        for v in lat_ints:
            tokens.append(int_off + v)
        tokens.append(TOK["LATTICE_END"])

    # ATOMS
    tokens.append(TOK["ATOMS_START"])
    for z in atomic_Zs:
        tokens.append(n_specials + int(z - 1))
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
        tokens.append(int_off + int(P))
    tokens.append(TOK["SHAPE_END"])

    # OFFSET
    tokens.append(TOK["OFFSET_START"])
    for o in offset:
        tokens.append(int_off + int(o))
    tokens.append(TOK["OFFSET_END"])

    # HI
    tokens.append(TOK["HI_START"])
    nx, ny, nz = grid_shape
    hi = tuple((offset[i] + P - 1) % grid_shape[i] for i in range(3))
    for h in hi:
        tokens.append(int_off + int(h))
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


def tile_full_coverage_offsets(grid_shape, P):
    """Return (offsets, local_slices) covering every voxel exactly once.

    Disjoint patches stride by P; for any axis n with n mod P != 0, one extra
    boundary patch is added at offset (n - P), and only its tail slice
    (P - tail : P) along that axis is used. Full grid is covered, no overlap.
    """
    def axis_specs(n):
        # (offset, local_start, local_stop) per patch along this axis.
        positions = list(range(0, n - P + 1, P))
        specs = [(x, 0, P) for x in positions]
        if n % P != 0:
            tail = n - (positions[-1] + P) if positions else n
            specs.append((n - P, P - tail, P))
        return specs

    nx, ny, nz = grid_shape
    sx = axis_specs(nx)
    sy = axis_specs(ny)
    sz = axis_specs(nz)
    offsets = []
    local_slices = []
    for ox, lx0, lx1 in sx:
        for oy, ly0, ly1 in sy:
            for oz, lz0, lz1 in sz:
                offsets.append((ox, oy, oz))
                local_slices.append((lx0, lx1, ly0, ly1, lz0, lz1))
    return offsets, local_slices


def build_all_patch_input_ids(
    density,         # (nx, ny, nz) float
    grid_shape,      # (nx, ny, nz)
    offsets,         # (n_patches, 3) array of (ix, iy, iz)
    Zs,              # (N,) int
    frac,            # (N, 3) float
    lmq_codec,
    vocab_offsets,
    P=14,
    pad_to=8192,
    lattice_params=None,  # (a, b, c, α, β, γ); required when lat-aware
):
    """Vectorized: return (n_patches, pad_to) int32 input_ids matrix.

    Layout matches tokenize_patch_for_eval; per-mat preamble is identical
    across patches (BOS, GRID, [LATTICE,] ATOMS, POS, SHAPE) — only
    OFFSET, HI, DENS blocks vary per patch. Built once via numpy ops, no
    Python per-patch loop. ``vocab_offsets["lat_aware"]`` (bool) selects
    the LATTICE-block-emitting layout; in that case ``lattice_params``
    must be supplied.
    """
    bounds, recon, clip_max = lmq_codec
    n_specials = vocab_offsets["n_specials"]
    int_off = n_specials + N_ATOMS  # 136 (pre-lat) or 138 (lat)
    density_offset = vocab_offsets["density_offset"]
    lat_aware = vocab_offsets.get("lat_aware", False)

    offsets_arr = np.asarray(offsets, dtype=np.int32)  # (n, 3)
    n = offsets_arr.shape[0]
    grid_np = np.asarray(grid_shape, dtype=np.int32)

    # ---- Per-mat preamble (constant across patches) ----
    pre = [TOK["BOS"]]
    pre += [TOK["GRID_START"], int_off + int(grid_shape[0]), int_off + int(grid_shape[1]),
            int_off + int(grid_shape[2]), TOK["GRID_END"]]
    if lat_aware:
        if lattice_params is None:
            raise ValueError("lat-aware dataset requires lattice_params")
        a, b, c, alpha, beta, gamma = lattice_params
        lat_ints = [
            int(round(a / LATTICE_LENGTH_RES_A)),
            int(round(b / LATTICE_LENGTH_RES_A)),
            int(round(c / LATTICE_LENGTH_RES_A)),
            int(round(alpha / LATTICE_ANGLE_RES_DEG)),
            int(round(beta  / LATTICE_ANGLE_RES_DEG)),
            int(round(gamma / LATTICE_ANGLE_RES_DEG)),
        ]
        if any(v < 0 or v >= N_INTS for v in lat_ints):
            raise ValueError(
                f"lattice quantization out of range: params={lattice_params} → "
                f"ints={lat_ints} (must be [0, {N_INTS}))"
            )
        pre += [TOK["LATTICE_START"]] + [int_off + v for v in lat_ints] + [TOK["LATTICE_END"]]
    pre += [TOK["ATOMS_START"]] + [n_specials + int(z - 1) for z in Zs] + [TOK["ATOMS_END"]]
    pre += [TOK["POS_START"]]
    for xyz in frac:
        for c in xyz:
            pre.extend(_pos_tokens(float(c), vocab_offsets))
    pre += [TOK["POS_END"]]
    pre += [TOK["SHAPE_START"], int_off + P, int_off + P, int_off + P, TOK["SHAPE_END"]]
    preamble = np.array(pre, dtype=np.int32)
    preamble_len = preamble.shape[0]

    # ---- Per-patch OFFSET / HI tokens (n, 3) ----
    offset_tokens = int_off + offsets_arr  # (n, 3)
    his = (offsets_arr + (P - 1)) % grid_np  # (n, 3) PBC wrap
    hi_tokens = int_off + his  # (n, 3)

    # ---- Per-patch DENS tokens via vectorized fancy-indexing extract ----
    # Build a (P, P, P) base index grid, broadcast-add each offset.
    # Result: patches[i] = density[ox[i]:ox[i]+P, oy[i]:oy[i]+P, oz[i]:oz[i]+P]
    iota = np.arange(P, dtype=np.int32)
    di, dj, dk = np.meshgrid(iota, iota, iota, indexing='ij')  # (P,P,P)
    # (n, P, P, P) absolute indices for each axis:
    idx_x = offsets_arr[:, 0, None, None, None] + di[None]  # (n, P, P, P)
    idx_y = offsets_arr[:, 1, None, None, None] + dj[None]
    idx_z = offsets_arr[:, 2, None, None, None] + dk[None]
    patches = density[idx_x, idx_y, idx_z]  # (n, P, P, P) float
    flat = patches.reshape(n, P * P * P)
    clipped = np.clip(flat, 0.0, clip_max)
    bins = np.searchsorted(bounds, clipped, side="right").astype(np.int32)
    dens_tokens = bins + density_offset  # (n, P^3)

    # ---- Assemble (n, pad_to) int32 array ----
    ids = np.full((n, pad_to), TOK["PAD"], dtype=np.int32)
    p = 0
    ids[:, p:p + preamble_len] = preamble[None, :]
    p += preamble_len
    ids[:, p] = TOK["OFFSET_START"]; p += 1
    ids[:, p:p + 3] = offset_tokens; p += 3
    ids[:, p] = TOK["OFFSET_END"]; p += 1
    ids[:, p] = TOK["HI_START"]; p += 1
    ids[:, p:p + 3] = hi_tokens; p += 3
    ids[:, p] = TOK["HI_END"]; p += 1
    ids[:, p] = TOK["DENS_START"]; p += 1
    ids[:, p:p + P ** 3] = dens_tokens; p += P ** 3
    ids[:, p] = TOK["DENS_END"]; p += 1
    ids[:, p] = TOK["EOS"]; p += 1
    if p > pad_to:
        raise ValueError(f"Token sequence too long: {p} > pad_to={pad_to}")
    return ids


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full-lmq-v2")
    model_preset = os.environ.get("TOMAT_MODEL", "200M")
    checkpoint_path = os.environ["TOMAT_CHECKPOINT"]
    lmq_path = os.environ["TOMAT_LMQ_PATH"]
    n_mats_cap = int(os.environ.get("TOMAT_EVAL_N_MATS", "10"))
    n_mats_skip = int(os.environ.get("TOMAT_EVAL_SKIP", "0"))
    # Optional: pin to a fixed mat-id set from data/eval_mat_ids.json
    # (e.g. "val_200" or "train_200") for apples-to-apples curves across runs.
    eval_mat_set = os.environ.get("TOMAT_EVAL_MAT_SET", "").strip()
    eval_batch = int(os.environ.get("TOMAT_EVAL_BATCH", "8"))
    decoder = os.environ.get("TOMAT_EVAL_DECODER", "median")
    if decoder not in ("argmax", "median", "mean"):
        raise ValueError(f"TOMAT_EVAL_DECODER must be argmax/median/mean, got {decoder!r}")
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

    # Vocab offsets — detect lat-awareness from meta.json's specials map.
    # Lat-aware datasets carry [LATTICE_START]/[LATTICE_END] in specials and
    # bump N_SPECIALS 18 → 20.
    specials = meta["vocab"].get("specials", {})
    lat_aware = "[LATTICE_START]" in specials or "LATTICE_START" in specials
    n_specials = N_SPECIALS_LAT if lat_aware else N_SPECIALS_PRELAT
    if len(specials) and len(specials) != n_specials:
        err(f"[eval-mat] WARN: meta specials count {len(specials)} != expected {n_specials}")
    pc = meta["vocab"]["position_codec"]
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    density_offset = n_specials + N_ATOMS + N_INTS + pos_total
    vocab_offsets = {
        "density_offset": density_offset,
        "pos_offset": n_specials + N_ATOMS + N_INTS,
        "pos_vocabs": pos_signed_vocabs,
        "pos_log_min": pc["log_min"],
        "pos_log_max": pc["log_max"],
        "n_specials": n_specials,
        "lat_aware": lat_aware,
    }
    err(f"[eval-mat] lat_aware={lat_aware}, n_specials={n_specials}, "
        f"DENSITY_OFFSET={density_offset}, density_range=[{density_offset}, {vocab_size})")

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

        # Decoder lives on-TPU: returns (B, Pos) float density predictions
        # (not logits / not bin indices). Avoids transferring full vocab dist.
        # The distribution is restricted to the density range and renormalized
        # before decoding — non-density-token mass doesn't contribute.
        DENS_LO = density_offset
        DENS_HI = density_offset + len(recon)
        decode_dens = jnp.asarray(recon, dtype=jnp.float32)  # (n_bins,) float

        if decoder == "argmax":
            @hax.named_jit(axis_resources=compute_mapping)
            def forward_decode(tokens_in):
                act = model.activations(tokens_in, key=None, attn_mask=None)
                head = model.get_lm_head()
                logits = hax.dot(act, head, axis=model.Embed)
                logits_arr = logits.array.astype(jnp.float32)  # (B, Pos, V)
                density_logits = logits_arr[..., DENS_LO:DENS_HI]  # (B, Pos, n_bins)
                bin_idx = jnp.argmax(density_logits, axis=-1)
                rho = decode_dens[bin_idx]  # (B, Pos)
                return hax.named(rho, (logits.axes[0], logits.axes[1]))
        elif decoder == "median":
            @hax.named_jit(axis_resources=compute_mapping)
            def forward_decode(tokens_in):
                act = model.activations(tokens_in, key=None, attn_mask=None)
                head = model.get_lm_head()
                logits = hax.dot(act, head, axis=model.Embed)
                logits_arr = logits.array.astype(jnp.float32)
                probs = jax.nn.softmax(logits_arr, axis=-1)
                density_probs = probs[..., DENS_LO:DENS_HI]
                sum_dens = density_probs.sum(axis=-1, keepdims=True) + 1e-12
                p_norm = density_probs / sum_dens
                cumP = jnp.cumsum(p_norm, axis=-1)
                # smallest bin where cumP >= 0.5
                bin_idx = jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32)
                bin_idx = jnp.clip(bin_idx, 0, len(decode_dens) - 1)
                rho = decode_dens[bin_idx]
                return hax.named(rho, (logits.axes[0], logits.axes[1]))
        else:  # mean
            @hax.named_jit(axis_resources=compute_mapping)
            def forward_decode(tokens_in):
                act = model.activations(tokens_in, key=None, attn_mask=None)
                head = model.get_lm_head()
                logits = hax.dot(act, head, axis=model.Embed)
                logits_arr = logits.array.astype(jnp.float32)
                probs = jax.nn.softmax(logits_arr, axis=-1)
                density_probs = probs[..., DENS_LO:DENS_HI]
                sum_dens = density_probs.sum(axis=-1, keepdims=True) + 1e-12
                p_norm = density_probs / sum_dens
                rho = jnp.einsum("bpv,v->bp", p_norm, decode_dens)
                return hax.named(rho, (logits.axes[0], logits.axes[1]))

        # Pick mp-IDs: either from the pinned JSON snapshot (preferred for
        # apples-to-apples curves across runs), or from the parquets directly.
        if eval_mat_set:
            mat_ids_path = os.environ.get(
                "TOMAT_EVAL_MAT_IDS_FILE",
                "gs://marin-eu-west4/tomat/eval/eval_mat_ids.json",
            )
            with fsspec.open(mat_ids_path, "r") as f:
                mat_ids_blob = json.load(f)
            if eval_mat_set not in mat_ids_blob:
                raise KeyError(f"set {eval_mat_set!r} not in {mat_ids_path}; have {list(mat_ids_blob)}")
            all_ids = mat_ids_blob[eval_mat_set]
            mp_ids = all_ids[n_mats_skip : n_mats_skip + n_mats_cap]
            err(f"[eval-mat] mat_set={eval_mat_set}, skip={n_mats_skip}, cap={n_mats_cap}; "
                f"eval on {len(mp_ids)} mats")
        else:
            import pyarrow.parquet as pq
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            parquet_glob = f"marin-eu-west4/tomat/tokenized/{label}/worker-*/*.parquet"
            shard_paths = sorted(fs.glob(parquet_glob))
            all_ids: list[str] = []
            target = n_mats_skip + n_mats_cap
            for shard in shard_paths:
                with fs.open(shard, "rb") as f:
                    tbl = pq.ParquetFile(f).read(columns=["task_id"])
                for tid in tbl.column("task_id").to_pylist():
                    if tid not in all_ids:
                        all_ids.append(tid)
                    if len(all_ids) >= target:
                        break
                if len(all_ids) >= target:
                    break
            mp_ids = all_ids[n_mats_skip : n_mats_skip + n_mats_cap]
            err(f"[eval-mat] skip={n_mats_skip}, cap={n_mats_cap}; eval on {len(mp_ids)} mats: {mp_ids}")

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
            Zs, frac, lattice_params = load_structure_from_zarr_attrs(group)
            err(f"[eval-mat] mat={mp_id}: grid={grid_shape}, atoms={len(Zs)}, "
                f"lattice=({lattice_params[0]:.2f},{lattice_params[1]:.2f},"
                f"{lattice_params[2]:.2f}) Å α,β,γ=("
                f"{lattice_params[3]:.1f},{lattice_params[4]:.1f},{lattice_params[5]:.1f})°")

            # Tile — full coverage, exactly one patch per voxel.
            offsets, local_slices = tile_full_coverage_offsets(grid_shape, P)
            n_patches = len(offsets)
            err(f"[eval-mat] tiling: {n_patches} patches (full coverage), grid={grid_shape}")

            rho_pred = np.zeros(grid_shape, dtype=np.float32)

            # Pad to multiple of eval_batch so every patch is evaluated
            # (TPU forward needs fixed batch shape).
            n_full_batches = (n_patches + eval_batch - 1) // eval_batch
            n_eval = n_full_batches * eval_batch
            pad_count = n_eval - n_patches
            offsets_padded = list(offsets) + [offsets[0]] * pad_count
            offsets_arr = np.asarray(offsets_padded, dtype=np.int32)

            # ---- Vectorized pre-tokenize: build all (n_eval, 8192) input_ids in one numpy pass ----
            t_tok0 = time.time()
            pad_to = 8192
            all_ids = build_all_patch_input_ids(
                density, grid_shape, offsets_arr, Zs, frac,
                lmq_codec, vocab_offsets, P=P, pad_to=pad_to,
                lattice_params=lattice_params,
            )  # (n_eval, 8192) int32
            t_tok = time.time() - t_tok0

            # Density-token positions are deterministic (same layout per patch).
            n_bins = len(recon)
            row0 = all_ids[0]
            is_dens = (row0 >= density_offset) & (row0 < density_offset + n_bins)
            dens_positions_t = np.where(is_dens)[0]
            if len(dens_positions_t) != P ** 3:
                err(f"[eval-mat] WARN: expected {P**3} density positions, got {len(dens_positions_t)}")
                continue
            # For each density INPUT position t, the prediction came from output @ t-1.
            pred_positions = dens_positions_t - 1  # (P^3,)

            # ---- TPU loop with depth-2 async pipelining ----
            # forward_decode returns (B, Pos) float32 density predictions on-TPU
            # (decoder = argmax|median|mean, restricted to density range).
            t_fwd0 = time.time()
            Batch = hax.Axis("batch", eval_batch)
            Pos = hax.Axis("position", pad_to)

            def dispatch(bi):
                start = bi * eval_batch
                tokens_np = all_ids[start : start + eval_batch]
                tokens_ha = hax.named(jnp.asarray(tokens_np), (Batch, Pos))
                return forward_decode(tokens_ha)  # async; NamedArray (Batch, Pos) float

            inflight = []  # list of (bi, future)
            depth = 2
            total = 0
            for bi in range(n_full_batches):
                inflight.append((bi, dispatch(bi)))
                while len(inflight) > depth or (bi == n_full_batches - 1 and inflight):
                    done_bi, fut = inflight.pop(0)
                    pred_floats = np.asarray(fut.array, dtype=np.float32)  # (B, Pos)
                    start = done_bi * eval_batch
                    pred_dens_floats = pred_floats[:, pred_positions]  # (B, P^3)
                    pred_blocks = pred_dens_floats.reshape(-1, P, P, P)
                    for k in range(eval_batch):
                        global_idx = start + k
                        if global_idx >= n_patches:  # padded entry, skip
                            continue
                        ox, oy, oz = offsets[global_idx]
                        lx0, lx1, ly0, ly1, lz0, lz1 = local_slices[global_idx]
                        rho_pred[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            pred_blocks[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        total += 1
            t_fwd = time.time() - t_fwd0
            err(f"[eval-mat] {mp_id}: processed {total} patches "
                f"(decoder={decoder}, tokenize={t_tok:.2f}s, "
                f"forward+decode={t_fwd:.2f}s, "
                f"{1000 * t_fwd / max(total, 1):.1f}ms/patch)")

            # NMAE over the full grid (every voxel evaluated exactly once).
            rho_true = density
            mae = np.mean(np.abs(rho_pred - rho_true))
            denom = np.mean(np.abs(rho_true))
            nmae = mae / max(denom, 1e-30)
            err(f"[eval-mat] {mp_id}: MAE={mae:.4e}, mean|ρ_true|={denom:.4e}, NMAE={nmae:.4%}")
            per_mat_results.append({
                "mp_id": mp_id,
                "grid_shape": list(grid_shape),
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

        # Machine-readable summary (also persists per-mat results to GCS so
        # downstream noise-calibration / bootstrap analysis can use them
        # without scraping iris job logs, which truncate per-mat lines).
        summary = {
            "checkpoint": checkpoint_path,
            "label": label,
            "model": model_preset,
            "mat_set": eval_mat_set,
            "n_mats": len(per_mat_results),
            "nmae_mean": float(nmaes.mean()) if per_mat_results else None,
            "nmae_median": float(np.median(nmaes)) if per_mat_results else None,
            "nmae_p99": float(np.percentile(nmaes, 99)) if per_mat_results else None,
            "per_mat": per_mat_results,
        }
        print(json.dumps(summary, indent=2))

        # Persist to GCS keyed by checkpoint + mat-set, so bootstrap noise
        # estimation can read per-mat values for any prior eval.
        if per_mat_results:
            # Levanter checkpointer lays out as <base_path>/<run_id>/step-N, where
            # base_path here is `<BUCKET>/results/<RESULTS_LABEL>/checkpoints` and
            # run_id == RESULTS_LABEL — so the path doubles the label, e.g.
            #   .../results/<RL>/checkpoints/<RL>/step-1000
            # Components from the tail: -1=step, -2=RL, -3=checkpoints, -4=RL.
            parts = checkpoint_path.rstrip("/").split("/")
            ckpt_tail = parts[-1]
            run_label = parts[-4]
            ms = eval_mat_set or "default"
            results_path = f"{BUCKET}/eval/results/{run_label}/{ms}/{ckpt_tail}.json"
            try:
                with fsspec.open(results_path, "w") as f:
                    json.dump(summary, f, indent=2)
                err(f"[eval-mat] persisted per-mat results to {results_path}")
            except Exception as e:
                err(f"[eval-mat] WARNING: failed to persist per-mat to GCS: {e}")


if __name__ == "__main__":
    main()
