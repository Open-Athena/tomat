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
    TOMAT_EVAL_MODE         "teacher" (default; one forward, true density fed
                            in) or "free" (autoregressive — the model's own
                            predictions feed back; v3-only). See specs/25 §1.
                            Free mode covers the whole grid (every patch),
                            same disjoint tiling as teacher-forced.
    TOMAT_EVAL_FREE_BATCH   (free mode) patches per forward — an HBM knob, not
                            a coverage cap (default 32). Free-running is ~P^3
                            sequential forwards/patch; cap cost via the mat
                            count (TOMAT_EVAL_N_MATS), not this.
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
from levanter.layers.attention import AttentionBackend, AttentionMask
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


def _pos_tokens_vec(coords, vocab_offsets):
    """Vectorized version of `_pos_tokens` — accepts any-shape numpy array of
    coords, returns array of shape ``(*coords.shape, 3)`` of absolute token IDs.

    Used by the v3 builder where atom positions are translated per-patch and
    we need to emit position tokens for shape ``(n_patches, n_atoms, 3)``
    coords in one numpy pass.
    """
    pos_off = vocab_offsets["pos_offset"]
    pos_vocabs = vocab_offsets["pos_vocabs"]  # (512, 256, 256)
    log_min = vocab_offsets["pos_log_min"]
    log_max = vocab_offsets["pos_log_max"]

    # Match FP16Codec._log_bin_index exactly: clip log_vals (not normalized),
    # truncate via .astype, force bin_index=0 when mag < floor. The canonical
    # codec truncates rather than rounds — `_pos_tokens` (the v2 sibling) uses
    # round, which is an off-by-one bug carried over; we fix it here.
    MAGNITUDE_FLOOR = 1e-15
    coords = np.asarray(coords, dtype=np.float64)
    mag = np.abs(coords)
    safe_mag = np.maximum(mag, MAGNITUDE_FLOOR)
    log_mag = np.log10(safe_mag)
    log_clipped = np.clip(log_mag, log_min, log_max)
    total_bins = 1 << 24
    normalized = (log_clipped - log_min) / (log_max - log_min)
    bin_index = (normalized * (total_bins - 1)).astype(np.int64)
    bin_index = np.clip(bin_index, 0, total_bins - 1)
    bin_index = np.where(mag < MAGNITUDE_FLOOR, 0, bin_index)

    # Three 8-bit chunks, MSB first.
    chunks = np.stack([
        (bin_index >> 16) & 0xff,
        (bin_index >> 8) & 0xff,
        bin_index & 0xff,
    ], axis=-1)  # (*coords.shape, 3)

    # First chunk gets sign bit (negative coords → add 256).
    first_half = 1 << 8
    signs_positive = coords >= 0
    chunks_first = np.where(signs_positive, chunks[..., 0], first_half + chunks[..., 0])
    chunks = np.concatenate([chunks_first[..., None], chunks[..., 1:]], axis=-1)

    # Cumulative offsets within position-codec vocab.
    cum = np.array([0, pos_vocabs[0], pos_vocabs[0] + pos_vocabs[1]], dtype=np.int64)
    return (pos_off + cum + chunks).astype(np.int32)  # (*coords.shape, 3)


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


def build_all_patch_input_ids_v3(
    density,         # (nx, ny, nz) float
    grid_shape,      # (nx, ny, nz)
    offsets,         # (n_patches, 3) array of (ix, iy, iz)
    Zs,              # (N,) int
    frac,            # (N, 3) float — global frac coords
    lmq_codec,
    vocab_offsets,
    P=19,
    pad_to=8192,
    lattice_params=None,
):
    """v3 builder: vectorized (n_patches, pad_to) int32 input_ids matrix.

    Differences from `build_all_patch_input_ids` (v2):
    - Atom positions are translated to each patch's frame (subtract
      ``offset / grid_shape`` and re-mod 1).
    - Drops SHAPE / OFFSET / HI blocks.
    - POS block now varies per patch — vectorized via `_pos_tokens_vec`.

    Layout:
        BOS | GRID | LATTICE | ATOMS | POS_translated | DENS | EOS
    """
    bounds, recon, clip_max = lmq_codec
    n_specials = vocab_offsets["n_specials"]
    int_off = n_specials + N_ATOMS
    density_offset = vocab_offsets["density_offset"]
    lat_aware = vocab_offsets.get("lat_aware", False)

    offsets_arr = np.asarray(offsets, dtype=np.int32)  # (n, 3)
    n = offsets_arr.shape[0]
    grid_np = np.asarray(grid_shape, dtype=np.int32)
    n_atoms = len(Zs)

    # ---- Static prefix (BOS, GRID, LATTICE, ATOMS): same across patches ----
    pre = [TOK["BOS"]]
    pre += [TOK["GRID_START"], int_off + int(grid_shape[0]), int_off + int(grid_shape[1]),
            int_off + int(grid_shape[2]), TOK["GRID_END"]]
    if not lat_aware:
        # v3 expects lat-aware (LMQ-era) tokenizer; fall back loud.
        raise ValueError("v3 builder requires lat_aware=True (LATTICE block in vocab)")
    if lattice_params is None:
        raise ValueError("v3 / lat-aware needs lattice_params")
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
            f"lattice quantization out of range: params={lattice_params} → ints={lat_ints}"
        )
    pre += [TOK["LATTICE_START"]] + [int_off + v for v in lat_ints] + [TOK["LATTICE_END"]]
    pre += [TOK["ATOMS_START"]] + [n_specials + int(z - 1) for z in Zs] + [TOK["ATOMS_END"]]
    static_pre = np.array(pre, dtype=np.int32)
    static_pre_len = static_pre.shape[0]

    # ---- Per-patch translated POS tokens ----
    # delta: (n, 3) fractional shift
    delta = offsets_arr.astype(np.float64) / grid_np.astype(np.float64)
    # translated coords: (n, n_atoms, 3) = (frac[None] - delta[:, None, :]) % 1
    translated = (np.asarray(frac, dtype=np.float64)[None, :, :]
                  - delta[:, None, :]) % 1.0
    # _pos_tokens_vec → (n, n_atoms, 3, 3): one extra trailing dim of 3 tokens.
    pos_tok = _pos_tokens_vec(translated, vocab_offsets)
    # Flatten to (n, n_atoms * 9): per-patch row of position tokens.
    pos_block = pos_tok.reshape(n, n_atoms * 9)

    # ---- Per-patch DENS tokens (vectorized fancy-indexing extract) ----
    iota = np.arange(P, dtype=np.int32)
    di, dj, dk = np.meshgrid(iota, iota, iota, indexing='ij')
    idx_x = (offsets_arr[:, 0, None, None, None] + di[None]) % grid_np[0]
    idx_y = (offsets_arr[:, 1, None, None, None] + dj[None]) % grid_np[1]
    idx_z = (offsets_arr[:, 2, None, None, None] + dk[None]) % grid_np[2]
    patches = density[idx_x, idx_y, idx_z]  # (n, P, P, P)
    flat = patches.reshape(n, P * P * P)
    clipped = np.clip(flat, 0.0, clip_max)
    bins = np.searchsorted(bounds, clipped, side="right").astype(np.int32)
    dens_tokens = bins + density_offset  # (n, P^3)

    # ---- Assemble ----
    ids = np.full((n, pad_to), TOK["PAD"], dtype=np.int32)
    p = 0
    ids[:, p:p + static_pre_len] = static_pre[None, :]
    p += static_pre_len
    ids[:, p] = TOK["POS_START"]; p += 1
    ids[:, p:p + n_atoms * 9] = pos_block; p += n_atoms * 9
    ids[:, p] = TOK["POS_END"]; p += 1
    ids[:, p] = TOK["DENS_START"]; p += 1
    ids[:, p:p + P ** 3] = dens_tokens; p += P ** 3
    ids[:, p] = TOK["DENS_END"]; p += 1
    ids[:, p] = TOK["EOS"]; p += 1
    if p > pad_to:
        raise ValueError(f"v3 token sequence too long: {p} > pad_to={pad_to}")
    return ids


# Free-running decode re-forwards the (growing) prefix each step; padding the
# forward to the next FREE_BUCKET multiple bounds the number of JIT shapes to
# ~ceil(max_seq_len / FREE_BUCKET) while avoiding a full-length forward at
# every early step. FREE_BUCKET MUST be a multiple of the flash-attention
# block size (1024) — flash attention rejects a query axis that isn't a
# block-size multiple. The token buffer is also padded to a FREE_BUCKET
# multiple so the largest bucket is valid too.
FREE_BUCKET = 1024


def _next_bucket(n):
    """Smallest FREE_BUCKET multiple >= n — the bucketed forward length."""
    return ((n + FREE_BUCKET - 1) // FREE_BUCKET) * FREE_BUCKET


def decode_density(density_logits, target_dens, decoder, decode_dens):
    """Decode per-position density-bin logits → (rho point estimate, EMD).

    Args:
      density_logits: (B, Pos, n_bins) raw logits over the density-token bins.
        MUST be the raw density logits — *not* a slice of the full-vocab
        softmax. Slicing then renormalizing is identical in exact arithmetic,
        but underflows fp32 to an all-zero distribution when the model
        confidently predicts a non-density token (density logits >~80 below
        the vocab max → exp() flushes to 0). An all-zero p_norm decodes
        (median) to the last bin — rho = decode_dens[-1], ~3e4 of garbage —
        while EMD = dot(0, ·) = 0 (fake-perfect): the NMAE-vs-NEMD blowup.
        softmax over the density logits alone max-subtracts within the density
        range, so it is always a valid normalized distribution.
      target_dens: (B, Pos) true density value (codec round-trip) for the EMD.
      decoder: "argmax" | "median" | "mean" — the point-estimate rule.
      decode_dens: (n_bins,) bin index → density value.

    Returns (rho, emd), each (B, Pos).
    """
    n_bins = decode_dens.shape[0]
    p_norm = jax.nn.softmax(density_logits, axis=-1)  # (B, Pos, n_bins)
    if decoder == "argmax":
        rho = decode_dens[jnp.argmax(density_logits, axis=-1)]
    elif decoder == "median":
        # smallest bin where the cumulative mass reaches 0.5
        cumP = jnp.cumsum(p_norm, axis=-1)
        bin_idx = jnp.clip(
            jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32), 0, n_bins - 1,
        )
        rho = decode_dens[bin_idx]
    elif decoder == "mean":
        rho = jnp.einsum("bpv,v->bp", p_norm, decode_dens)
    else:
        raise ValueError(f"unknown decoder: {decoder!r}")
    # mat-EMD: per-position Wasserstein-1 vs the delta target, E_P[|v - t|].
    # vmap (not a (B, Pos, n_bins) einsum) to avoid materializing the abs-diff.
    def _emd1(p_v, t_d):
        return jnp.dot(p_v, jnp.abs(decode_dens - t_d))
    emd = jax.vmap(jax.vmap(_emd1))(p_norm, target_dens)
    return rho, emd


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full-v3")
    model_preset = os.environ.get("TOMAT_MODEL", "200M")
    checkpoint_path = os.environ["TOMAT_CHECKPOINT"]
    lmq_path = os.environ["TOMAT_LMQ_PATH"]
    n_mats_cap = int(os.environ.get("TOMAT_EVAL_N_MATS", "10"))
    n_mats_skip = int(os.environ.get("TOMAT_EVAL_SKIP", "0"))
    # Optional: pin to a fixed mat-id set from data/eval_mat_ids.json
    # (e.g. "val_200" or "train_200") for apples-to-apples curves across runs.
    eval_mat_set = os.environ.get("TOMAT_EVAL_MAT_SET", "").strip()
    eval_batch = int(os.environ.get("TOMAT_EVAL_BATCH", "8"))
    # Attention backend: default (None) tries NVTE → falls back to the slow
    # reference impl when transformer_engine is absent. Set TOMAT_ATTN_BACKEND
    # =JAX_FLASH for the GPU eval path (pure-Pallas flash, ~5-10× faster, same
    # math). Flash is exact — differs from reference only by fp-rounding.
    attn_backend = os.environ.get("TOMAT_ATTN_BACKEND", "").strip()
    # Debug: if set, eval ONLY this mp_id and dump its per-voxel grids to GCS
    # (rho_pred / rho_true / emd_grid) — for pinning the NMAE-vs-NEMD bug.
    debug_dump_mat = os.environ.get("TOMAT_DEBUG_DUMP_MAT", "").strip()
    decoder = os.environ.get("TOMAT_EVAL_DECODER", "median")
    if decoder not in ("argmax", "median", "mean"):
        raise ValueError(f"TOMAT_EVAL_DECODER must be argmax/median/mean, got {decoder!r}")
    split = os.environ.get("TOMAT_EVAL_SPLIT", "validation")
    zarr_base = os.environ.get(
        "TOMAT_ZARR_BASE", f"{BUCKET}/rho_gga_raw/{split}"
    )
    seed = int(os.environ.get("TOMAT_SEED", "42"))
    # Eval mode: teacher-forced (default), free-running AR, or MaskGIT iterative.
    eval_mode = os.environ.get("TOMAT_EVAL_MODE", "teacher").strip()
    if eval_mode not in ("teacher", "free", "maskgit"):
        raise ValueError(f"TOMAT_EVAL_MODE must be teacher/free/maskgit, got {eval_mode!r}")
    free_batch = int(os.environ.get("TOMAT_EVAL_FREE_BATCH", "32"))

    # MaskGIT eval knobs.
    mg_k_steps = int(os.environ.get("TOMAT_EVAL_MG_K_STEPS", "12"))
    mg_decoder = os.environ.get("TOMAT_EVAL_MG_DECODER", decoder)  # default to TOMAT_EVAL_DECODER
    if mg_decoder not in ("argmax", "median", "mean"):
        raise ValueError(f"TOMAT_EVAL_MG_DECODER must be argmax/median/mean, got {mg_decoder!r}")

    # Partial-mask diagnostic: if set (comma-sep list of floats in (0,1]),
    # short-circuit the maskgit iterative loop. For each ratio r, mask r
    # fraction of density positions, leave the rest as GT, do ONE forward,
    # report NMAE on the masked positions only. Used to test whether the
    # model has learned anything at all (good at r=0.1) vs nothing (flat
    # at all r). Bypasses K-iteration entirely.
    mg_partial_ratios_env = os.environ.get("TOMAT_EVAL_MG_PARTIAL_RATIOS", "").strip()
    mg_partial_ratios = (
        [float(x) for x in mg_partial_ratios_env.split(",") if x.strip()]
        if mg_partial_ratios_env else []
    )

    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    # In MaskGIT mode the model was trained with a +1 vocab (MASK token at end).
    # Bump vocab_size to match the checkpoint's embedding table.
    MASK_ID: int | None = None
    if eval_mode == "maskgit":
        MASK_ID = vocab_size
        vocab_size = vocab_size + 1
    patch_size_meta = meta.get("patch_size")
    P = int(patch_size_meta) if isinstance(patch_size_meta, int) else 14
    tokenizer_version = meta.get("tokenizer_version", "v2")
    err(f"[eval-mat] label={label}, model={model_preset}, P={P}, vocab={vocab_size}, "
        f"tokenizer={tokenizer_version}, mode={eval_mode}"
        + (f", MASK_ID={MASK_ID}, mg_k_steps={mg_k_steps}" if eval_mode == "maskgit" else ""))

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
        attn_backend=AttentionBackend[attn_backend] if attn_backend else None,
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
        # (not logits / not bin indices). Avoids transferring the full vocab
        # distribution. See decode_density() for the point-estimate + EMD math.
        DENS_LO = density_offset
        DENS_HI = density_offset + len(recon)
        decode_dens = jnp.asarray(recon, dtype=jnp.float32)  # (n_bins,) float

        # Causal attention mask for AR-trained models — matches training-time
        # mask built by `LmExample.causal(...)` (`lm_model.py:62`). Levanter's
        # default of `attn_mask=None` materializes to FullMask (full
        # bidirectional) in every backend, NOT causal — see spec 41 for the
        # full audit. Passing `None` here would silently OOD-eval the model
        # under a mask it never saw at training time.
        _ar_causal_mask = AttentionMask.causal()

        @hax.named_jit(axis_resources=compute_mapping)
        def forward_decode(tokens_in):
            act = model.activations(tokens_in, key=None, attn_mask=_ar_causal_mask)
            head = model.get_lm_head()
            logits = hax.dot(act, head, axis=model.Embed)
            logits_arr = logits.array.astype(jnp.float32)  # (B, Pos, V)
            density_logits = logits_arr[..., DENS_LO:DENS_HI]  # (B, Pos, n_bins)
            # EMD target = the input token at the NEXT position: logits at
            # output position p predict input token p+1, so the target for the
            # prediction at p is tokens_in[..., p+1]. Shift left and pad the
            # last column (unused — the inference loop reads emd only at
            # pred_positions = dens_positions - 1, never the last position).
            tokens_arr = tokens_in.array  # (B, Pos)
            tokens_next = jnp.concatenate(
                [tokens_arr[..., 1:], tokens_arr[..., -1:]], axis=-1,
            )
            target_bins = jnp.clip(tokens_next - DENS_LO, 0, len(decode_dens) - 1)
            target_dens = decode_dens[target_bins]  # (B, Pos)
            rho, emd = decode_density(density_logits, target_dens, decoder, decode_dens)
            return (
                hax.named(rho, (logits.axes[0], logits.axes[1])),
                hax.named(emd, (logits.axes[0], logits.axes[1])),
            )

        # ---- Free-running (autoregressive) decode --------------------------
        # One recompute step: forward the prefix, read the density logits at
        # the frontier output position, median-decode → fed-back token; also
        # return the configured-decoder point estimate (rho) and the EMD vs
        # the true voxel density. No KV cache — the prefix is re-forwarded
        # each step, padded up to the next FREE_BUCKET multiple (see
        # specs/25 §1 for why recompute-first).
        @hax.named_jit(axis_resources=compute_mapping)
        def free_step(tokens_in, frontier, true_dens_i):
            # tokens_in: (Batch, Pos) prefix; frontier: i32 scalar output
            # position to read; true_dens_i: (Batch,) true voxel density.
            act = model.activations(tokens_in, key=None, attn_mask=_ar_causal_mask)
            head = model.get_lm_head()
            logits = hax.dot(act, head, axis=model.Embed)
            logits_arr = logits.array.astype(jnp.float32)          # (B, Pos, V)
            density_logits = logits_arr[..., DENS_LO:DENS_HI]      # (B, Pos, n_bins)
            dl = density_logits[:, frontier, :]                    # (B, n_bins)
            p = jax.nn.softmax(dl, axis=-1)
            cumP = jnp.cumsum(p, axis=-1)
            med_bin = jnp.clip(
                jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32), 0, len(recon) - 1,
            )
            next_tok = med_bin + DENS_LO                           # (B,) fed back
            if decoder == "argmax":
                rho = decode_dens[jnp.argmax(dl, axis=-1)]
            elif decoder == "median":
                rho = decode_dens[med_bin]
            else:  # mean — E[rho]
                rho = jnp.einsum("bv,v->b", p, decode_dens)
            td = true_dens_i.array                                 # (B,)
            emd = jax.vmap(lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t)))(p, td)
            B = tokens_in.axes[0]
            return (hax.named(next_tok, (B,)), hax.named(rho, (B,)),
                    hax.named(emd, (B,)))

        # ---- MaskGIT iterative-refinement forward --------------------------
        # One full-sequence bidirectional forward: tokens_in has density
        # positions that are either MASK_ID or already filled-in. Matches
        # the bidir mask used during MaskGIT training (see
        # `Qwen3MaskGITLMHeadModel.compute_next_token_loss` in
        # `marin/qwen3_density.py`).
        _mg_bidir_mask = AttentionMask(is_causal=False)

        @hax.named_jit(axis_resources=compute_mapping)
        def maskgit_forward(tokens_in):
            """Forward with full bidirectional attention. Returns density logits.

            tokens_in: (Batch, Pos) NamedArray.
            Returns: density_logits (B, Pos, n_bins) float32 raw array.
            """
            act = model.activations(tokens_in, key=None, attn_mask=_mg_bidir_mask)
            head = model.get_lm_head()
            logits = hax.dot(act, head, axis=model.Embed)
            logits_arr = logits.array.astype(jnp.float32)  # (B, Pos, V)
            return logits_arr[..., DENS_LO:DENS_HI]       # (B, Pos, n_bins)

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

        if debug_dump_mat:
            mp_ids = [debug_dump_mat]
            err(f"[eval-mat] DEBUG: single-mat per-voxel dump → {debug_dump_mat}")

        per_mat_results = []
        # Streaming knobs (introduced 2026-05-29 — see specs/41 §"Validation
        # plan" / the eval-bidir-bug postmortem).
        #   TOMAT_EVAL_DUMP_PREDICTIONS=1 → save per-mat (n_patches, P^3)
        #     pred/true arrays as `.../predictions/<mp_id>{out_suffix}.npz`
        #     siblings of the JSON summary. Used by `tomat analyze voxel-corr`
        #     to build the position-position prediction-correlation matrix
        #     (task #174, spec 41 §"Validation plan" point 4).
        #   We always flush per-mat summary writes to GCS so partial progress
        #     survives a SIGTERM / wall-clock deadline. Previously the eval
        #     wrote only at the end → a hard 4h wall-clock on the iris job
        #     wiped 100+ mats' work. Now N-1 mats are recoverable.
        dump_predictions = bool(int(os.environ.get("TOMAT_EVAL_DUMP_PREDICTIONS", "0")))
        # Per-position diagnostics — accumulated across all mats so the
        # dashboard can plot mean error at each density-block position
        # (0..P^3-1, row-major within the patch). Sums divided by pp_count
        # in the summary block. See specs/26 §"Open questions": this is the
        # answer to "is more training improving voxel-0 (structure→ρ) or
        # only the AR shortcut?" — non-trivial-and-dropping at voxel 0 →
        # real learning; flat at voxel 0 → only AR-conditional refinement.
        pp_mae_sum = np.zeros(P ** 3, dtype=np.float64)
        pp_emd_sum = np.zeros(P ** 3, dtype=np.float64)
        pp_abs_true_sum = np.zeros(P ** 3, dtype=np.float64)
        pp_count = 0

        # ---- Streaming-write helpers (intermediate-progress recovery) ------
        # Compute the output path once. Levanter writes
        # <base>/<run_id>/step-N and `tomat evals fire` sets base =
        # <BUCKET>/results/<RL>/checkpoints → parts[-4]=run_label, -1=step.
        _ckpt_parts = checkpoint_path.rstrip("/").split("/")
        _ckpt_tail = _ckpt_parts[-1]
        _run_label = _ckpt_parts[-4]
        _mode_suffix = {"free": "-free", "maskgit": "-maskgit"}.get(eval_mode, "")
        _ms = (eval_mat_set or "default") + _mode_suffix
        _out_suffix = os.environ.get("TOMAT_EVAL_OUTPUT_SUFFIX", "")
        _results_path = f"{BUCKET}/eval/results/{_run_label}/{_ms}/{_ckpt_tail}{_out_suffix}.json"
        _predictions_dir = f"{BUCKET}/eval/results/{_run_label}/{_ms}/predictions"

        def _bootstrap_ci(xs: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
            """(lo, hi) bootstrap CI for the mean. Fixed seed so successive
            intermediate writes are comparable (CI tightens as mats land).
            """
            if len(xs) < 2:
                return None
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
            means = xs[idx].mean(axis=1)
            return [
                float(np.percentile(means, 100 * alpha / 2)),
                float(np.percentile(means, 100 * (1 - alpha / 2))),
            ]

        def _build_summary() -> dict:
            nmaes = np.array([r["nmae"] for r in per_mat_results], dtype=np.float64)
            nemds = np.array([r["nemd"] for r in per_mat_results], dtype=np.float64)
            n = len(per_mat_results)
            return {
                "checkpoint": checkpoint_path,
                "label": label,
                "model": model_preset,
                "mode": eval_mode,
                "mat_set": eval_mat_set,
                "n_mats": n,
                "nmae_mean": float(nmaes.mean()) if n else None,
                "nmae_median": float(np.median(nmaes)) if n else None,
                "nmae_p99": float(np.percentile(nmaes, 99)) if n else None,
                "nmae_ci95": _bootstrap_ci(nmaes),
                "nemd_mean": float(nemds.mean()) if n else None,
                "nemd_median": float(np.median(nemds)) if n else None,
                "nemd_p99": float(np.percentile(nemds, 99)) if n else None,
                "nemd_ci95": _bootstrap_ci(nemds),
                "per_pos_mae": (pp_mae_sum / pp_count).astype(np.float32).tolist() if pp_count else None,
                "per_pos_emd": (pp_emd_sum / pp_count).astype(np.float32).tolist() if pp_count else None,
                "per_pos_abs_true": (pp_abs_true_sum / pp_count).astype(np.float32).tolist() if pp_count else None,
                "per_pos_n_patches": int(pp_count),
                "per_mat": per_mat_results,
            }

        def _flush_results():
            """Write current summary to GCS. Called after each mat."""
            if debug_dump_mat or not per_mat_results:
                return
            try:
                with fsspec.open(_results_path, "w") as f:
                    json.dump(_build_summary(), f, indent=2)
            except Exception as e:
                err(f"[eval-mat] WARN: failed to persist results: {e}")

        def _dump_mat_predictions(mp_id: str, pred: np.ndarray, true: np.ndarray,
                                  offsets_arr: np.ndarray):
            """Save per-mat (n_patches, P^3) prediction arrays for the
            voxel-position correlation diagnostic (spec 41 §"Validation plan").
            Skipped unless TOMAT_EVAL_DUMP_PREDICTIONS=1.
            """
            if not dump_predictions or debug_dump_mat:
                return
            path = f"{_predictions_dir}/{mp_id}{_out_suffix}.npz"
            try:
                import io as _io
                buf = _io.BytesIO()
                np.savez_compressed(
                    buf, pred=pred.astype(np.float32),
                    true=true.astype(np.float32),
                    offsets=offsets_arr.astype(np.int32),
                )
                with fsspec.open(path, "wb") as f:
                    f.write(buf.getvalue())
            except Exception as e:
                err(f"[eval-mat] WARN: failed to dump predictions for {mp_id}: {e}")

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

            # ---- Free-running branch -------------------------------------
            # Autoregressive whole-grid eval: EVERY patch (full disjoint
            # coverage — the same `tile_full_coverage_offsets` tiling the
            # teacher-forced path uses) is free-run, the model regenerating
            # its P^3 density tokens from its own predictions. `free_batch`
            # is the per-forward batch size (an HBM knob), NOT a coverage
            # cap — the loop sweeps all patches and NMAE/NEMD are computed
            # over the full grid, identically to teacher-forced.
            if eval_mode == "free":
                if tokenizer_version != "v3":
                    raise ValueError(
                        f"free-running eval is v3-only (v2 runs are hidden); "
                        f"got tokenizer={tokenizer_version}"
                    )
                P3 = P ** 3
                B = free_batch
                rho_pred = np.zeros(grid_shape, dtype=np.float32)
                emd_grid = np.zeros(grid_shape, dtype=np.float32)
                # Per-patch flat (n_patches, P^3) generation-order arrays —
                # only retained for `TOMAT_DEBUG_DUMP_MAT`, the diagnostic
                # path that captures the *shape* of free-running divergence
                # (graceful drift vs cold-start garbage).
                dump_per_patch = (debug_dump_mat == mp_id)
                if dump_per_patch:
                    rho_per_patch = np.zeros((n_patches, P3), dtype=np.float32)
                    true_per_patch = np.zeros((n_patches, P3), dtype=np.float32)
                n_full_batches = (n_patches + B - 1) // B
                offsets_padded = list(offsets) + [offsets[0]] * (
                    n_full_batches * B - n_patches
                )
                Batch = hax.Axis("batch", B)
                t_gen0 = time.time()
                seen_buckets: set[int] = set()
                for bi in range(n_full_batches):
                    off_arr = np.asarray(
                        offsets_padded[bi * B:(bi + 1) * B], dtype=np.int32,
                    )
                    # Truth buffer (preamble + DENS_START + true density);
                    # the density region is blanked and regenerated below.
                    full_ids = build_all_patch_input_ids_v3(
                        density, grid_shape, off_arr, Zs, frac, lmq_codec,
                        vocab_offsets, P=P, pad_to=8192,
                        lattice_params=lattice_params,
                    )
                    row0 = full_ids[0]
                    is_d = ((row0 >= density_offset)
                            & (row0 < density_offset + len(recon)))
                    dpos = np.where(is_d)[0]
                    if len(dpos) != P3:
                        raise ValueError(
                            f"free {mp_id}: {len(dpos)} density positions != "
                            f"P^3={P3} — v3 layout assumption broken"
                        )
                    d0 = int(dpos[0]) - 1                  # DENS_START index
                    seq_len = d0 + P3 + 3                  # + DENS_END + EOS

                    # Codec-round-tripped true patch densities (B, P^3) — the
                    # per-step EMD target. Row-major, matching the builder.
                    iota = np.arange(P, dtype=np.int32)
                    di, dj, dk = np.meshgrid(iota, iota, iota, indexing="ij")
                    gx = np.asarray(grid_shape, dtype=np.int32)
                    ix = (off_arr[:, 0, None, None, None] + di) % gx[0]
                    iy = (off_arr[:, 1, None, None, None] + dj) % gx[1]
                    iz = (off_arr[:, 2, None, None, None] + dk) % gx[2]
                    tp = density[ix, iy, iz].reshape(B, P3).astype(np.float32)
                    tbins = np.clip(
                        np.searchsorted(bounds, np.clip(tp, 0.0, clip_max),
                                        side="right"),
                        0, len(recon) - 1,
                    ).astype(np.int32)
                    emd_target = recon[tbins].astype(np.float32)

                    # On-device buffer, length padded to a FREE_BUCKET
                    # multiple (flash attention needs a block-size-multiple
                    # query axis); density region → PAD, regenerated.
                    buf_len = _next_bucket(seq_len)
                    tokens = jnp.asarray(full_ids[:, :buf_len].copy())
                    tokens = tokens.at[:, d0 + 1:].set(TOK["PAD"])
                    rho_b = jnp.zeros((B, P3), dtype=jnp.float32)
                    emd_b = jnp.zeros((B, P3), dtype=jnp.float32)
                    for i in range(P3):
                        frontier = d0 + i
                        bucket = _next_bucket(frontier + 1)
                        if bucket not in seen_buckets:
                            seen_buckets.add(bucket)
                            err(f"[eval-mat] free: compiling bucket={bucket}")
                        Pos = hax.Axis("position", bucket)
                        tok_ha = hax.named(tokens[:, :bucket], (Batch, Pos))
                        td_ha = hax.named(jnp.asarray(emd_target[:, i]), (Batch,))
                        nt, rho_i, emd_i = free_step(
                            tok_ha, jnp.asarray(frontier, jnp.int32), td_ha,
                        )
                        tokens = tokens.at[:, d0 + 1 + i].set(nt.array)
                        rho_b = rho_b.at[:, i].set(rho_i.array)
                        emd_b = emd_b.at[:, i].set(emd_i.array)
                    rho_b_np = np.asarray(rho_b)              # (B, P^3)
                    emd_b_np = np.asarray(emd_b)
                    rho_blocks = rho_b_np.reshape(B, P, P, P)
                    emd_blocks = emd_b_np.reshape(B, P, P, P)
                    # Scatter each real patch into the full grid via its
                    # local slice — exact disjoint coverage, same as the
                    # teacher-forced path.
                    for k in range(B):
                        gidx = bi * B + k
                        if gidx >= n_patches:              # padded entry
                            continue
                        ox, oy, oz = offsets[gidx]
                        lx0, lx1, ly0, ly1, lz0, lz1 = local_slices[gidx]
                        rho_pred[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            rho_blocks[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        emd_grid[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            emd_blocks[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        if dump_per_patch:
                            rho_per_patch[gidx] = rho_b_np[k]
                            true_per_patch[gidx] = tp[k]
                        # Per-position diagnostics (same as teacher path).
                        pp_mae_sum += np.abs(rho_b_np[k] - tp[k])
                        pp_emd_sum += emd_b_np[k]
                        pp_abs_true_sum += np.abs(tp[k])
                        pp_count += 1
                t_gen = time.time() - t_gen0

                # NMAE / NEMD over the full grid — identical to teacher-forced.
                rho_true = density
                mae = float(np.mean(np.abs(rho_pred - rho_true)))
                denom = float(np.mean(np.abs(rho_true)))
                mean_emd = float(np.mean(emd_grid))
                nmae = mae / max(denom, 1e-30)
                nemd = mean_emd / max(denom, 1e-30)
                err(f"[eval-mat] {mp_id} FREE: {n_patches} patches "
                    f"({n_full_batches}×B{B}) × {P3} steps, full grid in "
                    f"{t_gen:.1f}s, NMAE={nmae:.4%}, NEMD={nemd:.4%}")
                if dump_per_patch:
                    import io
                    _buf = io.BytesIO()
                    np.savez_compressed(
                        _buf, rho_pred=rho_pred, rho_true=rho_true,
                        emd_grid=emd_grid,
                        rho_per_patch=rho_per_patch,
                        true_per_patch=true_per_patch,
                        offsets=np.asarray(offsets, dtype=np.int32),
                        grid_shape=np.asarray(grid_shape),
                    )
                    _dump = f"{BUCKET}/eval/debug/{mp_id}-free-dump.npz"
                    with fsspec.open(_dump, "wb") as _f:
                        _f.write(_buf.getvalue())
                    err(f"[eval-mat] DEBUG: dumped per-voxel free-run "
                        f"grids → {_dump}")
                per_mat_results.append({
                    "mp_id": mp_id,
                    "grid_shape": list(grid_shape),
                    "n_patches": int(n_patches),
                    "n_atoms": int(len(Zs)),
                    "mae": mae,
                    "mean_emd": mean_emd,
                    "mean_abs_true": denom,
                    "nmae": nmae,
                    "nemd": nemd,
                })
                # Free mode already accumulates rho_b_np per-patch
                # generation-order — dump if requested.
                if dump_predictions:
                    rho_all = np.zeros((n_patches, P ** 3), dtype=np.float32)
                    true_all = np.zeros((n_patches, P ** 3), dtype=np.float32)
                    # Build (n_patches, P^3) by indexing the stitched grid
                    # back per-patch. Mirrors the scatter at lines 1148-1153.
                    iota = np.arange(P, dtype=np.int32)
                    di, dj, dk = np.meshgrid(iota, iota, iota, indexing="ij")
                    gx = np.asarray(grid_shape, dtype=np.int32)
                    for gi in range(n_patches):
                        ox, oy, oz = offsets[gi]
                        ix = (ox + di) % gx[0]
                        iy = (oy + dj) % gx[1]
                        iz = (oz + dk) % gx[2]
                        rho_all[gi] = rho_pred[ix, iy, iz].reshape(P ** 3)
                        true_all[gi] = density[ix, iy, iz].reshape(P ** 3)
                    _dump_mat_predictions(
                        mp_id, rho_all, true_all,
                        np.asarray(offsets, dtype=np.int32),
                    )
                _flush_results()
                continue

            # ---- MaskGIT iterative-refinement branch ----------------------
            # All density positions start as MASK_ID; K_steps iterations of
            # parallel confidence-driven filling (cosine schedule) until all
            # positions are decoded. Then decode each filled bin → ρ.
            if eval_mode == "maskgit":
                if tokenizer_version != "v3":
                    raise ValueError(
                        f"maskgit eval is v3-only; got tokenizer={tokenizer_version}"
                    )
                if MASK_ID is None:
                    raise RuntimeError(
                        "maskgit eval requires MASK_ID (vocab bump), "
                        "but MASK_ID is None — was TOMAT_EVAL_MODE=maskgit set?"
                    )
                P3 = P ** 3
                B = eval_batch
                rho_pred = np.zeros(grid_shape, dtype=np.float32)
                # Diagnostic: alternative decode path that skips the final
                # all-filled re-forward and uses the argmax-per-iter `filled_bins`
                # directly. Compare NMAE_reforward vs NMAE_filled to test the
                # "final re-forward is OOD (model never sees 0-mask inputs)"
                # hypothesis.
                rho_pred_filled = np.zeros(grid_shape, dtype=np.float32)
                emd_grid = np.zeros(grid_shape, dtype=np.float32)
                n_full_batches = (n_patches + B - 1) // B
                offsets_padded = list(offsets) + [offsets[0]] * (
                    n_full_batches * B - n_patches
                )
                Batch = hax.Axis("batch", B)
                pad_to = meta.get("pad_to") or 8192

                # Build one reference row to locate density positions.
                _ref_ids = build_all_patch_input_ids_v3(
                    density, grid_shape,
                    np.asarray([offsets[0]], dtype=np.int32),
                    Zs, frac, lmq_codec, vocab_offsets,
                    P=P, pad_to=pad_to, lattice_params=lattice_params,
                )
                row0 = _ref_ids[0]
                _is_d = (row0 >= density_offset) & (row0 < density_offset + len(recon))
                dens_positions = np.where(_is_d)[0]
                if len(dens_positions) != P3:
                    err(f"[eval-mat] MaskGIT {mp_id}: expected {P3} dens positions, "
                        f"got {len(dens_positions)}; skipping")
                    continue
                d0 = int(dens_positions[0])  # first density token index

                # Partial-mask diagnostic short-circuit: skip iterative
                # refinement, just measure NMAE on masked positions at fixed
                # mask ratios via ONE forward each. Reports per-ratio NMAE
                # and ρ-stats so we can tell "flat at all ratios" (architecture
                # broken) from "good at low r, flat at r=1" (cosine-prior
                # trivial-solution).
                if mg_partial_ratios:
                    rng = np.random.default_rng(seed=0)  # deterministic across mats
                    pm_results: list[dict] = []
                    t_pm0 = time.time()
                    for ratio in mg_partial_ratios:
                        # Build per-batch masked input; one forward each.
                        nmae_num = 0.0       # sum |pred-true| over masked positions, all batches
                        nmae_den = 0.0       # sum |true|       over masked positions, all batches
                        pred_vals: list[np.ndarray] = []
                        true_vals: list[np.ndarray] = []
                        for bi in range(n_full_batches):
                            off_arr_b = np.asarray(
                                offsets_padded[bi * B:(bi + 1) * B], dtype=np.int32,
                            )
                            full_ids_b = build_all_patch_input_ids_v3(
                                density, grid_shape, off_arr_b, Zs, frac, lmq_codec,
                                vocab_offsets, P=P, pad_to=pad_to,
                                lattice_params=lattice_params,
                            )
                            # Per-(example, position) mask: True ⇒ MASK_ID.
                            mask_bool = rng.random((B, P3)) < ratio          # (B, P3) bool
                            tok_b = full_ids_b.copy()
                            flat_idx_b = dens_positions[None, :].repeat(B, axis=0)
                            # Apply mask only where mask_bool is True.
                            for bi_b in range(B):
                                masked_idx_b = flat_idx_b[bi_b][mask_bool[bi_b]]
                                tok_b[bi_b, masked_idx_b] = MASK_ID
                            Pos_p = hax.Axis("position", pad_to)
                            tok_ha_p = hax.named(jnp.asarray(tok_b), (Batch, Pos_p))
                            dl_pm = np.asarray(maskgit_forward(tok_ha_p))   # (B, Pos, V_d)
                            dl_pm_dens = dl_pm[:, dens_positions, :]        # (B, P3, V_d)
                            pred_bin = dl_pm_dens.argmax(axis=-1)           # (B, P3)
                            pred_rho = recon[pred_bin]                      # (B, P3)
                            # True ρ via codec round-trip (apples-to-apples vs codec floor).
                            true_bin = full_ids_b[:, dens_positions] - density_offset
                            true_bin = np.clip(true_bin, 0, len(recon) - 1)
                            true_rho = recon[true_bin]                      # (B, P3)
                            # Accumulate over MASKED positions of REAL patches only.
                            for bi_b in range(B):
                                gidx = bi * B + bi_b
                                if gidx >= n_patches:
                                    continue
                                m = mask_bool[bi_b]
                                if not m.any():
                                    continue
                                p_vals = pred_rho[bi_b, m]
                                t_vals = true_rho[bi_b, m]
                                nmae_num += float(np.sum(np.abs(p_vals - t_vals)))
                                nmae_den += float(np.sum(np.abs(t_vals)))
                                pred_vals.append(p_vals)
                                true_vals.append(t_vals)
                        pv = np.concatenate(pred_vals) if pred_vals else np.zeros(0)
                        tv = np.concatenate(true_vals) if true_vals else np.zeros(0)
                        pm_nmae = nmae_num / max(nmae_den, 1e-30)
                        pm_results.append({
                            "mask_ratio": ratio,
                            "n_masked": int(pv.size),
                            "nmae": pm_nmae,
                            "pred_mean": float(pv.mean()) if pv.size else 0.0,
                            "pred_p50":  float(np.percentile(pv, 50)) if pv.size else 0.0,
                            "pred_p99":  float(np.percentile(pv, 99)) if pv.size else 0.0,
                            "true_mean": float(tv.mean()) if tv.size else 0.0,
                            "true_p50":  float(np.percentile(tv, 50)) if tv.size else 0.0,
                            "true_p99":  float(np.percentile(tv, 99)) if tv.size else 0.0,
                        })
                    t_pm = time.time() - t_pm0
                    err(f"[eval-mat] {mp_id} PARTIAL-MASK ({n_patches} patches, {t_pm:.1f}s):")
                    err(f"  {'r':>6s}  {'n_masked':>9s}  {'NMAE':>9s}  "
                        f"{'pred μ/p50/p99':>22s}  {'true μ/p50/p99':>22s}")
                    for r in pm_results:
                        err(f"  {r['mask_ratio']:>6.2f}  {r['n_masked']:>9d}  "
                            f"{r['nmae']:>9.4%}  "
                            f"{r['pred_mean']:>7.2f}/{r['pred_p50']:>6.2f}/{r['pred_p99']:>6.2f}    "
                            f"{r['true_mean']:>7.2f}/{r['true_p50']:>6.2f}/{r['true_p99']:>6.2f}")
                    # Surface r=1.0 NMAE (or first entry) as top-level so the
                    # aggregator doesn't KeyError; full per-ratio breakdown
                    # lives in `partial_mask`.
                    top_nmae = pm_results[-1]["nmae"] if pm_results else 0.0
                    per_mat_results.append({
                        "mp_id": mp_id,
                        "grid_shape": list(grid_shape),
                        "n_patches": int(n_patches),
                        "n_atoms": int(len(Zs)),
                        "nmae": top_nmae,
                        "nemd": top_nmae,        # no EMD computed in partial-mask path
                        "mae": 0.0,              # placeholder; not the primary metric here
                        "mean_emd": 0.0,
                        "mean_abs_true": 0.0,
                        "partial_mask": pm_results,
                    })
                    _flush_results()
                    continue  # skip the K-iter MaskGIT path for this mat

                # Per-iter NMAE tracking (nice-to-have diagnostic).
                iter_nmae: list[float] = []
                iter_nemd: list[float] = []

                t_gen0 = time.time()
                for bi in range(n_full_batches):
                    off_arr = np.asarray(
                        offsets_padded[bi * B:(bi + 1) * B], dtype=np.int32,
                    )
                    # Build ground-truth ids; replace density block with MASK_ID.
                    full_ids = build_all_patch_input_ids_v3(
                        density, grid_shape, off_arr, Zs, frac, lmq_codec,
                        vocab_offsets, P=P, pad_to=pad_to,
                        lattice_params=lattice_params,
                    )
                    # tokens: (B, pad_to) with density positions → MASK_ID
                    tokens = full_ids.copy()
                    tokens[:, dens_positions] = MASK_ID

                    # True densities for per-position EMD (codec round-trip).
                    gx = np.asarray(grid_shape, dtype=np.int32)
                    iota = np.arange(P, dtype=np.int32)
                    _di, _dj, _dk = np.meshgrid(iota, iota, iota, indexing="ij")
                    ix = (off_arr[:, 0, None, None, None] + _di) % gx[0]
                    iy = (off_arr[:, 1, None, None, None] + _dj) % gx[1]
                    iz = (off_arr[:, 2, None, None, None] + _dk) % gx[2]
                    tp = density[ix, iy, iz].reshape(B, P3).astype(np.float32)

                    # Which density positions are still masked (per example).
                    # shape (B, P3); bool.
                    still_masked = np.ones((B, P3), dtype=bool)

                    for k in range(mg_k_steps):
                        Pos = hax.Axis("position", pad_to)
                        tok_ha = hax.named(jnp.asarray(tokens), (Batch, Pos))
                        # (B, pad_to, n_bins) density logits
                        dl_full = np.asarray(maskgit_forward(tok_ha))
                        # Extract at density positions: (B, P3, n_bins)
                        dl_dens = dl_full[:, dens_positions, :]

                        # Softmax → confidence = max prob at each position.
                        probs = jax.nn.softmax(jnp.asarray(dl_dens), axis=-1)
                        probs_np = np.asarray(probs)  # (B, P3, n_bins)
                        confidence = probs_np.max(axis=-1)   # (B, P3)
                        picked_bin = probs_np.argmax(axis=-1) # (B, P3)

                        # Cosine schedule: how many positions to have filled
                        # after step k+1 (1-indexed).
                        frac_fill = 1.0 - math.cos((k + 1) / mg_k_steps * math.pi / 2.0)
                        n_fill_target = math.ceil(frac_fill * P3)
                        n_fill_target = min(n_fill_target, P3)

                        for b in range(B):
                            masked_pos_idx = np.where(still_masked[b])[0]
                            n_already_filled = P3 - len(masked_pos_idx)
                            n_to_fill_now = max(0, n_fill_target - n_already_filled)
                            n_to_fill_now = min(n_to_fill_now, len(masked_pos_idx))
                            if n_to_fill_now == 0:
                                continue
                            # Pick the most-confident among still-masked.
                            conf_masked = confidence[b, masked_pos_idx]
                            top_k = np.argsort(-conf_masked)[:n_to_fill_now]
                            to_fill = masked_pos_idx[top_k]
                            tokens[b, dens_positions[to_fill]] = (
                                density_offset + picked_bin[b, to_fill]
                            )
                            still_masked[b, to_fill] = False

                    # Force-fill any remaining MASK positions (last-step safety).
                    for b in range(B):
                        remaining = np.where(still_masked[b])[0]
                        if len(remaining) == 0:
                            continue
                        # Re-forward once more to fill the rest.
                        Pos = hax.Axis("position", pad_to)
                        tok_ha = hax.named(jnp.asarray(tokens[[b]]), (hax.Axis("batch", 1), Pos))
                        dl_b = np.asarray(maskgit_forward(tok_ha))[0]  # (pad_to, n_bins)
                        dl_r = dl_b[dens_positions[remaining], :]
                        tokens[b, dens_positions[remaining]] = (
                            density_offset + dl_r.argmax(axis=-1)
                        )
                        still_masked[b, remaining] = False

                    # Decode filled tokens → ρ using the configured decoder.
                    filled_bins = tokens[:, dens_positions] - density_offset  # (B, P3)
                    filled_bins = np.clip(filled_bins, 0, len(recon) - 1)
                    dl_arr = full_ids[:, dens_positions] - density_offset     # (B, P3) true bins
                    dl_arr = np.clip(dl_arr, 0, len(recon) - 1)

                    # DIAGNOSTIC PATH: ρ from the argmax-per-iter filled_bins,
                    # bypassing the final all-filled re-forward. If the
                    # re-forward is OOD, this should yield much better NMAE.
                    rho_b_filled = recon[filled_bins]               # (B, P3)

                    # Decode ρ point estimate (current primary path).
                    Pos_d = hax.Axis("position", pad_to)
                    tok_ha_f = hax.named(jnp.asarray(tokens), (Batch, Pos_d))
                    dl_full2 = np.asarray(maskgit_forward(tok_ha_f))
                    dl_dens2 = dl_full2[:, dens_positions, :]   # (B, P3, n_bins)
                    probs2 = np.asarray(jax.nn.softmax(jnp.asarray(dl_dens2), axis=-1))

                    if mg_decoder == "argmax":
                        rho_b = recon[dl_dens2.argmax(axis=-1)]  # (B, P3)
                    elif mg_decoder == "median":
                        cumP2 = np.cumsum(probs2, axis=-1)
                        bin_idx2 = np.clip(
                            (cumP2 < 0.5).sum(axis=-1), 0, len(recon) - 1,
                        )
                        rho_b = recon[bin_idx2]
                    else:  # mean
                        rho_b = np.einsum("bpv,v->bp", probs2, recon)

                    # EMD at each position vs true density.
                    true_dens_b = recon[dl_arr]  # (B, P3) codec-round-tripped
                    abs_diff = np.abs(recon[None, None, :] - true_dens_b[..., None])  # (B, P3, V)
                    emd_b = np.einsum("bpv,bpv->bp", probs2, abs_diff)  # (B, P3)

                    # Scatter into full grids + per-position diagnostics.
                    rho_b_flat = rho_b
                    rho_b = rho_b.reshape(B, P, P, P)
                    rho_b_filled_3d = rho_b_filled.reshape(B, P, P, P)
                    emd_b_grid = emd_b.reshape(B, P, P, P)
                    for k in range(B):
                        gidx = bi * B + k
                        if gidx >= n_patches:
                            continue
                        ox, oy, oz = offsets[gidx]
                        lx0, lx1, ly0, ly1, lz0, lz1 = local_slices[gidx]
                        rho_pred[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            rho_b[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        rho_pred_filled[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            rho_b_filled_3d[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        emd_grid[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            emd_b_grid[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        pp_mae_sum += np.abs(rho_b_flat[k] - tp[k])
                        pp_emd_sum += emd_b[k]
                        pp_abs_true_sum += np.abs(tp[k])
                        pp_count += 1

                t_gen = time.time() - t_gen0
                rho_true = density
                mae = float(np.mean(np.abs(rho_pred - rho_true)))
                mae_filled = float(np.mean(np.abs(rho_pred_filled - rho_true)))
                denom = float(np.mean(np.abs(rho_true)))
                mean_emd = float(np.mean(emd_grid))
                nmae = mae / max(denom, 1e-30)
                nmae_filled = mae_filled / max(denom, 1e-30)
                nemd = mean_emd / max(denom, 1e-30)
                # Diagnostic stats on the predicted-vs-true distribution shape
                # — helps tell "median is in top bin" from "median is reasonable
                # but biased" from "median ≈ truth on this mat".
                pf_mean = float(np.mean(rho_pred_filled))
                pf_p50 = float(np.percentile(rho_pred_filled, 50))
                pf_p99 = float(np.percentile(rho_pred_filled, 99))
                rf_mean = float(np.mean(rho_pred))
                rf_p50 = float(np.percentile(rho_pred, 50))
                rf_p99 = float(np.percentile(rho_pred, 99))
                t_mean = float(np.mean(rho_true))
                t_p99 = float(np.percentile(rho_true, 99))
                err(f"[eval-mat] {mp_id} MASKGIT: {n_patches} patches, "
                    f"K={mg_k_steps} iters, {t_gen:.1f}s")
                err(f"  NMAE(reforward,{mg_decoder})={nmae:.4%}  "
                    f"NMAE(filled_bins,argmax)={nmae_filled:.4%}  "
                    f"NEMD={nemd:.4%}")
                err(f"  rho_true        mean={t_mean:.3e}  p99={t_p99:.3e}")
                err(f"  rho_pred(reforw) mean={rf_mean:.3e}  p50={rf_p50:.3e}  p99={rf_p99:.3e}")
                err(f"  rho_pred(filled) mean={pf_mean:.3e}  p50={pf_p50:.3e}  p99={pf_p99:.3e}")
                per_mat_results.append({
                    "mp_id": mp_id,
                    "grid_shape": list(grid_shape),
                    "n_patches": int(n_patches),
                    "n_atoms": int(len(Zs)),
                    "mae": mae,
                    "mae_filled": mae_filled,
                    "mean_emd": mean_emd,
                    "mean_abs_true": denom,
                    "nmae": nmae,
                    "nmae_filled": nmae_filled,
                    "nemd": nemd,
                })
                _flush_results()
                continue

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
            builder = (
                build_all_patch_input_ids_v3
                if tokenizer_version == "v3"
                else build_all_patch_input_ids
            )
            all_ids = builder(
                density, grid_shape, offsets_arr, Zs, frac,
                lmq_codec, vocab_offsets, P=P, pad_to=pad_to,
                lattice_params=lattice_params,
            )  # (n_eval, 8192) int32
            t_tok = time.time() - t_tok0

            # True per-patch density values in row-major (matches the builder's
            # DENS-block extraction). Used for per-position diagnostics —
            # accumulated into pp_*_sum below.
            _iota = np.arange(P, dtype=np.int32)
            _di, _dj, _dk = np.meshgrid(_iota, _iota, _iota, indexing="ij")
            _gx = np.asarray(grid_shape, dtype=np.int32)
            _real_off = np.asarray(offsets, dtype=np.int32)
            _ix = (_real_off[:, 0, None, None, None] + _di) % _gx[0]
            _iy = (_real_off[:, 1, None, None, None] + _dj) % _gx[1]
            _iz = (_real_off[:, 2, None, None, None] + _dk) % _gx[2]
            all_true_patch = density[_ix, _iy, _iz].reshape(
                n_patches, P ** 3,
            ).astype(np.float32)

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
            emd_grid = np.zeros(grid_shape, dtype=np.float32)
            # Per-patch row-major predictions; only retained if dump_predictions.
            all_pred_patch = (
                np.zeros((n_patches, P ** 3), dtype=np.float32)
                if dump_predictions else None
            )
            for bi in range(n_full_batches):
                inflight.append((bi, dispatch(bi)))
                while len(inflight) > depth or (bi == n_full_batches - 1 and inflight):
                    done_bi, (rho_fut, emd_fut) = inflight.pop(0)
                    pred_floats = np.asarray(rho_fut.array, dtype=np.float32)  # (B, Pos)
                    emd_floats = np.asarray(emd_fut.array, dtype=np.float32)  # (B, Pos)
                    start = done_bi * eval_batch
                    pred_dens_floats = pred_floats[:, pred_positions]  # (B, P^3)
                    emd_dens_floats = emd_floats[:, pred_positions]
                    pred_blocks = pred_dens_floats.reshape(-1, P, P, P)
                    emd_blocks = emd_dens_floats.reshape(-1, P, P, P)
                    for k in range(eval_batch):
                        global_idx = start + k
                        if global_idx >= n_patches:  # padded entry, skip
                            continue
                        ox, oy, oz = offsets[global_idx]
                        lx0, lx1, ly0, ly1, lz0, lz1 = local_slices[global_idx]
                        rho_pred[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            pred_blocks[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        emd_grid[ox+lx0:ox+lx1, oy+ly0:oy+ly1, oz+lz0:oz+lz1] = \
                            emd_blocks[k, lx0:lx1, ly0:ly1, lz0:lz1]
                        # Per-position diagnostics: accumulate |err|, emd, |true|
                        # at each row-major position 0..P^3-1 in the patch.
                        pp_mae_sum += np.abs(pred_dens_floats[k] - all_true_patch[global_idx])
                        pp_emd_sum += emd_dens_floats[k]
                        pp_abs_true_sum += np.abs(all_true_patch[global_idx])
                        pp_count += 1
                        if all_pred_patch is not None:
                            all_pred_patch[global_idx] = pred_dens_floats[k]
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
            # mat-EMD: per-voxel Wasserstein-1 vs delta target, normalized by
            # mean |ρ_true| (same denominator as NMAE → directly comparable).
            mean_emd = float(np.mean(emd_grid))
            nemd = mean_emd / max(denom, 1e-30)
            err(f"[eval-mat] {mp_id}: MAE={mae:.4e}, EMD={mean_emd:.4e}, "
                f"mean|ρ_true|={denom:.4e}, NMAE={nmae:.4%}, NEMD={nemd:.4%}")
            if debug_dump_mat == mp_id:
                import io
                _buf = io.BytesIO()
                np.savez_compressed(
                    _buf, rho_pred=rho_pred, rho_true=rho_true, emd_grid=emd_grid,
                    grid_shape=np.asarray(grid_shape),
                )
                _dump = f"{BUCKET}/eval/debug/{mp_id}-dump.npz"
                with fsspec.open(_dump, "wb") as _f:
                    _f.write(_buf.getvalue())
                err(f"[eval-mat] DEBUG: dumped per-voxel grids → {_dump}")
            per_mat_results.append({
                "mp_id": mp_id,
                "grid_shape": list(grid_shape),
                "n_patches": total,
                "n_atoms": int(len(Zs)),
                "mae": float(mae),
                "mean_emd": mean_emd,
                "mean_abs_true": float(denom),
                "nmae": float(nmae),
                "nemd": float(nemd),
            })
            if all_pred_patch is not None:
                _dump_mat_predictions(
                    mp_id,
                    all_pred_patch[:total],
                    all_true_patch[:total],
                    np.asarray(offsets[:total], dtype=np.int32),
                )
            _flush_results()

        # Aggregate
        if per_mat_results:
            nmaes = np.array([r["nmae"] for r in per_mat_results])
            nemds = np.array([r["nemd"] for r in per_mat_results])
            err(f"[eval-mat] AGGREGATE over {len(nmaes)} mats:")
            err(f"  mean NMAE   : {nmaes.mean():.4%}    mean NEMD   : {nemds.mean():.4%}")
            err(f"  median NMAE : {np.median(nmaes):.4%}    median NEMD : {np.median(nemds):.4%}")
            err(f"  p99 NMAE    : {np.percentile(nmaes, 99):.4%}    p99 NEMD    : {np.percentile(nemds, 99):.4%}")

        # Per-position diagnostic curve (averaged across all real patches of
        # all mats). per_pos_mae[i] is the mean |pred - true| at row-major
        # position i ∈ [0, P^3) within the patch; per_pos_abs_true[i] gives
        # the denominator so the dashboard can compute per-position NMAE.
        # Voxel 0 (= "P(ρ | structure-only)") is the cold-start prediction
        # — if it's flat over training, the model is only learning the AR
        # shortcut; if it's dropping, real structure→ρ learning is happening.
        if pp_count > 0:
            err(f"[eval-mat] per-position diagnostic ({pp_count} patches):")
            per_pos_mae = (pp_mae_sum / pp_count).astype(np.float32)
            per_pos_emd = (pp_emd_sum / pp_count).astype(np.float32)
            per_pos_abs_true = (pp_abs_true_sum / pp_count).astype(np.float32)
            # A few canonical positions: voxel 0 (structure-only); voxel
            # P-1 (end of first 1D-row); voxel P (start of next 1D-row,
            # spatially adjacent to voxel 0 via y); voxel P^2 (start of
            # next 2D-slab, spatially adjacent to voxel 0 via x); mid; last.
            canon = [0, P - 1, P, P ** 2 - 1, P ** 2, (P ** 3) // 2, P ** 3 - 1]
            denom = float(per_pos_abs_true.mean())
            for i in canon:
                npm = float(per_pos_mae[i]) / max(denom, 1e-30)
                err(f"  i={i:>5d}: mae={per_pos_mae[i]:>9.3g}  emd={per_pos_emd[i]:>9.3g}"
                    f"  |true|={per_pos_abs_true[i]:>9.3g}  NMAE_global_denom={npm:.4%}")

        # Final summary write + stdout dump. _flush_results() during the loop
        # has already written N-1 incremental snapshots, so this is mostly a
        # belt-and-suspenders; the new piece is the printed JSON.
        summary = _build_summary()
        print(json.dumps(summary, indent=2))
        _flush_results()
        if per_mat_results and not debug_dump_mat:
            err(f"[eval-mat] persisted final per-mat results to {_results_path}")


if __name__ == "__main__":
    main()
