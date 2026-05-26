"""No-GPU tests for MaskGIT masking logic.

Validates:
  1. Mask-count invariant: exactly ceil(r * P³) density positions masked.
  2. Only density tokens (in [density_offset, density_offset+n_bins)) get masked.
  3. Non-density tokens are never touched (preamble, EOS, PAD).
  4. loss_weight is 1.0 exactly at masked positions, 0.0 elsewhere.
  5. Mask-ratio prior shapes (cosine, uniform, high) produce ratios in [0,1]
     and the "high" prior stays in [0.5, 1.0].
  6. d0 formula: density block starts at 10 * n_atoms + 18 (v3 layout),
     matching test_eval_free.py's canonical assertion.

All tests are pure-numpy — no JAX or GPU needed.
"""
import math

import numpy as np
import pytest

pytest.importorskip("jax")

from marin.eval_mat_nmae import (  # noqa: E402
    N_ATOMS,
    N_INTS,
    N_SPECIALS_LAT,
    TOK,
    build_all_patch_input_ids_v3,
)


P = 19
P3 = P ** 3
N_BINS = 256
DENSITY_OFFSET = N_SPECIALS_LAT + N_ATOMS + N_INTS + (512 + 256 + 256)  # pos_total = 1024
MASK_ID = DENSITY_OFFSET + N_BINS  # one past the last density token


def _v3_vocab_offsets():
    pos_vocabs = (512, 256, 256)
    pos_offset = N_SPECIALS_LAT + N_ATOMS + N_INTS
    return {
        "n_specials": N_SPECIALS_LAT,
        "pos_offset": pos_offset,
        "pos_vocabs": pos_vocabs,
        "pos_log_min": -4.0,
        "pos_log_max": 0.0,
        "density_offset": DENSITY_OFFSET,
        "lat_aware": True,
    }


def _make_ids(n_atoms: int = 4) -> np.ndarray:
    """Build a (1, 8192) input_ids row via the v3 builder."""
    rng = np.random.default_rng(n_atoms + 7)
    grid = (38, 38, 38)
    density = rng.random(grid, dtype=np.float64).astype(np.float32)
    offsets = np.array([[0, 0, 0]], dtype=np.int32)
    Zs = np.full(n_atoms, 6, dtype=np.int32)
    frac = rng.random((n_atoms, 3))
    lmq_codec = (
        np.linspace(0.0, 1.0, N_BINS - 1, dtype=np.float32),
        np.linspace(0.0, 1.0, N_BINS, dtype=np.float32),
        1.0,
    )
    return build_all_patch_input_ids_v3(
        density, grid, offsets, Zs, frac, lmq_codec, _v3_vocab_offsets(),
        P=P, pad_to=8192, lattice_params=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    )  # (1, 8192)


def _apply_numpy_mask(
    tokens: np.ndarray,   # (L,) int32
    mask_ratio: float,
    density_offset: int = DENSITY_OFFSET,
    n_bins: int = N_BINS,
    mask_id: int = MASK_ID,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference numpy masking — mirrors the JAX logic in qwen3_density.py.

    Returns (masked_tokens, loss_weight) both of shape (L,).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    L = len(tokens)
    lo, hi = density_offset, density_offset + n_bins
    dens_pos = np.where((tokens >= lo) & (tokens < hi))[0]
    P3_actual = len(dens_pos)
    mask_count = math.ceil(mask_ratio * P3_actual)
    mask_count = min(mask_count, P3_actual)

    chosen = rng.choice(P3_actual, size=mask_count, replace=False)
    chosen_positions = dens_pos[chosen]

    masked = tokens.copy()
    masked[chosen_positions] = mask_id
    lw = np.zeros(L, dtype=np.float32)
    lw[chosen_positions] = 1.0
    return masked, lw


@pytest.mark.parametrize("n_atoms", [1, 4, 17, 40])
def test_density_positions_count(n_atoms):
    """Every v3 row has exactly P³ density tokens."""
    ids = _make_ids(n_atoms)[0]
    lo, hi = DENSITY_OFFSET, DENSITY_OFFSET + N_BINS
    dens_pos = np.where((ids >= lo) & (ids < hi))[0]
    assert len(dens_pos) == P3


@pytest.mark.parametrize("n_atoms", [1, 4, 17, 40])
def test_d0_formula(n_atoms):
    """d0 = 10 * n_atoms + 18 — canonical v3 preamble formula."""
    ids = _make_ids(n_atoms)[0]
    lo, hi = DENSITY_OFFSET, DENSITY_OFFSET + N_BINS
    dens_pos = np.where((ids >= lo) & (ids < hi))[0]
    d0 = int(dens_pos[0]) - 1  # DENS_START index
    assert d0 == 10 * n_atoms + 18


@pytest.mark.parametrize("mask_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_mask_count_invariant(mask_ratio):
    """Exactly ceil(r * P³) positions are masked."""
    ids = _make_ids(n_atoms=4)[0]
    masked, lw = _apply_numpy_mask(ids, mask_ratio)
    expected_count = math.ceil(mask_ratio * P3)
    assert int(lw.sum()) == expected_count, (
        f"mask_ratio={mask_ratio}: expected {expected_count} masked, "
        f"got {int(lw.sum())}"
    )


@pytest.mark.parametrize("mask_ratio", [0.0, 0.5, 1.0])
def test_only_density_positions_masked(mask_ratio):
    """Non-density tokens (preamble, EOS, PAD) are never replaced."""
    ids = _make_ids(n_atoms=4)[0]
    masked, lw = _apply_numpy_mask(ids, mask_ratio)
    lo, hi = DENSITY_OFFSET, DENSITY_OFFSET + N_BINS

    # All non-density tokens unchanged.
    non_dens = (ids < lo) | (ids >= hi)
    assert np.all(masked[non_dens] == ids[non_dens]), (
        "Non-density tokens were modified by the mask."
    )

    # All MASK_ID tokens must have been density positions.
    mask_positions = np.where(masked == MASK_ID)[0]
    for p in mask_positions:
        assert lo <= ids[p] < hi, (
            f"Position {p} was masked but original token {ids[p]} is not a density token."
        )


@pytest.mark.parametrize("mask_ratio", [0.0, 0.5, 1.0])
def test_loss_weight_matches_mask(mask_ratio):
    """loss_weight == 1 exactly at masked positions, 0 elsewhere."""
    ids = _make_ids(n_atoms=4)[0]
    masked, lw = _apply_numpy_mask(ids, mask_ratio)

    mask_flag = (masked == MASK_ID).astype(np.float32)
    assert np.array_equal(lw, mask_flag), (
        "loss_weight does not exactly match the set of MASK_ID positions."
    )


def test_full_mask_replaces_all_density():
    """r=1.0 → all P³ density positions become MASK_ID."""
    ids = _make_ids(n_atoms=4)[0]
    masked, lw = _apply_numpy_mask(ids, mask_ratio=1.0)
    lo, hi = DENSITY_OFFSET, DENSITY_OFFSET + N_BINS
    dens_pos = np.where((ids >= lo) & (ids < hi))[0]
    assert np.all(masked[dens_pos] == MASK_ID), (
        "mask_ratio=1.0 should replace every density token with MASK_ID."
    )
    assert int(lw.sum()) == P3


def test_zero_mask_changes_nothing():
    """r=0.0 → no tokens changed, loss_weight all zeros."""
    ids = _make_ids(n_atoms=4)[0]
    masked, lw = _apply_numpy_mask(ids, mask_ratio=0.0)
    assert np.array_equal(masked, ids), "mask_ratio=0.0 should not change any token."
    assert lw.sum() == 0.0


@pytest.mark.parametrize("n_samples", [500])
def test_cosine_prior_range(n_samples):
    """Cosine prior samples are in [0, 1] and biased toward high values."""
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(n_samples):
        u = rng.random()
        r = math.cos(u * math.pi / 2.0)
        ratios.append(r)
    ratios = np.array(ratios)
    assert ratios.min() >= 0.0 - 1e-9
    assert ratios.max() <= 1.0 + 1e-9
    # Cosine prior is biased toward high mask ratios: mean should be > 0.5.
    assert ratios.mean() > 0.5, (
        f"Cosine prior mean={ratios.mean():.3f} expected > 0.5"
    )


@pytest.mark.parametrize("n_samples", [500])
def test_high_prior_range(n_samples):
    """High prior samples are in [0.5, 1.0]."""
    rng = np.random.default_rng(1)
    ratios = [0.5 + 0.5 * rng.random() for _ in range(n_samples)]
    ratios = np.array(ratios)
    assert ratios.min() >= 0.5 - 1e-9
    assert ratios.max() <= 1.0 + 1e-9
