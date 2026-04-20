"""Tests for `PatchTokenizer`."""

from pathlib import Path

import numpy as np
import pytest

from tomat.float_codec import FP16Codec
from tomat.tokenizers.patch import (
    ATOM_OFFSET,
    ATOM_END,
    INT_OFFSET,
    INT_VOCAB_SIZE,
    N_SPECIALS,
    PatchTokenizer,
    SPECIAL_TOKENS,
)


@pytest.fixture
def fake_structure():
    """Build a minimal pymatgen Structure (C-H in a 4Å cube)."""
    from pymatgen.core.structure import Structure
    return Structure(
        lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        species=["C", "H"],
        coords=[[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]],
    )


@pytest.fixture
def tokenizer():
    return PatchTokenizer(patch_size=8)


def test_vocab_layout_is_contiguous(tokenizer):
    v = tokenizer.vocab
    assert v.position_offset == INT_OFFSET + INT_VOCAB_SIZE
    assert v.density_offset == v.position_offset + v.position_vocab_size
    assert v.total_vocab_size == v.density_offset + v.density_vocab_size

    # Positions with 3-byte tomol codec: 512 + 256 + 256 = 1024
    assert v.position_vocab_size == 1024
    # Density with 2-token 9+12: 512 + 4096 = 4608
    assert v.density_vocab_size == 4608
    # Total: specials(16) + atoms(118) + ints(1024) + positions(1024) + density(4608)
    assert v.total_vocab_size == 16 + 118 + 1024 + 1024 + 4608


def test_extract_patch_with_pbc_wrap(tokenizer):
    rng = np.random.default_rng(0)
    density = rng.uniform(0, 1, size=(16, 16, 16))

    # Anchor at origin — simple slice.
    p0 = tokenizer.extract_patch(density, (0, 0, 0))
    assert p0.shape == (8, 8, 8)
    assert np.array_equal(p0, density[:8, :8, :8])

    # Anchor near the boundary — must wrap.
    p_wrap = tokenizer.extract_patch(density, (12, 12, 12))
    assert p_wrap.shape == (8, 8, 8)
    # x-indices 12,13,14,15,0,1,2,3 (PBC)
    expected = density[np.ix_(
        np.array([12, 13, 14, 15, 0, 1, 2, 3]),
        np.array([12, 13, 14, 15, 0, 1, 2, 3]),
        np.array([12, 13, 14, 15, 0, 1, 2, 3]),
    )]
    assert np.array_equal(p_wrap, expected)


def test_tokenize_structure(tokenizer, fake_structure):
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)
    sample = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(4, 4, 4),
    )
    tokens = tokenizer.tokenize(sample)

    # Expected count:
    #   BOS                                   1
    #   GRID_START  3 ints  GRID_END          5
    #   ATOMS_START  2 atoms  ATOMS_END       4
    #   POS_START  2*3*3tok  POS_END         20
    #   SHAPE_START 3 ints SHAPE_END          5
    #   OFFSET_START 3 ints OFFSET_END        5
    #   DENS_START 8³ × 2tok DENS_END      1026
    #   EOS                                   1
    # total                                1067
    expected = 1 + 5 + 4 + 20 + 5 + 5 + (2 * tokenizer.patch_size ** 3 + 2) + 1
    assert len(tokens) == expected

    # Special tokens present in the right order
    assert tokens[0] == SPECIAL_TOKENS["[BOS]"]
    assert tokens[-1] == SPECIAL_TOKENS["[EOS]"]
    assert SPECIAL_TOKENS["[DENS_START]"] in tokens
    assert SPECIAL_TOKENS["[POS_END]"] in tokens


def test_atom_tokens_are_in_range(tokenizer, fake_structure):
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16))
    sample = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(0, 0, 0),
    )
    tokens = np.array(tokenizer.tokenize(sample))
    # Extract atom tokens: after ATOMS_START until ATOMS_END.
    start = int(np.where(tokens == SPECIAL_TOKENS["[ATOMS_START]"])[0][0])
    end = int(np.where(tokens == SPECIAL_TOKENS["[ATOMS_END]"])[0][0])
    atom_toks = tokens[start + 1 : end]
    assert all(ATOM_OFFSET <= t < ATOM_END for t in atom_toks)
    # Carbon Z=6, Hydrogen Z=1 → offsets +5 and +0 respectively.
    assert atom_toks[0] == ATOM_OFFSET + (6 - 1)  # C
    assert atom_toks[1] == ATOM_OFFSET + (1 - 1)  # H


def test_all_tokens_within_vocab(tokenizer, fake_structure):
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16))
    sample = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(0, 0, 0),
    )
    tokens = tokenizer.tokenize(sample)
    assert min(tokens) >= 0
    assert max(tokens) < tokenizer.vocab.total_vocab_size


def test_random_offsets_uniform_coverage(tokenizer):
    rng = np.random.default_rng(42)
    offsets = tokenizer.random_offsets(grid_shape=(16, 16, 16), n=10_000, rng=rng)
    assert offsets.shape == (10_000, 3)
    # Each axis should cover the full range with no bias (chi² would be overkill).
    for axis in range(3):
        assert offsets[:, axis].min() == 0
        assert offsets[:, axis].max() == 15
        # Within ~5% of uniform mean (7.5)
        assert abs(offsets[:, axis].mean() - 7.5) < 0.3
