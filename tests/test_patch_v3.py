"""Tests for `PatchTokenizerV3`."""

import numpy as np
import pytest

from tomat.tokenizers.patch import (
    INT_OFFSET,
    LATTICE_ANGLE_RES_DEG,
    LATTICE_LENGTH_RES_A,
    SPECIAL_TOKENS,
)
from tomat.tokenizers.patch_v3 import PatchTokenizerV3


@pytest.fixture
def fake_structure():
    """Minimal pymatgen Structure (C-H in a 4Å cube)."""
    from pymatgen.core.structure import Structure
    return Structure(
        lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        species=["C", "H"],
        coords=[[0.1, 0.2, 0.3], [0.7, 0.6, 0.5]],
    )


@pytest.fixture
def tokenizer():
    return PatchTokenizerV3(patch_size=8)


def test_default_layout_has_no_shape_offset_hi(tokenizer, fake_structure):
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)
    sample = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(4, 4, 4),
    )
    tokens = tokenizer.tokenize(sample)

    # Expected count for default layout (no SHAPE / OFFSET / HI):
    #   BOS                                   1
    #   GRID_START  3 ints  GRID_END          5
    #   LATTICE_START 6 ints LATTICE_END      8
    #   ATOMS_START  2 atoms  ATOMS_END       4
    #   POS_START  2*3*3tok  POS_END         20
    #   DENS_START 8³ × 2tok DENS_END      1026
    #   EOS                                   1
    expected = 1 + 5 + 8 + 4 + 20 + (2 * tokenizer.patch_size ** 3 + 2) + 1
    assert len(tokens) == expected

    # SHAPE / OFFSET / HI markers are absent
    assert SPECIAL_TOKENS["[SHAPE_START]"] not in tokens
    assert SPECIAL_TOKENS["[OFFSET_START]"] not in tokens
    assert SPECIAL_TOKENS["[HI_START]"] not in tokens

    # Density / position markers are present
    assert SPECIAL_TOKENS["[DENS_START]"] in tokens
    assert SPECIAL_TOKENS["[POS_END]"] in tokens


def test_fallback_layout_emits_shape(tokenizer, fake_structure):
    """When patch_shape != default, SHAPE block is emitted."""
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)
    # Manually construct a fallback sample with patch_shape = (7, 8, 8).
    sample = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(4, 4, 4),
    )
    # Replace patch with a (7, 8, 8) crop
    sample = sample.__class__(
        task_id=sample.task_id,
        offset=sample.offset,
        patch_shape=(7, 8, 8),
        grid_shape=sample.grid_shape,
        lattice=sample.lattice,
        atomic_numbers=sample.atomic_numbers,
        frac_coords=sample.frac_coords,
        patch_density=sample.patch_density[:7, :, :],
    )
    tokens = tokenizer.tokenize(sample)
    assert SPECIAL_TOKENS["[SHAPE_START]"] in tokens
    # Still no OFFSET / HI
    assert SPECIAL_TOKENS["[OFFSET_START]"] not in tokens
    assert SPECIAL_TOKENS["[HI_START]"] not in tokens


def test_atoms_are_translated_to_patch_frame(tokenizer, fake_structure):
    """Atom positions in the token stream should reflect translation by
    -(ix/nx, iy/ny, iz/nz). Same atom + different offset → different
    position tokens."""
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)

    s_origin = tokenizer.make_sample("mp-x", density, fake_structure, offset=(0, 0, 0))
    s_shifted = tokenizer.make_sample("mp-x", density, fake_structure, offset=(8, 8, 8))

    toks_o = tokenizer.tokenize(s_origin)
    toks_s = tokenizer.tokenize(s_shifted)

    pos_start_o = toks_o.index(SPECIAL_TOKENS["[POS_START]"])
    pos_end_o = toks_o.index(SPECIAL_TOKENS["[POS_END]"])
    pos_start_s = toks_s.index(SPECIAL_TOKENS["[POS_START]"])
    pos_end_s = toks_s.index(SPECIAL_TOKENS["[POS_END]"])

    pos_o = toks_o[pos_start_o + 1: pos_end_o]
    pos_s = toks_s[pos_start_s + 1: pos_end_s]

    # Same number of position tokens (same number of atoms)
    assert len(pos_o) == len(pos_s)
    # But different values — translation took effect
    assert pos_o != pos_s


def test_roundtrip_recovers_translated_sample(tokenizer, fake_structure):
    """Tokenize → detokenize should recover the patch density + the
    translated frac_coords (atoms in patch frame)."""
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)
    original = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(3, 5, 7),
    )
    tokens = tokenizer.tokenize(original)
    recovered = tokenizer.detokenize(tokens)

    # Grid + patch shape preserved
    assert recovered.grid_shape == original.grid_shape
    assert recovered.patch_shape == original.patch_shape
    # Offset is collapsed to (0,0,0) in v3 — patch is in its own frame
    assert recovered.offset == (0, 0, 0)
    # Lattice preserved
    for got, want, res in zip(
        recovered.lattice, original.lattice,
        (LATTICE_LENGTH_RES_A,) * 3 + (LATTICE_ANGLE_RES_DEG,) * 3,
    ):
        assert abs(got - want) <= res
    # Atomic numbers preserved
    assert np.array_equal(recovered.atomic_numbers, original.atomic_numbers)
    # Translated frac_coords match what tokenize emitted
    expected_translated = tokenizer.translate_frac_coords(
        original.frac_coords, original.offset, original.grid_shape,
    )
    assert np.allclose(recovered.frac_coords, expected_translated, atol=1e-4)
    # Density precision
    rel_err = np.abs(recovered.patch_density - original.patch_density) / np.maximum(
        np.abs(original.patch_density), 1e-12
    )
    assert rel_err.max() < 1e-3


def test_roundtrip_recovers_fallback_shape(tokenizer, fake_structure):
    """Roundtrip works for fallback patch shape too — detokenize picks up SHAPE."""
    rng = np.random.default_rng(0)
    density = rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)
    original = tokenizer.make_sample(
        task_id="mp-fake",
        density=density,
        structure=fake_structure,
        offset=(0, 0, 0),
    )
    fallback = original.__class__(
        task_id=original.task_id,
        offset=original.offset,
        patch_shape=(7, 8, 8),
        grid_shape=original.grid_shape,
        lattice=original.lattice,
        atomic_numbers=original.atomic_numbers,
        frac_coords=original.frac_coords,
        patch_density=original.patch_density[:7, :, :],
    )
    tokens = tokenizer.tokenize(fallback)
    recovered = tokenizer.detokenize(tokens)
    assert recovered.patch_shape == (7, 8, 8)
    assert recovered.patch_density.shape == (7, 8, 8)


def test_translate_frac_coords_is_inverse_of_offset_shift():
    """Translation is just modular arithmetic; verify directly."""
    tk = PatchTokenizerV3(patch_size=8)
    fc = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
    # Offset (8,0,0) on a 16-grid means shift by 0.5 along x.
    out = tk.translate_frac_coords(fc, (8, 0, 0), (16, 16, 16))
    expected = np.array([[0.0, 0.5, 0.5], [0.6, 0.2, 0.3]])
    assert np.allclose(out, expected)


def test_default_patch_size_is_19():
    """v3 default patch size is 19 (vs 14 in v2)."""
    tk = PatchTokenizerV3()
    assert tk.patch_size == 19
    assert tk.default_patch_shape == (19, 19, 19)


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
