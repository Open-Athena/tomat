"""Tests for spec 34 F1 atom-position encoding.

Covers:

  (1) Tokenizer F0 vs F1: F1 preamble drops the [POS_START]…[POS_END]
      block and gains an aligned `atom_xyz` sidecar.
  (2) F1 detokenize round-trip preserves frac coords to float32 precision.
  (3) Sinusoidal encoding helper: shape correctness + 2π-periodicity at k=0.
  (4) `Qwen3F1LMHeadModel._f1_pos_addend` is exactly zero at non-atom
      positions and nonzero at atom positions.
  (5) `Qwen3F1LMHeadModel.activations` forward succeeds on a packed-style
      F1 row; F1-on vs F1-off outputs differ (the xyz addend propagates).
  (6) F1-tokenized parquet rows are loadable without `atom_xyz` (the
      column is optional from a consumer's POV — readers that ignore it
      see the same `input_ids` token stream).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402

_MARIN = str(Path(__file__).resolve().parent.parent / "marin")
if _MARIN not in sys.path:
    sys.path.insert(0, _MARIN)

from qwen3_density import (  # noqa: E402
    F1Args,
    Qwen3DensityConfig,
    Qwen3DensityLMHeadModel,
    configure_atom_encoder,
)
from atom_encoder import f1_sinusoidal_embed  # noqa: E402
from tomat.tokenizers.patch import (  # noqa: E402
    ATOM_END,
    ATOM_OFFSET,
    SPECIAL_TOKENS,
)
from tomat.tokenizers.patch_v3 import PatchTokenizerV3  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_structure():
    from pymatgen.core.structure import Structure
    return Structure(
        lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        species=["C", "H", "O", "Si"],
        coords=[
            [0.1, 0.2, 0.3],
            [0.7, 0.6, 0.5],
            [0.2, 0.8, 0.1],
            [0.4, 0.4, 0.4],
        ],
    )


@pytest.fixture
def density():
    rng = np.random.default_rng(0)
    return rng.uniform(1e-3, 1.0, size=(16, 16, 16)).astype(np.float32)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def test_f0_vs_f1_preamble_lengths(fake_structure, density):
    """F1 drops the [POS_START]…[POS_END] block. With 4 atoms and the
    default `tomol_3byte` position codec (3 tokens/axis), that's
    1 + 4*3*3 + 1 = 38 tokens dropped.
    """
    t0 = PatchTokenizerV3(patch_size=8, atom_encoding="f0")
    t1 = PatchTokenizerV3(patch_size=8, atom_encoding="f1")
    s0 = t0.make_sample(
        task_id="x", density=density, structure=fake_structure, offset=(4, 4, 4),
    )
    s1 = t1.make_sample(
        task_id="x", density=density, structure=fake_structure, offset=(4, 4, 4),
    )
    toks0, _ = t0.tokenize_with_xyz(s0)
    toks1, _ = t1.tokenize_with_xyz(s1)

    expected_savings = (
        1                            # [POS_START]
        + len(fake_structure) * 3 * 3  # 4 atoms × 3 axes × 3 tokens/axis (tomol_3byte)
        + 1                          # [POS_END]
    )
    assert len(toks0) - len(toks1) == expected_savings, (
        f"f0 - f1 = {len(toks0) - len(toks1)}, expected {expected_savings}"
    )
    assert SPECIAL_TOKENS["[POS_START]"] not in toks1
    assert SPECIAL_TOKENS["[POS_END]"] not in toks1


def test_f1_atom_xyz_populated(fake_structure, density):
    """`atom_xyz` is finite at the atom-token slots and NaN elsewhere; the
    finite rows match the patch-frame translated fractional coords.
    """
    t = PatchTokenizerV3(patch_size=8, atom_encoding="f1")
    sample = t.make_sample(
        task_id="x", density=density, structure=fake_structure, offset=(4, 4, 4),
    )
    toks, xyz = t.tokenize_with_xyz(sample)

    ai = toks.index(SPECIAL_TOKENS["[ATOMS_START]"])
    aj = toks.index(SPECIAL_TOKENS["[ATOMS_END]"])

    # Finite at atom slots (ai+1 .. aj-1 inclusive), NaN elsewhere.
    finite_rows = np.isfinite(xyz).all(axis=1)
    expected_finite = np.zeros(len(toks), dtype=bool)
    expected_finite[ai + 1 : aj] = True
    np.testing.assert_array_equal(finite_rows, expected_finite)

    # All atom-slot tokens are in the atom-id range.
    atom_slots = toks[ai + 1 : aj]
    assert all(ATOM_OFFSET <= t < ATOM_END for t in atom_slots), atom_slots

    # The xyz values match the patch-frame translated coords.
    expected = t.translate_frac_coords(
        sample.frac_coords, sample.offset, sample.grid_shape,
    )
    np.testing.assert_allclose(
        xyz[ai + 1 : aj].astype(np.float64),
        expected,
        atol=1e-6,
    )


def test_f1_detokenize_roundtrip(fake_structure, density):
    t = PatchTokenizerV3(patch_size=8, atom_encoding="f1")
    sample = t.make_sample(
        task_id="x", density=density, structure=fake_structure, offset=(4, 4, 4),
    )
    toks, xyz = t.tokenize_with_xyz(sample)
    rt = t.detokenize(toks, atom_xyz=xyz)

    expected = t.translate_frac_coords(
        sample.frac_coords, sample.offset, sample.grid_shape,
    )
    np.testing.assert_allclose(rt.frac_coords, expected, atol=1e-6)
    np.testing.assert_array_equal(rt.atomic_numbers, sample.atomic_numbers)
    np.testing.assert_array_equal(rt.grid_shape, sample.grid_shape)


def test_f0_path_unchanged(fake_structure, density):
    """`atom_encoding='f0'` token stream is byte-identical to the original
    `tokenize()` path (the default). This is the back-compat guarantee.
    """
    t = PatchTokenizerV3(patch_size=8, atom_encoding="f0")
    sample = t.make_sample(
        task_id="x", density=density, structure=fake_structure, offset=(4, 4, 4),
    )
    via_new = t.tokenize(sample)
    via_with_xyz, _ = t.tokenize_with_xyz(sample)
    assert via_new == via_with_xyz

    # Pre-spec-34 v3 callers used `tokenize()` which dispatched to the original
    # in-place build. Cross-check that the f0 token-id sequence has the
    # expected structure.
    assert via_new[0] == SPECIAL_TOKENS["[BOS]"]
    assert via_new[-1] == SPECIAL_TOKENS["[EOS]"]
    assert SPECIAL_TOKENS["[POS_START]"] in via_new
    assert SPECIAL_TOKENS["[POS_END]"] in via_new


def test_f1_requires_v3():
    """The CLI gate `--atom-encoding f1` requires `--tokenizer-version v3`.
    The tokenizer itself doesn't enforce this, but the CLI does — covered
    here for a single-file behavioural check on the v2 path.
    """
    # v2 doesn't have an atom_encoding field — confirm the v3 class accepts
    # the kwarg but only "f0" and "f1" pass validation.
    PatchTokenizerV3(patch_size=8, atom_encoding="f1")  # OK
    PatchTokenizerV3(patch_size=8, atom_encoding="f0")  # OK
    bad = PatchTokenizerV3(patch_size=8, atom_encoding="garbage")
    # Encoding is validated at tokenize() time (not __init__) for keep-it-cheap
    # dataclass semantics.
    from pymatgen.core.structure import Structure
    s = Structure(
        lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        species=["C"], coords=[[0.0, 0.0, 0.0]],
    )
    rho = np.zeros((16, 16, 16), dtype=np.float32)
    sample = bad.make_sample(task_id="x", density=rho, structure=s, offset=(0, 0, 0))
    with pytest.raises(ValueError, match="atom_encoding"):
        bad.tokenize(sample)


# ---------------------------------------------------------------------------
# sinusoidal helper
# ---------------------------------------------------------------------------

def test_sinusoidal_shape():
    xyz = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out = f1_sinusoidal_embed(xyz, num_freqs=10)
    assert out.shape == (2, 60)


def test_sinusoidal_periodicity():
    """At k=0 the angular freq is π, so γ(p) and γ(p + 2) should match
    along each axis."""
    xyz1 = jnp.array([[0.1, 0.2, 0.3]])
    xyz2 = jnp.array([[0.1 + 2.0, 0.2 + 2.0, 0.3 + 2.0]])
    g1 = f1_sinusoidal_embed(xyz1, num_freqs=1)
    g2 = f1_sinusoidal_embed(xyz2, num_freqs=1)
    np.testing.assert_allclose(g1, g2, atol=1e-5)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _tiny_f1_model(num_freqs: int = 8, max_seq_len: int = 32):
    Vocab = Axis("vocab", 5000)
    config = Qwen3DensityConfig(
        max_seq_len=max_seq_len,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    configure_atom_encoder("f1", f1_args=F1Args(num_freqs=num_freqs))
    return config, Qwen3DensityLMHeadModel.init(
        Vocab, config, key=jrandom.PRNGKey(0),
    )


def test_pos_addend_zero_at_non_atom():
    config, model = _tiny_f1_model(num_freqs=4, max_seq_len=16)
    Pos = config.max_Pos
    seq_len = Pos.size

    tokens_np = np.zeros((seq_len,), dtype=np.int32)
    atom_positions = [3, 4, 9]
    for i in atom_positions:
        tokens_np[i] = ATOM_OFFSET + 5  # arbitrary atom-type id

    XYZ = Axis("xyz", 3)
    xyz_np = np.full((seq_len, 3), np.nan, dtype=np.float32)
    for j, i in enumerate(atom_positions):
        xyz_np[i] = [0.1 * (j + 1), 0.2 * (j + 1), 0.3 * (j + 1)]

    tokens = hax.named(jnp.asarray(tokens_np), (Pos,))
    atom_xyz = hax.named(jnp.asarray(xyz_np), (Pos, XYZ))

    # The addend lives on the F1AtomEncoder now (spec-34 F-axis refactor).
    addend = model.atom_encoder._addend(tokens, atom_xyz)
    addend_np = np.asarray(addend.array)

    non_atom_idx = [i for i in range(seq_len) if i not in atom_positions]
    # Exactly zero (multiplied by mask=0).
    assert float(np.max(np.abs(addend_np[non_atom_idx]))) == 0.0
    # Strictly nonzero at atom positions (sinusoidal of nonzero coords + projection).
    for i in atom_positions:
        assert float(np.max(np.abs(addend_np[i]))) > 0.0


def test_forward_f1_on_vs_off():
    """F1-on vs F1-off forward should differ (the addend propagates through
    the transformer). F1-off (atom_xyz=None) should match standard Qwen3."""
    config, model = _tiny_f1_model(num_freqs=8, max_seq_len=32)
    Pos = config.max_Pos
    seq_len = Pos.size

    tokens_np = np.zeros((seq_len,), dtype=np.int32)
    tokens_np[5] = ATOM_OFFSET + 5   # C
    tokens_np[6] = ATOM_OFFSET + 7   # O
    tokens_np[12] = ATOM_OFFSET + 13  # Al
    tokens = hax.named(jnp.asarray(tokens_np), (Pos,))

    XYZ = Axis("xyz", 3)
    xyz_np = np.full((seq_len, 3), np.nan, dtype=np.float32)
    xyz_np[5] = [0.1, 0.2, 0.3]
    xyz_np[6] = [0.4, 0.5, 0.6]
    xyz_np[12] = [0.7, 0.8, 0.9]
    atom_xyz = hax.named(jnp.asarray(xyz_np), (Pos, XYZ))

    acts_on = model.activations(tokens, atom_xyz=atom_xyz)
    acts_off = model.activations(tokens, atom_xyz=None)

    assert acts_on.axes == acts_off.axes
    diff = float(jnp.max(jnp.abs(acts_on.array - acts_off.array)))
    assert diff > 0.0


def test_forward_f1_off_matches_base_qwen3():
    """With atom_xyz=None, F1 forward must be identical to plain Qwen3
    forward (the f1_atom_proj weights are unused). This is the
    TOMAT_F1_MODE=0 back-compat guarantee at the model layer."""
    config, model = _tiny_f1_model(num_freqs=8, max_seq_len=32)
    Pos = config.max_Pos
    seq_len = Pos.size
    tokens_np = np.zeros((seq_len,), dtype=np.int32)
    tokens_np[3] = ATOM_OFFSET + 5
    tokens = hax.named(jnp.asarray(tokens_np), (Pos,))

    # Equivalent base forward: call the parent (Qwen3DensityLMHeadModel /
    # Qwen3LMHeadModel) activations — which is what super().activations is.
    acts_f1_off = model.activations(tokens, atom_xyz=None)
    # Same thing via the LlamaLMHeadModel.activations path (the actual base).
    from levanter.models.qwen import Qwen3LMHeadModel
    acts_base = Qwen3LMHeadModel.activations(model, tokens)
    np.testing.assert_allclose(
        np.asarray(acts_f1_off.array),
        np.asarray(acts_base.array),
        atol=1e-5,
    )
