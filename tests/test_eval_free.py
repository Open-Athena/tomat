"""No-GPU tests for the free-running (autoregressive) eval's index math.

The free-running branch of `eval_mat_nmae.py` re-forwards a growing prefix
P^3 times per patch; correctness hinges on (a) `_next_bucket` padding each
forward to a bounded set of JIT shapes, and (b) the v3 DENS-block layout —
where DENS_START sits, that the density region is exactly P^3 tokens, and
where DENS_END / EOS land. A wrong `d0` feeds the model the wrong prefix and
silently corrupts the free-running number. No model / GPU needed.
"""
import numpy as np
import pytest

# `marin.eval_mat_nmae` imports the jax/levanter stack at module load — absent
# in lightweight CI, present in the local eval venv.
pytest.importorskip("jax")

from marin.eval_mat_nmae import (  # noqa: E402
    FREE_BUCKET,
    N_ATOMS,
    N_INTS,
    N_SPECIALS_LAT,
    TOK,
    _next_bucket,
    build_all_patch_input_ids_v3,
)

P = 19
P3 = P ** 3


def test_next_bucket_rounds_up_to_free_bucket_multiple():
    # Every bucketed forward length MUST be a FREE_BUCKET multiple — flash
    # attention rejects a query axis that isn't a multiple of its (1024)
    # block size. 6960 (a real seq_len) → 7168 is the case the first run hit.
    assert [_next_bucket(n) for n in (1, 1024, 1025, 2048, 3000, 6960)] == [
        1024, 1024, 2048, 2048, 3072, 7168,
    ]


def test_next_bucket_always_a_multiple():
    # Exhaustive: no input in the realistic range yields a non-multiple.
    assert all(_next_bucket(n) % FREE_BUCKET == 0 for n in range(1, 8193))


def _v3_vocab_offsets():
    """vocab_offsets for a lat-aware v3 tokenizer (token_mag_bits = [8,8,8])."""
    pos_vocabs = (512, 256, 256)
    pos_offset = N_SPECIALS_LAT + N_ATOMS + N_INTS
    return {
        "n_specials": N_SPECIALS_LAT,
        "pos_offset": pos_offset,
        "pos_vocabs": pos_vocabs,
        "pos_log_min": -4.0,
        "pos_log_max": 0.0,
        "density_offset": pos_offset + sum(pos_vocabs),
        "lat_aware": True,
    }


@pytest.mark.parametrize("n_atoms", [1, 4, 17, 40])
def test_v3_dens_block_layout(n_atoms):
    """The free branch derives `d0` (DENS_START index) and `seq_len` purely
    from the v3 layout (BOS|GRID|LATTICE|ATOMS|POS|DENS|EOS). Verify those
    derivations against an actually-built input row."""
    rng = np.random.default_rng(n_atoms)
    grid = (38, 38, 38)
    density = rng.random(grid, dtype=np.float64).astype(np.float32)
    offsets = np.array([[0, 0, 0], [19, 19, 19]], dtype=np.int32)
    Zs = np.full(n_atoms, 6, dtype=np.int32)
    frac = rng.random((n_atoms, 3))
    n_bins = 256
    lmq_codec = (
        np.linspace(0.0, 1.0, n_bins - 1, dtype=np.float32),  # boundaries
        np.linspace(0.0, 1.0, n_bins, dtype=np.float32),      # recon points
        1.0,                                                  # clip_max
    )
    vo = _v3_vocab_offsets()

    ids = build_all_patch_input_ids_v3(
        density, grid, offsets, Zs, frac, lmq_codec, vo,
        P=P, pad_to=8192, lattice_params=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    )
    row0 = ids[0]
    density_offset = vo["density_offset"]
    is_d = (row0 >= density_offset) & (row0 < density_offset + n_bins)
    dpos = np.where(is_d)[0]

    # The two derivations the free branch makes:
    d0 = int(dpos[0]) - 1          # DENS_START index
    seq_len = d0 + P3 + 3          # last used position (EOS) + 1

    assert len(dpos) == P3
    assert d0 == 10 * n_atoms + 18
    assert dpos.tolist() == list(range(d0 + 1, d0 + 1 + P3))
    assert int(row0[d0]) == TOK["DENS_START"]
    assert int(row0[d0 + P3 + 1]) == TOK["DENS_END"]
    assert int(row0[d0 + P3 + 2]) == TOK["EOS"]
    assert seq_len <= ids.shape[1]
