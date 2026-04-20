"""Roundtrip precision tests for the FP16-like log-uniform codec."""

import numpy as np
import pytest

from tomat.float_codec import FP16Codec


def nmae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / np.abs(a).sum())


@pytest.fixture
def codec() -> FP16Codec:
    # 8-decade range typical for electron densities in e/bohr³.
    return FP16Codec.tomol_3byte(log_min=-6.0, log_max=2.0)


def test_signed_shape(codec):
    vals = np.array([1.0, -1.0, 1e-3, -1e-3])
    comps = codec.encode_signed(vals)
    assert comps.shape == (4, 3)
    assert comps.dtype == np.int32
    assert (comps[:, 0] >= 0).all() and (comps[:, 0] < 512).all()
    assert (comps[:, 1] >= 0).all() and (comps[:, 1] < 256).all()
    assert (comps[:, 2] >= 0).all() and (comps[:, 2] < 256).all()


def test_signed_roundtrip_preserves_sign(codec):
    vals = np.array([1.0, -1.0, 3.14e-2, -3.14e-2])
    recon = codec.decode_signed(codec.encode_signed(vals))
    assert np.sign(recon).tolist() == np.sign(vals).tolist()


def test_signed_precision_6_sig_figs(codec):
    """24-bit over 8 log-decades → expect ~6 decimal digits of relative precision."""
    rng = np.random.default_rng(0)
    # log-uniform sample across the full range with random signs.
    logs = rng.uniform(codec.log_min, codec.log_max, size=50_000)
    signs = rng.choice([-1.0, 1.0], size=50_000)
    vals = signs * 10 ** logs

    recon = codec.decode_signed(codec.encode_signed(vals))
    rel_err = np.abs(recon - vals) / np.abs(vals)
    # Step size in log-space is (log_max - log_min) / 2^24 ≈ 4.8e-7 decades;
    # max relative error ≈ 10^4.8e-7 - 1 ≈ 1.1e-6. Allow 2× headroom.
    assert rel_err.max() < 3e-6
    assert nmae(vals, recon) < 1e-6


def test_zero_roundtrips_exactly(codec):
    vals = np.array([0.0, 1e-20, -1e-20])
    recon = codec.decode_signed(codec.encode_signed(vals))
    assert (recon == 0.0).all()


def test_clamp_above_log_max(codec):
    vals = np.array([1e10, -1e10])  # far outside [log_min, log_max]
    recon = codec.decode_signed(codec.encode_signed(vals))
    # Magnitudes clamp to 10^log_max; sign preserved.
    assert np.abs(recon[0] - 10 ** codec.log_max) / 10 ** codec.log_max < 1e-6
    assert recon[0] > 0 and recon[1] < 0


def test_unsigned_shape_and_precision(codec):
    rng = np.random.default_rng(1)
    logs = rng.uniform(codec.log_min, codec.log_max, size=10_000)
    vals = 10 ** logs
    comps = codec.encode_unsigned(vals)
    assert comps.shape == (10_000, 4)
    recon = codec.decode_unsigned(comps)
    rel_err = np.abs(recon - vals) / vals
    # 32-bit over 8 decades → step ≈ 1.9e-9 decades → rel_err max ≈ 4e-9.
    assert rel_err.max() < 1e-8


def test_se_sign_layout(codec):
    """SE ∈ [0,256) means positive; [256,512) means negative."""
    comps_pos = codec.encode_signed(np.array([1.0]))
    comps_neg = codec.encode_signed(np.array([-1.0]))
    assert comps_pos[0, 0] < 256
    assert 256 <= comps_neg[0, 0] < 512


# ---- two-token 9+12 variant -----------------------------------------------


def test_two_token_9_12_shape_and_vocabs():
    codec = FP16Codec.two_token_9_12(log_min=-6.0, log_max=2.0)
    assert codec.tokens_per_value_signed == 2
    assert codec.signed_vocabs == (512, 4096)
    assert codec.total_mag_bits_signed == 20  # 8 + 12

    comps = codec.encode_signed(np.array([1.0, -1.0, 1e-3, -1e-3]))
    assert comps.shape == (4, 2)
    assert (comps[:, 0] >= 0).all() and (comps[:, 0] < 512).all()
    assert (comps[:, 1] >= 0).all() and (comps[:, 1] < 4096).all()


def test_two_token_9_12_roundtrip_precision():
    """21-bit total (1 sign + 20 mag) over 8 decades → rel-err ~1e-5."""
    codec = FP16Codec.two_token_9_12(log_min=-6.0, log_max=2.0)
    rng = np.random.default_rng(42)
    logs = rng.uniform(codec.log_min, codec.log_max, size=20_000)
    signs = rng.choice([-1.0, 1.0], size=20_000)
    vals = signs * 10 ** logs
    recon = codec.decode_signed(codec.encode_signed(vals))
    rel_err = np.abs(recon - vals) / np.abs(vals)
    # Step = 8 / 2^20 ≈ 7.6e-6 decades → rel_err max ≈ 1.8e-5.
    assert rel_err.max() < 5e-5


def test_two_token_9_12_sign_in_first_token():
    codec = FP16Codec.two_token_9_12(log_min=-6.0, log_max=2.0)
    comps_pos = codec.encode_signed(np.array([1.0]))
    comps_neg = codec.encode_signed(np.array([-1.0]))
    assert comps_pos[0, 0] < 256
    assert 256 <= comps_neg[0, 0] < 512


# ---- one-token fp16 variant ----------------------------------------------


def test_fp16_1token_shape_and_vocab():
    codec = FP16Codec.fp16_1token(log_min=-6.0, log_max=2.0)
    assert codec.tokens_per_value_signed == 1
    assert codec.signed_vocabs == (65_536,)
    assert codec.total_mag_bits_signed == 15

    comps = codec.encode_signed(np.array([1.0, -1.0, 1e-3, -1e-3]))
    assert comps.shape == (4, 1)
    assert (comps >= 0).all() and (comps < 65_536).all()


def test_fp16_1token_roundtrip_precision():
    """16-bit total (1 sign + 15 mag) over 8 decades → rel-err ~6e-4."""
    codec = FP16Codec.fp16_1token(log_min=-6.0, log_max=2.0)
    rng = np.random.default_rng(7)
    logs = rng.uniform(codec.log_min, codec.log_max, size=10_000)
    signs = rng.choice([-1.0, 1.0], size=10_000)
    vals = signs * 10 ** logs
    recon = codec.decode_signed(codec.encode_signed(vals))
    rel_err = np.abs(recon - vals) / np.abs(vals)
    # Step = 8 / 2^15 ≈ 2.4e-4 decades → rel_err max ≈ 5.6e-4.
    assert rel_err.max() < 2e-3


def test_fp16_1token_sign_layout():
    codec = FP16Codec.fp16_1token(log_min=-6.0, log_max=2.0)
    comps_pos = codec.encode_signed(np.array([1.0]))
    comps_neg = codec.encode_signed(np.array([-1.0]))
    # First half is positive, second half is negative; boundary at 2^15 = 32768.
    assert comps_pos[0, 0] < 32_768
    assert comps_neg[0, 0] >= 32_768


def test_all_variants_zero_roundtrips_to_zero():
    for builder in (FP16Codec.tomol_3byte, FP16Codec.two_token_9_12, FP16Codec.fp16_1token):
        codec = builder(log_min=-6.0, log_max=2.0)
        recon = codec.decode_signed(codec.encode_signed(np.array([0.0, 1e-20, -1e-20])))
        assert (recon == 0.0).all(), f"zero round-trip failed for {builder.__name__}"
