"""No-GPU tests for `eval_mat_nmae.decode_density` — the density-distribution
decoder behind mat-NMAE / mat-NEMD.

The bug these guard (the NMAE-vs-NEMD blowup): the pre-fix eval computed the
model's density distribution as a *slice of the full-vocab softmax*
(`softmax(all_logits)[density_range]`, then renormalize). That slice underflows
fp32 to an all-zero vector when the model confidently predicts a non-density
token (density logits >~80 below the vocab max → exp() flushes to 0). An
all-zero distribution decodes — under the `median` rule — to the *last* bin
(`decode_dens[-1]`, ~1e2 of garbage here / ~3e4 with the real codec), while its
EMD is `dot(0, ·) = 0` (fake-perfect). `decode_density` takes the *raw* density
logits and softmaxes those, which max-subtracts within the density range and so
never underflows to all-zero.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marin.eval_mat_nmae import decode_density

N_BINS = 256
# Monotone bin → density-value map, wide dynamic range like the LMQ codec.
DECODE_DENS = jnp.asarray(np.geomspace(1e-2, 1e2, N_BINS), dtype=jnp.float32)
DECODERS = ["argmax", "median", "mean"]


@pytest.mark.parametrize("decoder", DECODERS)
def test_decode_shapes(decoder):
    rng = np.random.default_rng(0)
    B, P = 2, 7
    logits = jnp.asarray(rng.standard_normal((B, P, N_BINS)), jnp.float32)
    target = DECODE_DENS[jnp.asarray(rng.integers(0, N_BINS, (B, P)))]
    rho, emd = decode_density(logits, target, decoder, DECODE_DENS)
    assert rho.shape == (B, P) and emd.shape == (B, P)
    assert np.all(np.asarray(emd) >= 0.0)


def test_median_theorem_holds():
    """For the `median` decoder, `|rho - target| <= 2*emd` at every position —
    the inequality that bounds NMAE by 2*NEMD. It is exact math for any
    distribution; a violation means rho and emd were decoded from different
    distributions (the underflow bug does exactly that)."""
    rng = np.random.default_rng(1)
    B, P = 4, 32
    # Vary sharpness across positions: some logits sharp, some near-uniform.
    scale = jnp.asarray(rng.uniform(0.1, 12.0, (B, P, 1)), jnp.float32)
    logits = scale * jnp.asarray(rng.standard_normal((B, P, N_BINS)), jnp.float32)
    target = DECODE_DENS[jnp.asarray(rng.integers(0, N_BINS, (B, P)))]
    rho, emd = decode_density(logits, target, "median", DECODE_DENS)
    resid = np.abs(np.asarray(rho) - np.asarray(target)) - 2 * np.asarray(emd)
    assert resid.max() <= 1e-2, f"median theorem violated, worst {resid.max():.3e}"


@pytest.mark.parametrize("decoder", DECODERS)
def test_confident_nondensity_does_not_underflow(decoder):
    """THE regression test for the NMAE-vs-NEMD blowup.

    Full-vocab logits where the model is hugely confident about a *non-density*
    token, while its density sub-distribution is a sharp, correct spike at
    `true_bin`. The pre-fix slice-and-renormalize path underflows to all-zero
    here; `decode_density` (softmax over the raw density logits) stays correct.
    """
    n_vocab = 4 * N_BINS
    dens_lo = N_BINS  # density bins occupy [N_BINS, 2*N_BINS)
    true_bin = 30
    true_val = float(DECODE_DENS[true_bin])

    full = np.full((1, 1, n_vocab), -1e3, dtype=np.float32)
    bins = np.arange(N_BINS)
    # Sharp, correct density sub-distribution (Gaussian peak at true_bin)...
    full[0, 0, dens_lo:dens_lo + N_BINS] = -0.5 * (bins - true_bin) ** 2
    # ...but one non-density token is far more confident than ANY density bin.
    full[0, 0, 5] = 300.0
    full = jnp.asarray(full)

    # Pre-fix path: slice the full-vocab softmax + renormalize → all-zero.
    probs = jax.nn.softmax(full, axis=-1)
    dens_probs = probs[..., dens_lo:dens_lo + N_BINS]
    p_buggy = dens_probs / (dens_probs.sum(-1, keepdims=True) + 1e-12)
    assert float(jnp.max(p_buggy)) == 0.0, (
        "expected the full-softmax slice to underflow fp32 to all-zero — "
        "if it no longer does, this test's adversarial logits need sharpening"
    )

    # decode_density: softmax over the raw density logits → valid distribution.
    density_logits = full[..., dens_lo:dens_lo + N_BINS]
    target = jnp.asarray([[true_val]], dtype=jnp.float32)
    rho, emd = decode_density(density_logits, target, decoder, DECODE_DENS)
    rho_v, emd_v = float(rho[0, 0]), float(emd[0, 0])

    # rho must land near the true value — not the last-bin garbage (buggy
    # median → decode_dens[-1]) and not 0 (buggy mean → einsum with 0).
    assert true_val / 3 <= rho_v <= true_val * 3, (
        f"{decoder}: rho={rho_v:.3e} not near true {true_val:.3e} "
        f"(last-bin garbage = {float(DECODE_DENS[-1]):.3e})"
    )
    # ...and a sharp, correct prediction has small EMD.
    assert emd_v <= true_val, f"{decoder}: emd={emd_v:.3e} too large"
