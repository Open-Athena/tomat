"""Equivalence test for spec 39 — paged-KV decode vs legacy recompute on
the free-running eval path.

The spec-39 rewrite of `marin/eval_mat_nmae.py`'s free branch (paged-KV
decode via Levanter's `model.decode` + `DecodeState`) must match the
legacy recompute loop (one growing-prefix forward per step) on the
*same* (model, preamble, density-token feedback) trace to bf16
tolerance. This test reproduces both loops on a tiny model + synthetic
preamble, then asserts the per-step density-bin medians + EMDs agree.

Pass criterion: per-step rho (median-decode) max-abs-diff < 1e-5 AND
EMD max-abs-diff < 5e-3. Both modes feed the same sampled tokens back,
so any divergence compounds; a tight bound here means production NMAE
will agree to many decimal places.

The full-fat equivalence — same ckpt, same single mat, NMAE % agreeing
to 3+ decimal places — has to run on real hardware; that's a wall-clock
A/B once the paths are wired into the eval driver (`tomat evals fire`
on a real ckpt with `TOMAT_EVAL_FREE_LEGACY=1` vs unset).
"""
import numpy as np
import jax
import jax.random as jrandom
import jax.numpy as jnp
import haliax as hax
from haliax import Axis

from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.layers.attention import AttentionMask
from levanter.inference.page_table import PageTable
from levanter.inference.jit_scheduler import DecodeState


def _tiny_setup(seed: int = 0):
    """Build a tiny Qwen3 model + a fake preamble/density-token layout."""
    cfg = Qwen3Config(
        max_seq_len=64, hidden_dim=64, num_heads=4, num_kv_heads=2,
        num_layers=2, intermediate_dim=128,
    )
    Vocab = Axis("vocab", 256)
    key = jrandom.PRNGKey(seed)
    model = Qwen3LMHeadModel.init(Vocab, cfg, key=key)
    return model, Vocab


def _decode_dens():
    """Fake LMQ recon table: 32 density-token bins covering [0, 1.0]."""
    return jnp.linspace(0.0, 1.0, 32, dtype=jnp.float32)


def _run_legacy(model, preamble_np, n_steps, DENS_LO, DENS_HI, decode_dens,
                emd_targets):
    """Recompute-path: grow the prefix by 1 token per step; full forward
    each time. Returns ([B, n_steps] rho, [B, n_steps] emd, [B, n_steps]
    fed-back tokens). Mirrors `free_step` in eval_mat_nmae.py."""
    B, preamble_len = preamble_np.shape
    causal = AttentionMask.causal()
    Batch = Axis("batch", B)

    rho_out = np.zeros((B, n_steps), dtype=np.float32)
    emd_out = np.zeros((B, n_steps), dtype=np.float32)
    fed = np.zeros((B, n_steps), dtype=np.int32)

    # Token buffer, growing.
    buf = np.copy(preamble_np)
    n_bins = decode_dens.shape[0]
    for i in range(n_steps):
        seq_len = preamble_len + i  # tokens currently in buffer
        Pos = Axis("position", seq_len)
        tok_ha = hax.named(jnp.asarray(buf[:, :seq_len]), (Batch, Pos))
        act = model.activations(tok_ha, key=None, attn_mask=causal)
        head = model.get_lm_head()
        logits = hax.dot(act, head, axis=model.Embed)
        logits_arr = logits.array.astype(jnp.float32)
        # The frontier output position predicts the next density token; for
        # the i-th step it's the last position of the current buffer.
        dl = logits_arr[:, seq_len - 1, DENS_LO:DENS_HI]
        p = jax.nn.softmax(dl, axis=-1)
        cumP = jnp.cumsum(p, axis=-1)
        med = jnp.clip(jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32),
                       0, n_bins - 1)
        rho_i = decode_dens[med]
        emd_i = jax.vmap(
            lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t))
        )(p, emd_targets[:, i])
        next_tok = med + DENS_LO  # (B,)
        rho_out[:, i] = np.asarray(rho_i)
        emd_out[:, i] = np.asarray(emd_i)
        fed[:, i] = np.asarray(next_tok)
        # Append next_tok to buffer.
        col = np.asarray(next_tok)[:, None]
        buf = np.concatenate([buf, col], axis=1)
    return rho_out, emd_out, fed


def _run_paged(model, preamble_np, n_steps, DENS_LO, DENS_HI, decode_dens,
               emd_targets):
    """Paged-KV path: prefill the preamble once, then 1-token decode per
    step. Mirrors the new code in eval_mat_nmae.py's free branch."""
    B, preamble_len = preamble_np.shape
    n_bins = decode_dens.shape[0]
    full_len = preamble_len + n_steps
    page_size = 4  # tiny pages so we exercise multi-page allocation
    max_pages_per_seq = (full_len + page_size - 1) // page_size + 1
    pt = PageTable.init(
        max_pages=B * max_pages_per_seq + 4,
        max_seqs=B, page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
    )
    ds = DecodeState.init(pt, pad_token_id=0)
    for b in range(B):
        ds, _ = ds.reserve_slot(b)
    cache = model.initial_cache(pt.spec(), dtype=jnp.float32)

    # Prefill: flat layout.
    slot_ids_pre = hax.named(
        np.repeat(np.arange(B), preamble_len).astype(np.int32), "position",
    )
    pos_ids_pre = hax.named(
        np.tile(np.arange(preamble_len), B).astype(np.int32), "position",
    )
    tok_pre = hax.named(
        preamble_np.reshape(-1).astype(np.int32), "position",
    )
    ds, binfo = ds.allocate_for_seq(slot_ids_pre, pos_ids_pre)
    logits_pre, cache = model.decode(tok_pre, cache, binfo, pos_ids_pre)
    arr_pre = logits_pre.array.astype(jnp.float32)  # (B*preamble_len, V)
    last_pos_np = (
        np.arange(B, dtype=np.int32) * preamble_len + (preamble_len - 1)
    )
    last_logits = arr_pre[last_pos_np]  # (B, V)
    dl0 = last_logits[:, DENS_LO:DENS_HI]
    p0 = jax.nn.softmax(dl0, axis=-1)
    cumP0 = jnp.cumsum(p0, axis=-1)
    med0 = jnp.clip(jnp.sum(cumP0 < 0.5, axis=-1).astype(jnp.int32),
                    0, n_bins - 1)
    rho0 = decode_dens[med0]
    emd0 = jax.vmap(
        lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t))
    )(p0, emd_targets[:, 0])
    next_tok = med0 + DENS_LO

    rho_out = np.zeros((B, n_steps), dtype=np.float32)
    emd_out = np.zeros((B, n_steps), dtype=np.float32)
    fed = np.zeros((B, n_steps), dtype=np.int32)
    rho_out[:, 0] = np.asarray(rho0)
    emd_out[:, 0] = np.asarray(emd0)
    fed[:, 0] = np.asarray(next_tok)

    for i in range(1, n_steps):
        abs_pos = preamble_len + i - 1
        slot_ids = hax.named(np.arange(B, dtype=np.int32), "position")
        pos_ids = hax.named(
            np.full(B, abs_pos, dtype=np.int32), "position",
        )
        tok = hax.named(np.asarray(next_tok, dtype=np.int32), "position")
        ds, binfo = ds.allocate_for_seq(slot_ids, pos_ids)
        logits_step, cache = model.decode(tok, cache, binfo, pos_ids)
        arr = logits_step.array.astype(jnp.float32)  # (B, V)
        dl = arr[:, DENS_LO:DENS_HI]
        p = jax.nn.softmax(dl, axis=-1)
        cumP = jnp.cumsum(p, axis=-1)
        med = jnp.clip(jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32),
                       0, n_bins - 1)
        rho_i = decode_dens[med]
        emd_i = jax.vmap(
            lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t))
        )(p, emd_targets[:, i])
        next_tok = med + DENS_LO
        rho_out[:, i] = np.asarray(rho_i)
        emd_out[:, i] = np.asarray(emd_i)
        fed[:, i] = np.asarray(next_tok)
    return rho_out, emd_out, fed


def _run_paged_scan(model, preamble_np, n_steps, DENS_LO, DENS_HI,
                    decode_dens, emd_targets):
    """Same as `_run_paged` but the inner step is `lax.scan` — mirrors the
    production code's `paged_decode_scan` (one JIT per batch, not P^3)."""
    B, preamble_len = preamble_np.shape
    n_bins = decode_dens.shape[0]
    full_len = preamble_len + n_steps
    page_size = 4
    max_pages_per_seq = (full_len + page_size - 1) // page_size + 1
    pt = PageTable.init(
        max_pages=B * max_pages_per_seq + 4,
        max_seqs=B, page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
    )
    ds = DecodeState.init(pt, pad_token_id=0)
    for b in range(B):
        ds, _ = ds.reserve_slot(b)
    cache = model.initial_cache(pt.spec(), dtype=jnp.float32)

    # Prefill.
    slot_ids_pre = hax.named(
        np.repeat(np.arange(B), preamble_len).astype(np.int32), "position",
    )
    pos_ids_pre = hax.named(
        np.tile(np.arange(preamble_len), B).astype(np.int32), "position",
    )
    tok_pre = hax.named(
        preamble_np.reshape(-1).astype(np.int32), "position",
    )
    ds, binfo = ds.allocate_for_seq(slot_ids_pre, pos_ids_pre)
    logits_pre, cache = model.decode(tok_pre, cache, binfo, pos_ids_pre)
    arr_pre = logits_pre.array.astype(jnp.float32)
    last_pos_np = (
        np.arange(B, dtype=np.int32) * preamble_len + (preamble_len - 1)
    )
    last_logits = arr_pre[last_pos_np]
    dl0 = last_logits[:, DENS_LO:DENS_HI]
    p0 = jax.nn.softmax(dl0, axis=-1)
    cumP0 = jnp.cumsum(p0, axis=-1)
    med0 = jnp.clip(jnp.sum(cumP0 < 0.5, axis=-1).astype(jnp.int32),
                    0, n_bins - 1)
    rho0 = decode_dens[med0]
    emd0 = jax.vmap(
        lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t))
    )(p0, emd_targets[:, 0])
    next_tok = med0 + DENS_LO

    # Inner loop via scan.
    slot_ids = hax.named(jnp.arange(B, dtype=jnp.int32), "position")
    abs_pos_seq = jnp.broadcast_to(
        jnp.arange(preamble_len, preamble_len + n_steps - 1,
                   dtype=jnp.int32)[:, None],
        (n_steps - 1, B),
    )
    td_seq = jnp.asarray(emd_targets[:, 1:]).T  # (n_steps - 1, B)

    def body(carry, xs):
        ds, cache, prev_tok = carry
        abs_pos_i, td_i = xs
        pos_ids = hax.named(abs_pos_i, "position")
        tok = hax.named(prev_tok, "position")
        ds, binfo = ds.allocate_for_seq(slot_ids, pos_ids)
        logits, cache = model.decode(tok, cache, binfo, pos_ids)
        arr = logits.array.astype(jnp.float32)
        dl = arr[:, DENS_LO:DENS_HI]
        p = jax.nn.softmax(dl, axis=-1)
        cumP = jnp.cumsum(p, axis=-1)
        med = jnp.clip(jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32),
                       0, n_bins - 1)
        rho = decode_dens[med]
        emd = jax.vmap(
            lambda pv, t: jnp.dot(pv, jnp.abs(decode_dens - t))
        )(p, td_i)
        next_tok = med + DENS_LO
        return (ds, cache, next_tok), (rho, emd, next_tok)

    (_ds, _cache, _), (rho_rest, emd_rest, fed_rest) = jax.lax.scan(
        body, (ds, cache, next_tok), (abs_pos_seq, td_seq),
    )
    rho_out = np.zeros((B, n_steps), dtype=np.float32)
    emd_out = np.zeros((B, n_steps), dtype=np.float32)
    fed = np.zeros((B, n_steps), dtype=np.int32)
    rho_out[:, 0] = np.asarray(rho0)
    emd_out[:, 0] = np.asarray(emd0)
    fed[:, 0] = np.asarray(next_tok)
    rho_out[:, 1:] = np.asarray(rho_rest).T
    emd_out[:, 1:] = np.asarray(emd_rest).T
    fed[:, 1:] = np.asarray(fed_rest).T
    return rho_out, emd_out, fed


def test_paged_kv_matches_recompute():
    """End-to-end equivalence on a tiny synthetic problem."""
    B = 2
    preamble_len = 8
    n_steps = 6
    # Pretend density tokens occupy a slice of the (tiny) vocab.
    DENS_LO, DENS_HI = 200, 232  # 32-bin density block
    decode_dens = _decode_dens()  # 32 bins covering [0, 1]
    model, _ = _tiny_setup(seed=0)

    rng = np.random.default_rng(0)
    preamble = rng.integers(0, 200, size=(B, preamble_len), dtype=np.int32)
    # Synthetic EMD targets — uniform random floats in [0, 1].
    emd_targets = jnp.asarray(
        rng.uniform(0.0, 1.0, size=(B, n_steps)).astype(np.float32),
    )

    rho_l, emd_l, fed_l = _run_legacy(
        model, preamble, n_steps, DENS_LO, DENS_HI, decode_dens, emd_targets,
    )
    rho_p, emd_p, fed_p = _run_paged(
        model, preamble, n_steps, DENS_LO, DENS_HI, decode_dens, emd_targets,
    )
    rho_s, emd_s, fed_s = _run_paged_scan(
        model, preamble, n_steps, DENS_LO, DENS_HI, decode_dens, emd_targets,
    )

    # Per-step diagnostics so a failing test names the divergent step.
    def _diffs(rhoA, emdA, fedA, rhoB, emdB, fedB, label):
        out = []
        for i in range(n_steps):
            for b in range(B):
                drho = abs(float(rhoA[b, i] - rhoB[b, i]))
                demd = abs(float(emdA[b, i] - emdB[b, i]))
                if drho > 1e-5 or demd > 5e-3 or int(fedA[b, i]) != int(fedB[b, i]):
                    out.append((label, i, b, drho, demd,
                                int(fedA[b, i]), int(fedB[b, i])))
        return out

    bad = (
        _diffs(rho_l, emd_l, fed_l, rho_p, emd_p, fed_p, "legacy vs paged-loop")
        + _diffs(rho_l, emd_l, fed_l, rho_s, emd_s, fed_s, "legacy vs paged-scan")
    )
    assert not bad, (
        "paged-KV vs recompute disagree at:\n"
        + "\n".join(
            f"  [{lab}] step={i} b={b} drho={drho:.2e} demd={demd:.2e} "
            f"fed_A={fa} fed_B={fb}"
            for (lab, i, b, drho, demd, fa, fb) in bad
        )
    )

    # Same fed-back token stream → same downstream evolution.
    assert fed_l.tolist() == fed_p.tolist()
    assert fed_l.tolist() == fed_s.tolist()
