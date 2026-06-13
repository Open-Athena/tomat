"""Equivalence smoke for Levanter paged-KV `model.decode` vs the recompute
path that tomat's free-running mat-NMAE eval uses today.

Pass criterion: logits from the paged-KV path match the full-forward
path to ~bf16 tolerance at every probe position. If this passes we can
swap `marin/eval_mat_nmae.py:802-828` over to the cached path. If it
fails we learn exactly where (RoPE / pos_ids / page allocation / etc.).

Plan from `specs/39-fr-eval-paged-kv.md` (Path B). Reference:
`/Users/ryan/c/oa/levanter/tests/inference/test_paged_attention.py:303-384`
for the prefill-then-chunked-decode pattern.
"""
import pytest
# Heavy ML deps — CI's lightweight `uv sync` doesn't install them. Skip
# the whole module in environments without `jax`/`haliax`/Levanter (this
# smoke is a real-hardware bf16-equivalence A/B; CI never runs it green).
jax = pytest.importorskip("jax")
hax = pytest.importorskip("haliax")
pytest.importorskip("levanter")

import jax.random as jrandom
import jax.numpy as jnp
import numpy as np
from haliax import Axis

from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.layers.attention import AttentionMask
from levanter.inference.page_table import PageTable
from levanter.inference.jit_scheduler import DecodeState


def _build_tiny_model(key):
    cfg = Qwen3Config(
        max_seq_len=64, hidden_dim=64, num_heads=4, num_kv_heads=2,
        num_layers=2, intermediate_dim=128,
    )
    Vocab = Axis("vocab", 256)
    model = Qwen3LMHeadModel.init(Vocab, cfg, key=key)
    return model, Vocab, cfg


def main():
    B = 2
    prefix_len = 8
    n_steps = 5
    full_len = prefix_len + n_steps

    rng = np.random.default_rng(0)
    full_tok_np = rng.integers(0, 256, size=(B, full_len), dtype=np.int32)

    key = jrandom.PRNGKey(0)
    model, Vocab, cfg = _build_tiny_model(key)

    # --- (a) recompute path: one big forward over (B, full_len) ---
    Batch = Axis("batch", B)
    Pos = Axis("position", full_len)
    tok_ha = hax.named(full_tok_np, (Batch, Pos))
    # Paged decode hardcodes causal masking inside the kernel; the recompute
    # path's `attn_mask=None` would default to no mask → they'd disagree.
    # Force causal here so both modes see the same attention pattern.
    causal = AttentionMask.causal()
    act_full = model.activations(tok_ha, key=None, attn_mask=causal)
    head = model.get_lm_head()
    logits_full = hax.dot(act_full, head, axis=model.Embed)
    # axes: (Batch, Pos, Vocab); we'll probe the last `prefix_len-1` of prefill
    # and each of the n_steps decode positions.
    logits_full_arr = logits_full.array  # (B, full_len, V)

    # --- (b) paged path ---
    page_size = 4
    max_pages_per_seq = (full_len + page_size - 1) // page_size
    pt = PageTable.init(
        max_pages=B * max_pages_per_seq + 4,  # small headroom
        max_seqs=B,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
    )
    ds = DecodeState.init(pt, pad_token_id=0)
    for b in range(B):
        ds, _ = ds.reserve_slot(b)
    cache = model.initial_cache(pt.spec(), dtype=jnp.float32)

    # Prefill: flat over batch — slot_ids/pos_ids/tokens all 1-D `position` axis
    slot_ids_pre = hax.named(
        np.repeat(np.arange(B), prefix_len).astype(np.int32), "position",
    )
    pos_ids_pre = hax.named(
        np.tile(np.arange(prefix_len), B).astype(np.int32), "position",
    )
    tok_pre = hax.named(
        full_tok_np[:, :prefix_len].reshape(-1).astype(np.int32), "position",
    )
    ds, binfo = ds.allocate_for_seq(slot_ids_pre, pos_ids_pre)
    logits_pre, cache = model.decode(tok_pre, cache, binfo, pos_ids_pre)
    # logits_pre axes: (position=B*prefix_len, vocab)
    # Last-position-of-each-seq index = b*prefix_len + (prefix_len-1)
    for b in range(B):
        got = logits_pre.array[b * prefix_len + (prefix_len - 1)]
        want = logits_full_arr[b, prefix_len - 1]
        diff = np.max(np.abs(got - want))
        print(f"  prefill[b={b}] last-pos max-abs-diff: {diff:.2e}")
        assert diff < 5e-3, f"prefill mismatch b={b}: {diff}"

    # Per-step decode: B sequences × 1 new token each
    for s in range(n_steps):
        slot_ids = hax.named(np.arange(B, dtype=np.int32), "position")
        pos_ids = hax.named(
            np.full(B, prefix_len + s, dtype=np.int32), "position",
        )
        tok = hax.named(
            full_tok_np[:, prefix_len + s].astype(np.int32), "position",
        )
        ds, binfo = ds.allocate_for_seq(slot_ids, pos_ids)
        logits_step, cache = model.decode(tok, cache, binfo, pos_ids)
        # logits_step axes: (position=B, vocab)
        for b in range(B):
            got = logits_step.array[b]
            want = logits_full_arr[b, prefix_len + s]
            diff = np.max(np.abs(got - want))
            print(f"  step={s} b={b} max-abs-diff: {diff:.2e}")
            assert diff < 5e-3, f"step {s} b={b} mismatch: {diff}"

    print("PASS: paged-KV decode matches recompute within tolerance")


if __name__ == "__main__":
    main()
