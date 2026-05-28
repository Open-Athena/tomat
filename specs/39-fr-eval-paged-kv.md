# Spec 39: FR eval via Levanter paged-KV decode (v2 of spec 25 §1)

## Motivation

`specs/25-eval-dashboard-followups.md §1` shipped v1 of the free-running
eval as a recompute loop — "obviously correct, ~80 LOC, but ~1-2
min/patch". Spec 25 explicitly flagged v2: **"paged-KV-cache decode
loop, ~3000× faster, for the full 200-mat free-running eval"**, but
opted to defer because of cache-corruption risk at the time.

Two things changed since spec 25 (2026-05-22):
1. Levanter's paged-KV API has hardened: recent fixes through Nov 2025
   (commits `843ce72` switched to `List[KvPageCache]` for correct XLA
   memory reuse / ~5-6× speedup, `43751a7` added explicit decode-state
   reset, `a52830c` ("a near infinite number of bugs") closed the
   slot-allocation race). Inference engine + tests are in tree.
2. We hit the v1 ceiling for real: ~80 min/mat × n_mats=200 ≈ 270 h
   per (run, set, mode) cell. Even with spec-38 fan-out at N=20
   tasks, that's ~13 h per cell. Paged-KV would compress this to
   sub-minute.

## What Levanter exposes

(from a close read of `~/c/oa/levanter/src/levanter`)

- **API** — `qwen.py:338-383`:
  ```python
  logits, new_kv_cache = model.decode(
      input_ids,        # {Batch, Pos} or {Pos}
      kv_cache,         # ListCache[KvPageCache], from model.initial_cache()
      batch_info,       # PageBatchInfo (slot/page wiring)
      pos_ids,          # absolute token positions
      key=None,         # optional dropout RNG
  )
  ```
- **`PageBatchInfo`** — `inference/page_table.py:62-94` — describes
  `slot_ids`, `page_indices`, `seq_lens`, `cu_q_lens`, `new_token_dests`.
- **`KvPageCache`** — `layers/kv_cache.py:33-101` — paged K/V layout
  `[Page, Slot, 2·KVHeads, HeadDim]`.
- **`InferenceEngineConfig`** — `inference/engine.py` — auto-infers
  `max_pages` from HBM utilization budget.

The Qwen3 subclasses in `marin/qwen3_density.py`
(`Qwen3DensityLMHeadModel` / `Qwen3SSLMHeadModel`) override only
`compute_next_token_loss`; they inherit `decode` from base — drop-in
compatible, no overrides needed on our side.

## Plan

### Phase 1 — equivalence test (smoke; ~half a day)

Before swapping the eval loop, prove the paged-KV path matches the
recompute path on the same model + prefix. Single-mat, single-patch,
small B:

```python
def _assert_decode_matches_tf(model, prefix_tokens, n_steps):
    # 1. TF path: forward(prefix + dummies), read all density logits in one shot.
    # 2. KV path: feed prefix once, then `decode` step-by-step, read frontier logits.
    # Compare softmax(density_logits) at each step → max-abs-diff bound.
    ...
```

If max abs diff > 1e-3 (or similar bf16 budget), bail; that's the
cache-corruption flag spec 25 worried about.

### Phase 2 — eval loop rewrite (~150 LOC + tests)

Replace the JIT loop in `marin/eval_mat_nmae.py:802-828` + bucketed
forward at line 1010-1030 with:

```python
# Prefill the preamble (B tokens, ~512-1024 each) — one forward.
# Then P^3 decode steps per patch, each forwarding 1 token.
cache = model.initial_cache(InferenceEngineConfig(
    max_seq_len=8192, page_size=128, max_seqs=B, hbm_utilization=0.9,
))
logits_pre, cache = model.decode(preamble_tokens, cache, prefill_info, prefill_pos_ids)
frontier = preamble_len
density_buffer = jnp.zeros((B,), dtype=jnp.int32)
for step in range(P3):
    logits_1, cache = model.decode(
        density_buffer[..., None],          # 1 new token per seq
        cache,
        decode_info(slot_ids, page_indices, seq_lens=frontier+step, ...),
        pos_ids=jnp.full((B,), frontier+step),
    )
    # density logits → median bin → fed back
    ...
```

Drop the FREE_BUCKET ladder entirely (every step is shape `(B, 1)`,
JIT once).

### Phase 3 — wire `tomat evals fire --mode free`

No CLI change needed. Already on `--mode free`.

## Risk mitigation

1. **Equivalence test gates the change** (phase 1 above). If it
   diverges, dig deeper before merging.
2. **Keep the recompute loop behind `TOMAT_EVAL_FREE_LEGACY=1`** for a
   release cycle. Compare a few full-set NMAEs across both paths.
3. **Sharding parity.** Spec-25 footnote 6 (impl note 1) flags mesh
   sharding interactions with the page table. Use `auto_sharded()`
   (`qwen.py:335`) and confirm KV layout matches training's TP sharding.
4. **Causal-only attention.** Levanter's paged decode hardcodes the
   causal mask (see `attention.py:1701`). Our model is also strictly
   causal at inference — fine. Don't try to add non-causal flavors
   (e.g. bidirectional MaskGIT) on this path.

## Expected wins

| metric          | v1 recompute | v2 paged-KV    | factor   |
|-----------------|-------------:|---------------:|---------:|
| per-mat (200M)  | 67-95 min    | seconds-to-1m  | ~50-1000×|
| 200-mat full FR | 270 h        | < 1 h          | ~270×    |
| eval cost       | mat-stalls   | fully fungible | (HBM-bound) |

Spec 25's claim of "~3000×" was the upper bound assuming the recompute
loop spent ~all its time re-forwarding the prefix. Our measurements
show more time in actual generation than in prefix-recompute, so the
realistic speedup is ~50-300×. Still transformative for
"materials/minute" throughput.

## Out of scope

- Bigger batch / TPU pool tuning (separate concern; spec 38
  parallelism covers iris-fanout).
- MaskGIT one-shot mode for AR models — see spec 40 (mask-everything
  experiment).
