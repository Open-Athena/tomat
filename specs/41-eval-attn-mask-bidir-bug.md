# Spec 41: eval `attn_mask=None` ≠ causal — production AR eval has been bidirectional

## TL;DR

`marin/eval_mat_nmae.py` passes `attn_mask=None` to `model.activations`
in both the TF (`forward_decode` at line 773) and FR-recompute
(`free_step` at line 806) paths. `None` does **not** default to
causal — Levanter's VANILLA backend (`attention.py:401-402`) skips
masking entirely when `m is None`, and SPLASH/TPU
(`attention.py:1340-1341`) substitutes `FullMask`. Both → full
bidirectional attention.

Training uses causal attention
(`levanter/models/lm_model.py:62`: `attn_mask = AttentionMask.causal()`
inside `LmExample.causal(...)`), so every AR-trained model's TF + FR
NMAE number is a bidirectional-eval-of-causal-trained-model artifact,
not an honest AR measurement.

The MaskGIT path (line 847) is fine — it passes explicit
`AttentionMask(is_causal=False)`, which aligns with mg-* training
(`qwen3_density.py:555` builds the same `bidir_mask`).

## Evidence

### Backend dispatch (Levanter)

`~/c/oa/levanter/src/levanter/layers/attention.py`:

- `materialize_mask(None, ...) → None` (line 1106-1107).
- VANILLA `_simple_attention` (line 401-402):
  ```python
  if m is not None:
      weights = haliax.where(m, weights, -1e9)
  ```
  `m=None` → no masking → softmax over all keys for every query.
- SPLASH/TPU `_tpu_splash_attention` (line 1340-1341):
  ```python
  if mask is None:
      base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
  ```
  `mask=None` → `FullMask` (every position attends to every position).

No upstream substitution: `Qwen3LMHeadModel.activations` (`qwen.py:277`)
forwards `attn_mask` unchanged to `QwenTransformer` (`qwen.py:211`)
which forwards to `layers.fold(... mask=attn_mask ...)` →
`LlamaDecoderLayer.__call__` (`llama.py:323`) → `Attention.__call__`
(`attention.py:1571`) → `dot_product_attention(mask=...)`.

### Training is causal

`levanter/models/lm_model.py:62`:
```python
attn_mask = AttentionMask.causal(sliding_window=sliding_window)
```
inside `LmExample.causal(...)`, the standard supervised LM-example
builder used by the tomat trainer.

### Call sites in `marin/eval_mat_nmae.py`

| line | function           | mask passed                              | mode use     | correct? |
|-----:|--------------------|------------------------------------------|--------------|:--------:|
| 773  | `forward_decode`   | `attn_mask=None`                         | TF eval      | ❌       |
| 806  | `free_step`        | `attn_mask=None`                         | FR recompute | ❌       |
| 847  | `maskgit_forward`  | `attn_mask=AttentionMask(is_causal=False)` | maskgit    | ✅       |

The comment block at lines 830-838 reveals the author's mental model:

> *"Bidirectional attn is achieved by passing an explicit
> AttentionMask(is_causal=False) — materializes to None → full bidir."*

That line states `is_causal=False` is needed for bidirectional — but the
TF + FR sites already pass `None`, which has the same effect they think
only `is_causal=False` achieves.

## Impact on historical numbers

Every cell in the AR/TF and AR/FR columns of every results-summary doc
needs reinterpretation:

- **cont33k TF 0.91%, FR 386%** ([[free-running-divergence]] in memory,
  `specs/done/26-free-running-paths-forward.md` motivation). The TF
  number could be artificially low (lookahead via bidirectional read
  of GT density tokens at future positions) or artificially high (OOD
  for a causal-trained model). The FR-divergence 386% is read over
  bidirectional attention to a PAD-suffix — physical meaning unclear.
- **First-real-MG-and-SS results 2026-05-26** ([[first-real-mg-and-ss-results]]).
  AR-arm TF cells (cont33k, ar-ce-emd) and SS FR cells (93.9%
  closure of FR gap) are all affected.
- **SS sweep cells (91-97% FR NMAE)**. The "exposure bias gap"
  measurement is comparing bidirectional-vs-bidirectional, not
  bidirectional-vs-causal-AR, so the SS bridge effect we read is
  not the bridge effect the methods were designed to measure.
- **Subclass-init-cls postmortem (`specs/done/30-…`)** — the "re-eval
  under actually CE lens" item now also needs "and under causal-eval
  lens".

MaskGIT (mg-3, mg-4 partial-mask sweep) numbers stay valid (eval mode
matches training mode).

## Fix

```python
from levanter.layers.attention import AttentionMask
causal = AttentionMask.causal()

@hax.named_jit(axis_resources=compute_mapping)
def forward_decode(tokens_in):
    act = model.activations(tokens_in, key=None, attn_mask=causal)
    ...

@hax.named_jit(axis_resources=compute_mapping)
def free_step(tokens_in, frontier, true_dens_i):
    act = model.activations(tokens_in, key=None, attn_mask=causal)
    ...
```

No change to the maskgit path. No model code change. No training-side
change. Only the eval-script call sites need fixing.

## Validation plan

Before re-running the full FR/TF baseline matrix:

1. **Re-eval cont33k TF on val_200 with causal attn_mask.** Single
   short fire; ~minutes per the existing TF wall-clock (no FR).
   Compare with the old 0.91% bidir number.
2. **Re-eval cont33k FR on val_200 with causal attn_mask.** Single
   mat for a fast sanity check, then a small N (e.g. n=20) if step-1
   looks reasonable. Compare with old 386% (memory says
   [[free-running-divergence]] was real exposure bias / mode-collapsed
   biased-high).
3. **Re-eval mg-4 TF.** Mostly a control: maskgit path is unchanged,
   so its number should not move.
4. **Re-eval cont33k-SS cells.** Their FR-gap closure was measured
   under bidirectional eval; their true SS effect on causal AR could
   be larger OR smaller.

If the TF re-eval moves <1× (e.g. 0.91% → 0.5-2%), the impact on
prior conclusions is small but the rerun is still needed for honesty.
If it moves drastically (e.g. 0.91% → 10%), every comparison in
write-ups needs re-stating.

## Connection to other in-flight work

- **Spec 39 (paged-KV decode).** Levanter's paged decode hardcodes the
  causal mask inside the kernel (see spec 39 §"Risk mitigation"). Once
  the eval bug is fixed in the recompute path, the recompute path is
  causal-correct and equivalence-checking against paged-KV becomes
  meaningful again. `tests/test_paged_decode_smoke.py` already passes
  with an explicit `AttentionMask.causal()` on the recompute side —
  same fix that's needed in production eval.
- **Spec 40 (mask-everything inference).** Unaffected — that's a
  maskgit-side experiment.

## Out of scope

- Backporting the fix to the v1 paged-KV-cache spec attempt (spec 25
  §1) — that path was abandoned; the next-best replacement is spec 39
  paged-KV (covered separately).
- Re-baselining every memory / docs reference that cites old numbers
  in this single pass; correct as we re-use them.
