# 31 — Free-running-aware AR training: implementation plan

Follow-up to `specs/26-free-running-paths-forward.md`. MaskGIT (spec 26's
top recommendation) is already wired (`Qwen3MaskGITLMHeadModel` in
`marin/qwen3_density.py`). This spec scopes the **lightest AR-side
intervention** so we can decouple "can curriculum-on-AR move FR?" from
"does abandoning AR ordering (MaskGIT) close it?"

## Chosen method: token-level scheduled sampling (Bengio 2015)

Why first, not the other spec-26 options:

- **Pure training-loop change.** Same model, data, eval; only the forward
  pass during training mutates a fraction of density-token *inputs* with
  the model's own one-step-ahead prediction. No new architecture, no new
  collator, no vocab bump (contrast MaskGIT).
- **Reuses infra already in tree.** `Qwen3DensityLMHeadModel` already
  calls `self.activations(example.tokens, ...)` explicitly — that's the
  exact hook point. Density vocab range + `decode_all` are already
  plumbed through `DensityLossArgs`.
- **Cheap fine-tune.** Resume cont33k, 2–4k steps. Hours of TPU.
- **Calibrates even if it underperforms.** Per spec 26: "if SS pulls
  386% to <10% → AR-with-curriculum is real; barely moves → AR template
  is the binding constraint." We don't have that signal yet.
- **Caveat (spec 26 §"Updated recommendation"):** SS addresses the (b)
  OOD-AR-context failure, NOT the (a) bad-seed-at-voxel-0 failure
  (per-position TF logging: voxel-0 NMAE 196% at step-79999, worse than
  constant-mean baseline). Expect (b) to drop, (a) untouched. Still
  informative — it lower-bounds the AR ceiling.

## Implementation

Two-step forward inside
`Qwen3DensityLMHeadModel.compute_next_token_loss`:

1. **Pre-forward** wrapped in `jax.lax.stop_gradient`: forward
   `example.tokens` → logits → at each density position
   sample/argmax/median the next-token bin → `pred_tokens` buffer.
2. **Build mixed input.** Per density position, with probability ε
   replace `example.tokens[t+1]` with `pred_tokens[t]`. Non-density
   positions untouched (atom IDs etc. aren't under exposure-bias
   pressure). Use existing `density_lo / density_hi` from
   `DensityLossArgs`.
3. **Real forward** on the mixed input → existing `density_aware_loss`
   against the **original** ground-truth targets (model sees its own
   predictions as context, still graded against truth).

Bengio's "always-sample" variant restricted to density tokens,
ε scheduled.

### Files to touch

- `marin/qwen3_density.py`
  - Add `SSArgs` frozen dataclass + `configure_ss(args)` mirror of
    `configure_density_loss`.
  - Modify `Qwen3DensityLMHeadModel.compute_next_token_loss`: when
    `_SS_ARGS is not None` and `key is not None` (training), run
    pre-forward + mix + main forward. When `key is None` (eval), skip
    SS — eval is teacher-forced by construction.
  - Add `_jax_apply_ss(...)` helper analogous to `_jax_apply_maskgit`.
    Sampler modes: `"median"` (default — matches FR eval),
    `"argmax"`, `"sample"`.
- `marin/train_tomat_tpu.py`
  - Add env vars next to `TOMAT_MG_*`:
    - `TOMAT_SS_MODE` `0`/`1`
    - `TOMAT_SS_EPS_MAX` float, default 0.25
    - `TOMAT_SS_SAMPLER` `median`/`argmax`/`sample`
  - Wire only when `TOMAT_SS_MODE=1` AND `TOMAT_LMQ_PATH` is set.
    Disallow simultaneous `TOMAT_MG_MODE=1` (raise).
  - **ε schedule**: simplest is per-batch `ε ~ Uniform(0, ε_max)` inside
    the JIT, mirroring how MaskGIT samples its mask ratio per-example.
    No Python-side step counter, no re-config, no re-trace risk.

### No Levanter fork modification required

Hook is already `Qwen3DensityLMHeadModel.compute_next_token_loss`, which
we own. The `init` re-wrap from
`specs/done/30-levanter-init-cls-postmortem.md` already restores
dispatch to our subclass.

### Eval

No change. `eval_mat_nmae.py` `eval_mode={teacher,free}` already
exists; SS only affects training. Headline post-SS:
- TF NMAE: expected flat or mildly degraded.
- FR NMAE: the actual signal.

## Testing before TPU

1. **Unit test** (`tests/test_ss.py`, ~30 lines): synthetic density
   vocab range, fake logits, call `_jax_apply_ss` with ε=0 (assert
   `mixed == original`), ε=1 (all density positions replaced), ε=0.5
   (~50% replacement over a large batch). Verify non-density positions
   untouched.
2. **Local 50-step smoke** on `data/tokenized/val-smoke/`, tiny model,
   `TOMAT_SS_MODE=1`, ε_max=0.5. Confirm:
   - `[ss]` trace-time print fires (mirror `[maskgit]` pattern at
     `qwen3_density.py:502`).
   - Loss bounded.
   - No silent fallback — mirror the `RuntimeError` guard from
     `qwen3_density.py:481` (mg-2 lesson, memory
     [[feedback_smoke_before_sweep]]).
3. **Single-host ≤500-step TPU smoke** (v6e-4 if avail else v6e-16,
   5–10 min). Correctness + quick TF-NMAE eval at step-500 to confirm
   we haven't immediately wrecked the model.
4. **Only then**: fire cont33k SS fine-tune (~2k steps, hours).

## Open questions / judgment calls

- **ε schedule shape.** Per-batch Uniform(0, ε_max) vs Bengio's
  inverse-sigmoid ramp. Uniform is simpler and averages over the
  triangular distribution — likely fine; a2a if it underperforms.
- **ε_max.** Bengio's NLP defaults (~0.05) were for ~30-token
  sentences. Our P³=6859 rollout compounds error far more, so start
  larger: 0.25, sweep {0.1, 0.25, 0.5} if interesting.
- **Sampler.** `"median"` matches our FR eval decoder (a2a);
  `"sample"` is more faithful to the spec (true categorical from
  softmax). Try median first.
- **Will it close the gap?** Per memory [[free-running-divergence]] +
  spec 26: probably not fully. Voxel-0 (the (a) failure) is invariant
  to SS. Expect 386% → tens-of-percent at best, not <10%. If even that
  doesn't happen → MaskGIT becomes the only AR-side bet standing.
- **Pre-forward cost.** Doubles per-step FLOPs (extra forward).
  Budget: 2k SS steps ≈ 4k normal-cost steps' worth of TPU.

## Next concrete step

Read `Qwen3DensityLMHeadModel.compute_next_token_loss`
(qwen3_density.py lines 237–280) and `_jax_apply_maskgit` (lines
550–613) side-by-side. Draft `_jax_apply_ss` + `SSArgs` +
`configure_ss` modeled exactly on the MaskGIT structure. Land the
unit test in the same commit. Don't fire any TPU runs until both the
unit test and the local 50-step smoke pass.
