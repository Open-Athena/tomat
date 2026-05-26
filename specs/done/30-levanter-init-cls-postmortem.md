# Levanter `init` hard-coded class bug — post-mortem

**Date detected:** 2026-05-25
**Date introduced (in tomat):** 2026-04-XX (when `Qwen3DensityLMHeadModel` was added in commit 2214cdb "LMQ infrastructure")
**Scope of impact:** every "AR + EMD density-aware loss" run and every "MaskGIT" run between 2026-04-XX and 2026-05-25 — silently trained as plain causal AR with standard CE on the next-token objective.

## Summary

`levanter.models.qwen.Qwen3LMHeadModel.init` (and the same pattern in `LlamaLMHeadModel`, `GemmaLMHeadModel`, `MistralLMHeadModel`, `ApertusLMHeadModel`, `MixtralLMHeadModel`, `QwenLMHeadModel`) ends with:

```python
@classmethod
def init(cls, Vocab, config, *, key):
    ...
    return Qwen3LMHeadModel(transformer, embeddings, lm_head)  # hard-coded, NOT cls(...)
```

`Subclass.init(...)` therefore returns a base-class instance, discarding the subclass's method overrides. Our `Qwen3DensityLMHeadModel` and `Qwen3MaskGITLMHeadModel` overrode `compute_next_token_loss` to call `density_aware_loss` and `maskgit_aware_loss` respectively — but those overrides were unreachable from any model built via `config.build(...)`.

## Detection

While debugging an apparent mode-collapse in `train-mg-1`, three follow-up "mg-2" runs trained to bit-identical TL trajectories across three supposedly-different mask priors. Smoke isolation suggested either multi-host JAX or my recent code edits. The actual signal:

- `TL[step 1] = 9.828 ≈ ln(18570)` across cont33k (AR-EMD, allegedly), mg-1 (MaskGIT, allegedly), mg-2 variants (all variants), and mg-fix-verify (fixed).
- That value is the entropy of standard CE over the bumped vocab. EMD-at-init magnitude would be in the thousands. They were all CE.

Confirmed by reading `levanter/models/qwen.py:393`.

## Fix

Override `init` in each affected subclass to re-wrap the parent's tuple as the subclass instance:

```python
class Qwen3DensityLMHeadModel(Qwen3LMHeadModel):
    @classmethod
    def init(cls, Vocab, config, *, key):
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)
```

Same pattern in `Qwen3MaskGITLMHeadModel`. Verified on TPU: `train-mg-fix-verify` shows trace print firing with `prior='cosine' loss_type='emd' mask_id=18570 key_is_none=False` and step-0 TL = 38,861 (EMD-magnitude).

## Retracted conclusions

Every result attributed to "density-EMD" training in the AR series, and every conclusion drawn from "MaskGIT" training, is wrong about the loss/objective. Specifically:

- **cont33k 0.91% TF mat-NMAE / 386% FR mat-NMAE** — real numbers, but from plain CE training (not the EMD recipe). Stand as a CE baseline.
- **`mg-1` "mode collapse" investigation** (memory `mg-1-mode-collapse.md`) — the model under examination was a CE-trained AR LM with vocab+1, not a MaskGIT model. The constant-prediction artifact at all-MASK input was an untrained MASK_ID embedding's projection, not a property of MG training.
- **LR sweeps, cooldown ablations, EMD vs L1 ablations** in the AR series — all trained as CE. Comparisons against each other still hold (same loss); comparisons claimed about EMD specifically are noise.
- **All 3 "mg-2" sweep variants** (cos/uni/hi) — bit-identical CE runs.

## What this means going forward

Two things become open questions for the first time in the project:

1. **Does pure EMD beat pure CE in AR training?** `train-ar-emd-real` (fired 2026-05-25 on v5p-16 us-east5) is the first proper test. Compare TF + FR NMAE against cont33k's CE baseline.
2. **Does real MaskGIT work for charge-density prediction?** `train-mg-3-cos-emd` (fired 2026-05-25 on v5p-32 us-central1) is the first proper test.

A hopeful corollary (per Ryan's intuition): EMD might *structurally* help FR robustness — CE pushes all mass to the modal bin, while EMD spreads it across ρ-near bins, which should reduce error cascading under FR rollout.

## Process lessons

1. **Read library source when subclassing.** Look at the parent's `init`/`build`/`from_config` to confirm `cls(...)` vs `ClassName(...)` before relying on override behavior. Two minutes of reading would have caught this.
2. **Verify loss-init magnitude matches loss math.** Pure EMD over 16k log-spaced bins should give init loss in the thousands; CE should give `ln(V)`. This is a 30-second check that would have flagged the misattribution from day one. Added to memory ([[feedback_smoke_before_sweep]]).
3. **Always smoke before sweeping.** A 5-min single-host run before firing 4×3 = 12 v5p-32-hours of parameter sweep would have caught the bit-identical-loss problem in 5 minutes.
4. **Quickest-explanation-first when multiple runs give identical results.** Default hypothesis should be "the variable I think is varying isn't varying," not novel hardware claims. Took me 2 hours to converge on this.

## Open follow-ups

- File a Levanter upstream issue describing the systemic bug (every `*LMHeadModel.init` hard-codes its return class). Anyone subclassing trips this. **Not done — TODO.**
- Audit other Levanter patterns we rely on (data loader, optimizer build, checkpoint serialization). Initial audit (this session) found no other affected paths in our code, but the pattern of "library uses `cls` in some places and hard-codes in others" suggests more landmines may exist.
- All previously-published claims about EMD-vs-CE comparisons should be marked as untested.
