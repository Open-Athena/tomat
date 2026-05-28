# Spec 40: independent-given-MASK inference (parallel FR via mask-everything)

## The idea

At FR inference, replace ALL prior density tokens with a `MASK` token
instead of the model's own previous predictions. Each density position
is now predicted from an identical context (preamble + MASK at every
density slot). The predictions become **conditionally independent**
given the preamble, so all positions can be decoded in ONE forward
pass — same parallelism as TF mode.

This is a different question from spec 31 (scheduled sampling), where
the student conditions on its OWN previous outputs. There the
sequential dependency is preserved (still AR). Here we sever the
dependency at inference: each voxel prediction is
`p(density_i | preamble)` — a marginal — rather than the AR
conditional `p(density_i | preamble, density_<i>)`.

## Two regimes

### A. MaskGIT models — drop-in test

`mg-4-cos-ce` was trained for exactly this: predict masked positions
from unmasked context, with the mask ratio drawn from a cosine
schedule. At `r=1.0` (everything masked), the model has to imagine
every density value from the preamble alone — that IS the "mask
everything" experiment for free.

**Action**: run `tomat evals fire --mode maskgit --mg-partial-ratios
1.0 train-mg-4-cos-ce`. Memory `first-real-mg-and-ss-results` says
r=0.1 → 0.09% NMAE. Sweep `0.1,0.5,0.9,1.0` to see the
quality-vs-mask-ratio curve. r=1.0 is the asymptote: lowest
parallelism cost, highest information loss.

### B. AR-trained models — OOD without retraining

AR models like cont33k saw GT density tokens at every position during
training. Feeding them MASK in those slots is out-of-distribution. The
behavior is undefined: the model has no learned representation for
"density value is uncertain at this position; predict from preamble
only".

Options if you want this regime for an AR-trained model:

1. **Mask-augmented fine-tune**: take cont33k, fine-tune with
   randomly-masked density inputs (e.g. each density token replaced
   with MASK with probability `p_mask` ~ U[0, 1] per batch). The
   model learns to handle any mask density at inference. Closely
   related to BERT-style training; the discrete-diffusion line
   (Austin et al. 2021, Lou et al. 2023) calls this an absorbing-state
   diffusion.

2. **MaskGIT-style retraining**: drop the AR conditioning entirely;
   train mg-style directly. mg-4 is the existing example.

Option 1 preserves the AR strength (TF inference still works) while
opening up the parallel-inference option. Option 2 is the cleaner
formulation but loses the AR head.

## Connection to the small-P / patch-size experiments

Two arguments converge:

1. **For FR latency**: smaller P → less in-patch context per decode
   step → faster per step, better batch utilization (more patches in
   parallel). KV-cache (spec 39) handles the per-step cost; smaller P
   handles per-mat compounding.
2. **For "we don't really want all the other voxel predictions in our
   context"**: smaller P → less "wrong predictions tainting context"
   in FR mode. Aligned with the structural argument that nearby
   voxels carry the relevant signal.

If we ALSO go the independent-given-MASK route, the in-patch
context-pollution argument disappears (no AR conditioning), but the
small-P arguments for per-step efficiency persist.

## Suggested smoke (1 day of work)

1. Fire `tomat evals fire --mode maskgit --mg-partial-ratios
   0.1,0.5,0.9,1.0 train-mg-4-cos-ce` against val_200. ~hours per
   ratio in TF-equivalent compute.
2. Plot NMAE vs. mask ratio. This is the empirical "parallelism-
   vs-quality" curve for one trained model.
3. If r=1.0 gives <few-% NMAE, the parallel-inference regime is
   viable for tomat and the production-latency question collapses
   to "smaller / better MaskGIT". If r=1.0 is bad (say >30% NMAE),
   the in-patch context really does matter and we need the spec-39
   paged-KV path.

## Bigger question this opens

If mask-everything works (regime A succeeds at r=1.0): can we drop AR
training entirely for density prediction? MaskGIT's training objective
is "predict masked from unmasked"; at inference we just mask
everything. The AR ordering was never essential to the physics; it
was a convenient model architecture choice. Spec 28 (MaskGIT) was the
right direction; this spec is just the explicit "what's the latency
upper bound when we go all the way".

If mask-everything degrades sharply: the model genuinely needs
context across voxels to produce a reliable density. That's a
structural argument for spec 35 (E(3) structure encoder) or
ChargE3Net-style direct-voxel-prediction architectures — same end
goal (parallel inference) but via a different inductive bias.

## Out of scope

- KV-cache implementation (spec 39).
- Fan-out (spec 38).
- Training a mask-augmented variant of cont33k (cost: ~10 h v6e-16
  fine-tune; worth doing if the mg-4 r=1.0 smoke is promising).
