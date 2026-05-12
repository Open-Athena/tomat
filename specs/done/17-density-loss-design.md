# Density loss / eval design doc — beyond token-space CE

Status: **design**. Written 2026-04-23.

## ⇒ Current direction (top-of-doc summary)

After the design discussion, we converged on **two parallel paths**:

1. **Formulation X — 2-tok teacher-forced joint L_1** (this doc, L_1
   below): keep the existing `two_token_9_12` codec + corpus; precompute
   a (512, 4096) decode matrix; compute per-voxel `E[ρ] = P_A · D · P_B`
   with teacher-forced conditioning; L_1 loss vs ρ_true; no CE at
   density positions (penalize non-density probability mass via a
   constant ND-penalty term in the same L_1 unit). Bias: teacher-forced
   `P(B|A_true)` is used for every A branch of the joint, which is a
   small approximation error with ≈correct gradient direction.

2. **Formulation Y — LMQ 1-tok codec + clean L_1** (see
   [spec 18](./18-lmq-codec.md)): fit an empirical Lloyd-Max quantizer
   to the full train-full density distribution (16 k bins), retokenize
   under a 1-token codec, apply pure per-token L_1 (no joint needed).
   Also gives 2× more voxels per patch fitting in an 8 k context (up to
   cube P=19), so we get a free compute efficiency win. Costs: ~$10
   retokenize + ~8% embedding tax at 208 M.

Both kill CE at density positions, both directly optimize something
close to NMAE. Y is cleaner math; X gets there without a retokenize.

Dropped from earlier iterations: **L_3 (Gaussian-smoothed CE)** — felt
hacky, CE-in-bin-space is biased; **L_5/L_6 (mixture head / full
regression)** — diminishing returns vs the above. REINFORCE-style
loss is not worth the variance.

The rest of this doc is the original full design-space writeup for
reference. The actionable plan is in the two formulations above.

---

## Core problem

The model currently trains on **per-token cross-entropy** over a flat
sequence that mixes specials, atoms, positions, and density-codec
tokens. Density floats are encoded as 2 tokens each via
`two_token_9_12`:

- Token A (9 bits, vocab 512): log-magnitude bin.
- Token B (12 bits, vocab 4096): fine position within the bin.
- Together: a 21-bit discretization of log-density, range log_min=−4.13
  to log_max=4.97.

Under vanilla CE, the model is penalized **identically** for
predicting token_A=10 vs token_A=255 vs token_A=500 when the
ground-truth is token_A=256 — even though the decoded floats are
10^-4, 1.0, and 10^4 apart. **The loss has no ordinal structure.**

The model eventually learns the ordering (via smoothness in sequence
statistics), but the gradient signal is nowhere near as strong as it
could be. Specifically:

- Early training: the model wastes a lot of capacity learning that
  "adjacent bin" ≈ "similar output" before getting to density
  physics.
- At convergence: errors are no more "clustered near truth" than a
  label-smoothed CE would give.

This document maps out:
1. How to **eval** density reconstruction honestly (comparable to
   electrAI NMAE).
2. How to **train** with loss functions that understand the ordering.

## Part 1 — Eval (no model changes)

### Metrics

**M1** Token CE (current). Useful as smoothness proxy; not the target.

**M2** **Teacher-forced patch NMAE.** One forward pass per val sequence:
  - Extract `argmax(logits[density_positions])` (or `E[decode(token)]`
    from softmax, which is better for multimodal distributions).
  - Decode to floats via `FP16Codec.decode_signed`.
  - NMAE vs ground-truth decoded floats.
  - Patch-level mean, median, p99.

**M3** Autoregressive patch NMAE. Full greedy generation of V_patch×2
  density tokens. More honest (teacher-forcing conditions on actual
  prev tokens; AR uses predicted). ~1000× slower than M2. Run per-
  checkpoint, not per-eval-step.

**M4** **Mat-level NMAE** (per spec 11). Stitch disjoint/overlapping
  patches back into full grid; compute NMAE vs full Zarr. The number
  electrAI / charg3net report.

**M5** χ² = mean((ρ_pred − ρ_true)² / ρ_true_var). Secondary metric,
  matches our prior fidelity-sweep work.

### Infrastructure

- **Standalone eval script**: `scripts/eval_nmae.py` (or
  `marin/eval_nmae.py` if JAX/Levanter-based).
  - Loads a Levanter orbax checkpoint from GCS.
  - Builds the same model config as the training run.
  - Batches val sequences through teacher-forced forward pass.
  - Outputs a CSV of `{mp_id, patch_offset, nmae, chi2}`.
- **Levanter callback** for ongoing training-time tracking:
  - Add `compute_density_nmae(batch, logits, vocab)` hook in the
    Trainer's eval loop.
  - Log `eval/patch_nmae` alongside `eval/loss`.
  - Only adds a few ms per eval (same forward pass, extra argmax +
    decode).

### Expected-value vs argmax

The density codec is 2 tokens per voxel. Argmax picks the single most
likely bin — discards all other probability mass. **Expected-value**
decode:

```
pred_float = Σ_k softmax(logits)_k × codec.decode(k)
```

is often lower NMAE than argmax when the predicted distribution is
sharply unimodal (mass concentrated on the right bins). For multi-
modal distributions it can give weird "average of two modes"
results. **Recommendation: report both**, at least early on.

## Part 2 — Training loss (model change)

### Design axes

**Ordinal awareness.** The loss should know that bin 255 and bin 256
are closer (in reconstruction space) than bins 10 and 500.

**Autoregressive compatibility.** We want to keep greedy decoding
working (for actual density generation) — so the output at density
positions still needs to be a probability distribution the model can
sample from.

**Architectural minimalism.** Start with changes that don't break
the Qwen3-LM-on-flat-token-stream architecture. Only escalate if
simpler changes aren't enough.

### Candidate losses (ordered by increasing intrusiveness)

**L_0 (baseline): vanilla CE.** What we do today. No ordinal signal.

**L_1: CE + expected-value regularization.**

  ```
  L = Σ_t CE(logits_t, gt_token_t) + λ · |E_{softmax(logits_t)}[decode(t)] − decode(gt_token_t)|
  ```

  at density positions, else pure CE. The second term gives
  gradient signal that "moves the predicted distribution toward
  the correct decoded float." Fully differentiable, ~0% compute
  overhead, no architecture change. `λ` is a hyperparameter
  (start with small, e.g. 0.1).

**L_2: Earth mover (Wasserstein-1) distance.**

  On the 1D sorted bin distribution, W1 between predicted softmax
  and one-hot GT is:

  ```
  W1 = Σ_k |CDF_pred(k) − CDF_gt(k)|
  ```

  Simple to compute, respects ordinality naturally, less noisy
  gradients than L_1 expected-value on multimodal outputs.

**L_3: Ordinal label-smoothing / soft CE.**

  Replace one-hot GT with a Gaussian kernel centered on GT bin:

  ```
  gt_soft_k ∝ exp(−((k − gt) / σ)²)
  ```

  Then CE against soft target. Effectively encourages the model
  to predict distributions whose mass is near the truth. Trivial
  to implement, similar in spirit to W1 but cheaper.

**L_4: Regression head (dual-head arch).**

  Replace logits→CE at density positions with a continuous scalar
  head → L1 loss on the float directly. Keep LM head for non-
  density tokens. Best possible loss signal at density positions.

  Requires: Qwen3 subclass, masked loss, new generate() code path
  that reads from regression head at density positions. Biggest
  code change of the L_1–L_4 set.

  *Subtlety:* need to mask (density) tokens in the LM head's
  vocab so the LM head doesn't learn to predict them (otherwise
  the non-density-specific loss fights the density-specific one).

**L_5: Mixture output head.**

  Predict (μ_i, σ_i, π_i) for K mixture components per density
  voxel. Autoregressive sampling draws from the mixture. Handles
  multimodal outputs principled-ly. Bigger arch change, more
  params.

**L_6: Full regression model (non-autoregressive density).**

  Drop tokenization of density entirely; produce the whole
  (P, P, P) voxel tensor in one forward pass from the preamble
  context. Closer to electrAI's convolutional decoder. Very
  different training/eval dynamics. Out of scope for the next
  few weeks.

### Recommendation

**Sequence of experiments** (matched-compute ablations, 208M on
train-full for 4 B tokens each):

1. **L_0 re-baseline** — confirm the current 208M number on the
   Level-1 eval infra (teacher-forced patch NMAE). Gets us a
   comparable-to-electrAI number on day 1.
2. **L_3 (ordinal label-smoothing)** — simplest ordinality-aware
   loss, zero arch change. Expect NMAE drop of 10–30% based on
   prior lit on ordinal regression via CE.
3. **L_2 (W1)** — more principled; compare to L_3 to see if the
   extra principled-ness helps.
4. **L_1 (CE + E[x] L1)** — combines token-level sampling with
   float-level gradients; λ sweep over {0.01, 0.1, 1.0}.
5. **L_4 (dual-head)** — the bigger commitment. Only if L_1/2/3
   suggest this direction is worth the code effort.

Skip L_5/L_6 for now — diminishing returns vs engineering cost.

### Non-density loss terms

At non-density positions (specials, atoms, positions), vanilla CE
is fine — those are genuinely categorical. The ordinality issue is
specific to the density token stream. All the L_k variants above
only change the loss at density positions; non-density positions
keep pure CE.

## Phased plan

### Phase 1 — Eval infrastructure (this week)

- [ ] `scripts/eval_nmae.py`: standalone Levanter-based script.
  Forward-pass on a val corpus, teacher-forced patch NMAE
  (argmax + expected-value variants), writes per-patch CSV.
- [ ] Run against 208M (step 5999) and 1B (step 3999) checkpoints
  → first real NMAE numbers; document.
- [ ] Levanter callback: `eval/patch_nmae` logged every eval step
  for all future training runs.
- [ ] Per-mat NMAE via stitched-patch reconstruction (spec 11) —
  start implementation; defer to next week.

### Phase 2 — Ordinal-aware training (next week)

- [ ] L_3 implementation (Gaussian-smoothed CE on density
  positions). Simplest to add.
- [ ] Matched-compute 208M fine-tune from step-5999 checkpoint
  with new loss. ~4 B extra tokens; check if NMAE drops.
- [ ] L_2 (W1) as second experiment.
- [ ] Report: matrix of (loss variant × eval metric) for the
  first two variants.

### Phase 3 — Arch experiment (1–2 weeks)

- [ ] L_4 dual-head Qwen3 subclass. Careful about mask, generate
  path.
- [ ] Matched-compute 208M fresh train.
- [ ] Compare patch + mat-level NMAE against Phase 2 best.

### Phase 4 — Scale up the winner

Take the best-performing loss/arch from Phase 2–3 and scale to 1B+
with more tokens. By now we have enough tooling to know what we're
doing.

## Open questions

- **Should density NMAE be computed in linear or log space?**
  - electrAI reports linear-space NMAE = mean|ρ_pred − ρ_true|/mean|ρ_true|.
  - Our codec is log-space. An error "one bin off" in log-space is
    a different linear-space error depending on magnitude.
  - Likely: compute both, report linear (matches lit).

- **Is argmax or expected-value decoding the "right" comparison?**
  - Argmax matches autoregressive sampling exactly.
  - Expected-value uses more of the softmax info → probably better
    NMAE but not what the generative model does.
  - Recommendation: report both, use AR-decode NMAE as the headline
    "generation quality" number.

- **Should the loss operate on the magnitude-bin token only, or
  also the within-bin position token?**
  - Magnitude bin (token A) is where the big ordinality matters.
  - Within-bin token (token B) is finer-grained, errors there are
    small in linear space.
  - Could apply ordinal loss only to token A. Simpler.

## Related / adjacent

- `specs/11-per-mat-validation.md` — mat-level reconstruction eval.
- `specs/done/02-fidelity-sweep.md` — prior NMAE/χ² work on non-LM
  tokenizers (fit-only).
- `src/tomat/float_codec.py` — codec decode() path that any new
  loss function will call.
- charg3net / electrAI papers — reference NMAE numbers to compare to.
