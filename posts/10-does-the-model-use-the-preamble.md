# Does the model use the preamble?

A common worry with our setup: the model could converge to predicting
some "average" charge density regardless of what's in the preamble
(atoms, positions, lattice). If it does, then we're not actually
learning *electronic foundation model* behavior — we're learning a
shape-prior over densities and conditioning weakly (or not at all) on
material structure.

So we ran the simplest possible test on the current best ckpt
(`train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42` at step-62000):

| Condition | What's swapped | Expected if model uses it |
|---|---|---|
| **baseline** | true preamble | normal CE |
| **T1: shuffle** | replace preamble with another val mat's preamble (paired) | CE shoots up |
| **T3: random** | replace preamble tokens with uniform integers in vocab range | CE shoots up |

(We considered a T2 with zero/uniform preamble but the decoder is
deterministic in MaskGIT-CE inference, so all examples would collapse
to the same density — uninformative.)

## Result (n=8 patches, smoke run)

```
baseline:  mean CE = 6.81  ± 0.23  (se 0.08)
T1 shuffle: mean CE = 16.46 ± 2.24  (se 0.79)    — +9.65 nats
T3 random:  mean CE = 12.58 ± 0.57  (se 0.20)    — +5.77 nats
```

**The model is reading the preamble.** Both perturbations significantly
hurt the loss. Importantly:

- **T1 (real-but-wrong) hurts MORE than T3 (random noise).** A
  plausible-looking preamble from a different material *actively
  misleads* the model — it commits to a structure that's incompatible
  with the density it's being asked to predict. Random noise is more
  obviously OOD; the model might just fall back to a structureless prior.
- The variance also tells a story: T1 has σ=2.2 (big swings — for some
  pairs the swap-mat is similar enough that the prediction isn't
  catastrophic, for others it's wildly wrong), while T3 is tighter
  (σ=0.6 — random noise is uniformly bad).

## Method notes

- Forward-only patch eval, not material-level NMAE. Cheaper, and the
  patch is the natural unit of training.
- 8 val patches; smoke test, full fire at n=256 should give proper CIs.
- Density positions only contribute to the loss (we trained with
  `density_only=True`).
- Cost: ~$0.50 H100×8 wallclock for the smoke; full n=256 ≈ $3-5.

## Caveats

- We measured loss on the density region but with the preamble swapped.
  We did NOT remask the density region between conditions — same
  ground-truth tokens for all three modes. So this directly answers
  "does swapping the preamble change the model's prediction of the
  density?" — and yes, dramatically.
- We didn't test "preamble with right atoms, wrong positions" or
  "preamble with right positions, wrong atoms" (T4 in the original
  plan). Useful follow-ups; the +9.65 nat baseline gives us headroom
  to localize *what* about the preamble the model is using.

## Data

- Run output: `gs://marin-eu-west4/tomat/eval/preamble-test/train-mg-modal-h200x8-tz-v4-epochwin-step-62000.json`
- Harness: `scripts/preamble_vl_modal.py`
- Commit: `696b200`
