# tomat — state of experiments (2026-04-30)

Snapshot for Yael / Betsy. Living doc — update as numbers
land. See `specs/21-eval-noise-and-extended-runs.md` for the deeper write-up.

## TL;DR

- Best mat-NMAE so far: **2.95% val mean / 2.39% median** (200M, step
  14000 of the 16k run). Improvement from 8k's best of 3.65/3.11.
- **Reference points** (corrected from prior version of this doc):
  ChargE3Net SOTA 0.53%; **electrAI ResUNet 0.86% NMAE** on GGA+U at
  epoch 86 (12.1 M params, 26.7k samples — Hananeh, Meeting 26 / Apr 22);
  codec floor 0.18% (P=14, V=16k). The 1.02% number we'd been citing was
  stale — it was Hananeh's prior result on the same dataset before her
  longer training run finished.
- **Honest framing**: tomat 200M @ 2.95% vs electrAI 12.1 M @ 0.86%
  means we're 17× more params and ~3.4× higher NMAE — net per-param
  efficiency, electrAI is far ahead. Total data-exposure is comparable
  (~2M sample-iters either way), so it's not a "more training" gap.
  The bet — that transformer scaling closes this gap — is still
  unvalidated; the 1B-cont and beyond is what tests it.
- Two apples-to-apples caveats: our 77k val set is mixed-functional MP,
  theirs is GGA+U-only; we should slice our NMAE by functional before
  the next comparison.
- Switched from 200M → 1B and confirmed the bigger-model thesis: MFU
  jumps **8.7% → 13.0%** on v6e-32 (profile shows attention's share of
  step time falls 47% → 17%).
- 16k NMAE curve shows the **same cooldown regression** as the 8k run
  (val mean 2.95 → 5.79 over the last 1.6k steps, while EMD `eval/loss`
  keeps improving). Loss/decoder mismatch is now repeatable across runs.
- 1B cont-from-4711 in flight, MFU 13.17% (matches original 1B's MFU);
  1B partial-ckpt NMAE harvest still queued.

## What tomat is (one paragraph)

Voxel-density emulator: tokenize Materials Project DFT charge densities
into Qwen3-style sequences (P=14 patches, LMQ-v2 16k-bin codec for
1-token density), train a small LM on next-token + density loss, decode
back to predicted charge densities, evaluate per-mat NMAE against the
DFT ground truth. Goal: a foundation model that emulates DFT for new
materials at orders-of-magnitude lower compute.

## Run inventory

| run | model | hw | steps | status | best NMAE | wandb |
|---|---|---|---:|---|---|---|
| `train-full-lmq-v2-200M-bs128-emd-do-8k-tpu16` | 200M | v6e-16 | 8k | ✅ done | **3.65%** val mean (step 7000) | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-full-lmq-v2-200M-bs128-emd-do-8k-tpu16) |
| `train-full-lmq-v2-200M-bs128-emd-do-16k-tpu16` | 200M | v6e-16 | 16k | ✅ done | **2.95%** val mean (step 14000) | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-full-lmq-v2-200M-bs128-emd-do-16k-tpu16) |
| `train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32` | 1B | v6e-32 | 16k → 4711 | ❌ crashed (GH 502 on rebuild) | _harvest queued_ | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32) |
| `tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32` | 1B | v6e-32 | 11.3k | 🟡 in flight (step 535, MFU 13.17%) | — | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32) |
| `train-cont-from-7999-200M-bs128-emd-do-8k-tpu16` | 200M | v6e-16 | 8k | ❌ broken (LR=0; warm-start bug) | — | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-cont-from-7999-200M-bs128-emd-do-8k-tpu16) |
| `train-full-lmq-v2-200M-bs128-emd-density-only` | 200M | v6e-8 | 4k | ✅ done (early dev) | ~6% val mean | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-full-lmq-v2-200M-bs128-emd-density-only) |
| `train-full-lmq-v2-200M-bs128-l1-0.1add` | 200M | v6e-8 | 4k | ✅ done (CE+λ·L₁ ablation) | — | [link](https://wandb.ai/PrinceOA/tomat-lmq-P14/runs/train-full-lmq-v2-200M-bs128-l1-0.1add) |

Project view: <https://wandb.ai/PrinceOA/tomat-lmq-P14>

## Headline NMAE curve — 200M 8k EMD-density-only run

`val_200` and `train_200` are the canonical eval mat-sets (n=200 each,
seed-stable). NMAE is per-mat ‖ρ̂−ρ‖₁ / ‖ρ‖₁ × 100, aggregated as
{mean, median, p99} across mats.

| step | val mean | val median | val p99 | train mean | train median | train p99 | eval/loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 272  | 35.78 | 33.03 | 122.35 | 37.56 | 33.36 | 121.90 | — |
| 1000 | 60.25 | 59.33 | 156.11 | 60.96 | 60.46 | 145.58 | 1.66 |
| 2000 | 30.48 | 30.41 |  71.94 | 32.04 | 31.11 |  87.22 | — |
| 3000 | 13.13 | 11.91 |  46.51 | 13.92 | 12.70 |  52.57 | — |
| 4000 |  6.74 |  5.89 |  33.67 |  7.30 |  6.01 |  32.62 | — |
| 5000 |  6.47 |  5.52 |  31.74 |  7.25 |  5.71 |  29.10 | — |
| 6000 |  5.29 |  4.72 |  26.00 |  5.80 |  4.87 |  29.48 | — |
| 7000 |  **3.65** |  **3.11** |  15.02 |  4.24 |  3.25 |  33.13 | 1.13 |
| 7999 |  4.04 |  3.91 |  12.94 |  4.56 |  4.13 |  12.73 | 0.93 |

Note the **step-7000 → step-7999 regression** in NMAE means/medians while
EMD `eval/loss` keeps improving (1.13 → 0.93). Hypothesis: **loss-vs-decoder
mismatch sharpening** — EMD pulls the predicted distribution closer in W₁,
but the median (the L₁-optimal point estimator we read out) doesn't track.
The 16k from-scratch run will tell us whether this oscillates around ~4%
(mismatch confirmed) or saturates lower (underfit).

Disjoint sample (val_400b, n=400, fully disjoint from val_200): step-7999
val_400b mean=7.07% / median=5.34% / p99=64.42%. The wider noise band
suggests **n=200 is undersampled at the right tail** — large p99 events
matter and a single material can swing the mean by ~3 pp. NMAE estimator
variance is non-trivial; treat single-step movements ≤2 pp as noise.

## Headline NMAE curve — 200M 16k EMD-density-only run

Same recipe as the 8k run, but doubled in steps. New best landed at
step 14000 (mid-cooldown).

| step | val mean | val median | val p99 | train mean | train median |
|---:|---:|---:|---:|---:|---:|
|  1000 | — | — | — | 80.15 | 79.24 |
|  2000 | 52.37 | 52.40 | 133.50 | 52.76 | 51.05 |
|  3000 | 27.58 | 26.80 |  60.52 | 27.59 | 26.99 |
|  4000 | 20.64 | 20.63 |  52.23 | 20.99 | 20.91 |
|  5000 | 14.25 | 14.22 |  41.00 | 14.52 | 14.54 |
|  6000 |  8.72 |  8.08 |  29.51 |  8.99 |  8.97 |
|  7000 |  9.09 |  8.90 |  26.73 |  9.43 |  9.91 |
|  8000 |  6.20 |  5.29 |  25.84 |  6.63 |  5.74 |
|  9000 | 18.73 | 15.10 |  56.66 | 16.49 | 13.26 |
| 10000 |  6.92 |  6.88 |  20.16 |  6.87 |  7.05 |
| 11000 |  4.24 |  3.71 |  18.83 |  4.29 |  3.89 |
| 12000 |  4.39 |  4.03 |  15.99 |  4.49 |  4.19 |
| 13000 |  4.19 |  3.34 |  13.69 |  4.51 |  3.53 |
| **14000** | **2.95** | **2.39** |  **9.80** | **3.07** | **2.39** |
| 15000 |  3.59 |  2.68 |  13.10 |  3.93 |  2.64 |
| 15999 |  5.79 |  4.41 |  21.43 | — | — |

Two patterns:
1. **Cooldown-phase regression repeats** — last ~1.6k steps (the
   constant→cooldown LR transition lands around step 14400) take val
   mean from 2.95 → 5.79. Same direction as the 8k run's step-7000 →
   step-7999 bump (3.65 → 4.04). This is now repeatable signal for
   the EMD-loss-vs-median-decoder mismatch hypothesis.
2. **Step-9000 spike** (val mean 18.73 in an otherwise descending
   curve) is a single-point outlier — likely either a corrupt/preempt-
   mid-save ckpt or a sample-variance event. Re-evaluating against
   val_400b would disambiguate.

## Technical findings

### MFU: bigger model = better arithmetic intensity

200M / v6e-16 / BS=128: **MFU 8.7%**, step 2.08 s
1B / v6e-32 / BS=256: **MFU 13.0%** (1.49×), step 5.76 s

Profiler trace (XLA chrome trace from `jax-profile-step-20-45` artifact):

| category | 200M | 1B | Δ |
|---|---:|---:|---:|
| attention (splash MHA) | 47% | 17% | −30 pp |
| custom-call (matmul) | 21% | 40% | +19 pp |
| fusion | 28% | 30% | +2 pp |
| bias (bitcast_add) | 8% | 9.5% | +1.5 pp |
| comm (collective-permute / all-reduce) | 5.5% | 2.1% | −3.4 pp |

Reading: 200M was attention-dominated and HBM-bandwidth bound. 1B's
matmul shapes have higher arithmetic intensity, attention's relative
share collapsed, and **comm shrank despite host count doubling** —
we're moving toward compute-bound. Cheap headroom: per-chip BS↑
(BS=512), grad_ckpt off (if it fits). Parked as a post-cont
micro-bench (`tmp/launch-microbench-1B-mfu.sh`).

### Warm-start gotcha (LR=0 trap)

Levanter's `TrainLmConfig.initialize_from_checkpoint_path` does a
naive full-state load → optimizer step counter carries over → LR
schedule reads from optimizer's internal step → if source ckpt is past
cooldown, LR=0 and weights don't move. Fix: monkey-patch
`train_lm.load_checkpoint` to load only `subpath="model"` and stitch
back, leaving optimizer state fresh. Lives in `train_tomat_tpu.py`,
gated by `TOMAT_INIT_FROM_CHECKPOINT`.

### Loss / decoder mismatch

We train with EMD (Earth-Mover / W₁) on the predicted density
distribution but **decode at eval time with the L₁-optimal median**.
EMD doesn't constrain the median, only the mass distribution → the
median can drift even as `eval/loss` (EMD) improves. Open question:
either swap eval decoder for the EMD-optimal point estimator, or
add a median-regularization term to the loss.

## In flight

1. **`tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32`** (v6e-32,
   11.3k steps, ~17 hr ETA) — step 535 / 11289 (~5%), MFU 13.17% (matches
   the original 1B's MFU). Only 3 preempts in the first hour vs 14/min on
   the original — pool churn was unusually high yesterday. Healthy.
2. **200M 16k mat-NMAE harvest** — ✅ done (16/16 ckpts, both mat-sets).
3. **1B partial mat-NMAE harvest** — 0/6 harvested yet, queued behind the
   200M work. Wall: ~30-60 min more.
4. **Lattice-aware retokenize** (2026-04-30, Yael+Ryan call) — ✅ done.
   `train-full-lmq-v2-lat` / `val-full-lmq-v2-lat` on GCS. New 6-INT
   block holding `(a, b, c, α, β, γ)` between `[GRID_END]` and
   `[ATOMS_START]`, gated by 2 new specials (`[LATTICE_START]=18`,
   `[LATTICE_END]=19`). Vocab grows 18568 → 18570; N_SPECIALS 18 → 20.
   Per-axis voxel scale is now derivable as `length / grid_dim`.
   Stats: train 2,496,992 rows (~78k mats; 39 lat-overflow + 32 oversized
   skipped); val 138,752 rows (~4.3k mats; 1 + 2 skipped). Pad-overflow=0
   on both — the +8-token preamble cost fits cleanly in 8k seq_len.
   Spot-check on real train data: mp-2282417 → (4.20, 7.10, 7.10 Å,
   120°/90°/90°), mp-1803910 → triclinic cell — physically valid.
   `eval_mat_nmae.py` updated to auto-detect lat-awareness from
   `meta.json` and propagate lattice through the eval-time preamble.

## Tooling

- `tomat` CLI (`./tomat`, project root) — wraps wandb + iris + GCS into a
  single tool: `tomat runs ls/status/nmae`, `tomat iris ls/kill/logs/summary`,
  `tomat evals fire [--dry-run]/harvest`. Built today; replaces the
  inline-bash + ad-hoc python workflow.
- `OVERVIEW.md` (this doc) — kept current as numbers land.
- `specs/21-eval-noise-and-extended-runs.md` — deeper write-up incl. the
  full profiler breakdown.

## Open questions

- **Is the 8k step-7000 → 7999 NMAE bump real signal or n=200 noise?**
  16k re-eval will resolve.
- **Can we hit ChargE3Net's 0.53% with this approach?** Or is there a
  hard ceiling from the EMD-vs-median-decoder gap? Will be informed by
  the 16k saturation behavior + 1B-cont's NMAE curve.
- **Loss-vs-decoder mismatch fix:** new decoder vs new loss term?
- **Real MFU ceiling on v6e-32 for 1B?** 13% → 16-21% is plausible
  with BS=512 + grad_ckpt off. Quantified by post-cont micro-bench.

## Compute usage (MSRP-equivalent, sense-of-scale)

Public TPU rates: v6e preempt ~$1.15/chip-hr midpoint, on-demand $2.70-
2.97/chip-hr.

| run | TPU | wall | chip-hr | preempt-MSRP |
|---|---|---:|---:|---:|
| 200M 8k (orig) | v6e-16 | 7 hr | 112 | ~$130 |
| 200M 16k (done) | v6e-16 | 11 hr | 176 | ~$200 |
| 1B 16k (crashed at ~30%) | v6e-32 | 8.3 hr | 266 | ~$305 |
| 1B cont-from-4711 (in flight) | v6e-32 | ~17 hr est | ~545 | ~$625 |
| Mat-NMAE harvests (44 jobs × ~10 min) | v6e-8 | ~10 min/job | 60 | ~$70 |

Running spec-21 total so far: **~$1.3k preempt-MSRP** of work, with
the 1B-cont being the biggest single chunk.
