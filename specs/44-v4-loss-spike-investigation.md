# Spec 44: `train-mg-modal-h200x8-tz-v4-bs128-seed42` loss-spike investigation

## TL;DR

User noticed visually-recurring spikes in the TL trace of the v4 Modal
MaskGIT run. **Investigation finds no statistically significant
periodicity** — the spikes are single-step impulses (~0.20-0.28 above
local baseline), consistent with per-batch loss variance from a
data-loader reading consecutive same-material patches without
in-pipeline shuffle. Confidence: **high** for "no periodicity";
**medium-high** for "data-heterogeneity / no-shuffle" being the
mechanism. Recommendation: **ignore** (visual artifact of dense
sampling); separately, file a one-liner to enable `BlockShuffleConfig`
in the Modal bakeoff path to match the TPU trainer.

## Data + methodology

- Source: `https://tomat-runs-api.openathena.workers.dev/api/runs/train-mg-modal-h200x8-tz-v4-bs128-seed42/raw.parquet`
  (1.06 MB, 19,827 rows). All rows have `train/loss` populated, on a
  contiguous `global_step` grid (1 step per row, single skip from 9999
  → 10000 timestamp gap of 13,669 s ≈ 3.8 h = the v4 → v4-cont
  Modal-container restart).
- No eval rows (`eval/loss`, `eval/mat_nmae/*` all NaN — the bakeoff
  Modal path sets `steps_per_eval=max(steps, 1)` to disable mid-train
  eval, so spikes can't be eval-cadence artifacts).
- No lifecycle / preemption / failure rows logged.
- Detrending: rolling-median baseline (window=51 or 101 steps, centered)
  → residual `r = loss − base`. Loss decays monotonically 9.7 → 8.0
  over the run, so global thresholds would be biased — all checks use
  the per-step local residual.
- Spike detection: MAD-based robust z-score; clustered contiguous
  spikes into single events for inter-spike gap statistics.
- Periodicity tests: χ² of `step % P` for
  `P ∈ {2, 4, 8, 16, 32, 64, 100, 128, 200, 250, 256, 500, 1000}`;
  FFT of residual; autocorrelation lags 1–2000.

## Findings

### 1. Periodicity

**Not periodic in step-space.** χ² of `step % P` for any candidate
period gives p ≥ 0.099 (uniform-distribution null cannot be rejected at
p < 0.05) on the n=1277 weak-spike set (`resid > 0.10`):

| P    | exp/bin | χ²    | p-value | top-3 modulo bins      |
|------|---------|-------|---------|------------------------|
|   32 |    39.9 |  26.1 |   0.72  | (25,54) (9,48) (31,48) |
|  128 |    10.0 | 113.2 |   0.81  | (89,18) (6,17) (9,16)  |
|  256 |     5.0 |       |   ~     | similarly diffuse      |
| 1000 |     1.3 |       |   ~     | similarly diffuse      |

FFT of the residual: strongest peaks at ~93, ~127, and 3–5 step periods
— all consistent with white noise, no dominant frequency.

Autocorrelation: |ACF| ≤ 0.027 for all lags in [50, 2000]. (n=19,825
gives a 1σ noise floor ≈ 0.007; 0.027 is ~4σ but spread across many
lags with no single coherent peak.)

The 5 strongest spikes (z>4) sit at steps 269, 3283, 11607, 15868,
16509 — inter-spike gaps 3014, 8324, 4261, 641. **No fixed cadence.**

**Not periodic in wallclock either.** Inter-spike wallclock gaps for
the top-50 spikes range 10s → ~4500s, median 843s. No clustering near
common cadences (no 600s ckpt-save peak, no fixed eval-cycle peak).

### 2. Mechanism — top spikes are single-step impulses

For every top-10 spike (resid 0.23–0.28), neighbors at step±1 are back
to baseline. Examples (3 of the top 10):

```
peak step=17615 (resid=+0.280):
  step 17613  loss=8.240  resid=+0.045
  step 17614  loss=8.072  resid=−0.123
  step 17615  loss=8.469  resid=+0.280  <<<
  step 17616  loss=8.238  resid=+0.049
  step 17617  loss=8.211  resid=+0.022

peak step=5180 (resid=+0.266):
  step  5179  loss=8.975  resid=−0.013
  step  5180  loss=9.255  resid=+0.266  <<<
  step  5181  loss=8.987  resid=−0.002

peak step=11607 (resid=+0.254):
  step 11606  loss=8.774  resid=+0.099
  step 11607  loss=8.926  resid=+0.254  <<<
  step 11608  loss=8.675  resid=+0.003
```

`throughput/duration` at the spike steps is normal (3.05–3.16s).
Only 4 rows in the entire run have `throughput/duration > 5s`
(10037: 8.3s; 10097: 11.6s; 11122: 6.2s; 15365: 5.6s); none of those
coincide with a top-10 loss spike. So the spikes are **not** recompile
events, **not** preemption-recovery hiccups, **not** eval pauses.

### 3. Magnitude vs baseline noise

- Baseline-residual distribution (after 51-step centered-median detrend):
  `p50=0.044`, `p90=0.110`, `p99=0.174`, `max=0.294`.
- Baseline loss median: 8.73.
- Top spike (~0.28) is ~1.5 × p99 — well above noise, but in absolute
  terms ~3% of the loss value (8.47 → 8.73). Visually striking on
  a zoomed plot, statistically modest.
- The "visible spike rate" is high (~10× per 2k-step window with
  `resid > 0.10`) — the user's "regular spacing" perception is the
  cognitive bias of seeing pattern in dense noise.

### Leading hypothesis: no-shuffle data ordering

`scripts/train_smoke_modal.py:544-550` (the path that runs this
bakeoff) builds `LmDataConfig` without a `shuffle=` argument →
Levanter default is `False` → patches read in cache-shard order. With
`patches_per_material = 32` (M=32) and BS=128, each batch is 4
consecutive materials' patches. Batch-to-batch loss variance is then
dominated by per-material density-spectrum heterogeneity, not by
gradient noise on random subsets of the dataset.

This **fits** the observation (high-magnitude, low-autocorrelation,
single-step spikes) but is not provable from the parquet alone —
verifying would require either replaying the data loader (heavy) or
landing the shuffle fix and showing the residual variance drops.

The TPU trainer (`marin/train_tomat_tpu.py:656-666`) already enables
`BlockShuffleConfig(io_block_size=M, window_blocks=1024)` by default
(~32k-row window). The Modal bakeoff path is the outlier.

Alternative hypotheses considered + rejected:
- **JAX recompile spikes**: no; `throughput/duration` at peaks is
  baseline (~3.1s), and the 4 actual recompiles don't coincide.
- **Eval-cadence step**: no; zero eval rows in the parquet.
- **Checkpoint-save step**: would cluster modulo ~600s (10-min cadence)
  in wallclock; wallclock gaps are uniformly distributed.
- **Preemption/restart**: only one such event (step 10000, 13669s
  gap); no spike there, only a ~2% throughput step-up.
- **Sample-pack / mask-token-rate cycle in MaskGIT**: MaskGIT mask
  rate is per-batch IID from the cosine prior; no batch-index cycle.

## Recommendation

1. **Ignore the spikes** for this run; they are sub-3% transient
   fluctuations on individual batches and don't reflect a real
   training pathology.
2. **Separately** (small one-line fix, candidate for spec 45): plumb
   `BlockShuffleConfig` into the Modal bakeoff path in
   `scripts/train_smoke_modal.py` (lines 544 and 611) to match the TPU
   trainer. Expected effect on the loss curve: lower per-step variance,
   slightly smoother trace. Will not change asymptotic loss
   meaningfully, but will close one source-of-truth gap between TPU and
   Modal training paths and make spike-hunting easier in future runs.
3. **Future investigations of "regular" patterns**: rely on χ² /
   modulo concentration on a thresholded spike set rather than
   eyeballing — dense noise plus log-scale axes routinely produce
   pseudo-periodic visual artifacts.

## Artifacts

- Analysis scripts: `tmp/v4_spike_analysis.py`,
  `tmp/v4_spike_periodicity.py`, `tmp/v4_spike_visualize.py`,
  `tmp/v4_extra_checks.py` (gitignored).
- Raw parquet: `tmp/tomat-v4.parquet` (gitignored).

## Update: v4-cont-2 sawtooth (2026-06-02)

After cont-2 fired (steps 20001-34890 as of 2026-06-02 18:12 UTC), the
TL trace showed a pronounced sawtooth concentrated in the last third
of the run. Re-running the analysis on the full parquet (34,889 rows)
revealed the period missed by spec 44's original spike-detection (it
looked for impulse periodicity; the actual pattern is a slow within-
window decay + boundary jump).

### Measured period

- **Period: exactly 1024 steps** = `io_block_size × window_blocks /
  batch_size = 256 × 512 / 128 = 1024` (no jitter; full integer).
- ACF at lag 1024 (on smoothed-then-long-detrended TL):
  - pre-cont-2 (steps 5000-19999): **0.46**
  - v4-cont-2 (steps 20001-34890): **0.81** (3.3× stronger lock-in)
- Wallclock period: 1024 × 3.146 s/step ≈ **3222 s ≈ 53.7 min** in
  cont-2 (≈3252 s = 54.2 min in pre-cont-2; ~1% faster post-resume).
- Intra-window phase shape (residual after subtracting per-window
  mean, 14 windows in cont-2):
  - bin 0 (steps 0-31 within window): residual **+0.27**
  - bin 31 (steps 992-1023 within window): residual **−0.13**
  - **monotonic decrease** across the entire window (32 phase bins,
    no mid-window oscillation).
- Per-window swing `(first_loss - last_loss)`:
  - pre-cont-2: median **0.17** (sign mixed: 9 positive, 5 mixed)
  - v4-cont-2: median **0.48**, **14/14 windows positive** —
    every window's loss falls 0.2-0.8 nats from start to end.

### Cause (confidence: high)

The 1024-step period maps **exactly** to Levanter's
`BlockShuffleConfig` window: `train_smoke_modal.py` builds
`LmDataConfig` without an explicit `shuffle=` arg, picking up the
default `BlockShuffleConfig(io_block_size=256, window_blocks=512)`
(visible in the manifest's `data.shuffle` field). Examples are well-
shuffled within each 131,072-sequence window, but adjacent windows
hold physically adjacent regions of the source parquet shards (one
contiguous draw of ~4,000 materials × 32 patches/mat). The model
"learns" each window's material distribution over 1024 steps, then
at the boundary jumps to a fresh window of materials → loss spikes
up.

The effect is **3× stronger in cont-2** because cont-2 is ~entirely
in epoch 2:

- Dataset has ~2.48M sequences (per memory tomat-arch-key-facts) →
  one epoch ≈ 2.48M / 128 = **19,378 steps**.
- First epoch ends ~step 19378. v4-cont-2 (steps 20001+) is the
  start of epoch 2: every window is data the model last saw
  ~19k steps ago, long enough to partially forget → bigger upward
  jump at each window boundary.
- Falling mean loss also makes the fixed window-boundary jump a
  larger fraction of per-step loss, increasing visual prominence.

The user's hypothesis ("transition between materials' patches") is
**not** the mechanism. M=32 patches/mat × BS=128 means 4 mats are
consumed per step, and even the larger `io_block_size=256` block
holds 8 materials. Material-transition cadence is sub-step, not
1024 steps. The 1024-step cadence is the inter-window-boundary
distance.

### Recommended fix

Match the TPU trainer's shuffle config — tighter `io_block_size`,
bigger window, which both reduces window-boundary heterogeneity and
mixes more materials per window:

```python
# scripts/train_smoke_modal.py, in train_bakeoff_h200x8 (and the
# density variant) — pass shuffle to LmDataConfig:
from levanter.data.text.datasets import BlockShuffleConfig
data = LmDataConfig(
    ...,
    shuffle=BlockShuffleConfig(io_block_size=32, window_blocks=8192),  # ≈ TPU defaults
)
```

`io_block_size=32` = one material's patches per block (cache-
friendly sequential reads); `window_blocks=8192` puts all of one
parquet shard worth of materials in a single window. This shifts
the sawtooth period to ~2048 steps (32 × 8192 / 128) and makes
window-boundary jumps far smaller because each window already
covers a much larger slice of the dataset. Alternatively bumping
`window_blocks` alone (256 × 8192 = 2,097,152 sequences ≈ 0.85 of
the dataset) would essentially eliminate the boundary.

Will not change asymptotic loss; will visibly smooth the trace and
remove the within-window forgetting-then-relearning waste. Lands in
the same scripts/train_smoke_modal.py file flagged in spec 44's
original Recommendation #2.
