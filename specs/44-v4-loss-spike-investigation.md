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
