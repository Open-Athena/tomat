# LMQ codec analysis — audit vs Yael's adaptive Gaussian approach

Status: **draft, 2026-04-27**.

## Yael's reference

Paper: [Discrete Tokenization Unlocks Transformers for Calibrated Tabular
Forecasting](https://arxiv.org/abs/2603.07448) (Elmatad, 2026).
Repo: [yaelelmatad/RunTime-Public/train](https://github.com/yaelelmatad/RunTime-Public/tree/main/train).

Core idea: instead of one-hot CE, use **value-space Gaussian smoothing**:
for each bin `b = [start, end]`, target probability is the Gaussian integral

```
target_b = ½·[erf((end − actual)/(σ√2)) − erf((start − actual)/(σ√2))]
```

with **adaptive σ**:

```
σ_bin = √(σ_floor² + (k·width_bin)²)
```

So fine-grained bins (dense data regions) get tight Gaussians; coarse bins
(sparse regions) get wide Gaussians. Calibrated PDFs over the full output
distribution.

## Comparing our approach

| | tomat (current) | Yael (Gaussian smoothing) |
|---|---|---|
| **Loss at density positions** | `\|E[ρ] − ρ_true\|` (L_1 on expected float) + CE on full vocab | CE against soft target = ∫_b N(ρ_true, σ²)dx |
| **Bin recon point** | mean of values in bin (L2-optimal Lloyd-Max) | median (in her code, "median" stored) |
| **Loss optimizes** | minimum MAE / NMAE directly | calibrated PDF (likelihood of true ρ under predicted distribution) |
| **Multi-modal output** | collapses to single mean | preserves PDF shape |
| **Hyperparameter** | none beyond λ_l1 | σ_floor, k |

**Both formulations are reasonable** for different goals:
- **Pure NMAE (electrAI / charg3net target)**: L_1 (ours) is more direct.
- **Calibrated uncertainty / multi-modal outputs**: Gaussian smoothing wins.
- **Best of both**: combined loss `α·L_1 + β·CE(soft target)`.

## Mean vs median for Lloyd-Max recon

We use **mean** (`recon[i] = Σ x_j·w_j / Σ w_j`):
- L2-optimal: minimizes `Σ (x_j − recon)²` for samples in bin.
- For our log-spaced fine bins (used in fitting), the within-bin distribution
  can be skewed (heavy left tail for low-density bins; right tail for atom-core
  bins). Mean drifts toward the heavy tail.

**Median** (sample where bin's CDF crosses 0.5) is L1-optimal: minimizes
`Σ |x_j − recon|`. **For NMAE-targeted training, median is the better recon point.**

How big is the difference? Let me quantify by adding median-recon to the fit
output and comparing.

## Recommended fixes

1. **Fit script**: also compute and save `recon_median` alongside `recon` (=mean).
   Cheap; no impact on existing training.
2. **Codec API**: `LMQCodec.decode(idx, mode='mean'|'median')` — default to
   median for L1/NMAE evaluation, mean for L2 contexts. Tunable.
3. **Loss function**: option to enable `gaussian_ce_smoothing=True` with adaptive σ
   (à la Yael). Combine with L_1 as `α·L_1 + β·CE_soft`.
4. **Both adjustments are decoupled**: each is a self-contained change.

## Action items

- [ ] Refit codecs with `recon_median` field.
- [ ] Quantify mean-vs-median MAE delta on a small test set.
- [ ] Implement Gaussian-smoothed-CE loss option (in `qwen3_density.py`).
- [ ] Ablation: λ_l1=1, λ_l1=1+gauss, gauss-only, vanilla CE.

## Notes / caveats

- Yael's domain (race time prediction) is 1D output per sample. Ours is
  thousands of voxels per patch, autoregressive. Calibrated PDFs are nice
  but the autoregressive structure already provides per-token distributions
  — calibration matters less (we mostly care about the marginal).
- "Gaussian in value space" assumes the true value's *measurement uncertainty*
  is Gaussian. For DFT charge density that's a reasonable prior; for race
  times she argues it captures real noise. For our task it's mostly a
  smoothing-the-loss-landscape trick.
