# LMQ codec — empirical Lloyd-Max Quantizer for density

Status: **proposed**. Written 2026-04-23.

Related: [spec 17](./17-density-loss-design.md) (density loss design).

## Motivation

Two coupled issues with the current `two_token_9_12` codec:

1. **Loss has no ordinality.** Standard cross-entropy treats bin 256 vs
   bin 10 as equally wrong when the truth is bin 256. Spec 17 lays
   out the options; "Formulation X" (teacher-forced joint L_1 across
   both tokens of the 2-tok codec) is one viable answer but has
   teacher-forcing subtlety and a small A-B coupling bias.
2. **Log-uniform bin spacing isn't NMAE-optimal.** NMAE weights errors
   by where density actually sits in the distribution. Bins
   distributed according to the empirical PDF minimize MAE directly
   (Lloyd-Max quantizer).

A clean solution is to:

- Quantize densities with an **empirical Lloyd-Max quantizer** fit to
  the actual distribution across *all* train-full voxels.
- Collapse from **2 tokens per voxel to 1 token** per voxel. This
  eliminates the teacher-forcing joint-distribution issue (E[ρ] is a
  single dot product over one position's softmax).
- Use unsigned encoding — charge density is always ≥ 0, so the 2-tok
  codec's sign bit was dead weight anyway.

We call this the **LMQ codec**.

## Design

### Vocab size

| vocab | ≈ bits | embedding @ 208M | @ 1B | @ 3B | relative precision (uniform) |
|-------|-------|------------------|------|------|------------------------------|
| 8,192 | ~13 | 8 M (4%) | 17 M (2%) | ~30 M (~1%) | ~0.02% |
| **16,384** | ~14 | **17 M (8%)** | 34 M (3%) | 60 M (~2%) | **~0.01%** |
| 32,768 | ~15 | 34 M (16%) | 67 M (7%) | 120 M (~4%) | ~0.005% |
| 65,536 | ~16 | 67 M (33%) | 134 M (13%) | 240 M (~8%) | ~0.0025% |

**Proposed v1: 16,384 bins.** Sweet spot for 208M (8% embed cost,
tolerable), comfortable room at 1B and beyond. Lloyd-Max bins are
non-uniform, so the "effective relative precision" is better than
uniform for the high-density regions where NMAE is concentrated —
likely 10× better than the uniform benchmark above.

Precision target: electrAI/charg3net's NMAE floor is ~0.5%. Our
quantizer should contribute ≪ 1% to the total error budget. At 16 k
bins with NMAE-optimal placement, the quantizer contributes
approximately 0.01–0.05% — well below the floor.

### Fitting procedure

**All 130 B train-full voxels** (no subsampling — exact fit).

1. **Stream pass** via Modal fan-out (reuses `tomat-gcp-sa` secret,
   mounts `tomat-rho-gga-train` volume):
   ```python
   @app.function(volumes={MOUNT: volume}, cpu=1)
   def accumulate_histogram(mat_ids: list[str]) -> np.ndarray:
       hist = np.zeros(N_FINE_BINS, dtype=np.int64)
       for mat_id in mat_ids:
           density = load_rho_gga(f"{MOUNT}/label/{mat_id}.zarr").data['total']
           hist += np.bincount(
               np.clip(np.floor(density * FINE_SCALE), 0, N_FINE_BINS - 1).astype(np.int64),
               minlength=N_FINE_BINS,
           )
       return hist
   ```
   - N_FINE_BINS = 10⁶ (say, uniform bins over [0, 150]).
   - 77 k mats / 1000 workers = ~80 mats/worker = few minutes each.
   - Total ~5–10 min wall-clock.
   - Histogram size 10⁶ × 8 bytes = 8 MB — merges trivially.
2. **Fit Lloyd-Max** on the merged histogram (single CPU, seconds):
   ```python
   def lloyd_max_from_histogram(hist: np.ndarray, bin_centers: np.ndarray, n_bins: int):
       """Return (boundaries, reconstruction_points)."""
       # Equal-mass init: quantile boundaries from the CDF
       cdf = hist.cumsum() / hist.sum()
       init_boundaries = np.interp(np.linspace(0, 1, n_bins + 1)[1:-1], cdf, bin_centers)
       boundaries = init_boundaries
       for _ in range(20):
           # Reconstruction points: mean value within each output bin
           recon = weighted_means_of_bins(hist, bin_centers, boundaries)
           # Boundaries: midpoints between consecutive reconstruction points
           boundaries = 0.5 * (recon[:-1] + recon[1:])
       return boundaries, recon
   ```
3. **Save** the (16 k boundaries, 16 k reconstruction points) to
   `gs://marin-eu-west4/tomat/codecs/lmq-v1.npz`. Small (~128 KB).

### Codec API

Add to `src/tomat/float_codec.py`:

```python
@dataclass
class LMQCodec:
    """Empirical Lloyd-Max quantizer. Unsigned (density ≥ 0)."""
    boundaries: np.ndarray     # shape (n_bins - 1,)
    recon_points: np.ndarray   # shape (n_bins,)
    clip_max: float            # values above get the top bin

    @classmethod
    def load(cls, path: str) -> "LMQCodec":
        z = np.load(path); return cls(boundaries=z['bounds'], recon_points=z['recon'], clip_max=z['clip_max'].item())

    @property
    def vocab_size(self) -> int:
        return len(self.recon_points)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Map floats → bin indices via boundary search. Always 1 token/value."""
        return np.searchsorted(self.boundaries, np.clip(values, 0, self.clip_max))

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Map bin indices → float (reconstruction points)."""
        return self.recon_points[indices]
```

### Tokenizer integration

Extend `src/tomat/tokenizers/patch.py` + `ball.py` to accept an LMQCodec
at the density-codec slot. At 1 token per voxel, total sequence-length
budgets change dramatically:

| shape | 1-tok LMQ tokens | + preamble | fits 8k? | fits 16k? |
|-------|------------------|------------|----------|-----------|
| cube P=14 | 2,744 | 2,772 + 10·N_atoms | **99.9%** at 8k | 100% |
| cube P=19 | 6,859 | 6,887 + 10·N | 88% at 8k | 100% |
| cube P=22 | 10,648 | 10,676 + 10·N | 0% | 99.8% |
| cube P=25 | 15,625 | 15,653 + 10·N | 0% | 93% (8% OOM) |
| ball R²=200 | 11,837 | 11,866 + 10·N | 0% | 99%+ |

**Under 1-tok LMQ at 8k context we could push cube to P=19 (cf. the
2-tok codec's P=15 limit).** 2× more voxels per patch = 2× faster
per-voxel training on the same hardware. **This is an independent win
on top of the cleaner loss.**

### Loss at density positions (1-token codec)

Simplest form ever:

```python
# At each density-voxel position t, single token.
D_vec = codec.recon_points          # (16384,) constant, precomputed
P = softmax(logits[t])              # (vocab_size,) full vocab
P_density = P[DENSITY_OFFSET:DENSITY_OFFSET + 16384]
E_rho = np.dot(P_density, D_vec)    # expected density, scalar
P_ND = 1 - P_density.sum()          # total prob mass on non-density tokens
penalty_term = PENALTY * P_ND        # PENALTY ≫ ρ_max (e.g., 1000)
loss_voxel = abs(E_rho + penalty_term - ρ_true)
```

No joint, no teacher-forcing subtlety, no CE pollution. Just
differentiable L_1 in density space.

Per-patch NMAE loss (alternative):

```python
numerator   = sum(voxel_errors across patch)
denominator = sum(|ρ_true|) across patch  # precomputed at data-prep time
patch_nmae  = numerator / denominator
```

Directly optimizes pNMAE on each patch.

## Phased plan

### Phase A — fit + emit the codec (this week)
- [ ] `scripts/fit_lmq_codec_modal.py`: Modal-parallel stream → merged
      histogram → Lloyd-Max fit → save to GCS.
- [ ] `LMQCodec` class in `float_codec.py`, tests for encode/decode
      roundtrip, integration with `PatchTokenizer` / `BallTokenizer`.
- [ ] Sanity: re-tokenize val-smoke under LMQ, visualize a few recovered
      patches, confirm visual match.

### Phase B — retokenize + train (this week)
- [ ] Retokenize train-full at P=14/M=32 under LMQ (matched-baseline to
      existing train-full for direct comparison; not wasting M=256
      compute here). ~$8 + 10 min per the tokenize-to-GCS spec.
- [ ] Subclass Qwen3 with the new loss (single-position L_1 + ND
      penalty). Much simpler than Formulation X's 2-tok joint version —
      no decode matrix needed, just a 1D lookup.
- [ ] Matched-compute 208M training run against the new corpus + loss.
      Compare pNMAE + CE to vanilla-CE baseline on train-full.

### Phase C — ablation matrix (next week)
- [ ] LMQ-1tok + L_1 (this spec)
- [ ] 2-tok + L_1 teacher-forced joint (Formulation X per spec 17) —
      already planned, gives us a "was switching to LMQ worth the
      retokenize?" data point.
- [ ] Vanilla-CE baseline (existing 208M/1B)
- [ ] Optional: LMQ + vanilla-CE (isolates codec effect from loss
      effect) and 2-tok + Gaussian-smoothed-CE (isolates ordinality
      signal without float-space loss).

## Open questions

1. **Vocab size.** 16 k is the starter; do we also try 8 k and 32 k at
   207M? Cheap ablation given retokenize cost is low.
2. **Weighted Lloyd-Max variant?** For straight MAE, vanilla Lloyd-Max
   is optimal. If we want to trade vacuum-precision for atom-core-
   precision beyond what Lloyd-Max already gives, fit with
   sample-weighting by ρ. Unneeded in v1.
3. **Clip_max value.** Fit on train-full's distribution — what
   percentile? The 99.99th + safety margin (say p99.999 × 1.5).
4. **Near-zero handling.** ρ = 0 voxels should map to bin 0, not to a
   "smeared" near-zero bin. Explicitly handle in encode by snapping ρ <
   ε (e.g., 1e-15) to bin 0.
5. **Re-fit cadence.** If we later expand to GGA+U materials or other
   DFT functionals, the density distribution may shift. Plan to re-fit
   and retokenize. Version the codec (`lmq-v1`, `lmq-v2`, etc.).

## Adjacent specs

- [spec 17](./17-density-loss-design.md) — overall density-loss design,
  Formulation X (2-tok teacher-forced joint L_1).
- [spec 12](./12-tokenize-to-gcs.md) — 1000-wide tokenize infra that
  makes retokenization cheap.
- [spec 11](./11-per-mat-validation.md) — per-mat NMAE eval, the
  "target" metric this whole line of work is about.
- `src/tomat/float_codec.py` — the codec zoo this slots into.
