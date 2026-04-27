# tomat experiment matrix

Status: **2026-04-27**, prep for tomorrow's project meeting. Maps every
axis we want to sweep. Densely intersect-able — each cell is a candidate
ablation.

## Headline metric

**Mat-level NMAE** = `mean|ρ_pred − ρ_true| / mean|ρ_true|`, on held-out
materials, after stitching disjoint patch decodes.

References:
- ChargE3Net (current SOTA): **0.53%**
- electrAI: **1.02%** (PADS-fixed-PBC, GGA+U)
- Hananeh's PADS: ~12-14% (older)

Codec quantization floor (lower bound on what any model can reach):
- LMQ-v2-16k: 0.178%
- LMQ-v2-32k: 0.103%
- LMQ-v2-65k: 0.058%

## Axes

### 1. Codec
- `two_token_9_12` (legacy, 4608 density tokens, 21-bit signed log-uniform)
- **`lmq-v2-16k`** ← current
- `lmq-v2-32k`
- `lmq-v2-65k`
- (future) `lmq-v3-65k-median` (L1-optimal recon points)

### 2. Loss function
- vanilla CE (legacy)
- **CE + λ·L_1** (Formulation X / current; λ ∈ {0.1, 1.0})
- pure L_1 (`replace` mode, λ=∞)
- (proposed) value-space Gaussian-smoothed CE (Yael's adaptive σ approach)
- (proposed) hybrid: `α·L_1 + β·gauss-smooth-CE`

### 3. Patch shape
- **cube P=14** (current default)
- cube P=15, 19, 20 (8k / 16k context)
- ball R²=75 (cube-P=14-matched, never trained)
- ball R²=86, R²=138 (matched to higher cubes)

### 4. Patches per material (M, training-time random sampling)
- **M=32** (current; coverage C ≈ 0.05 = 5%)
- M=64 / 128 / 256 / 512 / 1024 (denser sweep — see spec 13)
- C ≈ 2-3 (Yael's "every voxel covered ≥ 2-3 times") needs M ≈ 1000-1500 at P=14

### 5. Tile stride at inference (mat-level eval)
- stride = P (disjoint, current)
- stride = P/2 (8× overlap; can measure inter-patch consistency at shared voxels)
- stride = P/4 (64× overlap; voxel-level consensus / smoothing)

### 6. Model size
- 30M (legacy baseline)
- **208M** (current; matched-compute baseline)
- 1B (multihost; planned, deferred until lower-scale data is in)
- 3B+ (down the road)

### 7. Train tokens / steps
- 4 B (1 epoch of train-full at M=32; current default for 4k steps × bs128 × 8k)
- 8 B (2× Chinchilla for 200M)
- 20 B (Chinchilla-optimal for 1B)
- 60 B+ (toward over-Chinchilla regime)

### 8. Voxel rasterization order (within a patch)
- **row-major C-order** (current; flat scan x slowest, z fastest)
- Hilbert curve (spatial locality preserved)
- Z-order / Morton (similar)

### 9. Boundary handling
- PBC wrap on patch extraction (current for cubes; spec 10 says default for balls too)
- edge clip with [BOUNDS] preamble (deprecated for balls)

## Active runs / data points

| variant | model | codec | loss | M | tokens | NMAE | status |
|---------|-------|-------|------|---|--------|------|--------|
| `train-full` | 208M | 2-tok | CE | 32 | 6.29 B | TBD (eval pending) | trained |
| `train-full-tpu16-1B` | 1B | 2-tok | CE | 32 | 4.19 B | TBD | trained |
| `train-full-lmq-v2-200M-bs128-l1-0.1add` | 208M | LMQ-16k | CE+0.1·L_1 | 32 | 4.0 B | **TBD (eval r7 in flight)** | trained |

## Highest-priority next runs (ranked)

1. **Get a real NMAE number** for the LMQ-v2 208M run (eval in flight, 7th attempt).
   This anchors all other experiments — without it we can't compare.
2. **λ ablation** at 208M: λ=1.0-add, λ=∞-replace (pure L_1), vanilla-CE-on-LMQ
   (isolates "new codec" from "new loss").
3. **Codec-vocab sweep** at 208M: 16k / 32k / 65k. We already have the corpora;
   each run is ~4 hr.
4. **M sweep** at 208M: 32 / 128 / 256 / 512.
5. **1B run** with the best (codec, loss) combo from steps 1-2. Worth saving
   the chips for after we know what's working.
6. **Median-vs-mean recon ablation**: refit one codec with median; compare.
   Confirms or refutes our L2-Lloyd-Max-recon-but-L1-loss audit concern.

## Open infra TODOs

- [ ] Implement Gaussian-smoothed-CE option in `qwen3_density.py` (Yael-style adaptive σ).
- [ ] Refit codec(s) with median recon points (audit fix).
- [ ] Wire patch-level NMAE-proxy into training-time eval (free metric — same forward pass already does L_1).
- [ ] Inter-patch consistency eval (overlap stride > 1 → measure variance at shared voxels).
- [ ] Ball tokenizer training run (spec 10 design done, never trained).
- [ ] HF-conversion path for checkpoints (enables Modal-side eval, off-cluster inference).

## Open spec status

| spec | title | status |
|------|-------|--------|
| 10 | Ball patches | design done, no run |
| 11 | Per-mat NMAE eval | partial (eval_mat_nmae.py debugging in flight) |
| 12 | tokenize-to-GCS | not built (still using volume + sync) |
| 13 | Ablation runbook | not used since LMQ pivot |
| 14 | Patch tokenization viz | static plot done (this session); no interactive |
| 15 | 3D ball voxel viz | not built |
| 16 | MPDB | Phase 1 live (data/mpdb.sqlite) |
| 17 | Density loss design | Formulation X built; Gaussian-smooth-CE pending |
| 18 | LMQ codec | live; median-recon refit pending |

## Notes for the meeting

- We have a working **LMQ-codec + density-L_1-loss** training pipeline. 208M run finished
  with eval/loss=1.158 — but loss in nats is not directly comparable to electrAI's
  voxel-NMAE %. The eval pipeline (mat-level NMAE) is on its 7th attempt; we expect
  a real % number this session.
- Codec quantization floor at 16k bins is **0.178%** — comfortably below electrAI's
  1.02% baseline and within striking distance of SOTA's 0.53%. **Codec is not the
  bottleneck**; model + loss is.
- Yael's value-space Gaussian smoothing (arXiv:2603.07448) is a complementary
  approach to our L_1 — different optimization target (calibrated PDFs vs minimum
  MAE). Worth ablating.
- Open data point: **median vs mean** Lloyd-Max recon. Our current code uses
  mean (L2-optimal); for L1/NMAE training, median is L1-optimal. Easy fix; needs
  quantification.
