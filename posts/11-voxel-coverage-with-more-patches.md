# Voxel coverage with more patches per material

**Status**: draft

---

## Setup

Tomat's v3 tokenization (P=19³ cube patches, random translations per material,
PBC-wrapped) currently uses `M=64` patches per material across `~77k` MP
training materials. Each patch covers `P³ = 6,859` voxels out of a
per-material grid of `V = nx·ny·nz` voxels.

Three new shards are being tokenized, disjoint from the existing one — taking
the union to an effective `M=256` per material (`64 → 128 → 192 → 256`). This
post asks: how much new voxel coverage does each additional shard buy?

**Question.** Is the 4th shard meaningfully better than stopping at 3?

---

## Closed-form coverage

For each material with volume `V` voxels, the probability a single uniformly
sampled patch covers any given voxel is `p = P³ / V` (no correction needed —
PBC wrap means every voxel is reachable from every patch origin). The expected
fraction of voxels covered by `M` independent patches per material is:

```
E[coverage_frac] = 1 − (1 − P³/V)^M
```

For mats with `V ≤ P³` this clips to 1, but in our MP train set there are
**zero** such mats (`p1` of V is 175,616; min is 25,600 voxels still ≫ 6,859).

---

## Distribution of mat sizes

Grid shapes pulled from `data/mpdb.sqlite` (`split=train`, n=77,427):

| pct  | V (voxels) |
|------|------------|
| p1   | 175,616    |
| p10  | 373,248    |
| p25  | 653,184    |
| p50  | 1,259,712  |
| p75  | 2,304,000  |
| p90  | 3,763,200  |
| p99  | 7,077,888  |
| max  | 21,952,000 |
| mean | 1,741,321  |

Mat volume spans a ~125× range (p1→p99). Since `p_hit = 6859/V`, this directly
drives a ~125× range in coverage difficulty: small mats saturate quickly, the
large-V tail crawls.

(Background on why the underlying _resolution_ — `Å/voxel` — is nearly uniform
across this distribution: see [post 06].)

---

## Aggregate coverage vs M

Mean (across mats) and voxel-weighted (across all corpus voxels) expected
coverage as a function of M:

| M (patches/mat) | mean E[cov] per mat | median per mat | voxel-weighted (corpus) |
|---:|---:|---:|---:|
| 32  | 21.0% | 16.0% |  — |
| **64**  (current, 1 shard)   | **35.1%** | 29.5% | **20.3%** |
| **128** (1 + 1 new shard)    | **53.1%** | 50.3% | **34.2%** |
| **192** (1 + 2 new shards)   | **64.2%** | 65.0% | **44.5%** |
| **256** (1 + 3 new shards)   | **71.7%** | 75.3% | **52.6%** |
| 320 (1 + 4 new shards)       | 77.1% | 82.6% | 59.0% |
| 384 (1 + 5 new shards)       | 81.2% | 87.7% | 64.3% |
| 512 (1 + 7 new shards)       | 86.7% | 93.9% | — |
| 1024                         | 95.7% | 99.6% | — |

Two ways to read "coverage":
- **Per-mat mean** treats every material as one unit, then averages. This
  weights small mats (which saturate fast) the same as large ones.
- **Voxel-weighted** sums covered voxels across the corpus and divides by total
  corpus voxels. This is the "what fraction of training data does the model
  ever see?" reading, and it's ~15–20pp lower because large mats are
  under-covered and dominate the corpus volume.

The "M=64 → ~12% coverage" note in older session memory was a coupon-collector
underestimate; the correct fresh closed-form numbers are 35% mean / 20%
voxel-weighted.

![voxel coverage vs M](/voxel-coverage-vs-M.png)

Left panel: aggregate (mean per-mat vs voxel-weighted) coverage as M grows
across the full corpus. Right panel: same curve split by mat-size quintile —
small mats hit 95%+ by M=128, the largest 10% are still under 50% even at
M=256.

---

## Per-bucket coverage by mat size

| volume bucket (voxels) | n_mats | median V | M=64 | M=128 | M=192 | M=256 |
|---|---:|---:|---:|---:|---:|---:|
| p0–p10  (smallest 10%) |  5,623 |    258k | **83.8%** | 96.8% | 99.3% | 99.8% |
| p10–p25                | 13,697 |    512k | 60.1% | 83.7% | 93.2% | 97.1% |
| p25–p50                | 18,573 |    885k | 38.7% | 62.1% | 76.5% | 85.3% |
| p50–p75                | 20,012 |   1.73M | 23.4% | 41.1% | 54.7% | 65.0% |
| p75–p90                | 11,753 |   2.76M | 14.3% | 26.5% | 36.9% | 45.9% |
| p90–p100 (largest 10%) |  7,769 |   4.42M |  8.7% | 16.7% | 23.9% | **30.4%** |

The bottom-decile (large mats) bucket is the bottleneck. Even at M=256, those
materials are still seeing under a third of their voxels. The top decile is
done by M=128.

---

## Per-shard marginals

Each shard adds 64 patches/mat. Marginal coverage gain per shard:

| shards | M | mean cov | Δ vs prev | new voxels per added patch |
|---:|---:|---:|---:|---:|
| 1 |  64 | 35.1% | +35.1pp | 5,513 (of 6,859 possible) |
| 2 | 128 | 53.1% | +18.0pp | 3,790 |
| **3** | **192** | **64.2%** | **+11.1pp** | **2,818** |
| **4** | **256** | **71.7%** | **+7.5pp**  | **2,188** |
| 5 | 320 | 77.1% | +5.4pp | 1,749 |
| 6 | 384 | 81.2% | +4.0pp | 1,426 |
| 7 | 448 | 84.3% | +3.1pp | — |
| 8 | 512 | 86.7% | +2.5pp | — |

The "new voxels per added patch" column is the most honest cost-effectiveness
metric: at M=0 the theoretical max is 6,859 (every patch is all-new); by the
3rd shard each new patch token is teaching the model ~2.8k new voxels, and by
the 5th it's down to 1.75k. The decay is monotonic, not sharply kneed.

---

## Knee analysis

When does the marginal gain per shard fall below threshold X?

| threshold | first shard where Δ < threshold | M | mean cov reached |
|---|---|---:|---:|
| <3.0pp | shard #8 | 512 | 86.7% |
| <2.0pp | shard #9 | 576 | 88.7% |
| <1.5pp | shard #11 | 704 | 91.7% |
| <1.0pp | shard #13 | 832 | 93.7% |
| <0.5pp | shard #17 | 1088 | 96.2% |

There is **no sharp knee**. Marginal coverage decays smoothly as `~1/M`. The
choice of "how many shards is enough" is a smooth tradeoff against
tokenization compute, not a clean transition.

---

## 3 vs 4 new shards

| | 3 new shards (M=256) | 4 new shards (M=320) | delta |
|---|---:|---:|---:|
| mean per-mat coverage | 71.7% | 77.1% | +5.4pp |
| voxel-weighted | 52.6% | 59.0% | +6.4pp |
| largest 10% of mats | 30.4% | ~36%   | ~+6pp |
| new voxels per patch (incremental) | 2,188 | 1,749 | −20% efficiency |

A 4th shard adds ~5–6pp of coverage and 20% less efficient per-patch than the
3rd. It's still adding meaningfully on the large-mat tail (which is where
coverage is starving), but the per-patch yield is monotonically declining.

The "3 vs 4" decision depends on what's limiting:
- If **patch-token budget** at train time is the constraint, the 3rd shard
  has 1.3× the new-voxel-per-patch yield of the 4th.
- If **tokenization-time compute / storage** is the constraint, all four
  shards have similar incremental cost (each is ~64 mat-patches × 77k mats =
  ~5M sequences); the 4th still buys real coverage on the under-served
  large-mat tail.
- The 4th shard is **not** wasted — it's not on the flat part of the curve.
  It's just a worse trade than the first three, by roughly the ratios above.

For full saturation (>95% mean coverage) you'd need ~16 shards (M≈1024).

---

## Caveats

- "Coverage" here means a voxel appears in at least one training patch
  somewhere across the whole corpus, not that it appears under every
  conditioning context. Even at 100% coverage, the model still has to
  generalize across the joint distribution of (preamble, neighbors,
  context-window-position).
- Patches sample uniform random offsets _per material_; the underlying
  distribution of structure is non-uniform within a material (atoms cluster).
  Random patch sampling treats voxel space as uniform, so empty/vacuum voxels
  get covered at the same rate as dense regions — for charge-density modeling
  this is probably fine (the model needs to learn vacuum prediction too) but
  worth noting if "valuable voxel coverage" is the question.
- These numbers are independent of token reuse across epochs: a re-shuffled
  epoch shows the same voxels again, not new ones. Adding shards is the only
  lever for new voxel exposure.

---

## Code

Analysis: `tmp/voxel_cov_analyze.py` (reads `data/mpdb.sqlite`,
closed-form). Plot: `tmp/voxel_cov_plot.py` →
`site/public/voxel-coverage-vs-M.png`. Both are derived from the existing
`scripts/analyze_voxel_coverage.py` (which sweeps P × M tables; this post fixes
P=19 and zooms in on M ∈ {64, 128, 192, 256, …}).

[post 06]: /posts/06-physical-scale-and-voxel-resolution
