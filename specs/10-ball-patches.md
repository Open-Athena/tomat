# Ball-shaped patches (ablation vs cube patches)

Status: **proposed**. Written 2026-04-23.

## Motivation

Current patch tokenization uses P³ cube patches (P=14) at a random
offset inside each material's grid. Cubes are simple (tile cleanly,
trivial preamble) but geometrically *uniform*: the center of the cube
and its corners are treated as equally "central" for learning
relationships between neighbors.

A **ball patch** is a set of voxels within integer squared-radius R² of
a chosen center voxel. This:

- Puts the "central" voxel at the center of the patch (cubes pick an
  arbitrary corner).
- Ranks voxels by isotropic radial distance, matching the physics
  (nearby atoms exert most influence; distant voxels are less
  informative for reconstructing the local density).
- Lets the model learn a rotation-equivariant-ish representation of
  local context (a cube is axis-aligned; a ball isn't).

Ball tokenization is an **ablation** vs cube tokenization: hold fixed
everything else (density codec, preamble format, atomic inventory,
model, training budget) and swap the patch shape + voxel traversal
order. Compare NMAE / train loss / per-voxel error under both.

## Definitions

Fix a ball center at integer voxel coordinates `(cx, cy, cz)` and a
squared-radius threshold `R²` (integer). The **ball** is

```
B(c, R²) = { (x, y, z) ∈ Z³ : (x−cx)² + (y−cy)² + (z−cz)² ≤ R² }
```

voxels inside the material's grid (bounds-clipped at edges).

**Canonical traversal order**: voxels sorted by

```
  (r², y−cy, x−cx, z−cz)       # primary: radial distance, then lex
```

(lex tiebreak is arbitrary but must be deterministic; we pick `(y, x,
z)` to match the prior patch-emission convention. We also flip signs to
keep a consistent hand of orientation.)

Canonical counts per r²-shell (number of integer triples `(i, j, k)`
with `i²+j²+k² = r²`):

| r²  | shell | cum_voxels | ≈cube P (V^⅓) |
|-----|-------|------------|---------------|
|  0  |     1 |          1 | 1.00          |
|  1  |     6 |          7 | 1.91          |
|  2  |    12 |         19 | 2.67          |
|  3  |     8 |         27 | 3.00          |
|  5  |    24 |         57 | 3.85          |
| 10  |    24 |        147 | 5.28          |
| 25  |    30 |        515 | 8.02          |
| 50  |    84 |      1,503 | 11.45         |
| **75**  | —   | **2,777**  | **14.06** ← matches cube **P=14** (2,744) |
| 76  |    24 |      2,801 | 14.10         |
| **86**  | —   | **3,407**  | **15.05** ← matches cube **P=15** (3,375) |
| 138 | —     |    ~6,859  | ~19 ← matches cube P=19 |
| 153 | —     |    ~8,025  | ~20 ← matches cube P=20 |

Counts computed by enumerating integer triples (i,j,k) with
i²+j²+k² ≤ R². Exact thresholds for ablation-matched comparisons:

- **ball r²≤75** ≈ cube P=14 (2,777 vs 2,744 voxels, +1.2%)
- **ball r²≤86** ≈ cube P=15 (3,407 vs 3,375 voxels, +0.9%)

## Token stream

Preamble differs from the cube case only in the shape block:

```
[BOS]
[GRID_START]   nx ny nz           [GRID_END]
[ATOMS_START]  Z₁ … Z_n            [ATOMS_END]
[POS_START]    (px₁ py₁ pz₁) …    [POS_END]
[RADIUS_START] R²                 [RADIUS_END]    # replaces SHAPE
[CENTER_START] cx cy cz           [CENTER_END]    # replaces OFFSET
[BOUNDS_START] x_lo x_hi y_lo y_hi z_lo z_hi [BOUNDS_END]  # bounding box of clipped voxels in grid; needed to decode edge-clipped balls
[DENS_START]   d_0 d_1 …          [DENS_END]      # density tokens in canonical order
[EOS]
[PAD] …
```

`[BOUNDS_*]` is new for balls (cube case derives bounds trivially from
`offset` + `shape`). Emitted only when the ball is clipped against the
grid edge.

New vocab tokens: `[RADIUS_START/END]`, `[CENTER_START/END]`,
`[BOUNDS_START/END]` (= 6 new specials; total 24 specials). No change
to integer / position-codec / density-codec blocks.

## Token budget

With two-token density codec (`two_token_9_12`, 2 tokens per voxel):

| ball target | n_voxels | density tokens | + preamble (~220) | context needed |
|-------------|----------|----------------|-------------------|----------------|
| cube P=14 match | 2,744 | 5,488 | 5,708 | **8k** ✓ |
| cube P=15 match | 3,375 | 6,750 | 6,970 | **8k** ✓ |
| r²≤163 | ~7,000 | 14,000 | 14,220 | **16k** ✓ |
| r²≤204 | ~8,000 | 16,000 | 16,220 | **16k** ✗ (just over) |

Preamble is +20 tokens vs cube (the extra `[RADIUS/CENTER/BOUNDS_*]`
block). Negligible.

## Sampling ball centers

For a material of grid shape (nx, ny, nz), sample M ball centers
uniformly across the full grid (any voxel can be a center). **Use PBC
wrap** — a ball centered at `(cx, cy, cz)` includes voxel
`((cx+dx) mod nx, (cy+dy) mod ny, (cz+dz) mod nz)` for each offset
`(dx, dy, dz)` in the canonical ball. This matches how cube patches
already work in `tokenize_patches.py` (`np.take(..., mode='wrap')`),
and matches the physical PBC of the underlying crystal.

Consequences of PBC-wrap:
- Every ball has exactly the same voxel count (no edge clipping).
- The `[BOUNDS_*]` block becomes redundant (can drop).
- Every voxel is an equally-valid center → uniform random sampling
  gives uniform exposure across all positions.
- Matches crystal physics: neighboring voxels across the grid edge are
  physically neighbors (by PBC), and the model should treat them as
  such.

~~Pre-PBC version (deprecated):~~ edge-clip with `[BOUNDS_*]` block.
Keep as a fallback if PBC reveals unexpected issues during training.

## Ablation protocol

Hold fixed: density codec, position codec, model (208M Qwen3), seed,
optimizer, steps, batch size, val-full / train-full splits.

Swap only: patch shape (cube vs ball) + voxel traversal order.

**Match voxel count per patch, not patch "radius"**: a cube P=14 ball
match (r²≤76) has ~2,744 voxels, same as P=14³. That's the fair
comparison.

Outputs to collect:

- Train loss + val loss at matched step budget.
- Per-patch NMAE / χ² on held-out full materials (see **#mat-level
  validation spec**, upcoming).
- Optional: ablate traversal order alone (emit cube voxels in radial
  order from cube center) to isolate the "sort by radial distance"
  effect from the "use spherical support" effect.

## Inference-time tiling: covering a full material with balls

Balls don't tessellate ℝ³ without gaps — unlike cubes — but that's fine,
because we want **overlapping** tiling at inference anyway (multiple
predictions per voxel → average for robustness, matches spec 11's
per-material-reconstruction flow). Key facts:

- Place ball centers on a **cubic lattice** with spacing `d`. Every
  voxel is within distance `d·√3/2` of some lattice center. For full
  coverage we need `d·√3/2 ≤ R` ⇒ `d ≤ 2R/√3 ≈ 1.155·R`.
- For `d = R·√2 ≈ 1.414·R` (slightly larger than the full-coverage
  threshold — matches a looser cubic pack), **some voxels get 1×
  coverage, others get 2×+**, average ≈ 1.5× across the grid.
  Non-uniform but bounded.
- Tighter spacing (`d ≤ R/√2 ≈ 0.707·R`, 8× more centers) gives ≥2×
  coverage everywhere — expensive but perfectly uniform.
- Combined with PBC-wrap (above), there are no edge artifacts to worry
  about — every ball has the same voxel count, every position is a
  valid center, coverage statistics are translation-invariant.

**For training**: uniform random center sampling. Each voxel gets
`~M·V_ball/V_grid` hits on average — non-uniform across mats (small
mats are over-covered) but fine for training since we see each mat
many times across epochs.

**For inference-time mat reconstruction (spec 11)**: pick a stride
based on the precision/cost tradeoff we want:
- stride = R·√2 (1.5× avg overlap): minimum for usable reconstruction.
- stride = R (2-3× avg overlap): better uniformity.
- stride = R/√2 (8× avg overlap): maximum voxel-consensus, 8× cost.

Recommend starting with stride = R (simple, moderate overlap) and
tuning if per-voxel variance is too high.

## Limitations / caveats

- **Ball tiling requires ≥1.5× overlap** at inference (can't be made
  disjoint). Minor compute overhead; an asset for voxel-consensus.
- **Variable shell sizes across r²** mean the model sees irregular
  "stride" in the density-token stream. Unlike cube patches where
  every voxel has the same local neighborhood structure, ball voxels
  at r²=5 (24 of them) get sandwiched between r²=4 (6 voxels) and r²=6
  (24 voxels) in the token stream — the model has to infer shell
  structure from sequence position.
- **Edge behavior (if not using PBC)** would break "every patch has
  the same geometry." PBC avoids this entirely.

## Recommendation

Defer until **LMQ-codec retokenize** (spec 18) lands — at which point
the ball tokenizer can reuse it directly. Balls are a cool-if-it-works
ablation, best run alongside the LMQ + density-L_1 loss work so the
comparison is apples-to-apples in the new regime.

## Non-goals (considered and dropped)

- **AR-consistency training over overlapping patches**: force the
  model's predictions on a shared voxel from two different patches'
  contexts to agree (self-consistency loss). Useful in semi-supervised
  settings where GT is scarce/noisy. We have per-voxel GT for every
  training material, so GT-supervised L_1 on many overlapping patches
  is strictly more informative than consistency regularization. Keep
  as a possible future unsup-/semi-sup lever if we ever work without
  GT.

## Links / adjacent specs

- `specs/done/02-fidelity-sweep.md` — prior analysis of alternative
  tokenizers (direct-float, Fourier, cutoff) that led to picking the
  current patch tokenizer.
- `specs/01-tokenization-strategies.md` — survey of tokenization
  schemes.
- `specs/04-patch-training.md` — original patch tokenizer design.
- `specs/11-per-mat-validation.md` (to be written) — per-material
  NMAE infra; a prerequisite if we want to evaluate ball tiling
  rigorously.
