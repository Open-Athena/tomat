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
uniformly from the grid interior (any voxel can be a center; edges get
clipped balls).

Open question: do we clip at the edge (ball near corner has fewer
voxels) or pad (ball near corner gets fake zero voxels)? Start with
**clip + record clipped bounds** — pad introduces fictitious signal.

Edge-clipped balls still fit the preamble: `[BOUNDS_*]` records the
actual clipped extent so the decoder knows the geometry.

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

## Limitations / caveats

- **Balls don't tile.** Full-material reconstruction requires
  overlapping balls and averaging / max-pooling of density estimates
  in overlap regions. Cube patches can reconstruct via disjoint
  tiling. This is the main architectural cost of balls.
- **Edge clipping** breaks the "every patch has the same geometry"
  invariant the model might exploit. Needs attention — possibly
  restrict ball centers to interior voxels (≥R from every edge), which
  reduces effective patch density near boundaries.
- **Variable shell sizes across r²** mean the model sees irregular
  "stride" in the density-token stream. Unlike cube patches where
  every voxel has the same local neighborhood structure, ball voxels
  at r²=5 (24 of them) get sandwiched between r²=4 (6 voxels) and r²=6
  (24 voxels) in the token stream — the model has to infer shell
  structure from sequence position.

## Recommendation

Defer until **P=15/8k cube retokenize** + **per-mat validation infra**
both land. Those are higher-certainty wins. Balls are a cool-if-it-works
ablation worth ~1 week of engineering, but won't move the needle alone
if the per-patch loss is already in a good place.

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
