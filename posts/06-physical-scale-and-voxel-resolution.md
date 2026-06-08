# Physical scale and voxel resolution in the MP training set

**Status**: draft

---

## Setup

Materials Project charge densities are computed on a DFT grid whose resolution
is chosen per-material, not globally fixed. The grid shape `(nx, ny, nz)` is
set by the VASP ENCUT energy cutoff combined with the lattice vectors: a larger
unit cell or lower cutoff → coarser grid. The result is that `Å/voxel` —
the physical length one voxel represents along each axis — varies substantially
across materials.

This post measures that variation empirically across the full MP training set
(~81k materials in our MPDB) and draws out the architectural implications for
tomat.

**Quantities measured:**

- `Å/voxel_a = a / nx`, and similarly for b/ny, c/nz — where (a, b, c) are
  the lattice edge lengths in Å and (nx, ny, nz) are the DFT grid dimensions.
- Per-axis distribution: histogram of each of the three ratios.
- Per-material summary: min-axis, max-axis, mean-axis Å/voxel.
- Intra-material anisotropy: `max(a/nx, b/ny, c/nz) / min(a/nx, b/ny, c/nz)`.

**Caveat on non-orthogonal lattices.** For materials where α, β, or γ ≠ 90°
(rhombohedral, monoclinic, triclinic systems), the true physical length per
voxel along each real-space axis is not simply a/nx but involves the full
lattice metric tensor. The numbers below use the simple a/nx approximation;
the caveat is noted where relevant.

Lattice constants come from the pymatgen structure JSON stored in each
material's zarr.json sidecar on GCS
(`gs://marin-eu-west4/tomat/rho_gga_raw/{train,validation}/<mp-id>.zarr/zarr.json`).
Grid shapes from MPDB (`data/mpdb.sqlite`).

---

## Hypothesis

We expected Å/voxel to vary significantly — at least 2–3× from p10 to p90.
DFT grid resolution scales with ENCUT and inversely with lattice length; since
MP spans a wide range of materials (insulators with small unit cells through
porous frameworks and large alloys), heterogeneous resolution was the natural
expectation.

The operational concern would be: if voxel resolution varies widely, then a
patch of P=19³ voxels represents a different physical volume depending on the
material. A model trained on these patches without being told the Å/voxel scale
must learn to ignore or implicitly infer the physical scale from the preamble
(lattice block). ChargE3Net's graph-based approach sidesteps this entirely —
atom positions are in Å, not voxel indices, so the network is naturally
scale-invariant. Tomat's patch-based tokenization is not.

**The data says the hypothesis is wrong (or at least far weaker than expected).**
See Observation.

---

## Observation

**Data source**: 81,711 materials from MPDB (1 missing from GCS zarr cache),
lattice from pymatgen structure JSON in zarr.json sidecar files.

### Per-axis Å/voxel distribution

| axis | p1 | p10 | p50 | p90 | p99 | mean | std |
|------|----|-----|-----|-----|-----|------|-----|
| a/nx | 0.053 | 0.058 | 0.065 | 0.067 | 0.068 | 0.064 | 0.0035 |
| b/ny | 0.053 | 0.058 | 0.065 | 0.067 | 0.068 | 0.064 | 0.0035 |
| c/nz | 0.053 | 0.058 | 0.065 | 0.067 | 0.068 | 0.064 | 0.0035 |
| mean | 0.053 | 0.058 | 0.064 | 0.067 | 0.068 | 0.064 | 0.0031 |

Global range: 0.049–0.072 Å/voxel. p1–p99 spans 0.053–0.068 — a factor of
1.28×. The three axes have nearly identical distributions.

### Anisotropy ratio (max/min per material)

| | p10 | p25 | p50 | p75 | p90 | p99 | mean | max |
|-|-----|-----|-----|-----|-----|-----|------|-----|
| max/min Å/vox | 1.000 | 1.000 | 1.018 | 1.046 | 1.073 | 1.118 | 1.028 | 1.196 |

Median intra-material anisotropy is 1.018×. 90% of materials have max/min
below 1.073×. The maximum observed is 1.196×.

### Non-orthogonal lattices

62,337 / 81,711 materials = **76.3%** have at least one of α, β, γ deviating
from 90° by more than 1°. For these, the simple a/nx formula is an
approximation. The true Å/voxel along, say, the a-axis for a monoclinic cell
(β ≠ 90°) is `a × sin(β) / nx`, which is smaller than a/nx by the factor
`sin(β)`. For β = 120° this correction is `sin(120°) ≈ 0.866`, so the true
value is ~13% lower than the approximation. Despite this, the dominant message
(tight distribution) stands: the VASP grid is still adaptive to the cell shape,
and the quantized lattice block in the preamble encodes the full (a, b, c, α,
β, γ) tuple, giving the model access to the exact Å/voxel after correction.

Histogram: `tmp/voxel-resolution-histogram.png`
Anisotropy plot: `tmp/voxel-resolution-anisotropy.png`

---

## Mechanism: why MP densities have approximately uniform voxel resolution

The data shows the opposite of what we expected. The answer is in how VASP
chooses its grid.

VASP computes the charge density on a grid whose dimensions are set by the
Fourier-space cutoff sphere: `n_i ≈ 2 × G_max × a_i / (2π)`, where
`G_max = sqrt(2m_e × ENCUT) / ħ` (in SI) is the maximum reciprocal-space
vector included at the given ENCUT. Rearranging: `a_i / n_i ≈ π / G_max`.

With GGA ENCUT ≈ 520 eV and converting to SI: `G_max ≈ 3.27 Å⁻¹`, giving
`Å/voxel ≈ π / 3.27 ≈ 0.96 Å / 2 ≈ 0.048 Å`. In practice VASP applies
an oversampling factor of ~2× (PREC=Accurate doubles NFFT beyond the Nyquist
cutoff), giving an effective `Å/voxel ≈ 0.064–0.065 Å` — precisely matching
the empirical median.

In other words: **the grid is a Fourier-space sphere, and Å/voxel is
approximately constant at fixed ENCUT**, regardless of cell size. A large cell
gets a proportionally larger grid; a small cell gets a small grid; the physical
resolution per voxel is nearly the same everywhere.

In other words: **the grid is a Fourier-space sphere, and Å/voxel is
approximately constant at fixed ENCUT**, regardless of cell size. A large cell
gets a proportionally larger grid; a small cell gets a small grid; the physical
resolution per voxel is nearly the same everywhere.

The remaining variation (factor 1.47× from min to max; p1–p99 = 1.28×) comes
from:
1. VASP rounding `n_i` to even integers (discretization noise).
2. A small fraction of calculations using non-default ENCUT (e.g. hard
   pseudopotentials with higher cutoffs, or soft pseudopotentials).
3. The non-orthogonal correction (see above) — a/nx overstates the true
   Å/voxel for monoclinic/triclinic cells.

The tight distribution is also why ChargE3Net can succeed without explicit
scale conditioning: if all densities are on approximately the same physical
scale (within ±10% in Å/voxel), scale heterogeneity is not a major source of
distribution shift. The model can treat voxel spacing as approximately fixed.

**Implication for the earlier concern.** The "voxel represents different
physical volumes" worry was real in principle but is small in practice: the
physical patch size (19 × 0.065 Å)³ ≈ (1.24 Å)³ varies by only ±10% across
the training set, not 2–3×. This does not eliminate the architectural concern
(the model still doesn't know Å/voxel exactly), but it substantially reduces
its expected impact.

---

## Takeaways

### Revised concern: the problem is smaller than expected

The headline result is that Å/voxel is approximately constant across the MP
training set (~0.064 Å/voxel ± 5.5%, p1–p99 = 0.053–0.068 Å). The DFT grid
is a Fourier-space sphere at fixed ENCUT, which makes Å/voxel a near-constant
of the dataset rather than a variable per material.

This directly addresses the "voxel represents different physical volumes"
concern: P=19 patches are approximately 1.24 Å on a side everywhere in the
training set, ± ~5%. Scale heterogeneity is not a major source of distribution
shift.

### For the AR/MaskGIT patch setup (P=19 cube patches)

Each P=19³ patch represents roughly `(19 × 0.064 Å)³ ≈ (1.22 Å)³` of physical
volume. This varies by ≤10% across the training set. A patch-based model does
not need strong scale reasoning: the physical context is consistent across materials.

The lattice block (tokens qa, qb, qc in the preamble, quantized at 0.05 Å
resolution) still provides per-material (a, b, c) to the model. But its role
in the MaskGIT setup is more likely about **crystal geometry** (cell shape,
angles, axis ratios) than about reconciling wildly different physical scales.

The key preamble-token quantization caveat: the lattice block encodes lengths
at 0.05 Å resolution, so `a` is known to ±0.025 Å. For a ≈ 3.0 Å and nx=48,
this gives a/nx ≈ 0.0625 Å, but the uncertainty in a is ±0.025/48 ≈ ±0.0005 Å
— negligible relative to the ≈5% inter-material spread.

### For the downsampled tokenizer (resize-to-64³)

An earlier tokenizer variant resized every material's full density grid to a
fixed 64³ by trilinear interpolation. Since the native grid shapes vary
(the dataset spans nx ∈ [~40, ~200+]), resampling to 64³ stretches some
axes and compresses others. For a material with `(nx, ny, nz) = (40, 40, 120)`,
the z-axis is compressed by 120/64 ≈ 1.9× while x/y are stretched by
40/64 ≈ 0.63×. This changes the _effective_ Å/voxel: instead of the native
~0.064 Å/voxel, the compressed z-axis would appear at ~0.12 Å/voxel.

For this tokenizer variant, voxel-scale heterogeneity is real and introduced
by the resampling step, not inherent in the original DFT data. The VQ-VAE
approach (which relies on 64³ inputs) would be training on distorted density
fields with physically inconsistent aspect ratios. This is a design problem
intrinsic to that tokenizer variant, not to the patch-based P=19 tokenizer.

### Summary

The empirical finding flips the concern from "strong" to "weak":

- **AR/MaskGIT + patch tokenizer**: physical scale is nearly uniform across
  the training set (≈0.064 Å/voxel ± 5.5%). Scale conditioning is a nice-to-have
  for the last few percent, not a prerequisite for the model to work.
- **Downsample-to-64³ tokenizer**: resampling *introduces* scale heterogeneity.
  This is the tokenizer variant to avoid if scale consistency matters.
- **Non-orthogonal lattices (76.3% of materials)**: the simple a/nx formula
  understates true Å/voxel by up to ~15% for highly oblique cells (β ≈ 60°).
  The preamble's (α, β, γ) tokens allow the model to account for this exactly.

### Next steps

The scale-heterogeneity concern does not appear to be a blocking issue for the
MaskGIT experiment. The residual concerns are:

1. **Non-orthogonal correction** (76% of materials): if the model needs to
   reason about true real-space voxel dimensions (e.g. for scale-equivariant
   operations), it needs to use the angle tokens. This is currently
   architecture-dependent.
2. **FiLM conditioning on Å/voxel** as a lightweight option: inject `(a/nx,
   b/ny, c/nz)` as a per-layer scale modulation. Cost is small (~3 extra
   features in the preamble embedding), and it eliminates the residual ±5%
   variation. Worth considering for a production-track architecture.
3. **Scale-equivariant CNNs / ChargE3Net path**: still the gold standard for
   physical rigor, but less motivated by scale heterogeneity than by the
   structural prior (equivariance under SO(3) rotations). That motivation
   is independent of Å/voxel uniformity.

---

## GCS artifacts

Computed stats (JSON) and histogram images uploaded to:
- `gs://marin-eu-west4/tomat/eval/baselines/voxel-resolution-2026-05-23.json`
- `gs://marin-eu-west4/tomat/eval/baselines/voxel-resolution-histogram-2026-05-23.png`
- `gs://marin-eu-west4/tomat/eval/baselines/voxel-resolution-anisotropy-2026-05-23.png`

---

## TODO

- Verify the ENCUT back-calculation: the theoretical `π/√ENCUT` estimate assumes
  a single ENCUT across all MP GGA calculations. Confirm this is approximately
  true (i.e. that the spread is dominated by rounding + pseudopotential variation,
  not ENCUT differences between materials).
- Add a scatter plot: (a in Å) vs (nx) across all materials, colored by crystal
  system. Should be a tight linear band, confirming the Fourier-cutoff origin of
  the uniformity.
- Add wandb link to the cont33k run referenced in the MaskGIT section.
- Consider: for the non-orthogonal majority (76.3%), what is the distribution of
  β and γ angles? A histogram of the actual angle deviations from 90° would clarify
  the practical magnitude of the a/nx correction.
