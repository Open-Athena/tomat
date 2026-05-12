# Discussion notes

Running record of design discussions, findings, and open questions outside
the core specs. New material appended at the bottom; major topics get their
own section. For implementation specs see `specs/`; for code see `src/`.

## Project frame

**tomat** = tokenized materials. The transformer/LLM analogue of
**electrAI** (recently renamed **RHOAR-Net**, "Rho Augmented Resolution
Network") — predict DFT-converged electron density $\rho(r)$ for periodic
crystals, but via a sequence model over a tokenized representation rather
than electrAI's 3D ResUNet over voxel grids.

Sibling to **tomol** (Will Held's molecule-tokenizer + Qwen3 transformer for
OMol25 S2EF). Inherits tomol's infra (Marin + Levanter + Qwen3 on TPU
v5p-8) and its float-encoding tricks (SE/M0/M1).

## Reference points

The **2.6%** NMAE figure we keep citing is **electrAI/RHOAR-Net's best
validation NMAE on Materials Project**, from a 100-epoch internal run
(Jan 2026 monthly review; 50-epoch runs cluster around 2.7–3.1% and were
"still learning"). It's an in-house ResUNet number, not the published
SotA.

**Published SotA on MP charge density: ChargE3Net** (Koker et al,
[npj Computational Materials 2024][ChargE3Net]): E(3)-equivariant GNN
trained on ~106k MP structures, reporting **0.52 ± 0.01% NMAE** on a
2000-sample MP test split. Baselines in the same paper: invDeepDFT
(0.86%), equiDeepDFT (0.80%). OA benchmarks electrAI against
ChargE3Net.

Context: Li et al 2025 ([arXiv:2402.12335], the paper electrAI
architecturally replicates) is a separate molecular-only line and
reports 0.14% on QM9, no MP number. I earlier conflated "the paper
electrAI replicates" with "the MP state of the art" — they're
different, and the correction was prompted by Betsy. See also the
feedback memory on needing real literature searches for negative
claims about the published field.

So the hierarchy of targets is roughly:

| reference | MP NMAE | role for tomat |
|---|---:|---|
| ChargE3Net (Koker et al 2024) | 0.52% | published SotA — longer-term target |
| electrAI / RHOAR-Net (OA 2026) | 2.60% | stepping-stone internal target |
| tomat reconstruction-floor only | varies by scheme | must be well below both to be competitive |

[arXiv:2402.12335]: https://arxiv.org/abs/2402.12335
[ChargE3Net]: https://www.nature.com/articles/s41524-024-01343-1

### Floor vs achieved-loss framing

**Reconstruction floor** = NMAE from `encode → decode` alone, with no model
in the loop. Measured by our fidelity sweep.

**Achieved loss** = total error of the trained pipeline = `floor +
prediction_error`. What electrAI's 2.6% is.

For tomat to beat electrAI: floor must be **well below 2.6%**, leaving
budget for the transformer to be imperfect. A floor approaching 2.6% is
disqualifying. A floor at 0.1% is a *prerequisite* to competing, not an
achievement. (Easy to confuse — I did, see git history.)

## NMAE-as-target caveat

From Yael's investigation surfaced in the Jan 2026 RHOAR-Net review: **MAE
and NMAE are dominated by high-density (near-nucleus) regions**, since
those voxels have large $\rho$ values and per-voxel errors there
contribute disproportionately. Per the table on slide 6, the ratio of
low-density to high-density error contribution varies by loss function:

| loss | low/high ratio |
|---|---:|
| MAE | 0.0047 |
| Hellinger | 0.03 |
| Jensen-Shannon | 0.02 |
| Weighted MAE | 2.91 |
| Chi-Squared | 15.11 |

A scheme could win on NMAE while badly mis-representing the chemistry
(bonds, charge transfer, all in low-density regions). Our cutoff
finding is the same observation from a different angle: top-density
voxels are easy (predictable from atomic positions); the hard part lives
elsewhere in the metric's blind spot.

**Action item**: report Chi-Squared, Jensen-Shannon, Hellinger, and
Weighted MAE alongside NMAE in the fidelity sweep. electrAI's repo has
implementations.

## Fidelity sweep findings (n=50, 128³)

Detailed table in `README.md`; key points:

- **Fourier dominates voxel-cutoff by ~2 orders of magnitude** at every
  sparsity level. Density information lives at low spatial frequency
  (slow oscillations capture bulk shape efficiently); high spatial
  frequency only matters for atomic cusps.
- **Cutoff (scheme 3) is disqualified standalone.** At 25% of top-density
  voxels, NMAE is still 18% (~7× over electrAI). The "top-K-by-density"
  ranking is structurally wrong for this task — it picks the easy part
  (near-nucleus voxels, predictable from atomic positions) and discards
  the hard part (bulk/bonding voxels). NMAE = 1 − mass_captured for
  cutoff by construction; bottom 95% of voxels carry ~50% of total mass.
- **Oxides are the worst case for Fourier**, 10–50× worse than other
  categories at each sparsity. The compact O 1s/2s contributions have
  high-|G| spectral content the lowpass loses. Concrete motivation
  for scheme 4 (Δρ).
- **Dataset skew**: first 50 mp-ids (alphabetical) are mostly oxides /
  intermetallics / "other", with no halides or oxyhalides. Halides are
  the sparsest category per Yael's table and the one cutoff was
  motivated by; stratified re-run is queued as a follow-up.

## Δρ (scheme 4) status

Pipeline scaffolded (`src/tomat/promolecule.py`,
`src/tomat/tokenizers/delta.py`). Three analytic promolecule-density
implementations:

- `GaussianPromolecule` — one isotropic Gaussian per atom. Smooth, no
  cusps. Crude but doesn't introduce artifacts when wrong.
- `SlaterPromolecule` — 2-electron Slater-1s core (Slater's-rule
  $Z_\mathrm{eff}$) + Gaussian valence. Has the exponential form Δρ
  wants in principle, but the Slater's-rule α is tuned for all-electron
  densities, not VASP's pseudopotential ones.
- `MultiShellSlaterPromolecule` — full multi-shell Slater-type orbitals
  per occupied shell with Slater's-rule Z_eff. Reproduces
  Clementi-Raimondi ζ values to ~1% on first-row atoms; worse on
  d-block. Valence-only mode approximates what VASP pseudopotential
  CHGCARs contain.

**Naming correction**: this module was briefly named `pads.py` — that
was wrong. OA's *PADS* (Pre-tabulated Atomic Density Superposition) is
a distinct, VASP-derived tabulated density used by RHOAR-Net to generate
low-resolution input densities without a VASP license at inference. The
technique here is a *promolecule-density subtraction* — a standard
chemistry tool (Independent Atom Model / IAM), not PADS.

**Result with `MultiShellSlaterPromolecule`**: Δρ-Fourier gives a small
oxide-specific improvement at low retention (~12% lower NMAE / ~70%
lower χ² at 1% coefs), no improvement at 5–25% coefs (PAW/all-electron
mismatch dominates there).

**Path to a real Δρ test**: replace the analytic promolecule with
**pseudopotential-matched valence densities** parsed from the VASP POTCAR
files (since our training data is pseudopotential CHGCARs, this is the
exact-match version). POTCARs are available on della
(`/home/ROSENGROUP/software/vasp/vasp_potcars/potpaw_PBE.64/`). Licensed —
don't commit; a fitted per-element radial-density npz is the portable
form.

## Context-length feasibility

This is the load-bearing analysis for whether the LLM idea works at all.

**Token cost per Fourier coef** with tomol's SE/M0/M1 float encoding:
3 tokens per real → 6 tokens per complex coef. **No coord overhead**
needed for `fourier-lowg-X%` because the lowest-|G| coefs are a
structure-independent ordering — sender and receiver agree on it.

**Marin context windows we've considered**:

| target | density tokens (after ~100-token structure prefix) | reachable Fourier fraction (direct float) | median NMAE | viable? |
|---|---:|---:|---:|---|
| 4k (tomol's training default) | ~3,900 | 0.06% | (worse than 0.25%) | **no** |
| 16k (Qwen3 default) | ~15,900 | 0.25% | 8.9% | **no** — 3× over electrAI budget |
| 64k | ~63,900 | 1% | 0.9% on median, 4.8% mean | tight on oxide tail |
| 256k+ | full | 5% | 0.10% | yes, but expensive |

So **direct-float Fourier tokenization is not viable at standard Marin
training contexts**. Three options to make it work:

1. **Longer context** (64k+): adds 4–16× compute per step; needs sequence
   parallelism on bigger TPU pods. Most direct path but most expensive.
2. **VQ on Fourier coefs**: replace SE/M0/M1 (6 tokens/coef) with one
   codebook token per `(Re, Im)` pair (1 token/coef, vocab ~4k). Same
   NMAE floor as the underlying Fourier scheme; just compresses the
   representation. Adds a codebook-training step.
3. **Voxel-patch VQ** (Betsy/Tim's suggestion on the doc; the
   [arXiv:2503.14304] paper Tim shared): skip Fourier. Tokenize 4³ or
   8³ voxel patches via a learned codebook. At 8³ patches, the full 128³
   grid is 4,096 tokens regardless of fidelity — fits any context. NMAE
   floor depends on codebook size.

[arXiv:2503.14304]: https://arxiv.org/pdf/2503.14304

For training-cost parity with tomol (4k context), **VQ is unavoidable**.

## Target context: 4k

Decided: aim for **4k context** as the training default, matching tomol.
Reasons: cheapest training step, sharpest pressure on the tokenization to
be compact, direct apples-to-apples comparison with tomol's scaling.

Implication: the next deliverable is a tokenization scheme that fits 128³
density into ~3.9k tokens at usable NMAE. Voxel-patch VQ is the leading
candidate.

## Voxel baseline (proposed)

Concrete simplest-thing-that-could-work for a near-term trainable baseline:

**Per-patch autoregression** — split the 128³ grid into 8³ patches
(4,096 patches per structure). Each training example is
`[structure_tokens + patch_position] → [512 voxel values via SE/M0/M1]`.

- Sequence length: ~100 (structure) + ~3 (position) + 1,536 (patch) ≈ 1.6k.
  Comfortable fit in 4k context.
- Training pairs per structure: 4,096 patches.
- Total training pairs at the 2,885-subset: ~12M. Plenty.
- Loses long-range coherence (patches don't see each other), so charge
  conservation, bond continuation across boundaries are not enforced.
  But it's a baseline, which is the point.

A coarser two-stage variant ("coarse-pass + per-patch refinement
conditioned on coarse") preserves more long-range structure at the cost
of a second model.

## Patches and VQ — clarifications

"Patches" gets used for two distinct things:

1. **Block-averaging** — replace each $k^3$ block with its mean. Lossy
   downsampling; one scalar per patch.
2. **VQ over patch shapes** — flatten each $k^3$ block to a $k^3$-D
   vector; a learned codebook of K prototypes maps each patch to one
   integer (codebook index). Preserves the patch's *shape* (up to
   quantization), one token per patch.

Most of the literature meaning "voxel patching for transformers" is
sense (2). The two are independent — you could downsample *and then* VQ.

VQ-VAE specifically is the *learned* version of (2): an autoencoder where
the bottleneck is quantized to a codebook, all trained jointly to
minimize reconstruction error. **Scheme 2 in our spec.**

**OOD risk**: codebook is fit to training-distribution patches; OOD
materials (e.g. exotic chemistries not seen during training) hit
prototypes that don't match well, causing extra reconstruction error
on top of the model's own generalization error. Mitigations: train
codebook on a wide chemistry distribution, periodically refit, use
overcomplete codebooks, or use Residual VQ (multiple codebooks in
series modeling the residual of the previous).

## Hierarchical / segmented context ideas

User suggestion: instead of `[structure] + [full density]` in one shot,
re-emit structure + a position prefix per chunk so chunks fit in context
individually. Two flavors:

1. **Spatial chunking** (your "voxel mode"): natural for cutoff/voxel
   schemes. Doesn't work cleanly for Fourier (every coef contributes
   everywhere). Lost: long-range coherence between chunks.
2. **Frequency-band chunking** (Fourier-specific analog): emit DC + low-|G|
   first, then medium, then high — autoregressive coarse-to-fine. Each
   band is much smaller than the next. Truncation at any band gives a
   meaningful (lower-fidelity) reconstruction, like JPEG progressive.

The endpoint of pushing chunking aggressively is "structure → spatial
output with positional conditioning" — which is what electrAI's ResUNet
already does, just without the transformer. The interesting transformer
angle is reusing pretrained scaffolding (Marin/Levanter) and inheriting
LLM scaling laws, not the chunking trick per se.

**Hierarchical-band autoregression** plausible bands for our 128³ grid
(rfftn output 128 × 128 × 65 = 1.07M complex coefs):

| band | $|G|^2$ range | # coefs | % of total |
|---|---|---:|---:|
| 0 | DC + ≤16 | ~270 | 0.03% |
| 1 | (16, 64] | ~1,900 | 0.18% |
| 2 | (64, 256] | ~15,000 | 1.4% |
| 3 | (256, 1024] | ~120,000 | 11% |
| 4 | > 1024 | ~930,000 | 87% |

Each band is 5–8× larger than the previous (logarithmic). A model could
spend most capacity on bands 0–2 (small + dense in chemistry info) and
VQ/skip 3–4. Coarse-to-fine generation, naturally autoregressive.
Whether this specific scheme exists in the literature for materials
densities I can't confirm — diffusion-on-Fourier work exists in general
but I don't know of a hierarchical-Fourier autoregressive density
predictor specifically.

## Diffusion direction

User is interested in diffusion-over-density as the natural framework for
SCF-trajectory training. The mapping is clean:

- **Forward (noising)** = SCF iteration, going backwards from converged
  to the initial-guess density (e.g. OA's PADS, or VASP SAD). Each
  "noising step" is an inverse SCF step (gives a less-converged density).
- **Reverse (denoising)** = predict $\rho_{i+1}$ given $\rho_i$.
- **Inference** = start from the initial-guess density, run reverse
  process to converged.

Training data per SCF run: the full sequence $(\rho_\mathrm{init},
\rho_1, \rho_2, \ldots, \rho_\mathrm{final})$. Score-matching /
flow-matching loss over this trajectory.

Compared to autoregressive LLM tokenization:

- ✓ Trajectory structure used directly (rather than ignored)
- ✓ No tokenization → no fidelity floor from quantization
- ✓ Continuous-output training is natural for continuous targets
- ✓ Existing materials-diffusion infra (e.g. MatterGen) is directly relevant
- ✗ Loses LLM scaling-law inheritance and Marin/Levanter alignment
- ✗ Diffusion sampling is many forward passes per inference; less
  amortizable than a single LLM autoregressive pass

Probably worth pursuing **both**: tokenized-LLM and diffusion-on-density,
as competing bets. They share data and evaluation infra; differ on the
modeling side.

## SCF-iterates as training data

Considered separately from the diffusion framing:

**Not chaotic** — SCF is deterministic and converges to a unique fixed
point for almost all MP materials (exceptions: some Mott insulators with
spin-symmetry breaking). The premise of "ML-shortcut DFT" isn't blocked
by chaos.

**Concerns about SCF iterates as LLM training data**:

1. Iterates of one material are highly correlated; per-example
   information content sublinear in N.
2. Trains the wrong skill (mid-converged → final is easier than
   initial-guess → final).
3. Code-specific: VASP's iterate trajectory depends on its mixing
   scheme. Model would learn VASP-mixing dynamics, not a universal
   refinement operator.
4. Disk cost: 10–100× to store, or re-run SCF (~as expensive as the
   original calc).

**Better data-richness levers**:

- Scale to MP's full 198k+ materials (already on S3)
- Multi-task: predict $\rho$ + ELF + electrostatic potential from the
  same SCF, train jointly
- Perturbed structures: small atomic displacements give many $(R,
  \rho)$ pairs per compound
- Higher-accuracy targets where available (CCSD(T), QMC): teach the
  model what DFT itself gets wrong, not just what it gets right

## Other open questions

- **Stratified sweep**: re-run with a halide/oxyhalide/oxide-balanced
  sample to see if cutoff looks better in its motivating regime.
- **Modal harness**: scale fidelity sweep to full 2,885 (or 198k+);
  user prefers Modal over EC2 due to credit availability.
- **Resampling to fixed physical voxel size**: would make a learned
  codebook chemistry-invariant (atom cusps look the same in every
  structure regardless of cell size). Cost: variable grid sizes per
  structure, complicating batching.
- **Multi-target sweep**: add ELF as a second target once
  electrAI's ELF dataset lands. Same arch, same sweep code.

## Tool lineage and the VASP dependency

tomat's data pipeline inherits electrAI's, which inherits Materials
Project's, which is VASP-computed. This chain creates a soft lock-in on
proprietary tooling worth being explicit about.

### Three orthogonal axes (to keep straight when discussing QC codes)

1. **System type — periodic vs molecular**: crystals/surfaces with PBCs
   vs. isolated finite clusters. Codes tend to specialize; some do both.
2. **Basis set — plane waves vs atomic orbitals**: periodic systems
   naturally use plane waves ($e^{iG \cdot r}$); molecular systems
   naturally use atom-centered Gaussians or Slaters. Orthogonal to (1)
   but strongly correlated in practice.
3. **Core treatment — all-electron vs pseudopotential**: compute the
   deep 1s/2s cores explicitly, or absorb them into a smoothed
   effective potential. Independent of (1) and (2).

VASP sits at (periodic, plane-wave, PAW-pseudopotential). Our CHGCARs
are shaped by all three choices; promolecule-density approximations
that don't account for them (especially the pseudopotential axis)
produce systematic errors near nuclei.

### Why VASP dominates materials QC

- Started ~1991; network effects of three decades of papers.
- PAW POTCAR library is genuinely high-quality across most of the
  periodic table (50+ elements, multiple variants each).
- MP picked it → everyone using MP data is downstream of VASP.
- Fast, well-validated, responsive developer team (closed-source
  funding model has kept pace with methodology advances).
- Per-group license cost (~low-k€/yr academic) is small vs. compute.

### OSS alternatives — realistic and their gaps

| code | basis | pseudopotentials | vs VASP for our work |
|---|---|---|---|
| [Quantum ESPRESSO][qe] | plane-wave | NC + US + PAW | closest drop-in; multiple PP libraries (SSSP, PSlibrary, GBRV); widely validated |
| [ABINIT][abinit] | plane-wave | NC + PAW | similar scope, stronger on excited states |
| [GPAW][gpaw] | real-space + plane-wave | PAW | same PP family as VASP, Python+ASE-friendly, slower for large systems |
| [CP2K][cp2k] | hybrid Gaussian + plane-wave | GTH | great for biomolecular; less common for crystals |
| [pyscf][pyscf] | Gaussian (+ periodic add-ons) | GTH, ECP | molecular-first; what Li et al used for QM9 |

[qe]: https://www.quantum-espresso.org/
[abinit]: https://www.abinit.org/
[gpaw]: https://gpaw.readthedocs.io/
[cp2k]: https://www.cp2k.org/
[pyscf]: https://pyscf.org/

Why none of them "won" — mostly path dependence:

- VASP papers beget VASP papers; reproducing published work is easier
  in the originating code
- POTCAR library quality/breadth is hard to match
- MP's choice locks in a whole ecosystem of downstream users
- Cost-to-switch greatly exceeds marginal benefit for most groups

### What an OSS-only stance would cost us, concretely

1. **Existing electrAI/MP CHGCARs become unusable as training data.**
   They encode VASP-PAW-PBE conventions; predicting them with an OSS
   code would still leave us downstream of VASP lineage.
2. **Recompute the training set.** QE over MP structures at comparable
   settings (PBE, similar k-mesh density). For the curated 2,885:
   probably few-hundred CPU-days, tractable. For full MP 198k+:
   hundreds of k CPU-days, real money.
3. **Re-establish the electrAI baseline.** Their 2.6% NMAE is on
   VASP-PAW data; if our training data shifts, we need to recompute
   the baseline against the new data too.
4. **Can't do POTCAR-matched pseudo-valence promolecule for existing
   data.** Would switch to QE's PSP files for the recomputed dataset
   (they have parsers; it's fine, just different).
5. **Gain full reproducibility.** Anyone can rerun end-to-end with no
   proprietary deps. Attractive for open science / Marin-community
   alignment.

### Interesting datapoint: Li et al are OSS already

The paper electrAI replicates uses **pyscf + GTH pseudopotentials +
GTH-TZV2P basis** for QM9 — entirely OSS. So an OSS-aligned tomat
isn't methodologically novel, just a fork from electrAI's choice of
pre-computed VASP data. The molecular side of the literature has
already moved this way; materials lag because of MP's dominance.

### The ML-centric observation

Once a density-prediction model is trained, **it's code-agnostic at
inference**: it predicts ρ given a structure, regardless of which DFT
code "would have computed it." The training-data choice fixes what
conventions the model emits (PAW vs all-electron cores, PBE vs other
XC, etc.), but not architectural lock-in. So:

- **Short-term pragma**: inherit electrAI's VASP-CHGCAR data for direct
  apples-to-apples comparison against the ResNet baseline.
- **Medium-term**: could recompute a held-out subset with QE for
  cross-code validation (and as a hedge against VASP licensing
  friction at scale).
- **Long-term**: if publishing broadly or integrating with Marin's
  open-science ethos, an OSS-data fork is a defensible reset.

Not a decision right now; a flag for the team.

### POTCAR availability summary

- Proprietary, bundled with VASP license; **MP does not redistribute**.
- MP metadata records *which* POTCAR variant was used (e.g. `O_s` vs
  `O_GW` vs `O_h`), so reproduction is possible given a license.
- `pymatgen.io.vasp.inputs.Potcar` can parse them **if we have them
  locally** — doesn't help otherwise.
- For a POTCAR-matched pseudo-valence promolecule density, we'd need a
  licensed VASP install + the specific POTCAR versions MP used. The
  rosengroup has both on della (`vasp/6.5.1` module; POTCARs under
  `/home/ROSENGROUP/software/vasp/vasp_potcars/potpaw_PBE.64/`).

## Glossary

### Quantum chemistry / DFT

| term | meaning |
|---|---|
| **DFT** | Density Functional Theory — quantum chemistry framework where the ground-state electron density $\rho(r)$ alone determines all properties (Hohenberg-Kohn theorem) |
| **SCF** | Self-Consistent Field — iterative solver for DFT/HF; converges when the density reproduces itself through the Kohn-Sham equations |
| **KS** | Kohn-Sham — the auxiliary single-particle equations DFT solves; orbitals $\psi_i$ are KS orbitals (not "real" wavefunctions) |
| **PAW** | Projector Augmented Wave — VASP's pseudopotential method; reconstructs all-electron-like density from valence-only computation |
| **PP** | pseudopotential — replaces the atomic core with a smoothed effective potential acting on valence electrons only; makes plane-wave calculations tractable |
| **XC** | exchange-correlation — the unknown / approximated part of the DFT energy functional; choices like LDA, PBE, GGA define the "level of theory" |
| **CCSD(T)**, **QMC** | higher-accuracy quantum methods, expensive, often used as gold-standard benchmarks for DFT |

### Densities and related fields

| term | meaning |
|---|---|
| **$\rho(r)$** | electron density at point $r$. Units: e/Å³. Integral over a region = expected electron count. |
| **$\Delta\rho$** | deformation density = $\rho - \rho_\mathrm{pro}$. Bonding-only; smaller dynamic range; smoother |
| **Promolecule / SAD / IAM** | Promolecule / Superposition of Atomic Densities / Independent Atom Model — all names for $\sum_\mathrm{atoms} \rho_\mathrm{atom}(r-R)$ |
| **PADS** (OA-specific) | *Pre-tabulated Atomic Density Superposition* — VASP-derived tabulated per-element radial densities used by RHOAR-Net to generate low-resolution *input* density guesses without a VASP license at inference. Not a Δρ subtraction. |
| **CHGCAR** | VASP's output file for $\rho(r)$ on a grid. Stored as $\rho \times V_\mathrm{cell}$ (so divide by `lattice.volume` to get e/Å³) |
| **ELFCAR** | VASP's output file for the **Electron Localization Function** $\eta(r) \in [0,1]$. Bonding-analysis quantity; not derivable from $\rho$ alone (needs kinetic energy density $\tau$) |
| **WAVECAR** | VASP's output file for the orbitals $\psi_i(r)$ themselves |

### Fourier / signal processing

| term | meaning |
|---|---|
| **G** | wavevector; integer triple $(h, k, l)$ in reciprocal space; identifies one 3D plane-wave basis function |
| **|G|²** | sum of squares — measures "how fast" that wave oscillates across the cell |
| **DC component** | the $G = (0,0,0)$ coefficient; equals the average density (proportional to total electron count) |
| **rfftn** | NumPy's N-dim FFT for **real** input; exploits conjugate symmetry to store only half the coefficients (1.07M instead of 2.10M for 128³) |
| **NMAE** | Normalized Mean Absolute Error = $\sum |\rho - \hat\rho| / \sum |\rho|$; the metric electrAI uses |

### ML

| term | meaning |
|---|---|
| **VQ** | Vector Quantization — encode continuous vectors as nearest entry in a learned codebook of prototypes; one integer per input vector |
| **VQ-VAE** | learned VQ — encoder + codebook + decoder trained jointly to minimize reconstruction error |
| **RVQ** | Residual VQ — multiple codebooks in series, each modeling the residual of the previous; cumulatively unlimited precision |
| **SE / M0 / M1** | tomol's float-encoding scheme: Signed Exponent (512 classes) + Mantissa digit 0 (256) + Mantissa digit 1 (256) per real value. ~4-digit precision in 3 tokens |
| **NTP** | Next-Token Prediction; the LLM training objective |

### Tools / orgs

| term | meaning |
|---|---|
| **VASP** | Vienna Ab initio Simulation Package — proprietary plane-wave DFT code; standard for materials. Fortran. |
| **MP** | Materials Project — public DFT-computed materials database, computed with VASP |
| **electrAI** / **RHOAR-Net** | OA's 3D ResUNet for electron density prediction (rebrand happened Jan 2026) |
| **tomol** | Will Held's molecule tokenizer + Qwen3 transformer for OMol25 S2EF |
| **Marin** | OA's training stack on TPU; Levanter + JAX + Haliax |
| **Levanter** | JAX-based LLM training library (Stanford) used by Marin |
| **Qwen3** | Alibaba's open-weights transformer family; tomol fine-tuned a 1B variant |
