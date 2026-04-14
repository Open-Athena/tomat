# Tokenization strategies for electron density

Summary of the [Tokenizing strategies for electron density] design doc (OA, "collection of possible paths forward and their tradeoffs — not a decision doc"), plus observations.

[Tokenizing strategies for electron density]: https://docs.google.com/document/d/1Mc7PfPHk_EdQVYMO_Ak_k47SrCw7MM5bJdh__1-FOPY/edit

## Context (from doc)

> Our super-resolution CNN approach to modeling electron density has not resulted in as quick of wins as we had hoped. Given that, we want to test other architectures to see how they perform.

Transformers are appealing because of scalability, cross-task generalization, and because OA already runs scalable transformer training on Marin — so the stack is mostly off-the-shelf once we have a tokenized ρ representation.

**The unresolved question is how to tokenize.** Six candidate schemes are enumerated below, roughly ordered from "most direct / longest sequences" to "most compressed / most preprocessing."

## Candidate schemes

### 1. Direct CHGCAR serialization

Tokenize the native CHGCAR layout: header (elements, counts, lattice, atomic coords) followed by the voxel grid.

- **Pros:** Data is already in this format. Minimal preprocessing.
- **Cons:** Context length explodes — a 128³ grid is ~2M voxels. Mitigation: tokenize 3D _patches_ of voxels instead of individual ones.

### 2. VQ-VAE

Learn a discrete latent code via an encoder/decoder, then train the transformer in the latent space. Could reuse the existing ResUNet as the encoder backbone — similar in spirit to [FOMO's approach].

- **Pros:** Builds directly on existing electrAI image approach.
- **Cons:** Tokens are not human-interpretable — hard to reason about or debug.

[FOMO's approach]: https://openreview.net/forum?id=uAzhODjALU

### 3. Highest-density voxels with cutoff (Yael)

Keep the CHGCAR header (atomic + lattice structure) but replace the full voxel grid with an ordered list of (coord, density) tuples: descending by density, cut off at a threshold. Threshold could be percentage-based, an absolute density cutoff, or top-N.

**Sparsity analysis from the doc (GGA dataset):**

| Category | n | <0.05 e/Å³ | <0.1 e/Å³ | <0.5 e/Å³ |
|---|---:|---:|---:|---:|
| Halide | 8,139 | 40.7% | 57.4% | 85.2% |
| Oxyhalide | 3,949 | 34.7% | 51.6% | 82.2% |
| Oxychalcogenide | 2,949 | 33.4% | 50.1% | 81.0% |
| Oxide | 19,953 | 18.1% | 37.6% | 77.8% |
| Chalcogenide | 10,454 | 18.1% | 39.0% | 83.4% |
| Other | 19,514 | 0.4% | 11.9% | 80.4% |
| Intermetallic | 21,282 | 0.0% | 9.3% | 82.3% |

i.e. most materials have ~80% of voxels below 0.5 e/Å³, so a density cutoff could compress significantly.

- **Pros:** Human-readable grammar. Doesn't waste context on empty voxels. Reasonable context length.
- **Cons:** Loses low-density information (generally less interesting but still matters). Cutoff must be low enough to capture bonding chemistry (not just nuclear-adjacent voxels, which are trivially predicted from atomic positions). Sparsity varies a lot by material type (intermetallics have almost no sparse voxels), so a single threshold may not generalize.

### 4. Deformation density prediction

Predict Δρ = ρ_DFT − ρ_PADS (superposition of isolated atomic densities) instead of absolute ρ. Input = crystal structure; output = deformation density. At inference, add the cheaply-computed PADS to the predicted Δρ.

- **Pros:**
  - Removes core-electron signal the model would otherwise re-derive from element identity
  - Δρ has smaller dynamic range, no nuclear cusps, values ~centered on zero — friendlier for tokenization
  - Highest-magnitude voxels are now the chemically meaningful ones (bonds, charge transfer), so schemes 1/2/3 all improve
- **Cons:** Requires a one-time PADS pass over the dataset, and a cheap atomic-superposition step at inference (sub-second, deterministic, no NN). Arguably less elegant — the model learns "less."

Note: this is orthogonal to the tokenization choice — it changes what's being tokenized, not how.

### 5. Fourier coefficients (reciprocal space)

Decompose ρ into plane waves; list (h, k, l, amplitude, phase) tuples sorted by |G|. This is how VASP stores charge density internally.

- **Pros:** Natural for periodic systems, well-understood |G|-truncation, meaningful ordering by spatial frequency.
- **Cons:** Number of coefficients scales with cell volume — large cells still produce long sequences. Complex-valued (amplitude + phase per coef).

### 6. Spherical harmonics expansion

Per-atom decomposition: ρ = Σ_atoms Σ_{l,m} R_{l,m}(r) Y_{l,m}(θ,φ). Token stream = lattice params + per atom (element, coord, list of (l, m, radial coef)). l_max controls accuracy-vs-length tradeoff.

- **Pros:** Compact, atom-centric (mirrors input structure). Fixed per-atom token count once l_max is chosen. Widely used in comp-chem (density fitting, pseudopotentials).
- **Cons:** Overlapping atomic contributions need careful handling. Need to choose radial basis + l_max. Less intuitive than real-space.

### 7. Gaussian mixture / RI density fitting

Represent ρ as Σ Gaussians (center, width, amplitude). Resolution-of-identity fitting from quantum chemistry. Sequence length directly tunable via number of Gaussians.

- **Pros:** Fixed, tunable sequence length _independent of cell size_ — unique among these schemes. Physically interpretable. Well-established fitting literature.
- **Cons:** Extra preprocessing fitting step. Reconstruction quality scales with N_Gaussians.

## Cross-cutting observations

**Orthogonality:** "Absolute vs deformation density" (scheme 4) is orthogonal to the spatial representation choice (1/2/3/5/6/7). Any of the spatial schemes can target either absolute ρ or Δρ, and Δρ improves _all_ the real-space ones.

**Human-readability axis:** 1 > 3 > 6 > 4 > 5 > 7 > 2 — matters for debugging, failure-case inspection, building intuition. Relevant early but less so at scale.

**Context-length axis:** 7 ≈ 6 < 3 < 5 < 1 (scales with cell volume unless patched). 2 depends on codebook/patch size. For Marin's current Qwen3 configs (max 16k context for tomol's 1B model), schemes 1 and 5 likely need truncation/patching even for modest cells.

**Precedent in `tomol`:** Will's molecule tokenizer uses a real-space, atom-centric scheme with per-axis SE+mantissa encoding for positions and forces. That's closest to scheme 6 in spirit (atom-centric structured tokens), but encodes positions/forces rather than density coefficients. It's a useful reference for the _encoding_ step — how to turn a float into tokens the LLM can predict — regardless of which overall scheme is chosen.

**Precedent in the literature:**
- **CrystaLLM** ([arXiv:2307.04340]): LLM over CIF strings — closest to scheme 1 in spirit but for structure only (no density).
- **MACE** / **Allegro** / equivariant GNNs: Scheme 6 style (spherical harmonics) but as continuous features, not discrete tokens.
- **M3GNet, CHGNet**: atom-centric feature extraction, not density-level.

No prior work I'm aware of tokenizes full 3D electron density for transformer training — so all of 1/3/5/6/7 would be somewhat novel.

[arXiv:2307.04340]: https://arxiv.org/abs/2307.04340

## Questions raised by the doc (my additions, not in the doc)

1. **Benchmark target**: what's "success" here? Matching electrAI's ~2–3% NMAE on MP with a transformer? Beating it? Or a different metric (e.g. downstream DFT convergence from the transformer's ρ)?
2. **Scale**: electrAI trains on ~2.9k MP materials; full S3 mirror has 198k+. Does a transformer approach need the larger corpus to outperform, and do we have the budget/infra to tokenize + train at that scale?
3. **Equivariance**: electrAI uses circular padding for PBC but no explicit rotational equivariance. Most of these token schemes are not equivariant either — does that matter? MACE-style equivariance has been a major win in MLIPs; it might also matter here.
4. **Hybrid approaches**: the doc treats the schemes as alternatives, but e.g. scheme 6 (SH expansion per atom) + scheme 3 (high-density voxel residuals) could combine atom-centric compactness with explicit non-local bond features. Worth considering.
5. **Tokenization fidelity**: for any lossy scheme (2/5/6/7), what's the reconstruction error floor? If scheme X loses 1% NMAE in reconstruction, the transformer can never beat 1% NMAE even with perfect prediction. This probably constrains the choice more than anything else.
