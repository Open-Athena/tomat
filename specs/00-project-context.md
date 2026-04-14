# Tomato: Tokenized Materials

Project context, positioning, and background on predecessor / sibling efforts.

## What is Tomato?

**tomato** = **to**kenized **ma**terials (with **to** intentionally echoing `tomol` = **to**kenized **mo**lecules).

The LLM/transformer analogue of [electrAI]: predict DFT-converged electron density for periodic crystals using a sequence model over a tokenized representation of ρ (and/or structure), instead of electrAI's 3D ResUNet doing super-resolution on voxel grids.

Per the OA design doc [Tokenizing strategies for electron density]: "our super-resolution CNN approach to modeling electron density has not resulted in as quick of wins as we had hoped. Given that, we want to test other architectures." The transformer path is appealing because OA already runs scalable transformer training on Marin, so the stack is mostly off-the-shelf once we commit to a tokenization.

**The unresolved question is how to tokenize.** See [`01-tokenization-strategies.md`](./01-tokenization-strategies.md) for the enumerated candidate schemes (CHGCAR, VQ-VAE, cutoff-thresholded voxels, deformation density, Fourier coefficients, spherical harmonics, Gaussian mixture).

This repository is currently a clean slate (empty git repo on `main`, no commits yet). The initial task is to scope the approach (pick a tokenization + training target) and stand up infrastructure.

[Tokenizing strategies for electron density]: https://docs.google.com/document/d/1Mc7PfPHk_EdQVYMO_Ak_k47SrCw7MM5bJdh__1-FOPY/edit

[electrAI]: https://github.com/Quantum-Accelerators/electrai

## Sibling / predecessor efforts

### electrAI (a.k.a. RHOAR-Net)

**Goal:** super-resolve / refine 3D electron-density fields for materials (periodic crystals).

- **Collaboration:** Rosen Group (Princeton) × Open Athena
- **Arch:** 3D ResUNet (`src/electrai/model/resunet.py`) / SRGAN-style generator (`srgan_layernorm_pbc.py`) with `InstanceNorm3d`, `PReLU`, PixelShuffle3d, circular padding for PBC
- **Input → Output:** cheap density guess (SAD = Superposition of Atomic Densities, or worsened DFT) → DFT-converged ρ, same voxel resolution (`n_upscale_layers=0` in current configs — it's refinement, not upscaling)
- **Datasets:** QM9 (molecules, 133k) and Materials Project (~2.9k periodic crystals). Full MP on S3 (`s3://materialsproject-parsed/chgcars/`, 198k+ samples).
- **Results (as of Jan 2026):** QM9 **0.15% NMAE** (matches paper's 0.14%); MP **~2.6–3.1% NMAE**
- **Loss:** NMAE (sum |pred − target| / sum target). Also explored Hellinger, χ², Jensen-Shannon, Weighted MAE.
- **Prior art:** Li/Sharir/Yuan/Chan (Caltech), _Image super-resolution inspired electron density prediction_, Nature Communications 2025 ([arXiv:2402.12335](https://arxiv.org/abs/2402.12335))
- **Upcoming:** ELF (Electron Localization Function) training (PR [#63]) — different scalar field, same arch as MVP

[#63]: https://github.com/Quantum-Accelerators/electrai/pull/63

**Takeaway for tomato:** electrAI treats the problem as _image super-resolution in 3D_. The hypothesis behind tomato is that the same material → property mapping can be learned by a sequence model over discrete tokens, the way tomol does for molecules.

### tomol (a.k.a. moltok)

**Goal:** tokenize molecular structures (OMol25) and train a Qwen3 LLM to predict energies/forces (the **S2EF** task).

- **Author:** Will Held (Marin community)
- **Arch:** Qwen3 transformer (HF: [ihxds/ToMol-marin-1B] — 1.4B params, 19 layers, `vocab_size=7293`, `max_seq_len=16384`, ~48k training steps)
- **Tokenizer:** [WillHeld/marin-tomol] — HuggingFace `WordLevel`, 7293 tokens
- **Dataset:** [WillHeld/Tomol25] — OMol25 tokenized, ~4M molecular structures sampled from DFT-annotated MD trajectories
- **Token format** (confirmed from actual val parquet, _not_ what the `moltok` README describes — see "Gotchas" below):
  ```
  [BOS]
  [ATOMS] [Z=6] [Z=1] ... [ATOMS_END]
  [PosX_SE:...] [PosX_M0:...] [PosX_M1:...]
  [PosY_SE:...] [PosY_M0:...] [PosY_M1:...]
  [PosZ_SE:...] [PosZ_M0:...] [PosZ_M1:...]
  [NL]          # per atom
  ...
  [FrcX_SE:...] [FrcX_M0:...] [FrcX_M1:...]
  [FrcY_SE:...] [FrcY_M0:...] [FrcY_M1:...]
  [FrcZ_SE:...] [FrcZ_M0:...] [FrcZ_M1:...]
  [NL]          # per atom
  ...
  [Eng_E:...] [Eng_M0:...] [Eng_M1:...] [Eng_M2:...]
  [EOS]
  ```
  Vocab breakdown: `PAD/BOS/EOS/NL/ATOMS/ATOMS_END/UNK` (7 specials) + `Z=1..118` (atomic numbers) + per-axis position/force **signed-exponent (SE)** (512 each × 6 axes) + per-axis position/force **mantissa** `M0`/`M1` (256 each × 2 × 6 axes) + energy `E`/`M0`/`M1`/`M2` (256 each × 4).
- **Related paper:** _Transformers Learn Molecular Structures Without Graph Priors_ (ICLR 2026 submission, anonymous/Aditi). 1B LLaMA2 → **Energy MAE 117.99 meV, Force MAE 18.35 meV/Å** on OMol25 val, competitive with eSEN 6M GNN.
  - Paper uses two-stage training: (1) autoregressive on discrete tokens, (2) fine-tune with continuous regression heads + bidirectional attention.
  - Will's approach differs: pure autoregressive, no continuous-head fine-tuning, uses Qwen3 instead of LLaMA2, uses RVQ codebooks.
- **Eval tasks (from `evaluate_omol25.py`):** S2EF + 7 chemistry tasks (conformers, distance_scaling, ie_ea, ligand_pocket, ligand_strain, protonation, spin_gap). Per Will's script, the tokenizer has _no_ charge/spin conditioning — geometry-only prediction.

[ihxds/ToMol-marin-1B]: https://huggingface.co/ihxds/ToMol-marin-1B
[WillHeld/marin-tomol]: https://huggingface.co/WillHeld/marin-tomol
[WillHeld/Tomol25]: https://huggingface.co/datasets/WillHeld/Tomol25

**Gotchas / status (2026-04-13):**
1. The README token format (`[P0:idx]`, `[FX0:idx]`, section delimiters like `[POS_END]`) does **not** match the tokens actually used in `WillHeld/Tomol25`. The real format is the `SE`/`M0`/`M1`/`M2` scheme above (floats encoded via signed exponent + mantissa digits).
2. The float → token mapping function (the _codebook_, in the general sense) is not checked in anywhere we've found. `tokenizer.json` only defines the vocab strings; whatever code converts a raw position/force/energy into the corresponding `SE`/`M0`/`M1` indices lives elsewhere. This blocks both (a) re-tokenizing new data and (b) decoding model outputs back to floats.
3. `codebook_mol_1m.pkl` (referenced by `serialize_molecules.py`, `build_rvq_codebooks.py`, `evaluate_omol25.py`) is not in the repo — described as an RVQ codebook, but appears to be a separate, earlier tokenization scheme that the model was _not_ actually trained on.
4. `fairchem.data.omol.evals` (imported by `evaluate_omol25.py`) does not exist in installed `fairchem-data-omol==0.1.1` — only `modules/evaluator.py` is present.

Net: tomol is a useful reference for arch, training setup (Marin + Levanter + Qwen3 on v5p-8 TPU), and S2EF task framing, but the eval pipeline is not runnable end-to-end without recovering the missing codebook from Will / Isaac, or re-tokenizing the data from scratch with a known encoder.

## Positioning: why tomato?

Same problem as electrAI — predict ρ_DFT for periodic crystals, evaluated against the same benchmarks (QM9, Materials Project, ELF coming). Different architectural bet: a sequence model over discrete tokens instead of a 3D ResUNet over voxel grids.

The case for it:
- **Scalability is a known strength of transformers.** electrAI's ~2.6–3.1% MP NMAE has been stubborn; bigger-is-better scaling laws don't obviously apply to ResUNets the way they do to transformers.
- **Multi-task / multi-source generalization** is easier: same tokenizer + same arch can learn from MP ρ, QM9 ρ, ELF, maybe eventually energy/force labels too.
- **Infra is already standing up**: Marin + Levanter + Qwen3 is proven at 30M–1B on TPU v5p-8 (tomol, protein-docs).

The case against:
- **No prior art tokenizes 3D ρ at useful fidelity** — every scheme in the design doc is somewhat novel.
- **Tokenization is lossy**; the NMAE floor set by reconstruction error may swamp any modeling gains.
- **electrAI's ResUNet has strong built-in inductive biases** (3D locality, translational equivariance via circular padding) that a vanilla transformer does not — the tomol/TATT result suggests transformers can learn such priors, but at a cost in data/compute.

## What's different from tomol

Materials vs molecules changes the problem meaningfully:

| Axis | tomol (molecules) | tomato (materials) |
|---|---|---|
| **Geometry** | Finite clusters, vacuum BCs | Periodic crystals, PBC |
| **Input representation** | Atoms + positions | Atoms + positions + **lattice vectors** |
| **Primary target** | Energy + forces (S2EF) | **Electron density field ρ** (and/or deformation density Δρ) |
| **Why this is harder to tokenize** | Just ~3·N_atoms float coords + scalars | A dense 3D scalar field (~10⁵–10⁶ voxels per structure) |
| **Dataset** | OMol25 (~4M) | Materials Project (~2.9k curated, 198k+ on S3) |

So tomato inherits tomol's _infra_ (Marin + Levanter + Qwen3 on TPU) and _encoding tricks_ (SE/mantissa float-to-token schemes for continuous quantities), but the central design question — how to tokenize a 3D density field — has no tomol analog.

## Technical approach (strawman)

- **Stack:** reuse Marin + Levanter + Qwen3 from tomol. Will's `experiments/tatt/` in the `will/tomol` branch of `marin-community/marin` has working v5p-8 configs at 30M, 600M, and 1B scales, plus parallel `protein_docs_*` experiments that show the same scaffolding generalizing to a different token stream. A tomato experiment file (`experiments/tatt/tomato_density_30m.py` or similar) is the minimum viable first step once a tokenized dataset exists.
- **Tokenizer:** pick from [the seven candidates](./01-tokenization-strategies.md). My initial lean (not a decision): scheme 3 (high-density voxels with cutoff) on Δρ (scheme 4), combined — keeps tokens interpretable, compresses well per the sparsity analysis, and the deformation-density trick removes the "model learns that ρ is big near nuclei" overhead.
- **Data:** electrAI's existing MP pipeline (`s3://materialsproject-parsed/chgcars/`, 198k+ GGA CHGCARs) is the natural source — identical supervision, directly comparable metrics. Start with the ~2.9k curated subset electrAI is training on now for fastest iteration / apples-to-apples comparison; scale up if needed.
- **Eval:** NMAE on the same MP validation split electrAI uses, so results are directly comparable. Stretch: downstream DFT convergence speedup if we get strong NMAE.

## Open questions

1. **Which tokenization?** The design doc is explicit that it's not a decision doc. We need to pick one (or a short list to prototype). See `01-tokenization-strategies.md`.
2. **Absolute ρ or Δρ?** Orthogonal to tokenization; the doc argues Δρ is strictly easier. Probably yes unless we see a specific reason otherwise.
3. **Relationship to electrAI:** "test other architectures" language in the doc implies tomato is a _sibling bet_, not a replacement — both efforts continue, with tomato judged on whether transformers can match/beat ResUNet at comparable compute. Worth confirming with Betsy / Hananeh / Rosen.
4. **Tokenization fidelity floor:** before investing in modeling, characterize reconstruction error for each candidate scheme. This is a small-scale empirical question (tokenize → detokenize → measure NMAE vs original) that could knock out schemes early.
5. **Tomol dependency:** tomol's per-axis SE/mantissa float encoding is reusable for continuous-valued tokens (density magnitudes, Fourier coefs, SH coefs). Salvaging the actual tomol tokenizer isn't necessary for tomato, but we can copy the encoding pattern.
6. **Dataset scope:** start with MP (matches electrAI), or push immediately into QM9 (simpler, molecule-only, matches prior-art replication benchmark)? QM9 first might be the fastest way to prove tokens-for-ρ is viable at all, before tackling periodic materials.

## Key references & links

### Tomol
- Repo (local): `$oa/tomol` — Will's serialization + eval scripts (incomplete eval pipeline; see Gotchas above)
- Marin experiments: `marin-community/marin`, branch `will/tomol`, `experiments/tatt/` — Qwen3 30M/600M/1B configs
- Paper (ICLR 2026, anon): `$oa/tomol/20331_Transformers_Adaptively_.pdf`
- HF: [ihxds/ToMol-marin-1B], [WillHeld/marin-tomol], [WillHeld/Tomol25]

### ElectrAI
- Repo: [Quantum-Accelerators/electrai]
- Local: `$oa/electrai` (notes in `CLAUDE.local.md`)
- WandB: [PrinceOA]
- Slack: [#oa-princeton-materials]
- Roadmap: [ElectrAI Dec '25–Mar '26] (Google Doc)

[Quantum-Accelerators/electrai]: https://github.com/Quantum-Accelerators/electrai
[PrinceOA]: https://wandb.ai/PrinceOA
[#oa-princeton-materials]: https://openathena.slack.com/archives/C09J8AUMBTL
[ElectrAI Dec '25–Mar '26]: https://docs.google.com/document/d/1H3DZyC6BM9Bu8f43P2Tlx0OPZzkwF9eVErG1mtpPF6E

### Design docs
- **[Tokenizing strategies for electron density]** — OA design doc enumerating candidate tokenization schemes. Summarized in `01-tokenization-strategies.md`.

### Literature / benchmarks
- **"Image super-resolution inspired electron density prediction"** (Li/Sharir/Yuan/Chan, Nature Communications 2025, [arXiv:2402.12335](https://arxiv.org/abs/2402.12335)): the paper electrAI replicates/extends.
- **CrystaLLM** ([arXiv:2307.04340](https://arxiv.org/abs/2307.04340)): LLM generating CIF strings for novel crystals — precedent for text-based crystal representation (but for structure, not density).
- **MACE** / **Allegro** / equivariant GNNs: spherical-harmonic features as continuous inputs; relevant to scheme 6.
- **MatterGen** (Microsoft, 2024): diffusion-based crystal generator.
- **UMA** / **fairchem-core** ([github](https://github.com/FAIR-Chem/fairchem)): Meta FAIR's universal MLIP — tangential (different task), but relevant framework if we later add energy/force heads.
