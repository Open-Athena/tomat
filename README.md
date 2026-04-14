# tomato 🍅
**to**kenized **ma**terials; LLM/transformer-based approach to predicting DFT-converged electron density for periodic crystals.

Uses a sequence model over a tokenized representation of $ρ$ (contrast with [electrAI]'s 3D ResUNet over voxel grids).

Sibling to [tomol] (**to**kenized **mo**lecules, Will Held's OMol25 S2EF
work). See [`specs/00-project-context.md`](./specs/00-project-context.md)
for positioning and the tomol/electrAI relationship.

[electrAI]: https://github.com/Quantum-Accelerators/electrai
[tomol]: https://huggingface.co/ihxds/ToMol-marin-1B

## Status

Very early. Repo exists to characterize the **reconstruction-error floor**
of candidate tokenization schemes before committing to training — the
transformer's achievable NMAE is bounded below by how much information
each scheme throws away on encode/decode.

Seven candidate schemes are enumerated in
[`specs/01-tokenization-strategies.md`](./specs/01-tokenization-strategies.md).
Three of the seven ("the easy three" — no trained VQ-VAE, no basis
choice, no RI-fitting step) are implemented so far.

## Preliminary results

**Data:** the electrAI-curated 2,885-entry MP subset on S3
(`s3://openathena/electrai/mp/chg_datasets/dataset_4/label/`), 128³
voxel grids. `label/` = DFT-converged ρ (the thing we want to tokenize).

**Metric:** NMAE = `sum(|ρ_reconstructed − ρ_original|) / sum(|ρ_original|)`.

What the sweep measures is the tokenizer's **reconstruction floor** — the
NMAE you get from `encode → decode` alone, with no model in the loop.
The transformer's total error on the same metric will be
`floor + prediction_error`, and electrAI reports ~2.6% achieved NMAE
(not a floor — that's their best end-to-end loss). So for tomato to
*beat* electrai, the floor needs to be well below 2.6%, leaving headroom
for the transformer to be imperfect. A floor above ~2.6% is an automatic
non-starter for the scheme (no amount of prediction accuracy recovers).

**Schemes:**

| scheme | implementation | what's kept | what's dropped |
|---|---|---|---|
| 1 — direct | `DirectTokenizer` | float32 copy of the density grid | nothing (sanity-check baseline) |
| 3 — voxel cutoff | `CutoffTokenizer` | top-K% voxels ranked by density value | all other voxels are zeroed |
| 5 — Fourier lowpass | `FourierTokenizer` | lowest-K% FFT coefficients by \|G\| | high-frequency modes |

### Overall (n=50 MP structures, 128³ grid)

| config | mean NMAE | median | min | max |
|---|---:|---:|---:|---:|
| `direct` | 2.15e-08 | 2.14e-08 | 2.10e-08 | 2.20e-08 |
| `cutoff-top-1pct` | 8.04e-01 | 8.29e-01 | 6.57e-01 | 9.29e-01 |
| `cutoff-top-5pct` | 5.01e-01 | 5.27e-01 | 2.78e-01 | 8.16e-01 |
| `cutoff-top-25pct` | 1.77e-01 | 1.75e-01 | 3.12e-02 | 4.55e-01 |
| `cutoff-top-100pct` | 2.15e-08 | 2.14e-08 | 2.10e-08 | 2.20e-08 |
| `fourier-lowg-1pct` | 4.78e-02 | **9.05e-03** | 2.68e-04 | 5.15e-01 |
| `fourier-lowg-5pct` | 9.16e-03 | **9.61e-04** | 9.17e-05 | 1.74e-01 |
| `fourier-lowg-25pct` | 9.08e-04 | **2.60e-04** | 3.47e-05 | 1.20e-02 |
| `fourier-lowg-100pct` | 5.40e-08 | 5.20e-08 | 2.95e-08 | 9.23e-08 |

### By material category (mean NMAE)

| config | oxide (n=18) | other (n=14) | intermetallic (n=11) | oxychalcogenide (n=4) | chalcogenide (n=3) |
|---|:---:|:---:|:---:|:---:|:---:|
| `cutoff-top-1pct` | 8.31e-01 | 8.26e-01 | 7.14e-01 | 8.48e-01 | 8.03e-01 |
| `cutoff-top-5pct` | 5.24e-01 | 5.53e-01 | 3.68e-01 | 5.50e-01 | 5.52e-01 |
| `cutoff-top-25pct` | 1.50e-01 | 2.03e-01 | 1.93e-01 | 1.62e-01 | 1.77e-01 |
| `fourier-lowg-1pct` | **1.13e-01** | 5.52e-03 | 1.91e-02 | 1.12e-02 | 5.18e-03 |
| `fourier-lowg-5pct` | **2.37e-02** | 4.23e-04 | 1.85e-03 | 1.01e-03 | 6.88e-04 |
| `fourier-lowg-25pct` | **2.01e-03** | 1.30e-04 | 4.96e-04 | 2.49e-04 | 3.35e-04 |

### Observations (n=50, preliminary)

**Budget framing.** We want `floor + prediction_error < 0.026`. The floor
is what the sweep measures; prediction error is the work the transformer
has to do. Lower floor = more budget for the model to be imperfect.

* **Fourier dominates voxel-cutoff at every sparsity level by ~2 orders
  of magnitude.** The chemically-interesting information lives in the
  low-spatial-frequency modes, not in the top-density voxels (which are
  concentrated near nuclei and don't carry bond/charge-transfer signal).
* **Cutoff (scheme 3) is a non-starter as-is.** At 25% of voxels it's
  still at 18% NMAE — already 7× over electrai's achieved loss before
  any model is trained. The rank-by-density criterion is backwards for
  this task: top-density voxels are near nuclei and trivially
  reconstructible from atomic positions, so the scheme is keeping the
  easy part and throwing away the hard part.
* **Fourier's budget at 5% coefs is comfortable in the median case
  (~2.5% budget) but tight in mean** (1.7% budget) — and is already
  *negative* for oxides in the mean (floor 2.4% vs target 2.6%, leaving
  ~0.2% budget for the transformer). Oxides at 1% coefs blow the budget
  outright (floor 11.3%).
* **Oxides are the worst case for Fourier**, 10–50× worse than every
  other category. Most likely the compact O core has non-negligible
  power at high \|G\|, so the lowpass leaves a residual. This is a
  concrete argument for **scheme 4 (Δρ)** next: subtracting PADS removes
  the atomic-core contribution and should flatten the category gap.
* **Dataset skew caveat:** the electrai-curated 2,885-subset has no
  halides or oxyhalides in the first n=50 (alphabetical by mp-id).
  Per the design doc's sparsity table, halides are the sparsest class
  and the one cutoff was originally motivated by — so cutoff may look
  relatively better if we re-run on a stratified sample. Doesn't change
  the headline (cutoff 18% at 25% kept is catastrophic), but worth
  confirming.

Raw CSV in [`results/sweep-n50.csv`](./results/sweep-n50.csv); regenerate
tables with `uv run scripts/summarize_sweep.py results/sweep-n50.csv`. See
[`specs/02-fidelity-sweep.md`](./specs/02-fidelity-sweep.md) for scope
notes and follow-ups.

## Running

```bash
spd                                            # project + venv setup
uv sync                                        # install deps
uv run pytest tests/                           # 17 tokenizer tests (no S3 IO)
uv run scripts/fidelity_sweep.py -n 10         # quick smoke test
uv run scripts/fidelity_sweep.py -n 50 -o tmp/sweep-n50.csv
```

CHGCARs are ~73 MB each. The first run against a given mp-id downloads
from S3 to `data/mp-cache/`; subsequent runs are local.

## Layout

```
pyproject.toml            # deps: pymatgen, numpy, click (marin stack deferred to PR 2)
src/tomato/
  tokenizers/
    base.py               # DensityTokenizer ABC (encode → decode → roundtrip)
    direct.py             # scheme 1
    cutoff.py             # scheme 3
    fourier.py            # scheme 5
  data/
    mp.py                 # S3 → pymatgen Chgcar, local caching
    classify.py           # material-type classifier (halide/oxide/intermetallic/...)
scripts/
  fidelity_sweep.py       # tokenize → detokenize → NMAE, per-scheme CSV output
specs/
  00-project-context.md   # positioning, sibling-project notes, open questions
  01-tokenization-strategies.md  # the 7 candidate schemes + tradeoffs
  02-fidelity-sweep.md    # this sweep's plan + current results
tests/
  test_tokenizers.py      # 17 roundtrip tests on synthetic densities (no IO)
```

## Follow-ups

- Add Δρ (scheme 4) variant. Requires a PADS (superposition of atomic
  densities) preprocessing step. Composes with schemes 1/3/5; doc
  predicts it improves all real-space representations.
- Copy grug's `launch.py`/`model.py`/`train.py` from
  `marin/experiments/grug/base/` and wire against the chosen tokenizer
  output — becomes the training entrypoint.
- Add a tomol-style SE/M0/M1 float codec on top of scheme 1 to get the
  "scheme 1 at production fidelity" NMAE number.
- Implement schemes 2 (VQ-VAE), 6 (SH), 7 (Gaussian / RI) only if the
  first three hit a fidelity floor above electrAI's ~2.6% NMAE.
