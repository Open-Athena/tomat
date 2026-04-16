# Tokenizer fidelity sweep

First empirical step for the tomat project: characterize the reconstruction
error floor each candidate tokenization scheme imposes, before investing in
modeling. The motivation is called out in both [`00-project-context.md`]
(open question 4: *characterize reconstruction error for each candidate
scheme — a small-scale empirical question that could knock out schemes
early*) and [`01-tokenization-strategies.md`] (additional question 5:
*tokenization fidelity — if scheme X loses 1% NMAE in reconstruction, the
transformer can never beat 1% NMAE even with perfect prediction*).

The transformer's total NMAE is `reconstruction_floor + prediction_error`
where the floor is what `encode → decode` alone loses before any model
runs. The MP reference point is electrAI (a.k.a. RHOAR-Net) at **2.60%
validation NMAE** (Jan 2026 monthly review, 100-epoch run) — OA's own
ResNet, not a published SotA (Li et al 2024 reports only on molecular
datasets, no MP number). For tomat to beat it, our floor needs to be
well below 2.6%, leaving headroom for transformer prediction error. A
scheme whose floor approaches or exceeds 2.6% is disqualified; a floor
well below it is a *prerequisite* to competing, not an achievement.

One more caveat on NMAE as a target: Yael's investigation in the same
Jan 2026 review shows the metric is dominated by high-density (near-
nucleus) voxels, so a scheme can score well on NMAE while being wrong
about the chemically interesting bonding regions. Reporting additional
metrics (Chi-Squared, Jensen-Shannon, Hellinger, Weighted MAE — all
already explored in the electrAI repo) alongside NMAE is a TODO.

[`00-project-context.md`]: ./00-project-context.md
[`01-tokenization-strategies.md`]: ./01-tokenization-strategies.md

## Scope of this PR

Three of the seven candidate schemes are implemented — the "easy three" that
don't require a trained VQ-VAE, an atom-centered basis choice, or a RI fitting
step. They're sufficient to sanity-check the pipeline and surface the first
real data point.

| Scheme | Module | Lossiness source |
|---|---|---|
| 1. Direct serialization | `tokenizers/direct.py` | (baseline: float32 round trip, ~lossless) |
| 3. High-density voxel cutoff | `tokenizers/cutoff.py` | voxels below cutoff are zeroed |
| 5. Fourier truncation | `tokenizers/fourier.py` | high-`\|G\|` coefficients dropped |

Scheme 4 (Δρ vs ρ) is orthogonal to the above and is a follow-up: requires a
PADS (superposition-of-atomic-densities) preprocessing step to compute.

Not in this PR: schemes 2 (VQ-VAE), 6 (spherical harmonics), 7 (Gaussian /
RI fit), and the tomol-style SE/M0/M1 float codec applied on top of
schemes 1/3/5.

## What's wired up

* `pyproject.toml` — mirrors `marin-experiments/tiny-tpu`, declaring
  `marin{,-levanter,-iris,-zephyr,-rigging}` dependencies plus `pymatgen`
  and `numpy`. No training code yet; the fidelity sweep deliberately
  doesn't import marin so it can run on CPU without pulling JAX/TPU deps.
* `src/tomat/tokenizers/` — ABC plus the three schemes; roundtrip-tested
  against synthetic Gaussian-lump densities in `tests/test_tokenizers.py`
  (no CHGCAR IO required).
* `src/tomat/data/mp.py` — fetches CHGCARs from
  `s3://openathena/electrai/mp/chg_datasets/dataset_4/` (same 2,885-entry
  curated MP subset electrAI trains on), caches under `data/mp-cache/`.
* `scripts/fidelity_sweep.py` — CLI that tokenizes→detokenizes→measures
  NMAE for each configured scheme over N samples, optional CSV output.

## Running the sweep

```bash
uv sync
uv run scripts/fidelity_sweep.py -n 5                   # quick smoke test
uv run scripts/fidelity_sweep.py -n 50 -o tmp/sweep.csv # first real data point
uv run scripts/summarize_sweep.py tmp/sweep.csv         # markdown tables
```

Each CHGCAR is ~76 MB on S3, so `-n 50` downloads ~3.8 GB into
`data/mp-cache/` the first run. Subsequent runs are local.

## First-pass results (n=50, 128³ grid)

See [`../README.md`](../README.md) for the full table. Headline: Fourier
truncation beats voxel-cutoff by ~2 orders of magnitude at every sparsity
level.

Δρ (scheme 4) has been scaffolded on top of Fourier with a crude Gaussian
promolecule-density baseline (`tomat.pads.GaussianPADS`, σ=0.4 Å). It
doesn't yet act as a physically faithful atomic-density model — real
atoms have core cusps (high-|G| content) that a Gaussian doesn't
reproduce, and our VASP CHGCARs use pseudopotentials that don't have
all-electron cores anyway. So the current Δρ variants bound "what the
pipeline does with an obviously-too-smooth PADS," not "the real Δρ
scheme's performance." Upgrading PADS to a Clementi-Raimondi
Slater-multi-shell or (better) pseudopotential-matched valence density
is the concrete next deliverable.

Key take-aways from this round:

1. **Scheme 5 (Fourier) is the only candidate of the three that clears
   the prerequisite.** At 5% coefs the median floor is 0.1%, leaving
   ~2.5% budget for the model. Mean is noisier (0.9%) because of the
   oxide tail — oxide mean floor alone is 2.4%, leaving only ~0.2%
   model budget, so oxides are the threshold case.
2. **Scheme 3 (voxel cutoff) is disqualified.** At 25% of voxels it's
   still at 18% NMAE — already 7× over electrai's achieved loss before
   any model runs. The "top-density voxels" ranking is backwards for
   this task: those voxels are near nuclei and are trivially
   reconstructible from atomic positions, so the scheme is keeping the
   easy part and throwing away the hard part.
3. **Category matters a lot for Fourier.** Oxides are 10–50× worse than
   every other category at each sparsity level. This is a concrete
   argument for pursuing scheme 4 (Δρ) next — subtracting the PADS
   removes the atomic-core structure that presumably drives this gap.
4. **Dataset skew.** The first 50 mp-ids in electrai's curated 2,885 set
   (alphabetical) contain no halides or oxyhalides. Stratified re-run
   is queued; doesn't change the headline (cutoff is disqualified
   regardless).

## Follow-ups (separate PRs)

1. **Add the Δρ variant** (scheme 4) once a PADS implementation exists — it
   composes with each of 1/3/5. The doc claims it improves all three, so
   we should measure the claim.
2. **Float codec** (tomol-style SE/M0/M1) layered on top of scheme 1. Gives
   the "scheme 1 at production fidelity" number as opposed to the lossless
   baseline reported here.
3. **Schemes 6 and 7** when we have a concrete reason to invest in the
   extra fitting/basis infrastructure — likely only if 1/3/5 all show
   floors above electrAI's ~2.6% NMAE.
4. **Grug template copy** — `launch.py` / `model.py` / `train.py` adapted
   from `submodules/marin/experiments/grug/base/`, wired against the
   chosen tokenizer output. Minimal value until a scheme is picked.

## Done criteria

Move this spec to `specs/done/` once we have:

- An NMAE-vs-sequence-length curve for each of schemes 1, 3, 5 over ≥50 MP
  samples.
- At least one explicit recommendation for which scheme(s) to pursue first
  in modeling (or a call to add 4/6/7 before deciding).
