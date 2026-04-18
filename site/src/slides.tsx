import type { SweepRow } from './types'
import { FractionPlot } from './FractionPlot'
import { ParetoPlot } from './ParetoPlot'

export interface SlideCtx {
  rows: SweepRow[] | null
}

export interface Slide {
  id: string
  title: string
  /** Short label shown in the thumbnail drawer. */
  thumb: string
  render: (ctx: SlideCtx) => React.ReactNode
}

const REPO = 'https://github.com/Open-Athena/tomat'

function Ext({ href, children }: { href: string; children: React.ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
}

export const slides: Slide[] = [
  {
    id: 'title',
    title: 'tomat — tokenizer fidelity for density prediction',
    thumb: 'Title',
    render: () => (
      <div className="slide-title">
        <h1>tomat <span aria-hidden>🍅</span></h1>
        <p className="subtitle">Tokenizer fidelity for electron-density prediction</p>
        <p className="byline">Ryan Williams · Open Athena · 2026-04-21</p>
      </div>
    ),
  },
  {
    id: 'problem',
    title: 'The problem',
    thumb: 'Problem',
    render: () => (
      <>
        <h2>The problem</h2>
        <p>
          Predict the electron density <code>ρ(r)</code> of a crystal structure
          with an LLM. Transformer alternative to electrAI's 3D ResUNet.
        </p>
        <p>Pipeline: structure → <strong>tokens</strong> → Qwen3 → tokens → <code>ρ(r)</code>.</p>
        <p>
          <strong>This talk:</strong> which tokenization scheme? Before training
          anything, characterize each candidate's reconstruction-error floor.
        </p>
      </>
    ),
  },
  {
    id: 'reference-points',
    title: 'Reference points',
    thumb: 'Targets',
    render: () => (
      <>
        <h2>What we're aiming at</h2>
        <ul>
          <li>
            <strong>ChargE3Net</strong> (Koker et al, npj Comp Mat 2024):
            <strong> 0.52% NMAE</strong> on Materials Project — published SotA.
          </li>
          <li>
            <strong>electrAI</strong> (OA in-house 3D ResUNet):
            <strong> 2.60% NMAE</strong> validation (Jan 2026 monthly review,
            100-epoch run) — stepping-stone target.
          </li>
        </ul>
        <p className="note">
          Both numbers are <em>achieved losses</em>, not tokenizer floors.
        </p>
      </>
    ),
  },
  {
    id: 'floor-vs-achieved',
    title: 'Floor vs. achieved loss',
    thumb: 'Floor',
    render: () => (
      <>
        <h2>Floor vs. achieved loss</h2>
        <p className="equation">
          <code>total_NMAE = reconstruction_floor + prediction_error</code>
        </p>
        <p>
          The floor is what <code>encode → decode</code> alone loses, before any
          model runs. For a tokenizer to be <em>competitive</em> with a target,
          its floor must sit well below that target — not near it.
        </p>
        <ul>
          <li>Floor ≈ 2.6% → zero budget for the model vs electrAI.</li>
          <li>Floor ≈ 0.5% → zero budget vs ChargE3Net.</li>
          <li>Floor &lt;&lt; target → there's room to train something useful.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'candidates',
    title: 'Candidate schemes',
    thumb: 'Schemes',
    render: () => (
      <>
        <h2>Seven candidate tokenization schemes</h2>
        <p><strong>In this sweep (easy three):</strong></p>
        <ul>
          <li><strong>1. Direct</strong> — float-per-voxel baseline (≈ lossless).</li>
          <li><strong>3. Cutoff</strong> — keep top-density voxels, zero the rest.</li>
          <li><strong>5. Fourier</strong> — keep low-|G| coefficients, drop the rest.</li>
        </ul>
        <p><strong>Next:</strong></p>
        <ul>
          <li><strong>4. Δρ</strong> — subtract a promolecule density (PADS), tokenize the residual. Composes with 1/3/5.</li>
        </ul>
        <p className="note">
          Deferred: 2 (VQ-VAE), 6 (spherical harmonics), 7 (Gaussian/RI fit).
        </p>
      </>
    ),
  },
  {
    id: 'sweep-setup',
    title: 'Sweep setup',
    thumb: 'Setup',
    render: () => (
      <>
        <h2>Empirical sweep</h2>
        <ul>
          <li><strong>Data:</strong> 50 Materials Project CHGCARs (electrAI's curated 2,885 subset, first 50 alphabetical).</li>
          <li><strong>Grid:</strong> 128³ voxels per sample.</li>
          <li><strong>Configs:</strong> 16 — cutoff × {'{1,5,25,100}'}%, Fourier × {'{0.25,0.5,1,5,25,100}'}%, Δρ-Fourier × {'{0.25,0.5,1,5,25}'}%.</li>
          <li><strong>Metrics:</strong> NMAE, χ², Hellinger, JSD, Weighted MAE — all on <code>encode → decode</code> roundtrip.</li>
        </ul>
        <p className="note">
          Live data: <Ext href={`${REPO}/blob/main/results/sweep-n50.csv`}>sweep-n50.csv</Ext>{' '}
          · <Ext href={`${REPO}/blob/main/scripts/fidelity_sweep.py`}>fidelity_sweep.py</Ext>.
        </p>
      </>
    ),
  },
  {
    id: 'plot-fraction',
    title: 'NMAE vs. fraction kept',
    thumb: 'Plot: fraction',
    render: ({ rows }) => (
      <div className="slide-plot">
        <h2>Reconstruction floor vs. fraction of representation kept</h2>
        {rows ? <FractionPlot rows={rows} metric="nmae" height={480} /> : <p className="note">Loading…</p>}
      </div>
    ),
  },
  {
    id: 'cutoff-red-herring',
    title: 'Cutoff is a red herring',
    thumb: 'Cutoff fail',
    render: () => (
      <>
        <h2>Cutoff (scheme 3) is disqualified</h2>
        <p>
          At 25% of voxels kept, median NMAE is still <strong>18%</strong> —
          already <strong>7×</strong> over electrAI's achieved loss before any
          model runs.
        </p>
        <p>
          <strong>Why:</strong> top-density voxels sit near nuclei. Those are
          trivially reconstructible from atomic positions. The cutoff scheme
          keeps the easy part and throws away the hard part (bonding regions).
        </p>
      </>
    ),
  },
  {
    id: 'fourier-wins',
    title: 'Fourier dominates',
    thumb: 'Fourier',
    render: () => (
      <>
        <h2>Fourier (scheme 5) clears the bar</h2>
        <ul>
          <li>At 5% coefs: median NMAE <strong>0.1%</strong> — ~3 orders of magnitude below cutoff.</li>
          <li>Leaves ~2.5% budget for the model vs electrAI.</li>
          <li>Mean NMAE is noisier (0.9%) — driven by the oxide tail.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'oxide-caveat',
    title: 'But: oxides',
    thumb: 'Oxides',
    render: () => (
      <>
        <h2>Oxides are the threshold case</h2>
        <p>
          Oxide mean NMAE at 5% coefs: <strong>2.4%</strong> — 10–50× worse
          than every other material category.
        </p>
        <p>
          <strong>Why:</strong> O 2p orbitals concentrate density at sharp
          cusps. Sharp cusps → high-|G| spectral content → a low-pass filter
          truncates exactly the content that matters.
        </p>
        <p>This is the concrete argument for pursuing scheme 4 (Δρ).</p>
      </>
    ),
  },
  {
    id: 'nmae-caveat',
    title: 'NMAE caveat (Yael)',
    thumb: 'NMAE caveat',
    render: () => (
      <>
        <h2>NMAE isn't everything</h2>
        <p>
          Yael (Jan 2026 review): NMAE is dominated by high-density voxels near
          nuclei — a scheme can score well on NMAE while getting the chemically
          interesting bonding regions wrong.
        </p>
        <p>
          Sweep reports 5 metrics in parallel: NMAE, χ², Hellinger, JSD,
          Weighted MAE. χ² is more sensitive to low-density regions and
          reshuffles the ranking somewhat.
        </p>
        <p className="note">
          Try the metric toggle on the live dashboard.
        </p>
      </>
    ),
  },
  {
    id: 'delta-pads',
    title: 'Δρ + PADS',
    thumb: 'Δρ',
    render: () => (
      <>
        <h2>Δρ — subtracting the promolecule</h2>
        <p>
          Scheme 4: compute a <strong>PADS</strong> (Promolecule Atomic Density
          Sum), subtract from ρ, tokenize the residual Δρ. At decode, add the
          PADS back.
        </p>
        <p>
          Current PADS: <strong>MultiShellSlater</strong> — Slater-type radial
          density per occupied shell, Z<sub>eff</sub> via Slater's rules,
          valence-only (crude approximation of VASP pseudopotentials).
        </p>
        <ul>
          <li>At 1% coefs: <strong>12%</strong> NMAE / <strong>70%</strong> χ² improvement.</li>
          <li>Oxides specifically: <strong>24%</strong> NMAE reduction at 1% coefs.</li>
          <li>At 5–25% coefs: no improvement (PAW/all-electron mismatch).</li>
        </ul>
      </>
    ),
  },
  {
    id: 'context-length',
    title: 'Context length reality',
    thumb: 'Context',
    render: ({ rows }) => (
      <div className="slide-plot">
        <h2>NMAE floor vs. tokens per structure</h2>
        {rows
          ? <ParetoPlot rows={rows} metric="nmae" height={460} />
          : <p className="note">Loading…</p>}
        <p className="note">
          Every scheme here assumes FP16 codec fidelity (3 tokens per real value,
          6 per complex). Vertical lines = 4k / 16k / 64k / 256k / 1M context.
        </p>
      </div>
    ),
  },
  {
    id: 'next-steps',
    title: 'Next steps',
    thumb: 'Next',
    render: () => (
      <>
        <h2>Next</h2>
        <ol>
          <li><strong>Better PADS:</strong> POTCAR-matched valence density (actual pseudopotential, not Slater orbitals). Should fix the 5–25% regression.</li>
          <li><strong>VQ-VAE baseline (Layer B):</strong> compress patches of ρ or Δρ to discrete tokens — unblocks 4k/16k context.</li>
          <li><strong>Float codec (tomol-style SE/M0/M1)</strong> on top of scheme 1 for the "scheme 1 at production fidelity" number.</li>
          <li><strong>Stratified halide sweep:</strong> first 50 mp-ids had no halides; stratify and re-run.</li>
          <li><strong>Modal harness:</strong> move CHGCAR-heavy jobs off laptop.</li>
        </ol>
      </>
    ),
  },
  {
    id: 'links',
    title: 'Links',
    thumb: 'Links',
    render: () => (
      <>
        <h2>Where to look</h2>
        <ul>
          <li>Live dashboard: <Ext href="https://tomat.oa.dev/">tomat.oa.dev</Ext></li>
          <li>Repo: <Ext href={REPO}>Open-Athena/tomat</Ext></li>
          <li>Sweep CSV: <Ext href={`${REPO}/blob/main/results/sweep-n50.csv`}>results/sweep-n50.csv</Ext></li>
          <li>Spec: <Ext href={`${REPO}/blob/main/specs/02-fidelity-sweep.md`}>specs/02-fidelity-sweep.md</Ext></li>
          <li>Notes: <Ext href={`${REPO}/blob/main/docs/discussion-notes.md`}>docs/discussion-notes.md</Ext></li>
        </ul>
      </>
    ),
  },
]
