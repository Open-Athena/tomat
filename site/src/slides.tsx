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
          <li><strong>4. Δρ</strong> — subtract an analytic promolecule density, tokenize the residual. Composes with 1/3/5.</li>
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
    id: 'pivot-patches',
    title: 'Pivot: patches, not full grids',
    thumb: 'Patches',
    render: () => (
      <>
        <h2>Pivot: patches, not full grids</h2>
        <p>
          At native resolution (40³ to 448³ rho_gga grids) no single-token-per-voxel
          scheme fits 8–16k context. Downsampling to 32³ / 64³ throws away exactly the
          information the DFT solver spent hours computing.
        </p>
        <p><strong>New direction:</strong> one training example = one <code>P × P × P</code> sub-cube
          of a material's native-resolution grid, prefixed with the material's atomic
          inventory + the patch's offset within the parent grid.</p>
        <ul>
          <li><strong>Any voxel is a valid anchor.</strong> Crystals are periodic, so
            the tokenizer extracts patches with PBC wrap — no "edge" to avoid.</li>
          <li><strong>M patches per material</strong> (default 32) → 4,305 val structures
            × 32 = ~138k training rows, each a distinct local context.</li>
          <li><strong>P = 14 → 14³ × 2 tokens/voxel = 5,488 density tokens;</strong>
            ~200-token preamble; total ~5.7k, fits 8k with 2k headroom.</li>
          <li><strong>Offset seed = free data augmentation;</strong> re-running preprocessing
            with a different RNG seed samples a different set of patches from the same crystals.</li>
        </ul>
      </>
    ),
  },
  {
    id: 'patch-layout',
    title: 'Patch token layout',
    thumb: 'Layout',
    render: () => (
      <>
        <h2>What each training example looks like</h2>
        <pre className="token-layout">
{`[BOS]
[GRID_START]   nx ny nz           [GRID_END]    `}<span className="comment">{`# parent grid shape`}</span>{`
[ATOMS_START]  Z₁ Z₂ … Zₙ         [ATOMS_END]   `}<span className="comment">{`# atomic numbers`}</span>{`
[POS_START]    ⟨x₁ y₁ z₁⟩ …        [POS_END]     `}<span className="comment">{`# frac coords (3-byte codec)`}</span>{`
[SHAPE_START]  P P P                [SHAPE_END]   `}<span className="comment">{`# patch dims`}</span>{`
[OFFSET_START] ix iy iz              [OFFSET_END]  `}<span className="comment">{`# patch anchor (low corner)`}</span>{`
`}<span className="hi">{`[HI_START]     hx hy hz              [HI_END]      `}<span className="comment">{`# wrapped high corner`}</span></span>{`
[DENS_START]   d₀ d₁ … d_{P³−1}     [DENS_END]    `}<span className="comment">{`# density (2-token 9+12 codec)`}</span>{`
[EOS]`}
        </pre>
        <p>
          <strong>Vocab (6,792 total):</strong> 18 specials + 118 atomic Zs +
          1,024 ints + 1,024 position-codec + 4,608 density-codec.
        </p>
        <p>
          <strong>Why <code>[HI_START]</code>?</strong>{' '}
          Encodes <code>(ix + P − 1) mod nx</code> per axis. On any axis where
          <code> hi &lt; lo</code> the patch crossed the PBC boundary. Derivable from
          <code> (grid, offset, P)</code> via modular arithmetic, but making it an
          explicit observable at layer 1 saves the model from learning it — 5 tokens
          out of ~5,700.
        </p>
      </>
    ),
  },
  {
    id: 'patch-example',
    title: 'Example: NaCl patch, P=4, 16³ grid',
    thumb: 'Example',
    render: () => (
      <>
        <h2>Example: NaCl patch, P=4, 16³ grid, offset=(14, 2, 5)</h2>
        <div className="token-example">
          <div className="row"><span className="idx">[0]</span><span className="tok">1</span><span className="lbl">[BOS]</span></div>
          <div className="row"><span className="idx">[1]</span><span className="tok">7</span><span className="lbl">[GRID_START]</span></div>
          <div className="row"><span className="idx">[2]</span><span className="tok">152</span><span className="lbl">int = 16 <span className="comment">(nx)</span></span></div>
          <div className="row"><span className="idx">[3]</span><span className="tok">152</span><span className="lbl">int = 16 <span className="comment">(ny)</span></span></div>
          <div className="row"><span className="idx">[4]</span><span className="tok">152</span><span className="lbl">int = 16 <span className="comment">(nz)</span></span></div>
          <div className="row"><span className="idx">[5]</span><span className="tok">8</span><span className="lbl">[GRID_END]</span></div>
          <div className="row"><span className="idx">[6]</span><span className="tok">3</span><span className="lbl">[ATOMS_START]</span></div>
          <div className="row"><span className="idx">[7]</span><span className="tok">28</span><span className="lbl">atom Z = 11 <span className="comment">(Na)</span></span></div>
          <div className="row"><span className="idx">[8]</span><span className="tok">34</span><span className="lbl">atom Z = 17 <span className="comment">(Cl)</span></span></div>
          <div className="row"><span className="idx">[9]</span><span className="tok">4</span><span className="lbl">[ATOMS_END]</span></div>
          <div className="row sep"><span className="idx">[10–29]</span><span className="tok">…</span><span className="lbl">POS block — 2 atoms × 3 coords × 3 tokens/coord</span></div>
          <div className="row"><span className="idx">[30]</span><span className="tok">9</span><span className="lbl">[SHAPE_START]</span></div>
          <div className="row"><span className="idx">[31–33]</span><span className="tok">140</span><span className="lbl">int = 4, 4, 4 <span className="comment">(P=4 cube)</span></span></div>
          <div className="row"><span className="idx">[34]</span><span className="tok">10</span><span className="lbl">[SHAPE_END]</span></div>
          <div className="row"><span className="idx">[35]</span><span className="tok">11</span><span className="lbl">[OFFSET_START]</span></div>
          <div className="row"><span className="idx">[36]</span><span className="tok">150</span><span className="lbl">int = 14 <span className="comment">(ix)</span></span></div>
          <div className="row"><span className="idx">[37]</span><span className="tok">138</span><span className="lbl">int = 2 <span className="comment">(iy)</span></span></div>
          <div className="row"><span className="idx">[38]</span><span className="tok">141</span><span className="lbl">int = 5 <span className="comment">(iz)</span></span></div>
          <div className="row"><span className="idx">[39]</span><span className="tok">12</span><span className="lbl">[OFFSET_END]</span></div>
          <div className="row hi"><span className="idx">[40]</span><span className="tok">13</span><span className="lbl">[HI_START]</span></div>
          <div className="row hi"><span className="idx">[41]</span><span className="tok">137</span><span className="lbl">int = 1 <span className="comment">(hx — WRAPS: 1 &lt; 14)</span></span></div>
          <div className="row hi"><span className="idx">[42]</span><span className="tok">141</span><span className="lbl">int = 5 <span className="comment">(hy — no wrap: 5 &gt; 2)</span></span></div>
          <div className="row hi"><span className="idx">[43]</span><span className="tok">144</span><span className="lbl">int = 8 <span className="comment">(hz — no wrap: 8 &gt; 5)</span></span></div>
          <div className="row hi"><span className="idx">[44]</span><span className="tok">14</span><span className="lbl">[HI_END]</span></div>
          <div className="row sep"><span className="idx">[45–174]</span><span className="tok">…</span><span className="lbl">DENS block — 4³ = 64 voxels × 2 tokens</span></div>
          <div className="row"><span className="idx">[175]</span><span className="tok">2</span><span className="lbl">[EOS]</span></div>
        </div>
        <p className="note">
          Total length 176 tokens. At production (P=14) the DENS block dominates:
          5,488 of ~5,700 tokens. This patch anchor <code>(14, 2, 5)</code> wraps on
          the x axis only — visible at <code>[41]</code> as <code>hx=1 &lt; lo_x=14</code>.
        </p>
      </>
    ),
  },
  {
    id: 'status',
    title: 'Status (2026-04-21)',
    thumb: 'Status',
    render: () => (
      <>
        <h2>Where we are now</h2>
        <ul>
          <li><strong>Pivoted to patch tokenization</strong> — each training example is a P³ sub-cube of a material&rsquo;s native-resolution density grid, prefixed with atomic inventory + grid/patch/offset metadata. Resolves the 16k-context constraint without downsampling.</li>
          <li><strong>30M Qwen3 scaffold</strong> — 6 layers, hidden=512, 8k context, 6,792-token vocab. Target hello-world: 100 steps, monotonically decreasing loss.</li>
          <li><strong>Wrap-aware tokens</strong> — <code>[HI_START]</code> block encodes the wrapped high corner per axis; <code>hi &lt; lo</code> directly flags PBC wrap to the model without requiring it to learn modular arithmetic.</li>
          <li><strong>Data pipeline</strong> — <code>tomat-rho-gga</code> Modal volume seeded with 4,305 native-resolution val-split Zarrs (~22 GB); patch tokenizer + parquet preprocessor shipped.</li>
        </ul>
      </>
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
          <li><strong>Smoke training on Modal</strong> — tokenize ~128 val structures → 30M Qwen3 on one A100 for 100 steps. Immediate-access compute surface to close the loop end-to-end.</li>
          <li><strong>Sweep over (codec × patch size × M × N)</strong> — preprocess + training grids defined; 6 of 9 (codec, P) combos fit 8k context.</li>
          <li><strong>Scale to Marin + GCP TPU Research Cluster</strong> — real training run on v5p-8 once the smoke loop is green. Scaffold already in place.</li>
          <li><strong>Full train split</strong> — smoke uses val-only (4,305 structures); train split adds 77k structures / ~390 GB.</li>
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
          <li>Specs: <Ext href={`${REPO}/blob/main/specs/04-patch-training.md`}>04-patch-training</Ext>, <Ext href={`${REPO}/blob/main/specs/05-modal-smoke.md`}>05-modal-smoke</Ext>, <Ext href={`${REPO}/blob/main/specs/done/02-fidelity-sweep.md`}>02-fidelity-sweep (done)</Ext></li>
          <li>Notes: <Ext href={`${REPO}/blob/main/docs/discussion-notes.md`}>docs/discussion-notes.md</Ext></li>
        </ul>
      </>
    ),
  },
]
