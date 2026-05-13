import { MatsMetadataPlots } from './MatsMetadataPlots'
import { ThemeToggle } from './theme'
import { TrajectoryPlot } from './TrajectoryPlot'

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
}

const WB_PROJECT = 'https://wandb.ai/PrinceOA/tomat-lmq-P19'

export function HomePage() {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')

  return (
    <>
      <header>
        <h1>tomat 🍅 — tokenized materials</h1>
        <ThemeToggle />
      </header>

      <p className="meta">
        LLM-based electron-density prediction for periodic crystals. We
        tokenize ρ(r) directly and train a transformer over the resulting
        token stream — a sequence-model alternative to 3D-ResUNet
        approaches over voxel grids.{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat">Open-Athena/tomat</ExtLink>{' '}
        · <a href="#/runs">live runs dashboard</a>.
      </p>

      <h2>Current best</h2>
      <p>
        <strong>200 M parameter Qwen3, v3 patch tokenizer + LMQ-v2-16k
        density codec.</strong> Best validation NMAE so far is{' '}
        <strong>1.73 %</strong> at step ~21 k of the{' '}
        <ExtLink href={`${WB_PROJECT}/runs/train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k-ext`}>
          <code>cont7k-ext</code>
        </ExtLink>{' '}run, with NEMD <strong>1.76 %</strong> at step 20 k.
        Run is a constant-LR continuation past Chinchilla-optimal and is
        still trending lower.
      </p>
      <table className="runs-table">
        <thead>
          <tr><th>model</th><th>params</th><th>val NMAE</th><th>val NEMD</th><th>notes</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><ExtLink href="https://arxiv.org/abs/2312.05388">ChargE3Net</ExtLink></td>
            <td>—</td><td><strong>0.523 %</strong></td><td>—</td>
            <td>equivariant GNN; published SOTA (Materials Project)</td>
          </tr>
          <tr>
            <td><strong>tomat <code>cont7k-ext</code></strong></td>
            <td>200 M</td><td><strong>1.73 %</strong> @ 21 k</td><td><strong>1.76 %</strong> @ 20 k</td>
            <td>v6e-16, 25 k+ steps, training</td>
          </tr>
          <tr>
            <td>LMQ-v2-16k codec floor</td>
            <td>—</td><td>0.18 %</td><td>—</td>
            <td>oracle: best NMAE if model outputs the codec posterior perfectly</td>
          </tr>
        </tbody>
      </table>
      <p className="note">
        Codec floor 0.18 % means our metric ceiling well exceeds SOTA;
        room is in the model, not the tokenization.
      </p>
      <div className="plot-card">
        <TrajectoryPlot url={`${base}/nmae-nemd-trajectories.json`} />
      </div>
      <p className="note">
        For live state of all runs (loss, step, MFU, lifecycle events,
        sigterms, preemptions over UTC wall-clock), see{' '}
        <a href="#/runs">the runs dashboard</a> — parquet sourced from
        wandb via <code>tomat runs sync</code>, served from R2 through a
        Cloudflare Worker.
      </p>

      <h2>Patch tokenization (v3, current era)</h2>
      <p>
        Each training example is one <code>P × P × P</code> sub-cube of a
        material's native-resolution charge density, prefixed with:
      </p>
      <ul>
        <li>The full grid shape <code>(nx, ny, nz)</code>.</li>
        <li>The material's lattice <code>(a, b, c, α, β, γ)</code>{' '}
          (added in v3-lat, 2026-04-30).</li>
        <li>The material's atomic inventory (Z + <em>per-patch-translated</em>{' '}
          fractional coordinates — v3 wraps atom positions relative to the
          patch's anchor, so the model never has to learn PBC modular
          arithmetic).</li>
      </ul>
      <p>
        At <code>P = 19</code> with the LMQ-v2 1-token-per-voxel density
        codec, each sequence is <code>19³ = 6,859</code> density tokens
        plus a small preamble — fits 8 k context. Vocab is{' '}
        <strong>~18.5 k tokens</strong> (20 specials + 118 atomic Z + ints
        for grid/positions/lattice + 16,384 LMQ density bins). Each
        material contributes <strong>M = 64</strong> randomly-sampled
        patches (one patch per sequence).
      </p>
      <div className="plot-card">
        <img src={`${base}/patch-tokenization.png`} alt="Patch tokenization schematic" style={{ width: '100%', height: 'auto' }} />
      </div>
      <p className="note">
        Earlier eras are documented separately: v2 (P=14, 2-token density
        codec, vocab ~6.8 k, SHAPE/OFFSET/HI preamble blocks) is archived
        in{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/docs/2026-04-30-overview-snapshot.md">
          <code>docs/2026-04-30-overview-snapshot.md</code>
        </ExtLink>. Sample tokenized rows for any current material are
        at <code>data/tokenized/train-full-v3/</code>.
      </p>

      <h2>LMQ-v2 density codec</h2>
      <p>
        The codec is the bridge between continuous ρ(r) values and discrete
        token ids. <strong>LMQ-v2</strong> (Lloyd–Max on log-density) fits
        empirical bin centers from data so that 1 token per voxel captures
        most of the information; with 16,384 bins the median NMAE floor is{' '}
        <strong>0.18 %</strong>, well below current SOTA — meaning the
        codec isn't the limiting factor. Previous eras used 2-token codecs
        ("two_token_9_12") with vocab ~4.6 k; v3 collapses to 1 token to
        unlock larger patches (P=19) at fixed context.
      </p>
      <div className="plot-card">
        <img src={`${base}/lmq-codecs.png`} alt="LMQ codec comparison" style={{ width: '100%', height: 'auto' }} />
      </div>
      <p className="note">
        Background on the codec design lives in{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/docs/lmq-vs-equal-mass.md">
          <code>docs/lmq-vs-equal-mass.md</code>
        </ExtLink>. The 1-token codec also let us define a
        decoder-independent loss (<strong>EMD</strong> on the per-voxel
        bin distribution) and a decoder-independent eval metric
        (<strong>NEMD</strong>) that's directly comparable to
        point-estimate models like electrAI.
      </p>

      <h2>Patch geometry — cube vs ball</h2>
      <p>
        Ball patches (ablation against cubes, matched by voxel count)
        contain all integer voxels <code>(i, j, k)</code> with{' '}
        <code>i² + j² + k² ≤ R²</code>. Voxels-on-shell at{' '}
        <code>i² + j² + k² = n</code> is{' '}
        <ExtLink href="https://oeis.org/A005875"><code>r₃(n)</code></ExtLink>{' '}
        — sum-of-three-squares; cumulative <code>V(n) = Σ r₃(k)</code>
        is{' '}<ExtLink href="https://oeis.org/A117609">OEIS A117609</ExtLink>,
        approaching <code>⁴⁄₃ π R³</code>.
      </p>
      <div className="plot-card">
        <img src={`${base}/ball-voxel-counts.png`} alt="Ball voxel counts vs R²" style={{ width: '100%', height: 'auto' }} />
      </div>
      <p className="note">
        Ablation-matched thresholds: <code>R²=75 ≈ cube P=14</code>{' '}
        (2,777 vs 2,744 voxels), <code>R²=86 ≈ P=15</code>,{' '}
        <code>R²=138 = P=19</code> exact (6,859), <code>R²=153 ≈ P=20</code>.
        We default to cubes for the current v3 runs; balls remain an
        available ablation.
      </p>

      <h2>Materials Project metadata — atomic counts &amp; grid sizes</h2>
      <p>
        Distributions across <strong>77,466 materials</strong> in the
        {' '}<code>train-full</code> split.{' '}
        <strong>Median n_atoms = 12, p99 = 88, max = 154</strong>; the
        long tail drives the preamble budget and which P fits at each
        context length.
      </p>
      <MatsMetadataPlots url={`${base}/mats-md.csv`} />
      <div className="plot-card">
        <img src={`${base}/preamble-dist.png`} alt="Preamble length distribution" style={{ width: '100%', height: 'auto' }} />
      </div>
      <table className="runs-table">
        <thead>
          <tr><th>context length</th><th>cube P (100% / 99% / 95% kept)</th><th>ball R² (100% / 99% / 95%)</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>8 k</strong></td><td>14 / 15 / 15</td><td>85 / 89 / 92</td></tr>
          <tr><td><strong>16 k</strong></td><td>19 / 19 / 19</td><td>145 / 149 / 151</td></tr>
          <tr><td><strong>32 k</strong></td><td>24 / 25 / 25</td><td>240 / 243 / 244</td></tr>
        </tbody>
      </table>
      <p className="note">
        P=19 fits 8 k context with 100 % of materials kept; current runs
        target this. Larger P (24+) unlocked by 32 k+ context is a future
        scaling axis — codec floor at 0.18 % NMAE doesn't bind, so
        bigger-context experiments are about model capacity, not codec.
      </p>

      <h2>Up next</h2>
      <ul>
        <li>
          <strong>Multi-epoch run to characterize overfitting.</strong>{' '}
          <code>cont7k-ext</code> resumed with target step{' '}
          <strong>80 k</strong> (~4.1 epochs through unique data). Want
          to see at what point train/val diverge — current best is on
          ~1.7 epochs in, so overfitting hasn't been informative yet.
        </li>
        <li>
          <strong>Cluster diagnosis for v5p host stall.</strong> v5p-16
          should be ~4× more efficient than v6e-16 for this memory-bound
          workload (per-chip HBM bandwidth ratio), but recent v5p runs
          show ~65 s wall-clock per step against ~2.5 s JAX inner step —
          63 s of unaccounted host-side dead time. Diagnostic launch
          with py-spy attached is running; results will flow into the
          {' '}<a href="#/runs">runs dashboard</a> + a follow-up writeup.
        </li>
        <li>
          <strong>Scaling-law characterization.</strong> 1 B runs have
          been blocked by v6e-32 cache-build brittleness + v5p
          coordinator RPCs; reattempt once the v5p issues are
          understood.
        </li>
      </ul>

      <h2>History</h2>
      <p className="meta">
        Earlier eras (pre-LMQ, P=14 with 2-token "two_token_9_12" codec,
        30 M / 208 M / 1 B models reported only in token-space cross-
        entropy) are preserved at{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/docs/2026-04-30-overview-snapshot.md">
          <code>docs/2026-04-30-overview-snapshot.md</code>
        </ExtLink>{' '}with the original run table, scaling-loss plot, and
        narrative. Those runs lived in the{' '}
        <ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14">
          <code>tomat-two_token_9_12-P14</code>
        </ExtLink>{' '}W&amp;B project; current runs are in{' '}
        <ExtLink href={WB_PROJECT}><code>tomat-lmq-P19</code></ExtLink>.
      </p>

      <h2>Past presentations</h2>
      <ul>
        <li>
          <strong>2026-04-21 weekly meeting</strong> —{' '}
          <a href="#/deck">live deck</a> ·{' '}
          <a href={`${base}/2026-04-21-weekly-meeting-deck.pdf`}>PDF</a>.
          Covers the pivot to patch tokenization and reconstruction-floor
          context that motivated it (pre-LMQ era).
        </li>
      </ul>

      <div className="footer">
        Active specs:{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/21-eval-noise-and-extended-runs.md">21-eval-noise-and-extended-runs</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/22-wandb-workspace-layouting.md">22-wandb-workspace-layouting</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/23-runs-dashboard.md">23-runs-dashboard</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/inference-cli.md">inference-cli</ExtLink>.
        {' '}Living overview:{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/OVERVIEW.md">
          <code>OVERVIEW.md</code>
        </ExtLink>.
        {' '}Datasets reference:{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/docs/datasets.md">
          <code>docs/datasets.md</code>
        </ExtLink>.
      </div>
    </>
  )
}
