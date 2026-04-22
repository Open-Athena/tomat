import { SmokeLossPlot } from './SmokeLossPlot'
import { ThemeToggle } from './theme'

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
}

export function HomePage() {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')

  return (
    <>
      <header>
        <h1>tomat 🍅 — tokenized materials</h1>
        <ThemeToggle />
      </header>

      <p className="meta">
        LLM-based electron-density prediction for periodic crystals. Transformer
        alternative to electrAI's 3D ResUNet — we tokenize ρ(r) directly and train
        a sequence model over the resulting token stream.{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat">Open-Athena/tomat</ExtLink>.
      </p>

      <h2>Patch tokenization</h2>
      <p>
        Each training example is one <code>P × P × P</code> sub-cube of a material's
        native-resolution density, prefixed with:
      </p>
      <ul>
        <li>The full grid shape <code>(nx, ny, nz)</code>.</li>
        <li>The material's atomic inventory (Z + fractional coordinates).</li>
        <li>The patch's low-corner anchor <code>(ix, iy, iz)</code>, its shape
          <code> (P, P, P)</code>, and the wrapped <strong>high corner</strong>{' '}
          <code>(hx, hy, hz) = (ix+P−1) mod nx</code>. On any axis where
          <code> hi &lt; lo</code> the patch crossed a PBC boundary — the model
          sees that as a direct observation rather than having to learn modular
          arithmetic.</li>
      </ul>
      <p>
        At <code>P = 14</code> with a 2-token-per-voxel density codec, each
        sequence is <code>14³ × 2 = 5,488</code> density tokens plus a ~200-token
        preamble — fits 8k context with headroom for a 100-atom structure. Vocab
        is 6,792 tokens total (18 specials + 118 atomic Z + 1,024 ints +
        1,024 position-codec + 4,608 density-codec).
      </p>

      <h2>Smoke training (30 M Qwen3, Modal A100)</h2>
      <div className="plot-card">
        <SmokeLossPlot url={`${base}/smoke-loss.csv`} />
      </div>
      <p className="meta">
        128 val-split materials × 32 random patches = 4,096 training sequences;
        6-layer Qwen3, hidden=512, tied embeddings, 8k context, batch 8, seed 42.
        Run tracked on{' '}
        <ExtLink href="https://wandb.ai/PrinceOA/tomat/runs/q824om6c">W&amp;B: rosy-durian-1</ExtLink>{' '}
        · provenance in{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/results/smoke.dvc">
          <code>results/smoke.dvc</code>
        </ExtLink>.
      </p>

      <h2>Up next</h2>
      <ul>
        <li><strong>(codec × patch_size × M × N) sweep</strong> — preprocess &amp;
          training grids already defined; 6 of 9 (codec, P) combos fit 8k context.</li>
        <li><strong>Marin + GCP TPU Research Cluster</strong> — real training
          run on v5p-8; current Modal A100 setup is the local smoke.</li>
        <li><strong>Full train split</strong> — smoke uses val-only (4,305
          structures, ~22 GB on the <code>tomat-rho-gga</code> Modal volume);
          train split adds 77 k structures / ~390 GB.</li>
      </ul>

      <h2>Past presentations</h2>
      <ul>
        <li>
          <strong>2026-04-21 weekly meeting</strong> —{' '}
          <a href="#/deck">live deck</a> ·{' '}
          <a href={`${base}/2026-04-21-weekly-meeting-deck.pdf`}>PDF</a>.
          Covers the pivot to patch tokenization and reconstruction-floor
          context that motivated it.
        </li>
      </ul>
      <p className="note">
        The live <a href="#/deck">deck</a> is a frozen snapshot of the most
        recent talk; check the PDF for the exact version shown at a given
        meeting.
      </p>

      <div className="footer">
        Specs:{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/04-patch-training.md">04-patch-training</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/05-modal-smoke.md">05-modal-smoke</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/06-oa-react-slides.md">06-oa-react-slides</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/done/02-fidelity-sweep.md">02-fidelity-sweep (done)</ExtLink>.
      </div>
    </>
  )
}
