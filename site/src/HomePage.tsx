import { ScalingLossPlot } from './ScalingLossPlot'
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

      <h2>Scale training runs (2026-04-22 / 23)</h2>
      <p>
        Same 30 M Qwen3 on <code>val-full</code> (4,305 mats × 32 patches = 137,696
        sequences), seed 42, 8k context. Runs live in the{' '}
        <ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14">
          <code>tomat-two_token_9_12-P14</code>
        </ExtLink>{' '}W&amp;B project.
      </p>
      <div className="plot-card">
        <ScalingLossPlot baseUrl={`${base}/run-histories`} />
      </div>
      <table className="runs-table">
        <thead>
          <tr><th>run</th><th>compute</th><th>bs (per-dev)</th><th>steps</th><th>MFU</th><th>tok/s</th><th>final loss</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-bs32-seed42">bs=32, A100:1</ExtLink></td>
            <td>Modal A100:1</td><td>32 (32)</td><td>2,560 / 5 k</td><td>12.4%</td><td>80 k</td><td>2.235 (OOM)</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-2gpu-bs32-seed42">bs=32, A100:2</ExtLink></td>
            <td>Modal A100:2</td><td>32 (16)</td><td>5 k</td><td>12.0%</td><td>157 k</td><td><strong>1.962</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs64-4gpu-bs64-seed42">bs=64, A100:4</ExtLink></td>
            <td>Modal A100:4</td><td>64 (16)</td><td>5 k</td><td>11.96%</td><td>313 k</td><td><strong>1.975</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs128-8gpu-bs128-seed42">bs=128, A100:8</ExtLink></td>
            <td>Modal A100:8</td><td>128 (16)</td><td>5 k</td><td>11.86%</td><td>624 k</td><td><strong>2.022</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-tpu-bs128-seed42">bs=128, TPU v6e-4</ExtLink></td>
            <td>Marin TPU v6e-4</td><td>128 (32)</td><td>1 k</td><td>10.25%</td><td><strong>792 k</strong></td><td>2.620</td>
          </tr>
        </tbody>
      </table>
      <p className="meta">
        Near-perfect data-parallel scaling across A100:2/4/8 at fixed
        per-device bs=16 (157 k → 313 k → 624 k tok/s = 2.0× per doubling,
        MFU stable ~12%). TPU v6e-4 ≈ <strong>10× A100:1 tok/s</strong>
        at same per-device batch — matching the hardware FLOPs ratio
        (v6e: 918 TFLOPs/chip × 4 = 12× an A100). A100:1 bs=32
        per-device=32 OOMed at step 2,560 on a 22-GiB attention-matrix
        allocation; per-device=16 fits comfortably.
      </p>

      <h2>Up next</h2>
      <ul>
        <li><strong>Full train split</strong> — 77 k structures / ~390 GB
          upload from della in progress ({' '}
          <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/08-della-seed-train-split.md">
            spec 08
          </ExtLink>); parallel tokenize + training-scale run per{' '}
          <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/07-full-train-scale.md">
            spec 07
          </ExtLink>.</li>
        <li><strong>(codec × patch_size × M × N) sweep</strong> — preprocess &amp;
          training grids already defined; 6 of 9 (codec, P) combos fit 8k context.</li>
        <li><strong>DVX-track val-full + raw zarrs</strong> (<ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/09-dvx-track-val-full.md">spec 09</ExtLink>)
          — drift-detection manifests after we caught a silent 1-in-80 GCS
          upload corruption via a ZSTD decompression error.</li>
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
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/07-full-train-scale.md">07-full-train-scale</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/08-della-seed-train-split.md">08-della-seed-train-split</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/09-dvx-track-val-full.md">09-dvx-track-val-full</ExtLink>,{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/specs/done/02-fidelity-sweep.md">02-fidelity-sweep (done)</ExtLink>.
        {' '}Datasets reference:{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/docs/datasets.md">
          <code>docs/datasets.md</code>
        </ExtLink>.
      </div>
    </>
  )
}
