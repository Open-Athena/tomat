import { ScalingLossPlot } from './ScalingLossPlot'
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
      <p>
        Real row from <code>train-full</code> —{' '}
        <ExtLink href="https://elvis.oa.dev/?m=mp-2282417">
          <code>mp-2282417</code>
        </ExtLink>{' '}(Y₃Si₃Ag₃, grid 64×108×108), P=14 patch at offset (5, 9, 44):
      </p>
      <pre className="tokens-example">{`[BOS]
[GRID_START]   64 108 108                             [GRID_END]
[ATOMS_START]  Y Y Y Si Si Si Ag Ag Ag                [ATOMS_END]
[POS_START]    (p236 p699 p1003  p240 p767 p1005  p0 p512 p768)  …  (+7 more atoms)  [POS_END]
[SHAPE_START]  14 14 14                               [SHAPE_END]
[OFFSET_START] 5 9 44                                 [OFFSET_END]
[HI_START]     18 22 57                               [HI_END]
[DENS_START]   d172 d909  d169 d4175  d168 d525  …  d158 d2204    # 5,488 density tokens = 2 × 14³
[DENS_END]
[EOS]
[PAD] × 2,586                                         # right-padded to 8,192`}</pre>
      <p className="note">
        Atom Zs render as element symbols (<code>Y</code>, <code>Si</code>,
        <code>Ag</code>). Position codec = 3 tokens/coord × 3 coords → 9 tokens/atom.
        Density codec emits 2 tokens per voxel. Helper at{' '}
        <ExtLink href="https://github.com/Open-Athena/tomat/blob/main/scripts/show_tokens.py">
          <code>scripts/show_tokens.py</code>
        </ExtLink>{' '}renders any parquet row in this form.
      </p>

      <h2>Tokenized datasets</h2>
      <p>
        All <code>two_token_9_12</code> density codec, P=14, pad_to=8192, seed 42.
        "Tokens (pad)" counts padded positions (what the model trains on at each
        step); actual non-pad content is ~5,700 tokens for a 9-atom patch (see above).
      </p>
      <table className="runs-table">
        <thead>
          <tr><th>label</th><th>split</th><th>mats</th><th>patches/mat</th><th>rows</th><th>tokens (pad)</th><th>on-disk (GCS)</th><th>notes</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><code>val-smoke-n2</code></td><td>val</td><td>2</td><td>32</td><td>64</td><td>524 K</td><td>— (local)</td><td>throwaway</td>
          </tr>
          <tr>
            <td><code>val-smoke</code></td><td>val</td><td>128</td><td>32</td><td>4,096</td><td>34 M</td><td>~33 MB</td><td>earliest smoke target</td>
          </tr>
          <tr>
            <td><code>val-full</code></td><td>val</td><td>4,305</td><td>32</td><td>137,696</td><td>1.13 B</td><td>1.49 GB</td><td>"4 k mats" — primary compute-scaling target</td>
          </tr>
          <tr>
            <td><code>val-full-m128</code></td><td>val</td><td>4,305</td><td>128</td><td>549,664</td><td>4.50 B</td><td>1.44 GB</td><td>4× more unique patches/mat</td>
          </tr>
          <tr>
            <td><strong><code>train-full</code></strong></td><td>train</td><td><strong>77,498</strong></td><td>32</td><td><strong>2,478,912</strong></td><td><strong>20.31 B</strong></td><td><strong>21.1 GB</strong></td><td>first run 2026-04-23 (headline dataset)</td>
          </tr>
        </tbody>
      </table>
      <p className="note">
        Raw densities: 86,192 Zarr directories on Princeton della at{' '}
        <code>/scratch/gpfs/ROSENGROUP/.../rho_gga/</code>
        {' '}(~412 GB total; ~5 MB / structure mean). Staged onto two Modal volumes —{' '}
        <code>tomat-rho-gga</code> (val, 22 GB) and <code>tomat-rho-gga-train</code>
        {' '}(train, 370 GB) — where the tokenize pipeline runs and emits parquet,
        which then syncs to <code>gs://marin-eu-west4/tomat/tokenized/</code>.
      </p>

      <h2>Scale training runs (2026-04-22 / 23)</h2>
      <p>
        Seed 42, 8k context, patch P=14. First 5 rows on <code>val-full</code>
        {' '}(4,305 mats × 32 patches = 137,696 sequences, ~1.1 B tokens) with
        30 M Qwen3. Later rows on <code>train-full</code> (77,498 mats ×
        32 patches = 2.48 M sequences, ~20 B tokens, 18× more) — the last of
        which swaps in a <strong>208 M Qwen3</strong> (hidden=1024, 12 layers,
        16 heads, bf16 compute + val). Click a legend entry to solo-highlight
        that trace; click again or outside the legend to unpin. Runs live in
        the{' '}
        <ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14">
          <code>tomat-two_token_9_12-P14</code>
        </ExtLink>{' '}W&amp;B project.
      </p>
      <p className="note">
        Note on dataset naming: despite the <code>val-full</code> label, we
        use it as compute-scaling <em>training</em> data — it's MP's <em>validation</em>
        {' '}split (~4 k mats); it was seeded to Modal first and smaller, so it was
        the natural early target. <code>train-full</code> (~77 k mats) is the
        proper train split. "4 k mats" and "77 k mats" are the semantic
        descriptions if the val/train labels get confusing.
      </p>
      <div className="plot-card">
        <ScalingLossPlot baseUrl={`${base}/run-histories`} />
      </div>
      <table className="runs-table">
        <thead>
          <tr><th>run</th><th>model</th><th>data</th><th>compute</th><th>bs (per-dev)</th><th>steps</th><th>tokens</th><th>FLOPs (×10¹⁸)</th><th>MFU</th><th>tok/s</th><th>final loss</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-bs32-seed42">A100:1 bs=32</ExtLink></td>
            <td>30M</td><td>val-full</td><td>Modal A100:1</td><td>32 (32)</td><td>2,560/5k (OOM)</td><td>0.67 B</td><td>0.32</td><td>12.4%</td><td>80 k</td><td>2.235</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs32-2gpu-bs32-seed42">A100:2 bs=32</ExtLink></td>
            <td>30M</td><td>val-full</td><td>Modal A100:2</td><td>32 (16)</td><td>5 k</td><td>1.31 B</td><td>0.62</td><td>12.0%</td><td>157 k</td><td><strong>1.962</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs64-4gpu-bs64-seed42">A100:4 bs=64</ExtLink></td>
            <td>30M</td><td>val-full</td><td>Modal A100:4</td><td>64 (16)</td><td>5 k</td><td>2.62 B</td><td>1.25</td><td>11.96%</td><td>313 k</td><td>1.975</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-5k-bs128-8gpu-bs128-seed42">A100:8 bs=128</ExtLink></td>
            <td>30M</td><td>val-full</td><td>Modal A100:8</td><td>128 (16)</td><td>5 k</td><td>5.24 B</td><td>2.49</td><td>11.86%</td><td>624 k</td><td>2.022</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/val-full-tpu-bs128-seed42">TPU v6e-4 bs=128</ExtLink></td>
            <td>30M</td><td>val-full</td><td>Marin TPU v6e-4</td><td>128 (32)</td><td>1 k</td><td>1.05 B</td><td>0.50</td><td>10.25%</td><td>792 k</td><td>2.620</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-bs256-seed42"><strong>TPU v6e-8 bs=256</strong></ExtLink></td>
            <td>30M</td><td><strong>train-full</strong></td><td>Marin TPU v6e-8</td><td>256 (32)</td><td>2 k</td><td><strong>4.19 B</strong></td><td><strong>2.00</strong></td><td>8.38%</td><td>1,297 k</td><td><strong>2.214</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu16-30M-bs512-seed42"><strong>TPU v6e-16 bs=512</strong> (multihost)</ExtLink></td>
            <td>30M</td><td>train-full</td><td>Marin TPU v6e-16 (4 hosts)</td><td>512 (32)</td><td>2 k</td><td><strong>8.39 B</strong></td><td><strong>4.00</strong></td><td>6.6%</td><td><strong>1,983 k</strong></td><td><strong>2.212</strong></td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu8-200M-bs128-val-bf16-seed42"><strong>TPU v6e-8 bs=128</strong> (+ val, bf16)</ExtLink></td>
            <td><strong>208M</strong></td><td>train-full</td><td>Marin TPU v6e-8</td><td>128 (16)</td><td>1,147 / 2 k</td><td>1.20 B</td><td>2.98</td><td>9.9%</td><td>294 k</td><td>2.261 (live)</td>
          </tr>
        </tbody>
      </table>
      <p className="meta">
        Near-perfect data-parallel scaling across A100:2/4/8 at fixed
        per-device bs=16 (157 k → 313 k → 624 k tok/s, MFU stable ~12%).
        TPU v6e-4 ≈ <strong>10× A100:1</strong> tok/s at same per-device
        batch (v6e: 918 TFLOPs/chip × 4 ≈ 12× an A100).
      </p>
      <p className="meta">
        <strong>train-full + larger compute.</strong> Same 30 M model but
        <strong> 18× more training data</strong> (77 k materials vs 4,305) →
        0.41 nats lower loss (2.62 → 2.21). On v6e-8 that's 1.3 M tok/s;
        the multihost v6e-16 stretch (4 VMs × 4 chips) adds another 1.57×
        throughput to <strong>2.04 M tok/s</strong>. MFU at 30 M stays low
        (8–10%) — the model's too small to saturate the chips. 4.2 B tokens
        through 30 M is <strong>~7× past Chinchilla-optimal</strong>.
      </p>
      <p className="meta">
        <strong>Larger model.</strong> Parallel run with a 200 M Qwen3
        (hidden=1024, 12 layers, 16 heads, tied embeddings) on the same
        train-full — right in Chinchilla's zone for 4 B tokens and
        exercising the TPU's native bf16 compute properly. First real
        validation split (256 held-out sequences) is wired in on this
        run too, so we'll have a generalization number alongside train
        loss for the first time.
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
