import { MatsMetadataPlots } from './MatsMetadataPlots'
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

      <h2>MP metadata — n_atoms / dims / voxels across train-full</h2>
      <p>
        Distributions across <strong>77,466 materials</strong> in the{' '}
        <code>train-full</code> split (same MP source as val-full; near-
        identical shape). Long-tail-of-atoms drives the preamble budget
        below.
      </p>
      <MatsMetadataPlots url={`${base}/mats-md.csv`} />
      <p className="note">
        Source: first row of each material's parquet preamble (extracted
        via <code>scripts/pull_preamble_stats_modal.py</code>; 77k-mat
        scan runs on Modal in &lt;90 s via the{' '}
        <code>tomat-gcp-sa</code> secret).
      </p>

      <h2>Preamble size + context-length budget</h2>
      <p>
        Each sequence is <code>preamble + density + EOS</code> tokens before
        padding. Preamble length scales with <strong>atom count N</strong>:
        {' '}<code>28 + 10·N</code> for cubes (<code>29 + 10·N</code> for balls,
        which trade <code>SHAPE+OFFSET+HI</code> for{' '}
        <code>RADIUS+CENTER+BOUNDS</code>). Density is{' '}
        <code>2 × V_patch</code> with our 2-token codec —{' '}
        <code>5,488</code> for P=14, <code>6,750</code> for P=15, etc.
      </p>
      <p>
        For the 77,466-material <code>train-full</code> corpus:
        <strong> median N = 12 atoms, p99 = 88, max = 154</strong>. The
        long-tail-of-atoms determines how many materials we drop at each
        patch-size × context-length combo:
      </p>
      <div className="plot-card">
        <img src={`${base}/preamble-dist.png`} alt="Preamble distribution" style={{ width: '100%', height: 'auto' }} />
      </div>
      <p>Largest P / R² that fits at each CL (fraction kept):</p>
      <table className="runs-table">
        <thead>
          <tr><th>context length</th><th>cube P (100% / 99% / 95%)</th><th>ball R² (100% / 99% / 95%)</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>8 k</strong></td><td>14 / 15 / 15</td><td>85 / 89 / 92</td></tr>
          <tr><td><strong>16 k</strong></td><td>19 / 19 / 19</td><td>145 / 149 / 151</td></tr>
          <tr><td><strong>32 k</strong></td><td>24 / 25 / 25</td><td>240 / 243 / 244</td></tr>
        </tbody>
      </table>
      <p className="note">
        <strong>Takeaway:</strong> P=15 at 8k context loses only{' '}
        <strong>6 mats out of 77,466</strong> (&lt;0.01%) — all at the
        very tail of the atom distribution (n_atoms ∈ {'{144, 152, 154}'},
        seq_len 8,218–8,318). P=16 at 8k doesn't fit at all. 16k unlocks
        up to P=19 (100% kept); 32k up to P=24. Ball thresholds are
        slightly bigger than cube-equivalent P because balls pack more
        voxels per bounding-cube than corners-inclusive cubes. The
        tokenizer skips overflow materials with a log message so we
        never crash mid-run.
      </p>

      <h2>Ball patches — geometry & math</h2>
      <p>
        Ball patches (ablation against cubes, matched by voxel count)
        contain all integer voxels <code>(i, j, k)</code> with{' '}
        <code>i² + j² + k² ≤ R²</code>. The number of voxels on the shell
        at exactly <code>i² + j² + k² = n</code> is{' '}
        <ExtLink href="https://oeis.org/A005875"><code>r₃(n)</code></ExtLink>{' '}
        — sum-of-three-squares — and the cumulative count{' '}
        <code>V(n) = Σ r₃(k)</code> is{' '}
        <ExtLink href="https://oeis.org/A117609">OEIS A117609</ExtLink>,
        asymptotically approaching the continuous ball volume{' '}
        <code>⁴⁄₃ π R³</code>.
      </p>
      <div className="plot-card">
        <img src={`${base}/ball-voxel-counts.png`} alt="Ball voxel counts vs R²" style={{ width: '100%', height: 'auto' }} />
      </div>
      <p className="note">
        Shell counts are spiky — <strong>32 values of n ≤ 200</strong>{' '}
        have <code>r₃(n) = 0</code> (integers of the form{' '}
        <code>4ᵃ(8b+7)</code> — Gauss, 1801). Cumulative counts tightly
        track <code>⁴⁄₃ π R³</code>. Ablation-matched thresholds:
        R²=75 ≈ cube P=14 (2,777 vs 2,744 voxels), R²=86 ≈ P=15,
        R²=138 = P=19 exact (6,859), R²=153 ≈ P=20.
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
            <td><strong>208M</strong></td><td>train-full</td><td>Marin TPU v6e-8</td><td>128 (16)</td><td>6 k</td><td><strong>6.29 B</strong></td><td><strong>15.55</strong></td><td>9.86%</td><td>293 k</td><td><strong>1.661</strong> (eval 1.683)</td>
          </tr>
          <tr>
            <td><ExtLink href="https://wandb.ai/PrinceOA/tomat-two_token_9_12-P14/runs/train-full-tpu16-1B-bs128-val-bf16-seed42"><strong>TPU v6e-16 bs=128</strong> (1B, multihost)</ExtLink></td>
            <td><strong>1B</strong></td><td>train-full</td><td>Marin TPU v6e-16 (4 hosts)</td><td>128 (8)</td><td>4 k</td><td>4.19 B</td><td><strong>43.20</strong></td><td><strong>17.53%</strong></td><td>250 k</td><td><strong>1.524</strong> (eval 1.537)</td>
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
        <strong>Larger model.</strong> 208 M Qwen3 (hidden=1024, 12 layers,
        16 heads, tied embeddings, bf16 compute) with a real val split
        (256 held-out sequences). Extended to <strong>loss 1.661
        on 6.29 B tokens</strong> (eval 1.683 / BPB 0.595) — 0.55 nats
        below the 30 M baseline at similar tokens. 15.55 EF compute.
      </p>
      <p className="meta">
        <strong>1 B scale.</strong> Qwen3 1 B (hidden=2048, 20 layers,
        16 heads, 5632 ffn) on v6e-16 multihost, 4 B tokens at bs=128
        → <strong>loss 1.524 (eval 1.537)</strong>. 0.137 nats better
        than 208 M on <em>half</em> the tokens — strong scaling signal.
        MFU jumped to <strong>17.5 %</strong> (vs 9.9 % at 208 M and
        ~8–10 % at 30 M), confirming the small-model-under-saturates-
        chip hypothesis. 1 B at 4 tokens/param is ~5× under Chinchilla-
        optimal, so there's still clean "more tokens" headroom.
      </p>
      <p className="note">
        <strong>NB:</strong> the train/eval <em>loss</em> and <em>BPB</em>{' '}
        numbers above are <strong>token-space cross-entropy</strong> — NOT
        directly comparable to electrAI / charg3net's voxel-space{' '}
        <strong>NMAE</strong> (2% / 0.5%). To produce a comparable NMAE we
        need to greedy-decode the density-codec tokens and compare decoded
        floats against ground truth; that eval infrastructure is in
        progress (spec 17).
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
