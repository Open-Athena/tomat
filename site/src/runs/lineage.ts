// Parent/child run lineage for the /runs dashboard.
//
// Tomat training runs frequently resume from a parent run's checkpoint via
// `tomat train --from-ckpt gs://…/<parent>/checkpoints/<parent>/step-N`,
// which pre-copies the parent ckpt into the new run's output dir so
// Levanter auto-resumes natively. The downside is that the parent
// reference never lands in the wandb run config — so we maintain it
// here, by hand, keyed off the canonical full run name.
//
// `parent_step` is the step-N suffix the parent's ckpt was at when the
// child resumed from it. Levanter's "step-N" means "step index N just
// completed" (0-indexed), so a parent that finished its `num_train_steps=S`
// produces a final `step-(S-1)` ckpt. That's why SS-* runs resume at
// step-79999 (cont33k's `num_train_steps=80000` → `step-79999` final ckpt).
//
// Leave `parent_step` undefined when it's not clearly known; the UI will
// still show the parent link (without the `@ step-N` tail).

export interface RunLineage {
  parent: string
  parent_step?: number
}

export const RUN_LINEAGE: Record<string, RunLineage> = {
  // cont7k-ext resumed from cont7k. The exact step is unclear from the name
  // (cont7k targeted 7000 steps but cont7k-ext's parent_step isn't recorded
  // in our specs/postmortems); leaving undefined.
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k-ext': {
    parent: 'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k',
  },

  // cont33k resumed from cont7k-ext at the intact-P19 step-33000 ckpt
  // (see specs/24-scaling-laws.md "Anchor: 200M P19 resume from step-33000").
  'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k': {
    parent: 'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k-ext',
    parent_step: 33000,
  },

  // SS sweep: all 6 fine-tune off cont33k's final step-79999 ckpt
  // (cont33k targeted `num_train_steps=80000` → final ckpt is step-79999).
  'train-ss-cont80k-emax025-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },
  'train-ss-cont80k-emax050-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },
  'train-ss-cont80k-emax075-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },
  'train-ss-cont80k-emax100-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },
  'train-ss-cont80k-eps1const-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },
  'train-ss-cont80k-hi-argmax-1': {
    parent: 'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k',
    parent_step: 79999,
  },

  // mg-4 extensions: both resume from mg-4-cos-ce at step-9999
  // (mg-4-cos-ce targeted num_train_steps=10000 → final ckpt step-9999).
  'train-mg-4-r1-ext': {
    parent: 'train-mg-4-cos-ce',
    parent_step: 9999,
  },
  'train-mg-4-cos-ext': {
    parent: 'train-mg-4-cos-ce',
    parent_step: 9999,
  },

  // v4-epochwin resumed v4 at step-40000 (Modal H200×8 MaskGIT). v4 ran 0 →
  // 40000; v4-epochwin picked up at 40001 and is targeting 90000. Internally
  // the epochwin label was actually two consecutive wandb runs (smoke
  // 40k-42k + cont 42k-90k) but they collapse into one results_label /
  // checkpoint stream, so the dashboard sees a single child card.
  'train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42': {
    parent: 'train-mg-modal-h200x8-tz-v4-bs128-seed42',
    parent_step: 40000,
  },

  // tz-11-bs256: BS=256 ablation forked from train-mg-tz-11 at step-20000
  // (warm-start). The TPU script ties wandb run_id to results_label, so we
  // get a fork at the wandb level; lineage glue here makes the dashboard
  // show one continuous trajectory through the BS bump.
  'train-mg-tz-11-bs256': {
    parent: 'train-mg-tz-11',
    parent_step: 20000,
  },

  // kl-s05-tpu-cont-v3: TPU resume of the KL-Gauss σ=0.5 Modal run at
  // step-20000 (its final ckpt; the from-scratch ts1 Modal run targeted
  // num_train_steps=20000).
  'kl-s05-tpu-cont-v3': {
    parent: 'train-mg-modal-h200x8-tz-fs-kl-s05-ts1-bs128-seed42',
    parent_step: 20000,
  },
}

/** Lineage entry for `runName`, or `null` if it has no recorded parent. */
export function lineageFor(runName: string): RunLineage | null {
  return RUN_LINEAGE[runName] ?? null
}

/** Short relationship label for an ancestor at `partIdx` in a `lineageInfo`
 *  of length `numAncestors`. `lineageInfo` is root→parent; the LAST entry
 *  is the immediate parent. Depth-from-current = `numAncestors - partIdx`:
 *  depth 1 → `'parent'`, 2 → `'grandparent'`, 3 → `'great-grandparent'`,
 *  ≥4 → `'ancestor #N'` (avoids piling on "great-"s past 3).
 *
 *  Used for the WallclockPlot legend's per-ancestor group title — the
 *  previous `ancestor: <full-name>` group title dominated plot width on
 *  multi-ancestor runs. The full ancestor name lives on hover (the trace's
 *  hovertemplate appends `· <relation> (<full-name>)`).
 */
export function ancestorRelation(partIdx: number, numAncestors: number): string {
  const depth = numAncestors - partIdx  // 1 = immediate parent, 2 = grandparent, …
  if (depth <= 0) return 'ancestor'
  if (depth === 1) return 'parent'
  if (depth === 2) return 'grandparent'
  if (depth === 3) return 'great-grandparent'
  return `ancestor #${depth}`
}

// ── Run segments (data-label / batch-size cutovers within one run) ──────────
//
// A "segment" is a half-open `[startStep, endStep)` slice of a run during
// which the data label and batch size were both constant. Used by
// `epochOfStep` (runMeta.ts) so the epoch coordinate is correct for runs
// that changed config mid-stream (e.g. shard cutover, BS bump).
//
// Conventions:
//   • Segments are contiguous and sorted by `startStep`.
//   • First segment's `startStep` is the run's first step (typically 0).
//   • Trailing segment's `endStep` is `Infinity` ("continues to the latest
//     step we see"), so we don't have to bump this table every time the run
//     logs new rows.
//   • `dataLabel` must be a key of `EPOCH_SEQUENCES` (runMeta.ts) — unknown
//     labels make `epochOfStep` return `null` rather than guess.
//
// This is NOT auto-derived from `annotations.ts`: annotations are display-only
// labels (e.g. "TS0", "TS1"), segments are arithmetic-bearing config. Author
// keeps them in lockstep when both apply.
export interface RunSegment {
  /** Inclusive lower bound. Typically 0 or the prior segment's `endStep`. */
  startStep: number
  /** Exclusive upper bound. `Infinity` means "until the latest step". */
  endStep: number
  /** Key into `EPOCH_SEQUENCES` — the tokenization-cache label active during
   *  this segment. Unknown labels make `epochOfStep` return `null`. */
  dataLabel: string
  /** `trainer.train_batch_size` active during this segment. */
  batchSize: number
}

export const SEGMENTS: Record<string, RunSegment[]> = {
  // v4-epochwin: TS0 trained on `train-full-v3` at BS=128 through step 76186,
  // then TS1 cut over to the re-seeded `train-full-v3-shard1` at BS=256.
  // Without this entry the manifest's sync-time (BS, label) drove epoch
  // math for the whole run, over-reporting by ~1.7× at the current step.
  // Per-segment numbers at step 90,982 (spec 54): 1.97 + 0.76 = 2.73.
  'train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42': [
    { startStep: 0,     endStep: 76186,    dataLabel: 'train-full-v3',        batchSize: 128 },
    { startStep: 76186, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
  ],
}

/** Segments for `runName`, or `null` if none are declared. */
export function segmentsFor(runName: string): RunSegment[] | null {
  return SEGMENTS[runName] ?? null
}

/** Walk the lineage map upward from `runName`, collecting every ancestor
 *  (parent, grandparent, …). Stops at the root or a cycle (defensive — there
 *  shouldn't be any). The starting `runName` itself is NOT included. */
export function ancestorsOf(runName: string): string[] {
  const out: string[] = []
  const seen = new Set<string>([runName])
  let cur: string | undefined = runName
  while (cur !== undefined) {
    const lin: RunLineage | undefined = RUN_LINEAGE[cur]
    if (!lin) break
    if (seen.has(lin.parent)) break  // cycle guard
    seen.add(lin.parent)
    out.push(lin.parent)
    cur = lin.parent
  }
  return out
}
