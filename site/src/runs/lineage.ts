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
}

/** Lineage entry for `runName`, or `null` if it has no recorded parent. */
export function lineageFor(runName: string): RunLineage | null {
  return RUN_LINEAGE[runName] ?? null
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
