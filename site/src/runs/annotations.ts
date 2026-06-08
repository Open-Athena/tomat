// Per-run plot annotations.
//
// Hand-curated markers that get rendered as vertical lines + small labels on
// the run-detail plot, calling out moments where something changed —
// typically a config tweak (LR, BS, schedule, data) or a deliberate
// intervention worth distinguishing from organic training dynamics.
//
// Keyed by the canonical full run name. Steps refer to the **concatenated
// lineage** trajectory (i.e. global step across the whole run-and-its-
// ancestors), so an annotation at `step: 40000` on a child run shows up
// at the post-resume point on the unified plot.
//
// To add new annotations: edit this file directly. The author signals their
// intent ("here's what changed") in `label`; the plot just renders vlines
// + labels.

export interface RunAnnotation {
  step: number
  label: string
  // Optional color override; defaults to a muted highlight color.
  color?: string
}

export const RUN_ANNOTATIONS: Record<string, RunAnnotation[]> = {
  // v4-modal lineage (TS-shard experimentation on Modal H200×8 MaskGIT).
  // v4 ran 0 → 40k; v4-epochwin took over from step-40k and is the live
  // continuation. The lineage glue (lineage.ts) concatenates v4's history
  // into v4-epochwin's plot, so step-N here means "step-N on the unified
  // trajectory" — which matches global_step values from wandb.
  'train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42': [
    {
      step: 40000,
      label: 'v4-epochwin smoke: shuffle 1024→78k blocks',
    },
    {
      step: 52000,
      label: 'cosine→constant LR (3.8e-6→3e-5, ~8× warm-restart spike)',
    },
    {
      step: 76186,
      label: 'TS0→TS1, BS 128→256, LR 3e-5→4e-5',
    },
  ],
  // 4-way LF ablation, KL-Gauss(σ=0.5) arm. Three consecutive resumes, each
  // re-extending num_train_steps → cosine schedule recomputed over the new
  // horizon. First two are pure "stretch the cosine"; the third (9k) also
  // flips to constant LR + epochwin shuffle (sawtooth-removal cutover).
  'train-mg-modal-h200x8-tz-fs-kl-s05-ts1-bs128-seed42': [
    {
      step: 1000,
      label: 'smoke (1k) → 6k extension',
    },
    {
      step: 6000,
      label: '6k → 20k extension; cosine re-stretched → big TL spike',
    },
    {
      step: 9190,
      label: 'cosine→constant LR + shuffle 1024→77568 (epochwin)',
    },
  ],
}

/** Annotations for `runName`, or empty list. */
export function annotationsFor(runName: string): RunAnnotation[] {
  return RUN_ANNOTATIONS[runName] ?? []
}
