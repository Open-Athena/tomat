// Canonical run-id fixtures used across the e2e suite. These are picked
// for stable data shape — see site/tests/e2e/README.md "adding a fixture
// run" for the criteria.
//
// Update guidance: if any of these gets purged from the backend or the
// data shape changes (lineage parent, segment count, eval set keys), bump
// the entry here AND grep the spec file mentioning it so the assertion
// thresholds are revisited.

export const FIXTURES = {
  /** Finished Modal H200×8 run, full summary fields populated, single
   *  segment (0 resumes). Used as the "happy path" baseline for
   *  card-header field coverage + per-x-mode rendering. */
  modalFinishedFull: 'train-mg-modal-h200x8-tz-fs-kl-s05-ts1-bs128-seed42',

  /** Finished iris run with 7 trainer_started events (multi-segment) AND
   *  a lineage parent registered in `RUN_LINEAGE`. Drives the lineage-glue
   *  test + the multi-segment phantom-bridge regression check. */
  irisLineageGlued: 'kl-s05-tpu-cont-v3',

  /** Running iris run with 4 segments, NOT registered in `RUN_LINEAGE`.
   *  Drives the 4-segment phantom-bridge assertion (target for the dotted
   *  gap-bridge cap). */
  irisRunningFourSeg: 'train-mg-kl-bin5-fs-tpu',

  /** Modal run with `eval.json` populated with all 4 set keys
   *  (val_200, train_200, val_200-maskgit, train_200-maskgit). Drives the
   *  MEvalTable assertions. */
  modalMEvalFull: 'train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42',
} as const

export type FixtureRunId = (typeof FIXTURES)[keyof typeof FIXTURES]
