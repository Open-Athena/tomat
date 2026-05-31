// % training: how much of the iris job's wallclock was spent in actual
// training (vs. waiting in queue, building images, JIT-compiling,
// crash-restarting, etc.).
//
// Formula:
//   total_wallclock = (finished_at | now) - submitted_at
//   training_seconds = sum over attempts t of:
//       overlap(attempt_window, training-active history span)
//     where the "training-active span" is the (timestamps[i_min], timestamps[i_max])
//     range of rows whose `train/loss` is non-null in the parquet.
//
// This is a deliberate approximation, not an exact measurement. Iris doesn't
// log "trainer process actually running with loss decreasing" — only the
// per-attempt (started_at, finished_at) window from the worker's POV, which
// includes startup + JIT-compile time before the first batch. We refine that
// by intersecting with the train/loss row span — a run that hit one train
// step had training-active wallclock, one that died during JIT did not.
//
// The training_seconds sum still slightly OVER-counts when an attempt is
// still in compile-and-startup at its end (we credit it through finished_at).
// But that's a small fraction vs the queue/build overhead the chip is meant
// to surface (typically the dominant non-training cost on iris).

import type { IrisAttempts, IrisJob } from './api'
import type { RunHistory } from './parquet'

export interface TrainingFraction {
  /** 0..1, or null if we can't compute (no job submission ts / no history). */
  pct: number | null
  /** Total wallclock from submission → now/finish, in seconds. */
  totalSec: number | null
  /** Training-active seconds (per the overlap with parquet train/loss span). */
  trainSec: number | null
  /** True if the run is still active (finished_at_ms == null) — caller may
   *  want to display "training: N% (so far)". */
  inProgress: boolean
}

/** First/last wallclock timestamp (seconds, epoch) where `train/loss` is set
 *  in the parquet. Returns `null` when there are no training rows. */
function trainingActiveSpan(history: RunHistory | null): { lo: number; hi: number } | null {
  if (!history) return null
  const col = history.cols.get('train/loss') ?? []
  let lo = Infinity, hi = -Infinity
  for (let i = 0; i < col.length; i++) {
    const v = col[i]
    if (v === null || v === undefined) continue
    const ts = history.timestamps[i]
    if (ts === null) continue
    if (ts < lo) lo = ts
    if (ts > hi) hi = ts
  }
  if (lo === Infinity) return null
  return { lo, hi }
}

export function computeTrainingFraction(
  job: IrisJob | null,
  attempts: IrisAttempts | null,
  history: RunHistory | null,
): TrainingFraction {
  const submittedMs = job?.submitted_at_ms ?? attempts?.submitted_at_ms ?? null
  const finishedMs = job?.finished_at_ms ?? attempts?.finished_at_ms ?? null
  const endMs = finishedMs ?? Date.now()
  const totalSec = submittedMs != null ? (endMs - submittedMs) / 1000 : null

  const span = trainingActiveSpan(history)
  if (totalSec == null || totalSec <= 0 || span == null || attempts == null) {
    return { pct: null, totalSec, trainSec: null, inProgress: finishedMs == null }
  }

  // Sum, over every attempt that has a started_at_ms, the overlap of
  // (started, finished|now) ∩ (span.lo, span.hi).
  let trainSec = 0
  for (const t of attempts.tasks) {
    for (const a of t.attempts) {
      if (a.started_at_ms == null) continue
      const aStart = a.started_at_ms / 1000
      const aEnd = a.finished_at_ms != null ? a.finished_at_ms / 1000 : endMs / 1000
      const lo = Math.max(aStart, span.lo)
      const hi = Math.min(aEnd, span.hi)
      if (hi > lo) trainSec += hi - lo
    }
  }
  const pct = Math.max(0, Math.min(1, trainSec / totalSec))
  return { pct, totalSec, trainSec, inProgress: finishedMs == null }
}

/** "3h 21m" / "47m" / "12s" — terse duration formatting for the chip. */
export function formatDurationShort(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.round(sec / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60), mm = m % 60
  return mm > 0 ? `${h}h ${mm}m` : `${h}h`
}
