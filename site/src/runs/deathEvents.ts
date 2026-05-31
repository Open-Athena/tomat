// Classifies an iris attempt's death by its (state, error, is_worker_failure)
// triple, into one of four buckets we render distinctly on the WallclockPlot.
//
// Bucket → color choices:
//   - preempt  (purple) → genuine GCP TPU preempt  (state='preempted',
//                         is_worker_failure=false). Matches the existing
//                         `cluster_preempt` vline color so the two layers
//                         visually rhyme — death-cause vlines surface the
//                         per-attempt timestamp instead of the +1 delta-derived
//                         time of the `cluster/preemptions` counter.
//   - cascade  (red)    → "Coscheduled sibling … bounced" — coscheduling
//                         cascade bug where one task dying takes its
//                         siblings down. Distinct color (the actionable
//                         death cause Ryan wants surfaced).
//   - failed   (orange) → other failure states: `failed`, `worker_failed`,
//                         `cosched_failed`, `killed`. Generic "this attempt
//                         died for some other reason" bucket.
//   - completed (none)  → `succeeded`/`completed` — terminal but not a death;
//                         no vline. Counted in the legend total though.
//
// In-progress attempts (no `finished_at_ms`) are not classified; the caller
// filters those out before invoking us.

import type { IrisAttempt } from './api'

export type DeathCause = 'preempt' | 'cascade' | 'failed' | 'completed'

export const DEATH_COLORS: Record<DeathCause, string> = {
  preempt: '#ba68c8',   // matches existing `cluster_preempt` purple
  cascade: '#ef4444',   // red — the coschedule-sibling-bounced cascade bug
  failed:  '#f59e0b',   // orange — generic failure
  completed: '#22c55e', // green (only rendered in the legend, not as a vline)
}

export const DEATH_LABELS: Record<DeathCause, string> = {
  preempt:   'GCP preempt',
  cascade:   'cascade (sibling)',
  failed:    'other failure',
  completed: 'completed',
}

/** Bucket one attempt by its final state + error. */
export function classifyDeath(a: IrisAttempt): DeathCause {
  const state = a.state
  if (state === 'succeeded' || state === 'completed') return 'completed'
  if (state === 'preempted') {
    return a.is_worker_failure ? 'cascade' : 'preempt'
  }
  // Some cosched-failure attempts come through with state='cosched_failed' or
  // 'failed' with `is_worker_failure=true` + the "Coscheduled sibling" error
  // string. Treat that prose as the canonical cascade signal so we don't miss
  // the cascade bug just because iris labelled the state differently.
  if ((a.error ?? '').toLowerCase().includes('coscheduled sibling')) return 'cascade'
  return 'failed'
}

/** Filter to attempts that have a recorded finish (= death event), classify
 *  each, and sort by finish time. Use this to drive both vline rendering and
 *  the death-cause counts in the legend. */
export function deathEventsFrom(attempts: IrisAttempt[]): Array<IrisAttempt & { cause: DeathCause }> {
  const out: Array<IrisAttempt & { cause: DeathCause }> = []
  for (const a of attempts) {
    if (a.finished_at_ms == null) continue
    const cause = classifyDeath(a)
    if (cause === 'completed') continue
    out.push({ ...a, cause })
  }
  out.sort((x, y) => (x.finished_at_ms! - y.finished_at_ms!))
  return out
}
