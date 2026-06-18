// Pure helpers for `MEvalTable.tsx`, factored out so the unit tests can
// import without hitting JSX (`node --test --experimental-strip-types` only
// handles plain TS, not TSX) or `api.ts`'s `import.meta.env` reads.

import type { EvalPoint, RunEval } from './api'

// ── eval-job regex (mirrors `EVAL_JOB_RE` in api.ts) ────────────────────────
// Lives here (not in api.ts) so the parsing logic stays node-testable
// without dragging the vite `import.meta.env` reads in api.ts. The api.ts
// `evalJobsByRun` re-uses this same regex string. Keep the two in sync.
export const EVAL_JOB_RE_SRC
  = '^/ryan/tomat-eval-(?:(maskgit|free)-)?(.+?)-(val_200|train_200)-step-(\\d+)(?:-task(\\d+))?$'

export type EvalModeBucket = 'oneshot' | 'maskgit' | 'free'

/** Parse one iris job id; returns null when the job isn't a tomat-eval job. */
export function parseEvalJobId(jobId: string): {
  mode: EvalModeBucket
  runLabel: string
  matSet: 'val_200' | 'train_200'
  /** 0-based `step_idx` (Levanter `info.step`). See `docs/step-conventions.md`. */
  step: number
  taskIdx: number | null
} | null {
  const re = new RegExp(EVAL_JOB_RE_SRC)
  const m = re.exec(jobId)
  if (!m) return null
  const [, modeInfix, runLabel, matSet, stepStr, taskStr] = m
  const mode: EvalModeBucket = modeInfix === 'maskgit' ? 'maskgit'
    : modeInfix === 'free' ? 'free'
    : 'oneshot'
  return {
    mode,
    runLabel,
    matSet: matSet as 'val_200' | 'train_200',
    step: Number(stepStr),
    taskIdx: taskStr != null ? Number(taskStr) : null,
  }
}

/** Colour bucket for a given NMAE or NEMD percentage. Thresholds chosen to
 *  read subtly: green = "healthy" (sub-1% NMAE is the SOTA range), amber =
 *  "worth watching", red = "diverged / mode-collapsed". */
export type Bucket = 'good' | 'mid' | 'bad' | 'none'

/** NMAE buckets: <1% green, 1-5% amber, >=5% red. */
export function nmaeBucket(pct: number): Bucket {
  if (!Number.isFinite(pct) || pct < 0) return 'none'
  if (pct < 1) return 'good'
  if (pct < 5) return 'mid'
  return 'bad'
}

/** NEMD buckets: <0.5% green, 0.5-3% amber, >=3% red. NEMD reads ~half NMAE
 *  for healthy runs (cf. cont33k NMAE 0.53% / NEMD 0.18% — memory
 *  `first-real-mg-and-ss-results`), so tighter thresholds keep the colour
 *  signal roughly matched in magnitude across the two metrics. */
export function nemdBucket(pct: number): Bucket {
  if (!Number.isFinite(pct) || pct < 0) return 'none'
  if (pct < 0.5) return 'good'
  if (pct < 3) return 'mid'
  return 'bad'
}

/** Unique steps across every set's points list, descending (latest first). */
export function stepsDescOf(evalSeries: RunEval | null): number[] {
  const sets = evalSeries?.sets
  if (!sets) return []
  const stepSet = new Set<number>()
  for (const k of Object.keys(sets)) {
    for (const pt of sets[k] ?? []) stepSet.add(pt.step)
  }
  return [...stepSet].sort((a, b) => b - a)
}

/** `(step|setKey)` → `EvalPoint` lookup map, indexed for O(1) cell rendering. */
export function pointsByStepSet(evalSeries: RunEval | null): Map<string, EvalPoint> {
  const m = new Map<string, EvalPoint>()
  const sets = evalSeries?.sets
  if (!sets) return m
  for (const k of Object.keys(sets)) {
    for (const pt of sets[k] ?? []) m.set(`${pt.step}|${k}`, pt)
  }
  return m
}

// ── m-eval job lifecycle bucket ─────────────────────────────────────────────
// iris's `JobState` proto enum, mirrored here as the wire-format string
// (server-side dump uses `JobState.Name(state)`, so it's UPPER_SNAKE_CASE).
// Authoritative source: `marin/lib/iris/src/iris/cluster/types.py`
// (`TERMINAL_JOB_STATES`).
//
// Why explicit? The previous version of `evalPhase` had a leaky
// `default → 'flight'` branch — so any state iris added (or any one we
// forgot) silently counted as "in flight" forever. That bit us with
// `KILLED` (worker_lost_spec on the bin5 m-evals): the run card showed
// `⏳ 3 m-evals` indefinitely with no live job behind it. Now: live and
// terminal states are spelled out, anything unknown buckets as 'flight'
// only after first being logged as a warning by the helper's caller (if
// they care). The test in `MEvalTable.test.ts` enumerates every state in
// these constants so they stay in sync with iris.

/** Live (non-terminal) iris JobState names — wire-format strings. */
export const IRIS_LIVE_JOB_STATES = ['PENDING', 'BUILDING', 'RUNNING'] as const

/** Terminal iris JobState names — wire-format strings. Mirrors
 *  `TERMINAL_JOB_STATES` in iris's `cluster/types.py`. */
export const IRIS_TERMINAL_JOB_STATES = [
  'SUCCEEDED', 'FAILED', 'KILLED', 'WORKER_FAILED', 'UNSCHEDULABLE',
] as const

/** Terminal states that should bucket as 'failed' (vs. 'done'). */
const IRIS_FAILED_JOB_STATES = new Set<string>([
  'FAILED', 'KILLED', 'WORKER_FAILED', 'UNSCHEDULABLE', 'CANCELLED',
])

/** Live states (incl. transient ones iris may surface in addition to the
 *  three canonical ones above). */
const IRIS_FLIGHT_JOB_STATES = new Set<string>([
  ...IRIS_LIVE_JOB_STATES,
  'SUBMITTED', 'QUEUED', 'ASSIGNED', 'UNKNOWN',
])

export type EvalPhase = 'flight' | 'done' | 'failed'

/** Coarse lifecycle bucket for an m-eval job's iris state.
 *
 *  Unknown / unrecognized states bucket as 'failed' (NOT 'flight') — a stale
 *  iris job from yesterday with a state the FE doesn't recognise is
 *  ≈definitely not in flight; treating it as a failure surfaces the issue
 *  (so the user can re-fire) without leaving a permanent hourglass on the
 *  run card. Add new states to one of the two sets above when iris adds
 *  them. */
export function evalPhase(job: { state: string }): EvalPhase {
  const s = job.state
  if (s === 'SUCCEEDED') return 'done'
  if (IRIS_FAILED_JOB_STATES.has(s)) return 'failed'
  if (IRIS_FLIGHT_JOB_STATES.has(s)) return 'flight'
  return 'failed'
}
