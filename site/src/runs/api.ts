// Tomat runs-dashboard API client.
//
// Talks to the `tomat-runs-api` Cloudflare Worker (see ../../../worker/),
// which serves the runs index + per-run parquet from OA's R2.
//
// API base URL is configurable via `VITE_RUNS_API_BASE` (build-time env)
// or `?api=…` query-string override (runtime). Defaults to the workers.dev
// URL once that's been registered for the OA account.

const QS_OVERRIDE = (() => {
  if (typeof window === 'undefined') return null
  try {
    return new URLSearchParams(window.location.search).get('api')
  } catch {
    return null
  }
})()

export const API_BASE: string =
  QS_OVERRIDE ??
  (import.meta.env.VITE_RUNS_API_BASE as string | undefined) ??
  'https://tomat-runs-api.openathena.workers.dev'

export interface RunsList {
  runs: string[]
  count: number
}

export interface RunManifest {
  schema_version: number
  synced_at: string
  run: {
    id: string
    name: string
    project: string
    entity: string
    state: string
    url: string
    created_at: string
    tags: string[]
    group: string | null
    config: Record<string, unknown>
  }
  summary: Record<string, number | string | boolean | null>
  history: {
    rows: number
    step_min: number | null
    step_max: number | null
    /** Latest non-null `global_step` from the history parquet. Authoritative
     *  training-step counter — populated even when `summary.global_step` is
     *  missing (fresh runs that haven't hit their first ckpt boundary).
     *  Optional: older manifests synced before this field was added won't
     *  carry it. */
    last_train_step?: number | null
    ts_min: number | null
    ts_max: number | null
  }
}

export async function fetchRuns(): Promise<RunsList> {
  const r = await fetch(`${API_BASE}/api/runs`)
  if (!r.ok) throw new Error(`fetchRuns ${r.status}`)
  return r.json()
}

export async function fetchManifest(runId: string): Promise<RunManifest> {
  const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/manifest.json`)
  if (!r.ok) throw new Error(`fetchManifest(${runId}) ${r.status}`)
  return r.json()
}

/**
 * Aggregated snapshot: runs index + every run's manifest + iris state in ONE
 * response. The Worker fans out to R2 server-side and edge-caches the result
 * (~30s TTL), so a tab with 50 runs no longer issues 50 manifest polls —
 * just one snapshot poll. Runs in the index without a synced manifest yet
 * show up as `runs[id] = null`; the dashboard drops them silently.
 *
 * See `worker/src/index.ts:handleRunsSnapshot` and `specs/23-runs-dashboard.md`
 * (Phase B "On-demand caching" section) for the design.
 */
export interface RunsSnapshot {
  synced_at: string
  count: number
  runs: Record<string, RunManifest | null>
  iris: IrisState | null
}

export async function fetchRunsSnapshot(): Promise<RunsSnapshot> {
  const r = await fetch(`${API_BASE}/api/runs-snapshot.json`)
  if (!r.ok) throw new Error(`fetchRunsSnapshot ${r.status}`)
  return r.json()
}

export function parquetUrl(runId: string): string {
  return `${API_BASE}/api/runs/${encodeURIComponent(runId)}/raw.parquet`
}

/** One checkpoint's mat-level eval. `nmae_*`/`nemd_*` are FRACTIONS (0.0117 =
 *  1.17%) — ×100 for display. (Note: distinct from the manifest summary's
 *  `eval/mat_nmae/...` values, which the watchdog logs already as percentages.) */
export interface EvalPoint {
  step: number
  n_mats: number | null
  nmae_mean: number | null
  nmae_p1: number | null
  nmae_p25: number | null
  nmae_median: number | null
  nmae_p75: number | null
  nmae_p99: number | null
  nemd_mean: number | null
  nemd_p1: number | null
  nemd_p25: number | null
  nemd_median: number | null
  nemd_p75: number | null
  nemd_p99: number | null
}

/**
 * Per-run mat-NMAE/NEMD series. `tomat evals sync` aggregates the canonical
 * per-step GCS eval-result JSONs (written by `eval_mat_nmae.py`) into one
 * `eval.json` per run on R2. This is the source of truth — the harvested
 * wandb points collapse to a single value in runs-sync's parquet merge.
 */
export interface RunEval {
  schema_version: number
  synced_at: string
  run: string
  sets: Record<string, EvalPoint[]>  // 'val_200' | 'train_200'
}

/** Fetch a run's per-step eval series; null when the run hasn't been
 *  eval-synced yet (no `eval.json` on R2 → 404). */
export async function fetchEval(runId: string): Promise<RunEval | null> {
  const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/eval.json`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchEval(${runId}) ${r.status}`)
  return r.json()
}

export interface IrisJob {
  state: string
  state_code: number
  preempts: number
  failures: number
  error: string | null
  exit_code: number | null
  submitted_at_ms: number | null
  started_at_ms: number | null
  finished_at_ms: number | null
  num_tasks: number
  /** Per-state task histogram, keyed by iris's `task_state_friendly`
   *  strings: `running`, `pending`, `building`, `completed`, `failed`,
   *  `killed`, `preempted`, ... Zero entries are dropped server-side;
   *  treat missing keys as zero.
   *
   *  Critical for distinguishing "RUNNING (all 4 healthy)" from a
   *  cascade-restart loop where job-level state says RUNNING but every
   *  task is `pending` (briefly running per cycle, never long enough for
   *  the job-level read). Optional: older snapshots written before this
   *  field was added won't carry it. */
  task_state_counts?: Record<string, number>
}

export interface IrisState {
  schema_version: number
  synced_at: string
  count: number
  jobs: Record<string, IrisJob>
}

export async function fetchIrisState(): Promise<IrisState> {
  const r = await fetch(`${API_BASE}/api/iris-state.json`)
  if (!r.ok) throw new Error(`fetchIrisState ${r.status}`)
  return r.json()
}

// ── per-attempt history (death events) ──────────────────────────────────────
// Mirror of iris's `AttemptReport`, plus epoch_ms fields added by
// `scripts/iris_attempts_dump.py` so the dashboard doesn't have to parse ISO
// timestamps. `state` is iris's `task_state_friendly` string (e.g.
// 'preempted', 'completed', 'running', 'failed').

export interface IrisAttempt {
  attempt_id: number
  worker_id: string
  state: string
  exit_code: number
  error: string
  is_worker_failure: boolean
  started_at: string
  started_at_ms: number | null
  finished_at: string
  finished_at_ms: number | null
}

export interface IrisAttemptsTask {
  task_id: string
  state: string
  started_at: string
  started_at_ms: number | null
  finished_at: string
  finished_at_ms: number | null
  exit_code: number
  error: string
  attempts: IrisAttempt[]
}

export interface IrisAttempts {
  schema_version: number
  label: string
  job_id: string
  synced_at: string
  job_state: string
  job_failure_count: number
  job_preemption_count: number
  completed_count: number
  submitted_at: string
  submitted_at_ms: number | null
  started_at: string
  started_at_ms: number | null
  finished_at: string
  finished_at_ms: number | null
  tasks: IrisAttemptsTask[]
}

/** Fetch the per-task attempt history sidecar for one training label.
 *  Null when the sidecar doesn't exist yet (new run, or not a training job). */
export async function fetchIrisAttempts(label: string): Promise<IrisAttempts | null> {
  const r = await fetch(`${API_BASE}/api/iris-attempts/${encodeURIComponent(label)}.json`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchIrisAttempts(${label}) ${r.status}`)
  return r.json()
}

/** wandb run name → iris job id. Our convention: iris job is `/ryan/<name>`. */
export function irisJobIdForRun(runName: string): string {
  return `/ryan/${runName}`
}

/**
 * An m-eval (mat-NMAE) job — an iris job that evaluates one checkpoint of one
 * run against one mat-set. `tomat evals fire` names them
 * `tomat-eval-<run_label>-<mat_set>-step-<N>`, so the name fully encodes which
 * run + checkpoint + split the job evaluates.
 */
export interface EvalJob {
  runLabel: string
  matSet: string  // 'val_200' | 'train_200'
  step: number
  jobId: string
  job: IrisJob
}

const EVAL_JOB_RE = /^\/ryan\/tomat-eval-(.+)-(val_200|train_200)-step-(\d+)$/

/** Group the iris snapshot's m-eval jobs by the run they evaluate. */
export function evalJobsByRun(iris: IrisState | undefined): Map<string, EvalJob[]> {
  const byRun = new Map<string, EvalJob[]>()
  if (!iris) return byRun
  for (const [jobId, job] of Object.entries(iris.jobs)) {
    const m = EVAL_JOB_RE.exec(jobId)
    if (!m) continue
    const [, runLabel, matSet, stepStr] = m
    const arr = byRun.get(runLabel) ?? []
    arr.push({ runLabel, matSet, step: Number(stepStr), jobId, job })
    byRun.set(runLabel, arr)
  }
  for (const arr of byRun.values()) {
    arr.sort((a, b) => a.step - b.step || a.matSet.localeCompare(b.matSet))
  }
  return byRun
}

export type EvalPhase = 'flight' | 'done' | 'failed'

/** Coarse lifecycle bucket for an m-eval job's iris state. */
export function evalPhase(job: IrisJob): EvalPhase {
  switch (job.state) {
    case 'SUCCEEDED':
      return 'done'
    case 'FAILED':
    case 'WORKER_FAILED':
    case 'CANCELLED':
      return 'failed'
    default:  // QUEUED, RUNNING, PENDING, SUBMITTED, UNKNOWN, …
      return 'flight'
  }
}
