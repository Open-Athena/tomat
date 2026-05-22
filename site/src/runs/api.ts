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
