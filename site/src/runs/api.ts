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
