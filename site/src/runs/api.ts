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
 * Per spec 43 (Phase A), each entry can also carry an `evals` index so the
 * dashboard renders the eval matrix without an extra round-trip per run.
 *
 * See `worker/src/index.ts:handleRunsSnapshot` and `specs/23-runs-dashboard.md`
 * (Phase B "On-demand caching" section) for the design.
 */
export type RunsSnapshotEntry = RunManifest & {
  evals?: EvalIndexEntry[]
  /** Spec 46 Phase A: tiny per-run MSRP summary inlined into the snapshot
   *  for the first-paint chip. Full breakdown lives in `cost.json` and
   *  is fetched on-demand by the detail page (`fetchRunCost`). */
  cost?: RunCostSummary
}

export interface RunsSnapshot {
  synced_at: string
  count: number
  runs: Record<string, RunsSnapshotEntry | null>
  iris: IrisState | null
  modal: ModalState | null
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

// ── eval-records (spec 43, Phase A) ────────────────────────────────────────
// One record per (step, set, mode, task) eval-child of a training run. The
// CFW serves them from R2 sidecars (`tomat/runs/<id>/evals/<key>.json`,
// plus an `index.json` pointer list). Phase A is read-only + backfill;
// `tomat evals fire`/`sync` start writing live in Phase B/C.

/** Pointer into a run's eval-records folder. Sorted by (step, set, mode, task). */
export interface EvalIndexEntry {
  key: string         // `<step>-<set>-<mode>` or `<step>-<set>-<mode>-task<i>`
  fired_at: string    // ISO-8601 (≈ GCS object mtime for backfill records)
  state: EvalRecordState
}

export type EvalRecordState = 'pending' | 'running' | 'succeeded' | 'failed' | 'killed'

/** Full record. Schema lifted directly from spec 43 §"Data model"; fields
 *  unknown to the dashboard (e.g. `job_ref`) pass through untouched. */
export interface EvalRecord {
  run_label: string
  ckpt_leaf: string | null
  step: number
  set: string                 // 'val_200' | 'train_200' | ...
  mode: string                // 'teacher' | 'free' | 'maskgit'
  task_idx?: number | null
  n_tasks: number
  n_mats: number | null
  eval_label: string | null
  model_preset: string | null
  mg_k_steps?: number | null
  free_batch?: number | null
  eval_batch?: number | null
  decoder?: string | null

  backend: 'iris' | 'modal' | null
  job_ref: Record<string, unknown> | null

  fired_at: string
  state: EvalRecordState
  state_synced_at: string | null
  started_at: string | null
  finished_at: string | null
  exit_code: number | null
  error_msg: string | null

  /** Bucket + path explicit per spec ([[cross-region-eval-egress]] —
   *  results may live in a mirror bucket distinct from the ckpt bucket). */
  gcs_result_path: string
  result: EvalRecordResult | null
  result_seen_at: string | null
}

/** The scalar(s) we copy out of the per-step result JSON for display.
 *  Kept narrow to keep records small; the full JSON is at `gcs_result_path`. */
export interface EvalRecordResult {
  n_mats: number | null
  nmae_mean: number | null
  nmae_median: number | null
  nmae_p99: number | null
  nemd_mean: number | null
  nemd_median: number | null
  nemd_p99: number | null
}

/** Fetch the eval-records index for one run. Empty array on 404 (no
 *  records yet — backfill hasn't been run, no evals fired). */
export async function fetchEvalsIndex(runId: string): Promise<EvalIndexEntry[]> {
  const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/evals`)
  if (r.status === 404) return []
  if (!r.ok) throw new Error(`fetchEvalsIndex(${runId}) ${r.status}`)
  return r.json()
}

/** Fetch one eval record by key. Null on 404 (record never landed / was
 *  cleared). */
export async function fetchEvalRecord(runId: string, key: string): Promise<EvalRecord | null> {
  const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/evals/${encodeURIComponent(key)}`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchEvalRecord(${runId}, ${key}) ${r.status}`)
  return r.json()
}

// ── cost.json (spec 46 Phase A: TPU MSRP) ───────────────────────────────────
// Estimated-MSRP equivalent retail value of one run's compute, written by
// `tomat cost compute` and surfaced as the "$X MSRP" chip on the run-detail
// page. Phase A models TPU iris runs (variant + chips + wallclock × rate);
// Modal runs ship a `breakdown[i].kind === "modal"` placeholder with null
// `msrp_usd` so the chip can render "Modal MSRP TBD" instead of $0.

/** One contributor: a TPU attempt cycle or a Modal function call.
 *  `msrp_usd` is null when the rate is unknown (Phase B Modal). */
export interface RunCostBreakdownRow {
  kind: 'tpu' | 'modal'
  label: string                // "v6e-16 preemptible", "Modal MSRP TBD"
  wallclock_sec: number
  rate_per_hr_usd: number | null
  msrp_usd: number | null
  source: string
  started_at_ms: number | null
  finished_at_ms: number | null
}

export interface RunCost {
  schema_version: number
  pricing_table_version: string  // "2026-06-02" — snapshot date
  computed_at: string             // ISO-8601 UTC
  is_complete: boolean
  msrp_usd: number
  breakdown: RunCostBreakdownRow[]
  notes: string[]
}

/** Tiny summary inlined into runs-snapshot for first-paint. The detail
 *  page's tooltip lazy-fetches the full RunCost via `fetchRunCost`. */
export interface RunCostSummary {
  msrp_usd: number
  is_complete: boolean
  pricing_table_version: string
  has_modal_pending: boolean
}

/** Fetch one run's full MSRP record. Null when the sidecar doesn't exist
 *  yet (run hasn't been `tomat cost compute`-d). */
export async function fetchRunCost(runId: string): Promise<RunCost | null> {
  const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}/cost.json`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchRunCost(${runId}) ${r.status}`)
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
  /** Cleaned first line of `error` (stripped of iris's
   *  `Exit code: N. stderr:` wrapper). Server-side convenience added in
   *  schema v2 by `scripts/iris_attempts_dump.py`. Optional for back-compat
   *  with v1 sidecars; dashboard falls back to client-side cleanup. */
  error_first_line?: string | null
  /** Bucket label for the failure mode — e.g. "JAX mesh ValueError (eval
   *  boundary)", "OOM", "cascade (sibling died)". Same regex set as
   *  `site/src/runs/errorClassification.ts`. Schema v2; dashboard falls
   *  back to client-side classification. */
  error_classification?: string | null
  is_worker_failure: boolean
  started_at: string
  started_at_ms: number | null
  finished_at: string
  finished_at_ms: number | null
}

/** Flat per-attempt-per-task summary record (schema v2). Sorted by
 *  `trainer_started_ts_ms`. Mirrors `attempts_summary` in
 *  `iris_attempts_dump.py`. */
export interface IrisAttemptSummary {
  task_id: string
  attempt_id: number
  trainer_started_ts_ms: number | null
  ended_ts_ms: number | null
  state: string
  exit_code: number
  error_first_line: string | null
  error_classification: string | null
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
  /** Schema v2: flat per-attempt summary records ordered by start time.
   *  Optional for back-compat. The dashboard builds the same list
   *  client-side from `tasks[*].attempts` when missing. */
  attempts_summary?: IrisAttemptSummary[]
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

// ── Modal snapshot (tomat modal sync) ───────────────────────────────────────
// Mirrors the python schema in `tomat::modal_sync`. App state names are
// lowercase `AppState` enum stems (`deployed`, `stopped`, `initializing`,
// `stopping`, ...). Input statuses are lowercase `GenericResult.GenericStatus`
// stems (`pending`, `success`, `failure`, `terminated`, `timeout`,
// `init_failure`, `internal_failure`, `idle_timeout`).

export interface ModalInput {
  input_id: string
  task_id: string | null
  status: string
  status_code: number
}

export interface ModalFunctionCall {
  function_call_id: string
  function_name: string
  module_name: string
  inputs: ModalInput[]
  /** Set when the call-graph probe failed (e.g. invalid fc-id, deleted). */
  error?: string
}

export interface ModalApp {
  description: string
  state: string
  state_code: number
  n_running_tasks: number
  created_at_ms: number | null
  stopped_at_ms: number | null
  /** fc-id → probe result. Empty when no --fc-id was passed to `tomat modal sync`. */
  function_calls: Record<string, ModalFunctionCall>
}

export interface ModalState {
  schema_version: number
  synced_at: string
  count: number
  apps: Record<string, ModalApp>
}

export async function fetchModalState(): Promise<ModalState> {
  const r = await fetch(`${API_BASE}/api/modal-state.json`)
  if (!r.ok) throw new Error(`fetchModalState ${r.status}`)
  return r.json()
}

/** A run is "Modal-hosted" when its name carries our convention's
 *  `-modal-` infix (e.g. `train-mg-modal-h200x8-tz-v3-bs128-seed42`).
 *  This is a name-pattern heuristic, not a wandb-tag check — keeps the
 *  match dead-simple and avoids waiting for the wandb session to open. */
export function isModalRun(runName: string): boolean {
  return /-modal-/.test(runName)
}

/** Pick the most-relevant Modal app for a Modal run. We have one
 *  app (`tomat-train-smoke`) that hosts all training fires for now,
 *  so this just returns the most recently created `deployed` app
 *  matching that description — falling back to the most-recent
 *  `stopped` if none are deployed. */
export function modalAppForRun(modal: ModalState | null | undefined, runName: string): ModalApp | null {
  if (!modal) return null
  if (!isModalRun(runName)) return null
  const candidates = Object.values(modal.apps).filter((a) =>
    a.description === 'tomat-train-smoke',
  )
  if (candidates.length === 0) return null
  candidates.sort((a, b) => (b.created_at_ms ?? 0) - (a.created_at_ms ?? 0))
  const deployed = candidates.find((a) => a.state === 'deployed')
  return deployed ?? candidates[0]
}
