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

// In dev (`pnpm dev`) default to the staging worker, so the dev
// `[fire]`/`[retry]` buttons hit a write-enabled endpoint that isn't
// exposed on `tomat.oa.dev`. Prod builds and any explicit override
// (env var or `?api=…`) bypass this fallback.
const DEV_DEFAULT_API_BASE = 'https://tomat-runs-api-staging.openathena.workers.dev'
const PROD_DEFAULT_API_BASE = 'https://tomat-runs-api.openathena.workers.dev'

export const API_BASE: string =
  QS_OVERRIDE ??
  (import.meta.env.VITE_RUNS_API_BASE as string | undefined) ??
  (import.meta.env.DEV ? DEV_DEFAULT_API_BASE : PROD_DEFAULT_API_BASE)

export interface RunsList {
  runs: string[]
  count: number
}

/** One wandb run that contributed to a logical run (spec 61 §2.2 — 1 run :
 *  N wandb runs). Today most runs have N=1 (1 wandb : 1 segment / fire);
 *  post-trainer-change (commit 61f5853) `tomat train --resume` fires get
 *  their own wandb run with id `<results_label>-<fire-tail>` so a single
 *  segment can accumulate N>1 wandb runs over its lineage of resumes. */
export interface WandbRunRef {
  entity: string
  project: string
  /** wandb's internal `r.id` (used for URL construction). */
  run_id: string
  /** wandb display name. Typically `<results_label>(-<fire-tail>)?`. */
  name: string
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
  /** Spec 61 §2.2 — all wandb runs that contributed to this logical run,
   *  ordered by `created_at` ascending. Optional (older manifests + Modal
   *  runs may omit). Length-1 array is the common case today; >1 once
   *  per-fire wandb-id derivation has run for ≥2 fires of the same segment. */
  wandb_run_ids?: WandbRunRef[]
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
  /** `pending-fires.json` — Modal training fires `tomat modal pending-fire add`
   *  recorded ahead of their wandb session opening. Renders placeholder cards
   *  for runs not yet in `runs`. Optional for back-compat with snapshots from
   *  before the feed was added. */
  pending_fires?: PendingFires | null
}

export async function fetchRunsSnapshot(): Promise<RunsSnapshot> {
  const r = await fetch(`${API_BASE}/api/runs-snapshot.json`)
  if (!r.ok) throw new Error(`fetchRunsSnapshot ${r.status}`)
  return r.json()
}

/** Heartbeat the GCE-VM monitor writes every 2 min after `runs-sync-active`.
 *  Lets the dashboard surface "cron last fired Xm ago" — when the chip goes
 *  red (≥ ~5 min stale) the VM is silent and badges + sparklines will start
 *  drifting. */
export interface CronHeartbeat {
  ts: string
  host?: string
  last_run_count?: number
  last_failure_count?: number
}

export async function fetchCronHeartbeat(): Promise<CronHeartbeat | null> {
  const r = await fetch(`${API_BASE}/api/cron-heartbeat.json`)
  // 404 (R2 object not yet written) → null; the FE renders "—" in that case
  // rather than spamming console errors.
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchCronHeartbeat ${r.status}`)
  return r.json()
}

export function parquetUrl(runId: string): string {
  return `${API_BASE}/api/runs/${encodeURIComponent(runId)}/raw.parquet`
}

/** One checkpoint's mat-level eval. `nmae_*`/`nemd_*` are FRACTIONS (0.0117 =
 *  1.17%) — ×100 for display. (Note: distinct from the manifest summary's
 *  `eval/mat_nmae/...` values, which the watchdog logs already as percentages.) */
export interface EvalPoint {
  /** 0-based `step_idx` (Levanter `info.step`) from the per-step ckpt name.
   *  See `docs/step-conventions.md`. */
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
  /** 0-based `step_idx` (Levanter `info.step`). See `docs/step-conventions.md`. */
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
  /** MaskGIT-only: diagnostic-path NMAE from argmax-per-iter filled_bins,
   *  bypassing the OOD final all-filled re-forward. For absorbing-trained
   *  MG models, `nmae_*` (the re-forward) is GIGO; `nmae_filled_*` is the
   *  honest in-distribution number. See memory `k1-oneshot-on-mg-is-gigo`. */
  nmae_filled_mean: number | null
  nmae_filled_median: number | null
  nmae_filled_p99: number | null
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
 * run against one mat-set + mode. `tomat evals fire` names them
 *   `tomat-eval-<run_label>-<mat_set>-step-<N>`              (oneshot/teacher; K=1)
 *   `tomat-eval-maskgit-<run_label>-<mat_set>-step-<N>`      (K=12 schedule)
 *   `tomat-eval-free-<run_label>-<mat_set>-step-<N>`         (free-running)
 *   `…-task<i>` suffix for fan-out across multiple iris tasks.
 *
 * The previous regex collapsed the mode infix into `runLabel` (so a
 * `tomat-eval-maskgit-train-…` job's runLabel parsed as
 * `maskgit-train-…`, never matching any actual run). Now we split out the
 * `mode` infix explicitly so the MEvalTable can join in-flight jobs to
 * the correct (run, step, set, mode) cell.
 */
export type EvalMode = 'oneshot' | 'maskgit' | 'free'

export interface EvalJob {
  runLabel: string
  matSet: string  // 'val_200' | 'train_200'
  mode: EvalMode
  /** 0-based `step_idx` parsed from the iris job name (`…-step-{step_idx}`).
   *  See `docs/step-conventions.md`. */
  step: number
  /** Ablation variant from `--output-suffix` (e.g. `K1`). null when the job
   *  was fired with no suffix. Goes through `evalSetKey` to land the job in
   *  the right MEvalTable column. */
  variant: string | null
  taskIdx: number | null
  jobId: string
  job: IrisJob
}

// Capture the optional mode infix (`maskgit-` or `free-`), then runLabel,
// then matSet, step, and optional `-taskN` suffix. NB: lazy `(.+?)` on
// runLabel so the suffix groups bind correctly. Source string is exported
// from `./MEvalTable.helpers` so node-test can exercise the same regex
// without dragging the `api.ts` `import.meta.env` evaluation in.
import { EVAL_JOB_RE_SRC } from './MEvalTable.helpers'

const EVAL_JOB_RE = new RegExp(EVAL_JOB_RE_SRC)

/** Map the iris job name's mode infix to the column-set key the eval.json
 *  / MEvalTable uses. `oneshot`/`teacher` shares the bare set key
 *  (`val_200`), `maskgit` appends `-maskgit`, `free` appends `-free`. An
 *  optional ablation `variant` (from `tomat evals fire --output-suffix`,
 *  e.g. `K1`) appends as a further suffix so e.g. an honest-K=1 maskgit
 *  ablation lands in `val_200-maskgit-K1` instead of the legacy K=12
 *  bucket. Sync writes the same setKey shape (see `tomat evals_sync`). */
export function evalSetKey(matSet: string, mode: EvalMode, variant?: string | null): string {
  const base = mode === 'oneshot' ? matSet : `${matSet}-${mode}`
  return variant ? `${base}-${variant}` : base
}

/** Group the iris snapshot's m-eval jobs by the run they evaluate. */
export function evalJobsByRun(iris: IrisState | undefined): Map<string, EvalJob[]> {
  const byRun = new Map<string, EvalJob[]>()
  if (!iris) return byRun
  for (const [jobId, job] of Object.entries(iris.jobs)) {
    const m = EVAL_JOB_RE.exec(jobId)
    if (!m) continue
    const [, modeInfix, runLabel, matSet, stepStr, variantStr, taskStr] = m
    const mode: EvalMode = modeInfix === 'maskgit' ? 'maskgit'
      : modeInfix === 'free' ? 'free'
      : 'oneshot'
    const arr = byRun.get(runLabel) ?? []
    arr.push({
      runLabel, matSet, mode, step: Number(stepStr),
      variant: variantStr ?? null,
      taskIdx: taskStr != null ? Number(taskStr) : null,
      jobId, job,
    })
    byRun.set(runLabel, arr)
  }
  for (const arr of byRun.values()) {
    arr.sort((a, b) => a.step - b.step || a.matSet.localeCompare(b.matSet)
      || a.mode.localeCompare(b.mode)
      || (a.variant ?? '').localeCompare(b.variant ?? ''))
  }
  return byRun
}

// ── eval-fire request (FE → CFW → R2 → autosync.sh) ─────────────────────────
// FE POSTs a small request blob to the runs API; the worker appends it to
// `tomat/eval-fire-requests/<ts>-<rand>.json` on R2; the GCE-VM cron
// (`scripts/autosync.sh`) lists the prefix and calls `tomat evals fire` for
// each pending entry. This avoids a public live `tomat evals fire` endpoint
// (TPU spend) while still giving the dashboard a one-click "fire this cell"
// affordance.

export interface EvalFireRequest {
  run_id: string
  /** 0-based `step_idx` (Levanter `info.step`). Matches the GCS ckpt
   *  `step-{step_idx}` directory. See `docs/step-conventions.md`. */
  step: number
  mat_set: string         // 'val_200' | 'train_200'
  mode: EvalMode          // 'oneshot' | 'maskgit' | 'free'
  /** Optional override of the eval-fire `--eval-label`; the autosync runner
   *  picks a sensible default per-run if missing. */
  eval_label?: string
  /** Free-form reason for audit (logged with the fire request). */
  reason?: string
}

export interface EvalFireResponse {
  ok: boolean
  request_id?: string
  error?: string
}

/** Submit an eval-fire request. Returns the response from the CFW endpoint
 *  (which only writes the request to R2 — actual firing happens out-of-band
 *  via the GCE-VM autosync cron). */
export async function fireEval(req: EvalFireRequest): Promise<EvalFireResponse> {
  const r = await fetch(`${API_BASE}/api/eval/fire`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  let body: EvalFireResponse
  try { body = await r.json() } catch { body = { ok: false, error: `non-JSON response (${r.status})` } }
  if (!r.ok && body.ok !== false) body.ok = false
  return body
}

// `evalPhase` + `EvalPhase` live in `MEvalTable.helpers.ts` so they can be
// exercised by `node --test --experimental-strip-types` without dragging
// `api.ts`'s `import.meta.env` reads in. Re-exported here so existing
// imports (`from './api'`) keep working.
export { evalPhase, type EvalPhase } from './MEvalTable.helpers'

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
  /** Modal app id (`ap-…`). Optional — older snapshots predate the sync writing it. */
  app_id?: string
  /** fc-id → probe result. Empty when no --fc-id was passed to `tomat modal sync`. */
  function_calls: Record<string, ModalFunctionCall>
  /** function_name → function_id (`fu-…`) for every function in this app.
   *  Populated by `AppGetLayout`. Used by the Modal-icon deep-link to
   *  build `?activeTab=logs&functionId=<fu>&fcId=<fc>` — the per-call
   *  filtered logs view. Empty for snapshots predating the `AppGetLayout`
   *  capture; in that case the link falls back to the app overview. */
  function_ids?: Record<string, string>
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

// ── pending-fires.json (Modal training fires not yet wandb-manifested) ─────
// Written by `tomat modal pending-fire add` immediately after `fn.spawn(...)`
// in `scripts/train_smoke_modal.py`, so the /runs dashboard can render a
// placeholder card during the 15-30 min pre-wandb window (container starting,
// JAX compile, ckpt load). GC'd by `tomat runs sync` once the wandb manifest
// lands on R2. Schema mirrors the python helper in `tomat::pending_fire_add`.

export interface PendingFire {
  run_id: string
  fc_id: string
  app_name: string
  function_name: string
  spawned_at_ms: number
  hardware: string
  model_preset: string
  batch_size: number
  seed: number
  intended_steps: number
  results_label: string
  wandb_project: string
  loss_type: string
  kl_sigma: number | null
  maskgit_prior: string
  tags: string[]
}

export interface PendingFires {
  schema_version: number
  synced_at: string
  fires: PendingFire[]
}

export async function fetchPendingFires(): Promise<PendingFires | null> {
  const r = await fetch(`${API_BASE}/api/pending-fires.json`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`fetchPendingFires ${r.status}`)
  return r.json()
}

/** Find the `ModalFunctionCall` probe (if any) for one pending fire's fc_id.
 *  `tomat modal sync --fc-id <fc>` probes only fc-ids that were passed at
 *  sync time; without that probe we just have app-level state. The pending
 *  fire still renders fine — the placeholder card falls back to the app's
 *  state alone. */
export function modalFcForPending(
  modal: ModalState | null | undefined,
  fc_id: string,
): ModalFunctionCall | null {
  if (!modal) return null
  for (const app of Object.values(modal.apps)) {
    const fc = app.function_calls[fc_id]
    if (fc) return fc
  }
  return null
}

/** Resolve the `ModalFunctionCall` for a given run_id, via pending-fires +
 *  modal-state. Returns null when no pending-fire is recorded for the run
 *  (true for any Modal run fired before the spawn wrapper started writing
 *  pending-fire entries) or when the fc isn't in the Modal state snapshot
 *  yet. Used by `RunCardData.modalFc` so the run-detail Modal link can
 *  deep-link to the specific call. When multiple pending-fires exist for
 *  the same run_id (re-fires after a kill), prefers the most recently
 *  spawned one. */
export function pendingFcForRun(
  pendingFires: PendingFires | null | undefined,
  modal: ModalState | null | undefined,
  runId: string,
): ModalFunctionCall | null {
  const fires = (pendingFires?.fires ?? []).filter((f) => f.run_id === runId)
  if (fires.length === 0) return null
  fires.sort((a, b) => b.spawned_at_ms - a.spawned_at_ms)
  return modalFcForPending(modal, fires[0].fc_id)
}

/** Same as `pendingFcForRun` but returns the raw fc-id, even when the Modal
 *  state snapshot hasn't probed the fc yet. Useful for the per-run Modal
 *  link (which only needs the fc-id, not the call's runtime state). */
export function pendingFcIdForRun(
  pendingFires: PendingFires | null | undefined,
  runId: string,
): string | null {
  const fires = (pendingFires?.fires ?? []).filter((f) => f.run_id === runId)
  if (fires.length === 0) return null
  fires.sort((a, b) => b.spawned_at_ms - a.spawned_at_ms)
  return fires[0].fc_id
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
