// Rich per-run header chrome — the same block that appears on both the /runs
// dashboard's card view (inside `RunCard`) and the per-run detail page
// (`#/runs/<name>`, rendered by `RunDetail` above the chart).
//
// The header is a 2-column grid:
//   • LEFT: pin icon (cards only), status dots, iris badge (or wandb fallback),
//     the run name + wandb link, eval chip, parent chip, tag chips, then the
//     hw/model/BS/loss/target/suffix line, then the created/logged/synced line.
//   • RIGHT: step counter + progress bar, sparkline + freshness tag, syncrate
//     (steps/min/h/6h), latest-summary metrics (tr/ev/MFU/MT/MV/ep/FLOP).
//
// `RunHeaderRich` is a presentation-only component: it takes a `RunCardData`
// (the same snapshot the card uses — manifest + iris job + history + eval
// jobs + color) and renders. No fetching, no react-query inside.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Tooltip } from '../Tooltip'
import {
  evalPhase, fetchRunCost, isModalRun,
  type EvalJob, type IrisAttempts, type IrisJob, type ModalApp,
  type ModalFunctionCall, type RunCost, type RunCostSummary, type RunManifest,
} from './api'
import { computeTrainingFraction, formatDurationShort } from './trainingFraction'
import { lineageFor } from './lineage'
import type { RunHistory } from './parquet'
import {
  epochBreakdownAtStep,
  epochOfStep,
  formatStepCount,
  freshnessColor,
  HW_COLORS,
  nEpochsOf,
  nFlopsOf,
  numTrainStepsOf,
  parseRunName,
  parseTargetSteps,
  recentStepPoints,
  secsAgo,
  stepsInWindow,
  timeAgo,
} from './runMeta'
import { formatFlops, useFlopUnit } from './flops'
import { tagsFor, type RunTag } from './tags'

// ── small SVG sparkline (no plotly) ─────────────────────────────────────────
export function Sparkline({
  pts, width = 160, height = 36, color = '#22863a',
}: {
  pts: { x: number; y: number }[]
  width?: number
  height?: number
  color?: string
}) {
  if (pts.length < 2) {
    return <svg width={width} height={height} />
  }
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  const xmin = Math.min(...xs), xmax = Math.max(...xs)
  const ymin = Math.min(...ys), ymax = Math.max(...ys)
  const xspan = xmax - xmin || 1, yspan = ymax - ymin || 1
  const pad = 2
  const scaledX = (x: number) => pad + ((x - xmin) / xspan) * (width - 2 * pad)
  const scaledY = (y: number) => height - pad - ((y - ymin) / yspan) * (height - 2 * pad)
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaledX(p.x).toFixed(1)} ${scaledY(p.y).toFixed(1)}`).join(' ')
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  )
}

// ── shared types ────────────────────────────────────────────────────────────

export interface RunCardData {
  id: string
  manifest: RunManifest | null
  job: IrisJob | null
  history: RunHistory | null
  evalJobs: EvalJob[]
  color: string
  err: string | null
  /** Per-task attempt history sidecar (death events + per-attempt windows).
   *  Currently only fetched on the run-detail page; index cards leave this
   *  null. Used by RunHeaderRich to render a "% training" chip. */
  attempts?: IrisAttempts | null
  /** Modal app (+ any probed function calls) hosting this run, when the
   *  run name matches the `-modal-` convention. Null for TPU runs or
   *  when the modal snapshot doesn't have a matching app. Used by
   *  `ModalBadge` to replace the wandb-state fallback with real Modal
   *  state (cache-build / queued / running / failed). */
  modalApp?: ModalApp | null
  /** Per-run Modal function-call binding, when known. The fc-id comes from
   *  `pending_fires.json`'s `run_id → fc_id` map (the spawn wrapper records
   *  it at fire time); the resolved `ModalFunctionCall` (state, container)
   *  comes from `modal-state.json` via `modalFcForPending`. Null when no
   *  pending-fire is recorded for this run (true for any Modal run fired
   *  before the spawn wrapper added pending-fire bookkeeping), OR when
   *  `tomat modal sync --fc-id <fc>` hasn't probed the fc yet (state
   *  unknown). For the latter case use `modalFcId` instead — the raw fc-id
   *  is available even without a Modal-state probe. */
  modalFc?: ModalFunctionCall | null
  /** Raw per-run Modal fc-id from pending-fires, independent of whether the
   *  Modal-state snapshot has probed it. Used by the Modal-icon deep link
   *  (which only needs the id, not the runtime state). Null only when no
   *  pending-fire is recorded for this run. */
  modalFcId?: string | null
  /** Spec 46 Phase A: tiny MSRP summary inlined in `/api/runs-snapshot.json`.
   *  Used by the "$X MSRP" chip on cards + detail-page header. Null when
   *  the run hasn't been `tomat cost compute`-d yet. */
  cost?: RunCostSummary | null
}

// ── status badges/dots ──────────────────────────────────────────────────────

// Fallback for runs not in the iris snapshot (e.g. Modal runs, or runs created
// after the last `tomat iris sync`). State here comes from wandb's view —
// "running" / "finished" / "crashed" / "failed" — and is laggier than iris.
// Italic styling marks it as lower-trust.
const WANDB_STATE_BG: Record<string, string> = {
  running: '#22863a',
  finished: '#0366d6',
  crashed: '#cb2431',
  failed: '#cb2431',
  killed: '#6a737d',
}

const IRIS_STATE_STYLES: Record<string, { bg: string; fg: string }> = {
  RUNNING:       { bg: '#22863a', fg: '#fff' },
  PENDING:       { bg: '#d4a017', fg: '#fff' },
  BUILDING:      { bg: '#d4a017', fg: '#fff' },
  SUCCEEDED:     { bg: '#0366d6', fg: '#fff' },
  FAILED:        { bg: '#cb2431', fg: '#fff' },
  KILLED:        { bg: '#6a737d', fg: '#fff' },
  WORKER_FAILED: { bg: '#cb2431', fg: '#fff' },
  UNSCHEDULABLE: { bg: '#cb2431', fg: '#fff' },
}

// Map iris task-state names → 1-letter abbreviations for the compact
// per-task histogram chip (e.g. `RUNNING (1r/3p)` = 1 running, 3 pending).
// Order here is the order we render — running first, then pending /
// building (in-flight states), then terminal/failure states. Keys NOT in
// this map fall back to their first letter (last-ditch fallback).
const TASK_STATE_LETTER: Record<string, string> = {
  running: 'r',
  pending: 'p',
  building: 'b',
  completed: 'c',
  failed: 'f',
  killed: 'k',
  preempted: 'x',
  unschedulable: 'u',
}
const TASK_STATE_ORDER = [
  'running', 'pending', 'building', 'completed',
  'failed', 'killed', 'preempted', 'unschedulable',
]

/** Render the per-task-state histogram tail like ` (1r/3p)`. Returns ''
 *  when every task is in the job-level state (so the badge stays clean
 *  for the all-healthy case). Returns '' too when the snapshot pre-dates
 *  `task_state_counts`. */
function taskStateTail(job: IrisJob): string {
  const tsc = job.task_state_counts
  if (!tsc) return ''
  const entries = Object.entries(tsc).filter(([, v]) => v > 0)
  if (entries.length === 0) return ''
  // All-healthy: job is RUNNING and every task is also running. Skip the tail.
  if (job.state === 'RUNNING'
      && entries.length === 1
      && entries[0][0] === 'running'
      && entries[0][1] === job.num_tasks) {
    return ''
  }
  // Same idea for terminal states: if every task matches the job state,
  // the chip would be redundant noise. (PENDING/BUILDING jobs typically
  // have N pending/building tasks; collapse those too.)
  const jobStateLower = job.state.toLowerCase()
  if (entries.length === 1
      && entries[0][0] === jobStateLower
      && entries[0][1] === job.num_tasks) {
    return ''
  }
  const parts: string[] = []
  const seen = new Set<string>()
  for (const k of TASK_STATE_ORDER) {
    const v = tsc[k]
    if (v && v > 0) {
      parts.push(`${v}${TASK_STATE_LETTER[k] ?? k[0]}`)
      seen.add(k)
    }
  }
  // Any leftover (unknown) states — show with first-letter fallback.
  for (const [k, v] of entries) {
    if (!seen.has(k) && v > 0) {
      parts.push(`${v}${TASK_STATE_LETTER[k] ?? k[0]}`)
    }
  }
  return parts.length > 0 ? ` (${parts.join('/')})` : ''
}

/** Verbose tooltip line: `tasks: 4 pending` etc. */
function taskStateLine(job: IrisJob): string {
  const tsc = job.task_state_counts
  if (!tsc) return ''
  const entries = Object.entries(tsc).filter(([, v]) => v > 0)
  if (entries.length === 0) return ''
  // Render in our canonical order, then any leftovers alphabetically.
  const sorted: [string, number][] = []
  const seen = new Set<string>()
  for (const k of TASK_STATE_ORDER) {
    const v = tsc[k]
    if (v && v > 0) { sorted.push([k, v]); seen.add(k) }
  }
  for (const [k, v] of [...entries].sort()) {
    if (!seen.has(k) && v > 0) sorted.push([k, v])
  }
  return `tasks (${job.num_tasks}): ` + sorted.map(([k, v]) => `${v} ${k}`).join(', ')
}

/** Detect "iris parent says RUNNING but every task is pending and we have
 *  failures" — the crash-loop pathology where iris keeps bouncing the
 *  gang because each attempt dies in <3 min. Symptoms: `state == RUNNING`,
 *  `task_state_counts.pending == num_tasks` (zero running tasks),
 *  `failures > 0`. The job-level state lies; per-task state and the
 *  failure counter tell the truth.
 *
 *  Two data sources, in priority order:
 *    1. `job.task_state_counts` — populated by recent `tomat iris sync`
 *       passes (schema with that field). Fast path: one read.
 *    2. `attempts.tasks[*].state` — populated by `iris attempts-dump`
 *       sidecar (always present when the sidecar is present, regardless
 *       of which iris-state.json schema is live).
 *
 *  When neither source is available (e.g. older runs without an attempts
 *  sidecar AND an older iris-state schema), we conservatively return false
 *  — better to underflag than to mislabel a healthy run.
 *
 *  See specs/45-dashboard-tz11-surfacing.md for the standing rule.
 */
export function isCrashLoop(job: IrisJob, attempts?: IrisAttempts | null): boolean {
  if (job.state !== 'RUNNING') return false
  if (job.failures <= 0) return false
  const tsc = job.task_state_counts
  if (tsc) {
    const pending = tsc.pending ?? 0
    const running = tsc.running ?? 0
    if (running > 0) return false
    if (pending + (tsc.building ?? 0) < job.num_tasks) return false
    return true
  }
  // Fallback: derive per-task running/pending counts from the attempts
  // sidecar. This makes the crash-loop badge work even when the iris-state
  // snapshot was written by an older schema (no task_state_counts) — as
  // long as the per-label attempts sidecar is fresh.
  if (attempts) {
    let running = 0
    let pendingOrBuilding = 0
    for (const t of attempts.tasks) {
      if (t.state === 'running') running++
      else if (t.state === 'pending' || t.state === 'building') pendingOrBuilding++
    }
    if (running > 0) return false
    if (pendingOrBuilding < job.num_tasks) return false
    return true
  }
  return false
}

export function IrisBadge({ job, attempts, incomplete }: {
  job: IrisJob
  attempts?: IrisAttempts | null
  incomplete?: boolean
}) {
  // `incomplete`: iris says SUCCEEDED but the run stopped well short of its
  // step target — almost always a preemption whose clean SIGTERM exit (0)
  // iris mis-buckets as success. Show it as its own burnt-orange state so the
  // card doesn't read as a healthy finish (iris won't re-enqueue it).
  const showIncomplete = incomplete && job.state === 'SUCCEEDED'
  // Crash-loop: parent RUNNING but tasks all pending + failures > 0. The
  // iris state field is lying; honor the contradiction by labeling the
  // badge for what's actually happening (a restart cascade). Red/orange
  // (not green) — distinct from a healthy RUNNING.
  const crashLoop = !showIncomplete && isCrashLoop(job, attempts ?? null)
  const style = showIncomplete
    ? { bg: '#b4632a', fg: '#fff' }
    : crashLoop
      ? { bg: '#cb2431', fg: '#fff' }
      : IRIS_STATE_STYLES[job.state] ?? { bg: '#888', fg: '#fff' }
  // Compact per-task-state histogram. Replaces the old `(p=N, f=M)`
  // preemption/failure counters — those still appear in the tooltip but
  // are noisy on the badge itself once a run has churned for hours.
  // Falls back to the old `(p,f)` format when `task_state_counts` is
  // missing (older snapshots) so nothing regresses.
  const stateTail = taskStateTail(job)
  const legacyTail = stateTail
    ? ''
    : (job.preempts > 0 || job.failures > 0
       ? ` (p=${job.preempts}, f=${job.failures})` : '')
  // Crash-loop: lead with failure count — that's the metric the human
  // cares about ("how many failed restarts so far?"). Per-task histogram
  // stays in the tooltip.
  const crashLoopTail = ` (f=${job.failures})`
  const tail = crashLoop ? crashLoopTail : (stateTail || legacyTail)
  const tooltipParts: string[] = []
  if (showIncomplete) {
    tooltipParts.push(
      `iris reported SUCCEEDED, but the run ended early — likely preempted `
      + `with a clean SIGTERM exit. iris will not re-enqueue it.`,
    )
  } else if (crashLoop) {
    tooltipParts.push(
      `iris parent is RUNNING but 0 of ${job.num_tasks} tasks have started; `
      + `${job.failures} task-level failures so far — restart cascade.`,
    )
  } else if (job.error) {
    tooltipParts.push(job.error)
  } else {
    tooltipParts.push(`iris state=${job.state}`)
  }
  const taskLine = taskStateLine(job)
  if (taskLine) tooltipParts.push(taskLine)
  tooltipParts.push(`preempts=${job.preempts} failures=${job.failures}`)
  const label = showIncomplete
    ? 'INCOMPLETE'
    : crashLoop
      ? 'CRASH-LOOP'
      : job.state
  return (
    <Tooltip content={tooltipParts.join(' · ')}>
      <span
        style={{
          backgroundColor: style.bg, color: style.fg,
          padding: '1px 6px', borderRadius: 3,
          fontSize: '0.75rem', fontFamily: 'monospace',
        }}
      >
        {label}{tail}
      </span>
    </Tooltip>
  )
}

// ── Modal badge ─────────────────────────────────────────────────────────────
// For runs hosted on Modal (no iris job, name matches `-modal-`), surface
// real Modal state — app deployed/stopped + per-input fc state — instead of
// the stale wandb-session-state fallback. Replaces the old wandb-running
// chip for Modal runs; the iris case is untouched.
const MODAL_APP_STATE_STYLES: Record<string, { bg: string; fg: string }> = {
  deployed:      { bg: '#22863a', fg: '#fff' },
  initializing:  { bg: '#d4a017', fg: '#fff' },
  ephemeral:     { bg: '#22863a', fg: '#fff' },
  detached:      { bg: '#0366d6', fg: '#fff' },
  stopping:      { bg: '#b4632a', fg: '#fff' },
  stopped:       { bg: '#6a737d', fg: '#fff' },
  disabled:      { bg: '#6a737d', fg: '#fff' },
}

// Modal `GenericResult.GenericStatus` letters for the per-input tail chip,
// e.g. `deployed (1p)` = 1 pending input, `deployed (1f)` = 1 failed input.
// We collapse `success → S`, `failure → F`, etc.; `pending` (status 0) is
// the most actionable signal — a queued fc that hasn't started its container.
const MODAL_INPUT_STATUS_LETTER: Record<string, string> = {
  pending: 'p',
  success: 'S',
  failure: 'F',
  terminated: 'T',
  timeout: 'O',
  init_failure: 'I',
  internal_failure: 'X',
  idle_timeout: 'D',
}

function modalInputTail(call: ModalFunctionCall): string {
  if (call.inputs.length === 0) return ''
  // Aggregate per-status. We render only the non-success ones if there's
  // at least one terminal-bad status, since the success count is implied
  // by the app being `deployed` + the chip itself being shown.
  const counts: Record<string, number> = {}
  for (const inp of call.inputs) {
    counts[inp.status] = (counts[inp.status] ?? 0) + 1
  }
  const parts: string[] = []
  for (const [k, v] of Object.entries(counts)) {
    parts.push(`${v}${MODAL_INPUT_STATUS_LETTER[k] ?? k[0]}`)
  }
  return parts.length > 0 ? ` (${parts.join('/')})` : ''
}

export function ModalBadge({ app, wandbStale = false }: { app: ModalApp; wandbStale?: boolean }) {
  // Surface the most-recent fc's input statuses on the badge tail.
  // We sort fc-ids descendingly (ULID-ish so this is created-time order)
  // and take the latest; the tooltip lists all of them.
  const fcs = Object.values(app.function_calls)
  fcs.sort((a, b) => b.function_call_id.localeCompare(a.function_call_id))
  const latestFc = fcs[0]
  // Distinguish four cases the bare app-state hides:
  //   1. `deployed + n_running_tasks > 0`: containers are actually executing
  //      → label "RUNNING (Nr)" in green. The Modal app-state stays
  //      `deployed` regardless of whether work is happening, so without
  //      this branch every actively-training run reads as "DEPLOYED (1p)"
  //      and a casual reader can't tell training from idle.
  //   2. `deployed + no running tasks + all fcs successful`: the fc(s)
  //      finished. Modal leaves the app deployed for ~10 min before
  //      tearing it down, so a successfully-completed run would otherwise
  //      read "DEPLOYED (1S)" — actively misleading for a done job. Label
  //      "SUCCEEDED" in green.
  //   3. `deployed + no running tasks + no fcs at all`: zombie/idle app
  //      between fires. Label "IDLE" in amber.
  //   4. Anything else: pass through (`stopped`, `stopping`, mid-failure
  //      fcs, etc.) so unusual states stay visible.
  const isActuallyRunning = app.state === 'deployed' && app.n_running_tasks > 0
  const allInputs = fcs.flatMap((fc) => fc.inputs)
  const isAllSucceeded = app.state === 'deployed'
    && app.n_running_tasks === 0
    && fcs.length > 0
    && allInputs.length > 0
    && allInputs.every((i) => i.status === 'success')
  const isDeployedIdle = app.state === 'deployed'
    && app.n_running_tasks === 0
    && fcs.length === 0
  let label: string
  let style: { bg: string; fg: string }
  let tail: string
  if (isActuallyRunning && wandbStale) {
    // Modal container is alive but wandb hasn't logged in >10 min →
    // cache build / JAX compile / ckpt load in progress, not training.
    label = 'BUILDING'
    style = { bg: '#b08800', fg: '#fff' }
    tail = ` (${app.n_running_tasks}r)`
  } else if (isActuallyRunning) {
    label = 'RUNNING'
    style = { bg: '#22863a', fg: '#fff' }
    tail = ` (${app.n_running_tasks}r)`
  } else if (isAllSucceeded) {
    label = 'SUCCEEDED'
    style = { bg: '#22863a', fg: '#fff' }
    tail = ''
  } else if (isDeployedIdle) {
    label = 'IDLE'
    style = { bg: '#b08800', fg: '#fff' }
    tail = ''
  } else {
    label = app.state.toUpperCase()
    style = MODAL_APP_STATE_STYLES[app.state] ?? { bg: '#888', fg: '#fff' }
    tail = latestFc
      ? modalInputTail(latestFc)
      : (app.n_running_tasks > 0 ? ` (${app.n_running_tasks}r)` : '')
  }
  // Build a verbose tooltip.
  const tooltipLines: string[] = [
    `modal: app=${app.description} state=${app.state}`,
  ]
  if (app.n_running_tasks > 0) {
    tooltipLines.push(`running tasks: ${app.n_running_tasks}`)
  }
  for (const fc of fcs) {
    if (fc.error) {
      tooltipLines.push(`fc ${fc.function_call_id.slice(0, 16)}…: ERROR ${fc.error}`)
      continue
    }
    const summary = fc.inputs.map((i) =>
      `${i.status}${i.task_id ? ` (${i.task_id.slice(0, 10)}…)` : ''}`,
    ).join(', ')
    tooltipLines.push(
      `fc ${fc.function_call_id.slice(0, 16)}… (${fc.function_name}): ${summary || 'no inputs'}`,
    )
  }
  return (
    <Tooltip content={tooltipLines.join(' · ')}>
      <span
        style={{
          backgroundColor: style.bg, color: style.fg,
          padding: '1px 6px', borderRadius: 3,
          fontSize: '0.75rem', fontFamily: 'monospace',
        }}
      >
        {label}{tail}
      </span>
    </Tooltip>
  )
}

// Two/three dots per card — iris job state + wandb run state, plus a
// modal dot when the run is Modal-hosted. Replaces the old italic
// distinction: colour shows each source's health, so a healthy run
// reads as green-green dots and an iris/wandb disagreement is visible
// at a glance. A hollow dot = that source has nothing (e.g. no iris
// job for Modal runs; no modal app for TPU runs).
export const DOT = {
  green: '#22c55e', amber: '#f59e0b', blue: '#3b82f6',
  orange: '#b4632a', red: '#ef4444', grey: '#6a737d',
}

function modalDotColor(app: ModalApp): string {
  // Real state preference: any non-success fc input → red; pending → amber;
  // deployed + n_running_tasks > 0 → green; deployed but no running task and
  // no fc activity → amber (deployed-but-idle, e.g. between fires);
  // stopped/stopping → grey.
  const allInputs = Object.values(app.function_calls).flatMap((fc) => fc.inputs)
  if (allInputs.some((i) =>
    i.status === 'failure' || i.status === 'init_failure'
    || i.status === 'internal_failure' || i.status === 'terminated'
    || i.status === 'timeout' || i.status === 'idle_timeout',
  )) return DOT.red
  if (allInputs.some((i) => i.status === 'pending')) return DOT.amber
  if (allInputs.length > 0 && allInputs.every((i) => i.status === 'success')) {
    return DOT.blue
  }
  switch (app.state) {
    case 'deployed':
    case 'ephemeral':
      return app.n_running_tasks > 0 ? DOT.green : DOT.amber
    case 'initializing':
      return DOT.amber
    case 'stopping':
      return DOT.orange
    case 'stopped':
    case 'disabled':
      return DOT.grey
    default: return DOT.grey
  }
}

function irisDotColor(job: IrisJob, incomplete: boolean): string {
  switch (job.state) {
    case 'RUNNING': {
      // Cascade-restart: job-level state lingers on RUNNING because some
      // task is briefly RUNNING each cycle, but the per-task histogram says
      // 0 are actually running. Drop to amber so the dot doesn't lie.
      const tsc = job.task_state_counts
      if (tsc && job.num_tasks > 0) {
        const running = tsc.running ?? 0
        if (running === 0) return DOT.amber
      }
      return DOT.green
    }
    case 'PENDING':
    case 'BUILDING': return DOT.amber
    case 'SUCCEEDED': return incomplete ? DOT.orange : DOT.blue
    case 'FAILED':
    case 'WORKER_FAILED':
    case 'UNSCHEDULABLE': return DOT.red
    default: return DOT.grey // KILLED / unknown
  }
}

// wandb's `state` lags, and a crashed run often never flips off "running" —
// so a "running" run that hasn't logged in >10 min is treated as stale (grey).
function wandbDotColor(state: string, logAgeSec: number | null): string {
  if (state === 'running') {
    return logAgeSec != null && logAgeSec < 600 ? DOT.green : DOT.grey
  }
  if (state === 'finished') return DOT.blue
  if (state === 'crashed' || state === 'failed') return DOT.red
  return DOT.grey // killed / unknown
}

function Dot({ color, hollow, title }: { color?: string; hollow?: boolean; title: string }) {
  return (
    <Tooltip content={title}>
      <span
        style={{
          display: 'inline-block', width: 9, height: 9, borderRadius: '50%',
          flexShrink: 0,
          backgroundColor: hollow ? 'transparent' : color,
          border: hollow ? '1px solid #555' : 'none',
        }}
      />
    </Tooltip>
  )
}

function StatusDots({ job, modalApp, manifest, incomplete, lastLogTs }: {
  job: IrisJob | null
  modalApp: ModalApp | null
  manifest: RunManifest | null
  incomplete: boolean
  lastLogTs: number | null
}) {
  const logAgeSec = lastLogTs != null ? Math.max(0, Date.now() / 1000 - lastLogTs) : null
  const loggedStr = lastLogTs != null ? `last logged ${secsAgo(lastLogTs)}` : 'never logged'
  // For Modal runs we surface a Modal dot in place of the iris-slot's
  // hollow-empty dot. Runs not matching the `-modal-` pattern (TPU
  // runs) leave the modal slot off the row entirely — no point
  // claiming "no Modal app" for a run that was never meant to run there.
  return (
    <span style={{ display: 'inline-flex', gap: 3, alignItems: 'center' }}>
      {job
        ? <Dot
            color={irisDotColor(job, incomplete)}
            title={(() => {
              const lines = [`iris: ${job.state}`]
              const tl = taskStateLine(job)
              if (tl) lines.push(tl)
              return lines.join(' · ')
            })()}
          />
        : (modalApp
            ? <Dot
                color={modalDotColor(modalApp)}
                title={`modal: ${modalApp.description} ${modalApp.state}`}
              />
            : <Dot hollow title="iris: no job (e.g. a Modal run)" />)}
      {manifest
        ? <Dot
            color={wandbDotColor(manifest.run?.state, logAgeSec)}
            title={`wandb: ${manifest.run?.state} · ${loggedStr}`}
          />
        : <Dot hollow title="wandb: no data yet" />}
    </span>
  )
}

// ── m-eval chip ─────────────────────────────────────────────────────────────
// m-eval (mat-NMAE) jobs are iris jobs named `tomat-eval-<run>-<set>-step-<N>`;
// `evalJobsByRun` (api.ts) groups them per run. The chip surfaces only the
// actionable states — in-flight or failed. Once every eval has finished, the
// MT/MV numbers themselves are the signal, so the chip hides.
function EvalChip({ jobs }: { jobs: EvalJob[] }) {
  if (jobs.length === 0) return null
  let flight = 0, failed = 0, done = 0
  for (const ej of jobs) {
    const p = evalPhase(ej.job)
    if (p === 'flight') flight++
    else if (p === 'failed') failed++
    else done++
  }
  if (flight === 0 && failed === 0) return null
  const label = flight > 0
    ? `⏳ ${flight} m-eval${flight > 1 ? 's' : ''}`
    : `⚠ ${failed} m-eval${failed > 1 ? 's' : ''} failed`
  return (
    <Tooltip content={`${jobs.length} m-eval job(s): ${flight} in flight · ${done} done · ${failed} failed`}>
      <span style={{
        backgroundColor: flight > 0 ? '#d4a017' : '#cb2431', color: '#fff',
        padding: '1px 6px', borderRadius: 3,
        fontSize: '0.75rem', fontFamily: 'monospace',
      }}>
        {label}
      </span>
    </Tooltip>
  )
}

// ── parent chip ─────────────────────────────────────────────────────────────

const isModifiedClick = (e: React.MouseEvent) =>
  e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0

// Short label = strip the long `train-full-v3-200M-bs128-emd-do-` prefix etc.
// Replicated here from RunsTimelinePlot.shortLabel via a regex match — the
// caller can also pass an explicit `shortLabel` to short-circuit.
function defaultShortLabel(id: string): string {
  return id
    .replace(/^train-full-v3-200M-bs128-emd-do-/, '')
    .replace(/^train-full-/, '')
    .replace(/^train-/, '')
}

/** Compact "← from <parent> @ step-N" chip. Renders nothing when the run has
 *  no recorded lineage in `RUN_LINEAGE`. Click behaviour:
 *   - if `onScrollToParent` returns true, the parent's card was scrolled into
 *     view (no further action);
 *   - else, if we know the parent's wandb URL, open that in a new tab;
 *   - else, navigate to `#/runs/<parent>` (the parent's detail page). */
function ParentChip({
  runId, parentWandbUrl, parentColor, onScrollToParent,
}: {
  runId: string
  parentWandbUrl: string | null
  /** Optional color swatch for the parent — when supplied, rendered as a
   *  small left-edge dot so the chip visually ties to the parent's trace
   *  color on the timeline. */
  parentColor?: string | null
  onScrollToParent?: (parentId: string) => boolean
}) {
  const lin = lineageFor(runId)
  if (!lin) return null
  const stepTail = lin.parent_step != null ? ` @ step-${lin.parent_step}` : ''
  const tipBase = `resumed from ${defaultShortLabel(lin.parent)}${stepTail}`
  const tip = onScrollToParent
    ? (parentWandbUrl
        ? `${tipBase}. Click to scroll to the parent's card (or open its wandb).`
        : `${tipBase}. Click to scroll to the parent's card.`)
    : `${tipBase}. Click to open the parent's detail page.`
  // On the detail page (no scroll target), the chip becomes a real link to the
  // parent's detail page — middle/cmd-click then opens it in a new tab.
  const href = onScrollToParent
    ? (parentWandbUrl ?? '#')
    : `#/runs/${lin.parent}`
  return (
    <Tooltip content={tip}>
      <a
        href={href}
        onClick={(e) => {
          if (isModifiedClick(e)) return
          // Don't bubble: the card body's onClick toggles pin.
          e.stopPropagation()
          e.preventDefault()
          if (onScrollToParent) {
            const scrolled = onScrollToParent(lin.parent)
            if (!scrolled && parentWandbUrl) {
              window.open(parentWandbUrl, '_blank', 'noopener')
            }
          } else {
            window.location.hash = `#/runs/${lin.parent}`
          }
        }}
        style={{
          fontSize: '0.7rem',
          fontFamily: 'monospace',
          color: '#9aa6c2',
          textDecoration: 'none',
          background: 'rgba(120,140,200,0.10)',
          border: '1px solid rgba(120,140,200,0.30)',
          borderRadius: 10,
          padding: '1px 8px',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 5,
        }}>
        {parentColor && (
          <span style={{
            display: 'inline-block',
            width: 8, height: 8, borderRadius: '50%',
            background: parentColor,
            flex: '0 0 auto',
          }} />
        )}
        ← {defaultShortLabel(lin.parent)}{stepTail}
      </a>
    </Tooltip>
  )
}

// ── tag chips (per-run) ─────────────────────────────────────────────────────
// Curated tags for this run. Clicking a chip navigates to
// `#/runs?tags=<tag>` — same routing the omnibar uses, so the dashboard
// opens already filtered to that tag.

function navigateToRunsWithTag(tag: RunTag) {
  const params = new URLSearchParams()
  params.set('tags', tag)
  window.location.hash = `#/runs?${params.toString()}`
}

function TagChips({ runId }: { runId: string }) {
  const tags = tagsFor(runId)
  if (tags.length === 0) return null
  return (
    <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 4,
                   alignItems: 'center' }}>
      {tags.map((t) => (
        <Tooltip key={t} content={`filter /runs to tag "${t}"`}>
          <a
            href={`#/runs?tags=${encodeURIComponent(t)}`}
            onClick={(e) => {
              if (isModifiedClick(e)) return
              e.stopPropagation()
              e.preventDefault()
              navigateToRunsWithTag(t)
            }}
            style={{
              fontSize: '0.66rem',
              fontFamily: 'monospace',
              color: '#9aa6c2',
              textDecoration: 'none',
              background: 'rgba(120,140,200,0.08)',
              border: '1px solid rgba(120,140,200,0.22)',
              borderRadius: 8,
              padding: '0px 6px',
              lineHeight: 1.5,
            }}
          >
            {t}
          </a>
        </Tooltip>
      ))}
    </span>
  )
}

// ── MSRP chip (spec 46 Phase A) ─────────────────────────────────────────────
// Renders "$X MSRP" + tooltip with the per-segment breakdown. Honest framing:
// "Estimated MSRP" / "equivalent retail value — not actual billing", with the
// pricing-table snapshot date. Do NOT change copy to "cost" / "spend" — see
// spec 46's "Important framing" section + `feedback_no_funding_in_public_repo`
// memory.

function formatMsrp(n: number): string {
  // Cents granularity at small values, whole dollars once we're past $10.
  if (n < 10) return `$${n.toFixed(2)}`
  if (n < 1000) return `$${Math.round(n)}`
  return `$${(n / 1000).toFixed(1)}k`
}

function formatWallclockShort(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`
  if (sec < 3600) return `${(sec / 60).toFixed(0)}m`
  if (sec < 86400) return `${(sec / 3600).toFixed(1)}h`
  return `${(sec / 86400).toFixed(2)}d`
}

function CostBreakdownTooltipBody({ cost, summary }: {
  cost: RunCost | null
  summary: RunCostSummary | null
}) {
  // Render copy that survives the spec's framing constraints. The summary
  // bits (total + snapshot date) come first so the user has them even when
  // the full breakdown is still fetching.
  const total = cost?.msrp_usd ?? summary?.msrp_usd ?? null
  const tableV = cost?.pricing_table_version ?? summary?.pricing_table_version ?? ''
  const isComplete = cost?.is_complete ?? summary?.is_complete ?? false
  return (
    <div style={{ fontFamily: 'sans-serif', fontSize: '0.78rem', lineHeight: 1.45,
                  maxWidth: 360 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Estimated MSRP</div>
      <div style={{ borderTop: '1px solid #555', margin: '4px 0' }} />
      {cost == null ? (
        <div style={{ color: '#bbb', fontStyle: 'italic' }}>loading breakdown…</div>
      ) : cost.breakdown.length === 0 ? (
        <div style={{ color: '#bbb', fontStyle: 'italic' }}>
          no billable segments — this run has no attempts sidecar nor
          iris-state job data.
        </div>
      ) : (
        <div style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>
          {cost.breakdown.map((row, i) => {
            if (row.kind === 'modal' && row.msrp_usd == null) {
              return (
                <div key={i} style={{ color: '#d4a017' }}>
                  Modal MSRP TBD <span style={{ color: '#888' }}>(Phase B)</span>
                </div>
              )
            }
            const wallclock = formatWallclockShort(row.wallclock_sec)
            const rate = row.rate_per_hr_usd != null
              ? `$${row.rate_per_hr_usd.toFixed(2)}/hr` : '—'
            const msrp = row.msrp_usd != null
              ? `$${row.msrp_usd.toFixed(2)}` : '—'
            return (
              <div key={i}>
                {row.label} · {wallclock} · {rate} → {msrp}
                <div style={{ color: '#888', fontSize: '0.68rem', marginLeft: 12 }}>
                  {row.source}
                </div>
              </div>
            )
          })}
        </div>
      )}
      <div style={{ borderTop: '1px solid #555', margin: '4px 0' }} />
      <div>
        Total: <b>{total != null ? `$${total.toFixed(2)}` : '—'}</b>
        {isComplete ? '' : ' (so far)'}
      </div>
      {tableV && (
        <div style={{ color: '#888', marginTop: 3, fontSize: '0.72rem' }}>
          Snapshot pricing {tableV} — equivalent retail value, not actual billing.
        </div>
      )}
      {cost && cost.notes.length > 0 && (
        <div style={{ color: '#b08800', marginTop: 4, fontSize: '0.7rem' }}>
          {cost.notes.map((n, i) => <div key={i}>{n}</div>)}
        </div>
      )}
    </div>
  )
}

function CostChip({ runId, summary }: {
  runId: string
  summary: RunCostSummary | null
  /** Caller-side: was this a Modal run? Kept for future re-use even
   *  though Phase B no longer special-cases Modal-only on the chip side
   *  (real msrp_usd now lands on every sync). */
  isModalOnly?: boolean
}) {
  // Lazy-fetch the full breakdown on tooltip hover. Until the user hovers,
  // we render only what's in the snapshot's summary (msrp_usd + snapshot
  // date) so first-paint is one R2 round-trip lighter per run.
  const costQ = useQuery({
    queryKey: ['runCost', runId],
    queryFn: () => fetchRunCost(runId),
    enabled: false,  // gated; we call refetch() on mouse-enter below
    staleTime: 60_000,
  })
  if (summary == null) {
    // No sidecar yet — hide the chip rather than render an alarming
    // placeholder. The `runs sync` flow auto-populates cost.json on
    // every sync (spec 46 Phase A/B), so the gap is briefly the time
    // between run creation and the next sync. `isModalOnly` is no
    // longer special-cased — Modal cost is now computed too.
    return null
  }
  // Phase B writes real Modal msrp_usd values via wandb-tracked wallclock,
  // so a non-null summary with msrp_usd > 0 always renders as $X MSRP.
  // `has_modal_pending` only fires now when a Modal segment ships with
  // null msrp_usd (the "no usable wallclock signal yet" placeholder),
  // which we keep separate from the `$X MSRP` chip's happy path.
  const label = summary.has_modal_pending && summary.msrp_usd === 0
    ? 'Modal MSRP TBD'
    : `${formatMsrp(summary.msrp_usd)} MSRP`
  return (
    <Tooltip content={
      <CostBreakdownTooltipBody cost={costQ.data ?? null} summary={summary} />
    }>
      <span
        onMouseEnter={() => { if (!costQ.data && !costQ.isFetching) costQ.refetch() }}
        style={{ color: '#9aa6c2', fontFamily: 'monospace', cursor: 'help' }}>
        {label}
      </span>
    </Tooltip>
  )
}

// ── RunHeaderRich ───────────────────────────────────────────────────────────

const navigate = (path: string) => {
  window.location.hash = `#/${path}`
}

export interface RunHeaderRichProps {
  data: RunCardData
  /** wandb URL for the parent run (looked up by the caller from the visible
   *  manifest map). When set, the ParentChip will use it as a fallback. */
  parentWandbUrl?: string | null
  /** Trace color the parent uses on the timeline plot — rendered as a small
   *  swatch on the ParentChip so the eye can map a child card to its parent
   *  trace at a glance. */
  parentColor?: string | null
  /** Try to scroll the parent's card into view (cards-only). When omitted
   *  (detail page), the ParentChip falls back to navigating to the parent's
   *  detail page. */
  onScrollToParent?: (parentId: string) => boolean
  /** When true (the default on cards), the run name is a link to the detail
   *  page. On the detail page itself we render it as plain text. */
  linkRunName?: boolean
}

export function RunHeaderRich({
  data, parentWandbUrl, parentColor, onScrollToParent, linkRunName = true,
}: RunHeaderRichProps) {
  const { id, manifest, job, history, evalJobs, err, attempts, modalApp } = data
  const incomplete = isIncomplete(data)
  // "wandbStale" = wandb hasn't received a metric in a while. We do NOT
  // require `run.state === 'running'` for this — per
  // feedback_no_wandb_for_state, wandb's run state is unreliable (it
  // stamps "crashed" on any client disconnect, including intentional
  // kills). Just look at the last-log timestamp, treat anything older
  // than 10 min as silent. Modal/iris signals are what drive RUNNING vs
  // BUILDING vs UNKNOWN downstream.
  const lastLogTs = manifest?.history?.ts_max ?? null
  const wandbStale = lastLogTs != null && (Date.now() / 1000 - lastLogTs) >= 600

  const meta = parseRunName(id)
  const hwColor = meta.hardwareKind ? HW_COLORS[meta.hardwareKind] : '#888'
  const wbUrl = manifest?.run?.url
  // Prefer `trainer.num_train_steps` from the manifest config (authoritative,
  // reflects most-recent resume target) over the run-name parse (which freezes
  // the *original* target — e.g. cont7k-ext was named …-8k but is now resumed
  // to 80k, so the name says "8k" while the trainer is targeting 80k).
  const trainerNumSteps = numTrainStepsOf(manifest)
  const targetSteps = trainerNumSteps ?? (meta.targetSteps ? parseTargetSteps(meta.targetSteps) : null)
  const targetLabel = trainerNumSteps != null
    ? formatStepCount(trainerNumSteps)
    : (meta.targetSteps ?? null)
  // Steps completed. Three possible sources, in order of preference:
  //   1. `history.last_train_step` — max non-null `global_step` from the
  //      parquet's train-step rows. Updated EVERY sync from the parquet
  //      column, so it's current even between ckpt boundaries.
  //      Authoritative; preferred. Optional on older manifests.
  //   2. `summary.global_step` — Levanter's training step from wandb summary.
  //      Only updated on ckpt save / eval-end, so stale or missing on
  //      runs that haven't hit their first ckpt boundary yet.
  //   3. `summary._step` — wandb's auto log counter. AVOID: bumped by
  //      `cluster/*` preempt-watch pings too, so it overcounts real training
  //      progress for runs whose training has stalled or hasn't started.
  // Both (1) and (2) are 0-indexed (Levanter convention) so we display +1.
  const ltsRaw = manifest?.history?.last_train_step
  const gsRaw = manifest?.summary?.['global_step']
  const lastGlobalStep =
    typeof ltsRaw === 'number' ? ltsRaw
    : typeof gsRaw === 'number' ? gsRaw
    : null
  const stepsDoneRaw = lastGlobalStep != null ? lastGlobalStep + 1 : null
  const stepsDone = stepsDoneRaw != null
    ? (targetSteps != null ? Math.min(stepsDoneRaw, targetSteps) : stepsDoneRaw)
    : null
  const progressPct = stepsDone != null && targetSteps != null
    ? Math.min(100, (stepsDone / targetSteps) * 100)
    : null
  const trainLoss = manifest?.summary?.['train/loss']
  const evalLoss = manifest?.summary?.['eval/loss']
  const mfu = manifest?.summary?.['throughput/mfu']
  // MT/MV — latest mat-NMAE on the {train,val}_200 mat snapshots. Stored already
  // as a percentage (cont7k-ext logs 1.73 = 1.73%), so display as-is — no ×100.
  const mtNmae = manifest?.summary?.['eval/mat_nmae/train_200/mean']
  const mvNmae = manifest?.summary?.['eval/mat_nmae/val_200/mean']
  // Epoch displayed in the chip. For runs declared in `SEGMENTS` (lineage.ts),
  // compute segment-aware epoch-at-current-step so the chip + tooltip
  // breakdown agree. Otherwise fall back to `nEpochsOf` (the original
  // total_tokens-driven scalar) so undeclared runs render identically.
  const epochBreakdown = stepsDoneRaw != null
    ? epochBreakdownAtStep(stepsDoneRaw, manifest)
    : null
  const epochSegmented = stepsDoneRaw != null && epochBreakdown != null
    ? epochOfStep(stepsDoneRaw, manifest)
    : null
  const nEpochs = epochSegmented ?? nEpochsOf(manifest)
  const nFlops = nFlopsOf(manifest)
  // FLOP-unit preference (`?fopu=EF|PF|TF|sci`). Default EF. Same hook on
  // every card so the chip group below the plot drives every per-card pill.
  const [flopUnit] = useFlopUnit()

  const sparkPts = useMemo(() => history ? recentStepPoints(history) : [], [history])
  const lastTs = sparkPts.length > 0 ? sparkPts[sparkPts.length - 1].x : null
  const freshSec = lastTs != null ? Math.max(0, Date.now() / 1000 - lastTs) : null

  // % training: fraction of submitted→finished wallclock spent in
  // training-active windows (per the attempts sidecar × the train/loss-row
  // span). null when we don't have the sidecar / no training has happened
  // yet — in which case the chip just hides.
  const trainFrac = useMemo(
    () => computeTrainingFraction(job, attempts ?? null, history),
    [job, attempts, history],
  )

  // "Steps in last 1m / 1h / 6h" — anchored to the latest log timestamp (so a
  // run that logged its last step 2 min ago still reports its 1m rate from
  // when it was actually stepping).
  const rate1m = useMemo(() => history ? stepsInWindow(history, 60) : null, [history])
  const rate1h = useMemo(() => history ? stepsInWindow(history, 3600) : null, [history])
  const rate6h = useMemo(() => history ? stepsInWindow(history, 6 * 3600) : null, [history])

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr auto',
      gap: '0.5rem 1rem',
      overflow: 'hidden',
    }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <StatusDots
            job={job}
            modalApp={modalApp ?? null}
            manifest={manifest}
            incomplete={incomplete}
            lastLogTs={lastLogTs}
          />
          {job && <IrisBadge job={job} attempts={attempts} incomplete={incomplete} />}
          {!job && modalApp && <ModalBadge app={modalApp} wandbStale={wandbStale} />}
          {!job && !modalApp && (() => {
            // No iris job AND no matched Modal app → we lack a live
            // ground-truth signal. Wandb's `crashed` state is unreliable
            // (stamps on any client disconnect, including intentional
            // kills) — DON'T infer running/crashed from wandb. BUT
            // wandb's `finished` state IS reliable: the trainer only
            // calls `wandb.finish()` after a clean exit; nothing else
            // can write that state. So we can show FINISHED when
            // wandb says so, even without iris/Modal corroboration.
            // All other wandb states (`crashed`, `failed`, `running`,
            // `killed`) collapse to UNKNOWN per the memory
            // `feedback_no_wandb_for_state`.
            const wandbState = manifest?.run?.state
            if (wandbState === 'finished') {
              return (
                <Tooltip content="wandb explicitly reported `finished` (trainer called wandb.finish() cleanly)">
                  <span
                    style={{
                      backgroundColor: '#1f6feb',
                      color: '#fff', padding: '1px 6px', borderRadius: 3,
                      fontSize: '0.75rem', fontFamily: 'monospace',
                    }}>
                    FINISHED
                  </span>
                </Tooltip>
              )
            }
            return (
              <Tooltip content={
                manifest
                  ? `no iris job or modal app matched; wandb last state was "${wandbState}" but we don't infer state from wandb (only "finished" is reliable)`
                  : 'no iris job, no modal app, no manifest — nothing to derive state from'
              }>
                <span
                  style={{
                    backgroundColor: '#6a737d',
                    color: '#fff', padding: '1px 6px', borderRadius: 3,
                    fontSize: '0.75rem', fontFamily: 'monospace',
                  }}>
                  UNKNOWN
                </span>
              </Tooltip>
            )
          })()}
          {linkRunName ? (
            <a
              href={`#/runs/${id}`}
              onClick={(e) => {
                if (isModifiedClick(e)) return
                e.preventDefault()
                navigate(`runs/${id}`)
              }}
              style={{ fontFamily: 'monospace', fontSize: '0.9rem' }}
            >
              {id}
            </a>
          ) : (
            <span style={{ fontFamily: 'monospace', fontSize: '0.9rem',
                           color: '#ddd' }}>{id}</span>
          )}
          {wbUrl && (
            <Tooltip content="open this run in wandb">
              <a href={wbUrl} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.75rem', color: '#888',
                  display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                {/* wandb.svg ships as the full horizontal lockup
                    (icon + text + tagline); crop to just the leftmost
                    icon square via overflow:hidden + width constraint. */}
                <span style={{ display: 'inline-block', width: 18, height: 18,
                  overflow: 'hidden', verticalAlign: 'middle' }}>
                  <img src="/wandb.svg" alt="wandb"
                    style={{ height: 18, width: 'auto', display: 'block' }} />
                </span>
                ↗
              </a>
            </Tooltip>
          )}
          {isModalRun(id) && (() => {
            // The modal LINK is independent of whether the run still has
            // a live `modalApp` association. The badge-side `RunsPage` drops
            // `data.modalApp` for stale runs (so the badge stops misclaiming
            // RUNNING), but a finished Modal run's logs are still useful to
            // open — gate on the run-name pattern instead.
            //
            // Modal URL pattern: /apps/<workspace>/main/deployed/<app_name>
            // Workspace is open-athena per `modal profile current`. We host
            // all training fires under one app (`tomat-train-smoke`); the
            // live `data.modalApp.description` matches that when present,
            // and `modalAppForRun` filters by the same constant, so falling
            // back to it is safe even without a live association.
            //
            // Deep-link to the specific function call when we have a
            // pending-fires-bound fc-id for this run. The plain app URL
            // would land on combined-across-all-fires logs — mostly noise
            // from sibling runs hosted under the same app. Fall back to
            // app URL only when no fc is recorded (e.g. older Modal runs
            // fired before the spawn wrapper added pending-fire bookkeeping).
            const liveApp = data.modalApp
            const appName = liveApp?.description ?? 'tomat-train-smoke'
            // Prefer `data.modalFcId` (raw fc-id from pending-fires; available
            // even if `tomat modal sync --fc-id <fc>` hasn't probed this fc
            // yet). `data.modalFc` only resolves when the probe has run,
            // which is intermittent for newly-fired runs.
            const fcId = data.modalFcId ?? data.modalFc?.function_call_id ?? null
            const appId = liveApp?.app_id ?? null
            // function_name comes from the probed fc only — we don't have it
            // from pending-fires alone.
            const fnName = data.modalFc?.function_name ?? null
            const fnId = (fnName && liveApp?.function_ids)
              ? liveApp.function_ids[fnName] ?? null
              : null
            // Modal URL pattern for per-call App Logs view (verified empirically
            // 2026-06-08, ryan's working example):
            //   /apps/<ws>/main/<ap-id>?activeTab=logs&functionId=<fu>&fcId=<fc>
            // The `logs` tab (NOT `calls`) is the most useful landing page —
            // full-width log stream + timestamp histogram, all filtered to
            // this specific call. Fallback chain:
            //   1. Have app_id + function_id + fc_id → full deep-link
            //   2. Have app_id + fc_id (no fn_id yet) → app + fc filter
            //   3. Have app_id alone → app overview (ap-id path)
            //   4. Nothing → app overview by name
            let appUrl: string
            if (appId && fnId && fcId) {
              appUrl = `https://modal.com/apps/open-athena/main/${appId}?activeTab=logs&functionId=${fnId}&fcId=${fcId}`
            } else if (appId && fcId) {
              appUrl = `https://modal.com/apps/open-athena/main/${appId}?activeTab=logs&fcId=${fcId}`
            } else if (appId) {
              appUrl = `https://modal.com/apps/open-athena/main/${appId}`
            } else {
              appUrl = `https://modal.com/apps/open-athena/main/deployed/${appName}`
            }
            const fcs = liveApp ? Object.values(liveApp.function_calls) : []
            fcs.sort((a, b) => b.function_call_id.localeCompare(a.function_call_id))
            const latestFc = fcs[0]
            const tooltip = fnId && fcId
              ? `open Modal logs (${fnName})\nfc ${fcId}`
              : fcId
                ? `open Modal app\nfc ${fcId} (function_id not yet captured — link may not pre-filter)`
                : liveApp
                  ? `open Modal app (${appId ?? appName})\n(no pending-fire fc-id for this run)`
                    + (latestFc ? `\nlatest fc seen in app: ${latestFc.function_call_id}` : '')
                  : `open Modal app (${appName})\n(this run has finished — no live app association)`
            return (
              <Tooltip content={tooltip}>
                <a href={appUrl} target="_blank" rel="noreferrer"
                  style={{ fontSize: '0.75rem', color: '#888',
                    display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                  <img src="/modal.png" alt="modal"
                    style={{ height: 18, width: 'auto', verticalAlign: 'middle' }} />
                  ↗
                </a>
              </Tooltip>
            )
          })()}
          {data.job && (() => {
            // iris dashboard URL pattern: `https://iris.oa.dev/#/job/%2Fryan%2F<id>`
            // (per iris-dashboard-share-pattern memory). The iris job-id is
            // the dict key in iris-state.json (`/ryan/<run-label>`); the
            // `IrisJob` object itself has no `job_id` field — earlier code
            // read `data.job.job_id` and got `undefined` here, so all iris
            // links rendered as `/#/job/undefined`. Reconstruct from `data.id`
            // (the run label) plus the standard `/ryan/` prefix that the
            // server-side `tomat iris sync` uses (matches
            // `trainingRunIdFromIrisJob` in RunsPage.tsx).
            const irisJobId = `/ryan/${data.id}`
            const irisUrl = `https://iris.oa.dev/#/job/${encodeURIComponent(irisJobId)}`
            return (
              <Tooltip content={`open iris job (${irisJobId})`}>
                <a href={irisUrl} target="_blank" rel="noreferrer"
                  style={{ fontSize: '0.75rem', color: '#888',
                    display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                  <img src="/tpu.png" alt="iris"
                    style={{ height: 18, width: 'auto', verticalAlign: 'middle' }} />
                  ↗
                </a>
              </Tooltip>
            )
          })()}
          <EvalChip jobs={evalJobs} />
          <ParentChip
            runId={id}
            parentWandbUrl={parentWandbUrl ?? null}
            parentColor={parentColor}
            onScrollToParent={onScrollToParent}
          />
          <TagChips runId={id} />
        </div>
        <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', color: '#ccc',
                      display: 'flex', flexWrap: 'wrap', gap: '0.4rem 0.9rem' }}>
          {meta.hardware && (
            <span><span style={{ color: hwColor, fontWeight: 600 }}>{meta.hardware}</span></span>
          )}
          {meta.model && <span>{meta.model}</span>}
          {meta.batchSize && <span>BS={meta.batchSize}</span>}
          {meta.lossType && <span>{meta.lossType}</span>}
          {targetLabel && (
            <Tooltip content={trainerNumSteps != null ? 'trainer.num_train_steps (authoritative)' : 'parsed from run name'}>
              <span>target {targetLabel}</span>
            </Tooltip>
          )}
          {meta.suffix && <span style={{ color: '#888' }}>· {meta.suffix}</span>}
        </div>
        <div style={{ marginTop: '0.3rem', fontSize: '0.75rem', color: '#888' }}>
          {manifest && (
            <>
              created {timeAgo(manifest.run?.created_at)}
              {manifest.history.ts_max != null && (
                <>
                  {' · '}
                  <span style={{
                    color: freshnessColor(
                      Math.max(0, Date.now() / 1000 - manifest.history.ts_max)),
                  }}>
                    logged {secsAgo(manifest.history.ts_max)}
                  </span>
                </>
              )}
              {' · '}synced {timeAgo(manifest.synced_at)}
              {manifest.run?.entity && manifest.run?.project && (
                <> · {manifest.run?.entity}/{manifest.run?.project}</>
              )}
            </>
          )}
          {err && <span style={{ color: 'crimson' }}>err: {err}</span>}
        </div>
      </div>
      <div style={{ minWidth: 200, fontSize: '0.8rem', color: '#ccc', textAlign: 'right' }}>
        {/* Top: step counter + progress bar */}
        {stepsDone != null && (
          <div>
            {/* Once complete (steps done == target) just show it once,
                compact — "step 80k" rather than "step 80,000 / 80k". */}
            {stepsDone === targetSteps && targetLabel
              ? <>step <b>{targetLabel}</b></>
              : <>step <b>{stepsDone.toLocaleString()}</b>{targetLabel ? ` / ${targetLabel}` : ''}</>}
            {progressPct != null && (
              <div style={{
                height: 4, marginTop: 4, backgroundColor: '#333', borderRadius: 2,
              }}>
                <div style={{
                  width: `${progressPct}%`, height: '100%',
                  backgroundColor: hwColor, borderRadius: 2,
                }} />
              </div>
            )}
            {/* Epoch counter — first-class measure of training progress, sits
                directly below the step counter so it's visible at a glance
                regardless of which x-axis the WallclockPlot is showing.
                Hidden when the data label / batch size needed for the
                computation isn't in the manifest. Segmented runs (declared
                in lineage.ts SEGMENTS) get a tooltip showing the per-
                segment breakdown so the discrepancy with single-(BS,label)
                math is auditable from the chip itself. */}
            {nEpochs != null && (
              <Tooltip content={
                epochBreakdown && epochBreakdown.length > 0
                  ? (
                    <div style={{ fontFamily: 'monospace', fontSize: '0.72rem' }}>
                      <div style={{ marginBottom: 4 }}>
                        epochs at step {stepsDoneRaw?.toLocaleString()}, by segment:
                      </div>
                      {epochBreakdown.map(({ segment, endStep, epochs }, i) => (
                        <div key={i}>
                          {`#${i + 1} ${segment.dataLabel}, BS=${segment.batchSize}, `}
                          {`${segment.startStep.toLocaleString()} → ${endStep.toLocaleString()}: `}
                          <b>{Number.isFinite(epochs) ? epochs.toFixed(2) : '—'}</b>
                        </div>
                      ))}
                      <div style={{ marginTop: 4 }}>
                        total: <b>{nEpochs.toFixed(2)}</b>
                      </div>
                    </div>
                  )
                  : 'fractional passes over the training set, computed from total_tokens / (epoch_sequences × train_seq_len)'
              }>
                <div style={{ marginTop: 3, fontSize: '0.78rem' }}>
                  epoch <b>{nEpochs.toFixed(2)}</b>
                </div>
              </Tooltip>
            )}
          </div>
        )}
        {/* Sparkline: step vs wallclock over last 6h. Flat = preempt/restart. */}
        {sparkPts.length >= 2 && freshSec != null && (
          <div style={{ marginTop: 6, display: 'flex', justifyContent: 'flex-end',
                        alignItems: 'center', gap: 6 }}>
            <Tooltip content={`last log: ${new Date(lastTs! * 1000).toLocaleString()}`}>
              <span
                style={{
                  fontSize: '0.68rem', color: freshnessColor(freshSec),
                  fontFamily: 'monospace',
                }}>
                ● {secsAgo(lastTs!)}
              </span>
            </Tooltip>
            <Sparkline pts={sparkPts} color={freshnessColor(freshSec)} />
          </div>
        )}
        {/* Step-rate over last 1m / 1h / 6h. Anchored to last log timestamp
            (a run that died 2m ago shows steps in the 1m before that). */}
        {(rate1m != null || rate1h != null || rate6h != null) && (
          <Tooltip content="steps in last 1m / 1h / 6h, ending at latest log">
            <div style={{ marginTop: 2, fontFamily: 'monospace', fontSize: '0.7rem',
                          color: '#aaa' }}>
              {rate1m != null && <>+{rate1m}/min</>}
              {rate1h != null && <> · +{rate1h.toLocaleString()}/h</>}
              {rate6h != null && <> · +{rate6h.toLocaleString()}/6h</>}
            </div>
          </Tooltip>
        )}
        {/* Latest summary metrics */}
        <div style={{ marginTop: 4, fontFamily: 'monospace', fontSize: '0.72rem' }}>
          {typeof trainLoss === 'number' && <>tr {trainLoss.toFixed(3)}</>}
          {typeof evalLoss === 'number' && <> · ev {evalLoss.toFixed(3)}</>}
          {typeof mfu === 'number' && <> · MFU {mfu.toFixed(1)}%</>}
          {typeof mtNmae === 'number' && <> · MT {mtNmae.toFixed(2)}%</>}
          {typeof mvNmae === 'number' && <> · MV {mvNmae.toFixed(2)}%</>}
          {data.cost != null && (
            // Spec 46 Phase A/B: chip renders only when a cost.json sidecar
            // has been computed (via `tomat runs sync` or `tomat cost
            // compute`). Modal runs without a sidecar yet hide silently
            // rather than show "TBD" — they'll appear on the next sync.
            <> · <CostChip
              runId={id}
              summary={data.cost}
              isModalOnly={isModalRun(id)}
            /></>
          )}
          {nFlops != null && (
            <> · {flopUnit === 'sci'
              ? <>{formatFlops(nFlops, 'sci')} FLOP</>
              : formatFlops(nFlops, flopUnit)}
            </>
          )}
          {nFlops != null && data.cost?.msrp_usd != null && data.cost.msrp_usd > 0 && (
            <>
              {' · '}
              <Tooltip content={`$${data.cost.msrp_usd.toFixed(2)} MSRP / ${(nFlops / 1e18).toFixed(1)} EF`}>
                <span><b>${(data.cost.msrp_usd / (nFlops / 1e18)).toFixed(2)}</b>/EF</span>
              </Tooltip>
            </>
          )}
        </div>
        {/* % training chip — sum(train-active windows ∩ attempt windows) /
            (now or finished_at − submitted_at). Only renders when we have
            the attempts sidecar AND a parquet train/loss span to overlap
            with; otherwise the chip is silently dropped. */}
        {trainFrac.pct != null && trainFrac.totalSec != null && trainFrac.trainSec != null && (
          <Tooltip content={
            `training time / iris wallclock\n`
            + `numerator: Σ (per-attempt window) ∩ (train/loss row span)\n`
            + `denominator: ${trainFrac.inProgress ? 'now' : 'finished_at'} − submitted_at\n`
            + `\nan approximation: it credits compile/startup time up to the\n`
            + `first train-step row as non-training (good), but credits any\n`
            + `compile-and-die attempt as zero training (also good).`
          }>
            <div style={{ marginTop: 3, fontFamily: 'monospace', fontSize: '0.72rem',
                          color: '#9aa6c2' }}>
              training: <b style={{ color: '#ddd' }}>{(trainFrac.pct * 100).toFixed(0)}%</b>
              {' '}({formatDurationShort(trainFrac.trainSec)} / {formatDurationShort(trainFrac.totalSec)})
              {trainFrac.inProgress && <span style={{ color: '#888' }}> · so far</span>}
            </div>
          </Tooltip>
        )}
      </div>
    </div>
  )
}

// Re-export for RunsPage's local use (it owns the broader card chrome).
export { isModifiedClick }

// iris reports SUCCEEDED whenever the job process exits 0 — including a run
// that was preempted and shut down cleanly on SIGTERM. So a "SUCCEEDED" run
// that never reached its step target is really incomplete + inactive (iris
// will not re-enqueue it). Flag that so it doesn't read as a healthy finish.
export function isIncomplete(c: RunCardData): boolean {
  if (c.job?.state !== 'SUCCEEDED') return false
  // `global_step` is 0-indexed (a finished N-step run ends at N-1), so steps
  // completed = global_step + 1; "incomplete" = that falls short of target.
  const gs = c.manifest?.summary?.['global_step']
  const lastGlobalStep = typeof gs === 'number' ? gs : null
  const target = numTrainStepsOf(c.manifest)
  return lastGlobalStep != null && target != null && lastGlobalStep + 1 < target
}

