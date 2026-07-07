// Chronological listing of what's actually happened to a run.
// Sourced from real infra data only: iris-attempts sidecar (per-task
// start/finish + death reason), wandb history (`lifecycle/trainer_started`,
// `cluster/preemptions`), and Modal app state. No time-based heuristics —
// only events that were actually reported by an upstream system.
//
// Per-attempt causation: each iris attempt is rendered as a single row
// captioned with a `trainer_started #N` primary line and a "died Xm Ys
// later · step S · <classification>" sub-line. This is the dashboard
// fulfilling the standing rule in specs/45-dashboard-tz11-surfacing.md
// — when the iris parent says RUNNING but tasks are pending and
// failures > 0, the user needs to see WHY each restart died.

import { useMemo } from 'react'
import type { IrisAttempt, IrisAttempts, ModalApp } from './api'
import { classifyDeath } from './deathEvents'
import { classifyErrorMessage, errorFirstLine } from './errorClassification'
import type { RunHistory } from './parquet'
import { WandbIcon } from './WandbIcon'

interface Event {
  ts_ms: number
  source: 'iris' | 'wandb' | 'modal'
  label: string
  detail?: string
  /** Indented sub-line rendered below the main row — primary use is the
   *  per-attempt "died Xm Ys later · step S · <classification>" line
   *  attached to a trainer_started row. */
  subline?: {
    text: string
    cls?: 'info' | 'warn' | 'error' | 'ok'
  }
  cls?: 'info' | 'warn' | 'error' | 'ok'
}

const SRC_COLOR: Record<Event['source'], string> = {
  iris: '#8b9bff',     // soft blue
  wandb: '#facc15',    // yellow — distinct from the orange trainer_started vlines
  modal: '#22c55e',    // green
}

// Source-column rendering: brand icons for wandb / modal, text for iris
// (no first-party logo, and the colored-text reads cleanly anyway).
function SourceCell({ source }: { source: Event['source'] }) {
  if (source === 'wandb') {
    return <WandbIcon title="wandb" />
  }
  if (source === 'modal') {
    return (
      <img
        src="/modal.png"
        alt="modal"
        title="modal"
        style={{ height: 18, width: 'auto', verticalAlign: 'middle' }}
      />
    )
  }
  return <span style={{ color: SRC_COLOR.iris }}>iris</span>
}

const CLS_BG: Record<NonNullable<Event['cls']>, string> = {
  info: 'transparent',
  warn: 'rgba(245, 158, 11, 0.08)',
  error: 'rgba(239, 68, 68, 0.10)',
  ok: 'rgba(34, 197, 94, 0.07)',
}

function formatTs(ms: number): string {
  const d = new Date(ms)
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

/** "3m 41s" / "47s" / "2h 14m". */
function formatDelta(ms: number): string {
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60), rem_s = s % 60
  if (m < 60) return rem_s > 0 ? `${m}m ${rem_s}s` : `${m}m`
  const h = Math.floor(m / 60), rem_m = m % 60
  return rem_m > 0 ? `${h}h ${rem_m}m` : `${h}h`
}

/** A logical restart "cycle" — one `attempt_id` across one or more tasks. */
interface AttemptCycle {
  attempt_id: number
  /** Earliest task-attempt start within the cycle. */
  started_at_ms: number
  /** Latest finish; null if any task hasn't finished yet (still running). */
  finished_at_ms: number | null
  /** Per-task entries that made up this cycle. Sorted by task_id. */
  perTask: Array<{
    task_id: string
    attempt: IrisAttempt
  }>
}

/** Group `IrisAttempt`s across tasks into per-attempt-id cycles. Iris
 *  re-uses the same `attempt_id` across the gang's tasks for a coordinated
 *  restart, so this groups them naturally. */
function buildCycles(attempts: IrisAttempts): AttemptCycle[] {
  const byId = new Map<number, AttemptCycle>()
  for (const t of attempts.tasks) {
    for (const a of t.attempts) {
      if (a.started_at_ms == null) continue
      let c = byId.get(a.attempt_id)
      if (!c) {
        c = {
          attempt_id: a.attempt_id,
          started_at_ms: a.started_at_ms,
          finished_at_ms: null,
          perTask: [],
        }
        byId.set(a.attempt_id, c)
      } else if (a.started_at_ms < c.started_at_ms) {
        c.started_at_ms = a.started_at_ms
      }
      c.perTask.push({ task_id: t.task_id, attempt: a })
    }
  }
  // Finish-time rule: a gang restart cycle is considered DEAD as soon as
  // ANY task in it finishes with a non-success state — iris bounces the
  // siblings, but their per-attempt `finished_at` doesn't always get
  // written before the bug-report snapshot, so the sibling records read
  // as "still running" even though the cycle effectively ended. We pick
  // the earliest finish as the cycle's death time (= when the trigger
  // task fell over). If EVERY task finished (clean completion), we use
  // the latest finish (the gang's final stop time).
  for (const c of byId.values()) {
    const finished = c.perTask
      .map((e) => e.attempt.finished_at_ms)
      .filter((ms): ms is number => ms != null)
    if (finished.length === 0) {
      c.finished_at_ms = null  // every task still running → cycle alive
    } else if (finished.length === c.perTask.length) {
      c.finished_at_ms = Math.max(...finished)  // clean finish across all tasks
    } else {
      // Some task died, others are listed as still-pending. Iris has
      // started bouncing the gang; mark the cycle dead at the trigger's
      // finish time.
      c.finished_at_ms = Math.min(...finished)
    }
  }
  const out = [...byId.values()]
  // Sort each cycle's perTask by task index so the picked-trigger task
  // is deterministic for ties.
  for (const c of out) {
    c.perTask.sort((x, y) => x.task_id.localeCompare(y.task_id))
  }
  // Cycles in attempt-id order (== chronological for iris's monotonic
  // counter).
  out.sort((a, b) => a.attempt_id - b.attempt_id)
  return out
}

/** Pick the per-task attempt that "caused" the cycle to die: skip cascade
 *  victims (sibling-bounced) in favor of the actual trigger if any task
 *  has a non-cascade error. The trigger is the task whose own `finished_at`
 *  is set + carries a non-cascade error; the gang's other tasks are
 *  usually marked `preempted` (cascade) without their own `finished_at`. */
function pickTrigger(cycle: AttemptCycle): { task_id: string; attempt: IrisAttempt } {
  // Prefer a task that actually finished + has a non-cascade error.
  // That's iris's "this is the task that triggered the bounce" signal.
  for (const e of cycle.perTask) {
    if (e.attempt.finished_at_ms == null) continue
    const cause = classifyDeath(e.attempt)
    if (cause !== 'cascade' && (e.attempt.error || '').trim()) return e
  }
  // Next: any finished task (even if cause is cascade) — it's the one
  // with an actual death timestamp.
  for (const e of cycle.perTask) {
    if (e.attempt.finished_at_ms != null) return e
  }
  // No finishes — fall back to any non-cascade.
  for (const e of cycle.perTask) {
    const cause = classifyDeath(e.attempt)
    if (cause !== 'cascade') return e
  }
  return cycle.perTask[0]
}

/** Walk wandb history's `global_step` column and find the max step seen
 *  inside the (`started_at_ms`, `finished_at_ms`) window. Used to caption
 *  per-attempt sub-lines with the step at which the attempt died. */
function maxStepBetween(history: RunHistory | null, startMs: number, endMs: number | null): number | null {
  if (!history) return null
  const end = endMs ?? Date.now()
  const ts = history.timestamps
  const gs = history.cols.get('global_step') ?? []
  let max = -Infinity
  for (let i = 0; i < history.rowCount; i++) {
    const t = ts[i]
    if (t == null) continue
    const tms = t * 1000
    if (tms < startMs || tms > end) continue
    const v = gs[i]
    if (v == null) continue
    const n = Number(v)
    if (n > max) max = n
  }
  return max === -Infinity ? null : max
}

export function RecentEvents({ attempts, modalApp, history }: {
  attempts: IrisAttempts | null
  modalApp: ModalApp | null
  history: RunHistory | null
}) {
  const events = useMemo<Event[]>(() => {
    const out: Event[] = []

    // Build per-attempt-cycle rows. Each cycle = one logical restart.
    // We emit a single `trainer_started #N` row per cycle (collapsing
    // the N per-task started events into one), with a sub-line describing
    // the outcome.
    const cycles = attempts ? buildCycles(attempts) : []
    // Lookup wandb-history attempt termini windowed by cycle start/end
    // — used to caption sub-lines with the step at which the attempt died.
    for (const c of cycles) {
      const trigger = pickTrigger(c)
      const cause = classifyDeath(trigger.attempt)
      // Prefer the server-classified fields (schema v2) when available;
      // fall back to client-side regex for v1 sidecars + dev runs against
      // a freshly-dumped attempt.
      const classification = trigger.attempt.error_classification
        ?? classifyErrorMessage(trigger.attempt.error)
      const firstLine = trigger.attempt.error_first_line
        ?? errorFirstLine(trigger.attempt.error)
      const step = maxStepBetween(history, c.started_at_ms, c.finished_at_ms)
      const taskCountSuffix = c.perTask.length > 1
        ? ` · ${c.perTask.length} tasks`
        : ''
      const taskShort = trigger.task_id.split('/').pop() ?? trigger.task_id
      let subline: Event['subline'] | undefined
      let cls: Event['cls'] = 'ok'
      if (c.finished_at_ms == null) {
        // Cycle still in flight — caption with elapsed time.
        const elapsed = Date.now() - c.started_at_ms
        const stepBit = step != null ? ` · step ${step}` : ''
        subline = {
          text: `alive · started ${formatDelta(elapsed)} ago${stepBit}`,
          cls: 'ok',
        }
        cls = 'ok'
      } else {
        const delta = c.finished_at_ms - c.started_at_ms
        const stepBit = step != null ? ` · step ${step}` : ''
        const isPreempt = cause === 'preempt'
        const isCascade = cause === 'cascade'
        // Outcome phrasing: preempt is a soft death (purple/warn), cascade
        // and other failures read as error.
        const verb = isPreempt
          ? 'preempted'
          : (trigger.attempt.state === 'succeeded' || trigger.attempt.state === 'completed')
            ? 'completed'
            : 'died'
        const completed = verb === 'completed'
        // Caption: prefer the classified label; fall back to the cleaned
        // first line (already trimmed to ≤80 chars by the classifier).
        let causeLine: string
        if (completed) {
          causeLine = 'succeeded'
        } else if (isPreempt) {
          causeLine = 'GCP preempt'
        } else if (isCascade && !classification) {
          causeLine = 'cascade (sibling died)'
        } else {
          causeLine = classification ?? firstLine ?? trigger.attempt.state
        }
        subline = {
          text: `${verb} ${formatDelta(delta)} later${stepBit} · ${causeLine}`,
          cls: completed ? 'ok' : isPreempt ? 'warn' : 'error',
        }
        cls = completed ? 'ok' : isPreempt ? 'warn' : 'error'
      }
      out.push({
        ts_ms: c.started_at_ms,
        source: 'iris',
        label: `trainer_started #${c.attempt_id}`,
        detail: `task ${taskShort}${taskCountSuffix}`,
        subline,
        cls,
      })
    }

    // iris attempts: also emit the job-level lifecycle events (submitted,
    // started, finished). Per-task finish rows are NOW collapsed into
    // each cycle's sub-line above — no separate `task N failed` row.
    if (attempts) {
      if (attempts.submitted_at_ms) {
        out.push({
          ts_ms: attempts.submitted_at_ms, source: 'iris',
          label: 'submitted',
          detail: attempts.job_id,
          cls: 'info',
        })
      }
      if (attempts.started_at_ms) {
        out.push({
          ts_ms: attempts.started_at_ms, source: 'iris',
          label: 'started', cls: 'ok',
        })
      }
      if (attempts.finished_at_ms) {
        out.push({
          ts_ms: attempts.finished_at_ms, source: 'iris',
          label: `${attempts.job_state}`,
          detail: `failures=${attempts.job_failure_count} preempts=${attempts.job_preemption_count}`,
          cls: attempts.job_state === 'KILLED' || attempts.job_state === 'FAILED'
            ? 'error' : attempts.job_state === 'SUCCEEDED' ? 'ok' : 'info',
        })
      }
    }

    // Modal: app state, function-call inputs.
    if (modalApp) {
      if (modalApp.created_at_ms) {
        out.push({
          ts_ms: modalApp.created_at_ms, source: 'modal',
          label: `app ${modalApp.state}`,
          detail: modalApp.description,
          cls: 'info',
        })
      }
      if (modalApp.stopped_at_ms) {
        out.push({
          ts_ms: modalApp.stopped_at_ms, source: 'modal',
          label: 'app stopped',
          cls: 'warn',
        })
      }
      for (const fc of Object.values(modalApp.function_calls ?? {})) {
        // We don't have per-input timestamps from Modal — surface the
        // statuses with the app's created_at as a fallback (so they
        // appear in chronological order with the app event).
        for (const inp of fc.inputs ?? []) {
          if (inp.status === 'success' || inp.status === 'failure') {
            // No ts for these — fold under the modal app's lifetime.
            // Skip rather than fake a timestamp.
            continue
          }
        }
        // Surface ERROR on the fc itself if any
        if (fc.error) {
          const fcId = fc.function_call_id ?? '???????'
          const errStr = typeof fc.error === 'string' ? fc.error : String(fc.error)
          out.push({
            ts_ms: modalApp.created_at_ms ?? Date.now(),
            source: 'modal',
            label: `fc ${fcId.slice(3, 10)}… ERROR`,
            detail: errStr.slice(0, 100),
            cls: 'error',
          })
        }
      }
    }

    // wandb history: trainer_started, sigterm, cluster_preempt.
    //
    // When the iris attempts sidecar is populated, we DROP the wandb
    // `trainer_started` rows here — every started cycle is already
    // surfaced above as an iris `trainer_started #N` row with a per-attempt
    // sub-line, and emitting them twice would be confusing redundancy.
    // The wandb-side row stays as a backstop for runs WITHOUT iris
    // attempts data (e.g. Modal-hosted training, or a sidecar that hasn't
    // synced yet).
    if (history) {
      const hasIrisAttempts = !!attempts && cycles.length > 0
      const trainerStarts: number[] = []
      const sigterms: number[] = []
      const preempts: number[] = []
      let firstStepTs: number | null = null
      let lastStepTs: number | null = null
      let maxStep = -Infinity
      const ts_arr = history.timestamps
      const trainerStartedCol = history.cols.get('lifecycle/trainer_started') ?? []
      const sigtermCol = history.cols.get('lifecycle/sigterm_received') ?? []
      const preemptDeltaCol = history.cols.get('cluster/preempts_delta') ?? []
      const globalStepCol = history.cols.get('global_step') ?? []
      for (let i = 0; i < history.rowCount; i++) {
        const ts_s = ts_arr[i]
        if (ts_s == null) continue
        const ts = ts_s * 1000
        if (trainerStartedCol[i] != null) trainerStarts.push(ts)
        if (sigtermCol[i] != null) sigterms.push(ts)
        const pd = preemptDeltaCol[i]
        if (pd != null && pd > 0) preempts.push(ts)
        const gs = globalStepCol[i]
        if (gs != null) {
          if (firstStepTs == null) firstStepTs = ts
          lastStepTs = ts
          const gsN = Number(gs)
          if (gsN > maxStep) maxStep = gsN
        }
      }
      if (firstStepTs != null) {
        out.push({
          ts_ms: firstStepTs, source: 'wandb',
          label: 'first train step', cls: 'ok',
        })
      }
      if (lastStepTs != null && firstStepTs != null && lastStepTs > firstStepTs && maxStep > 0) {
        out.push({
          ts_ms: lastStepTs, source: 'wandb',
          label: `latest train step gs=${maxStep}`, cls: 'info',
        })
      }
      if (!hasIrisAttempts) {
        for (const ts of trainerStarts) {
          out.push({ ts_ms: ts, source: 'wandb', label: 'trainer_started', cls: 'info' })
        }
      }
      for (const ts of sigterms) {
        out.push({ ts_ms: ts, source: 'wandb', label: 'sigterm received', cls: 'warn' })
      }
      for (const ts of preempts) {
        out.push({ ts_ms: ts, source: 'wandb', label: 'cluster preempt', cls: 'warn' })
      }
    }

    return out.sort((a, b) => b.ts_ms - a.ts_ms)
  }, [attempts, modalApp, history])

  if (events.length === 0) {
    return (
      <div style={{
        marginTop: '1rem', padding: '0.6rem 1rem',
        border: '1px dashed #2a2a2a', borderRadius: 4,
        color: '#888', fontSize: '0.85rem',
      }}>
        no events yet — iris/Modal/wandb hasn't reported anything for this run
      </div>
    )
  }

  return (
    <div style={{
      marginTop: '1rem',
      border: '1px solid #2a2a2a', borderRadius: 4,
      backgroundColor: '#181818',
    }}>
      <div style={{
        padding: '0.5rem 0.8rem', borderBottom: '1px solid #2a2a2a',
        fontSize: '0.8rem', color: '#aaa', display: 'flex', justifyContent: 'space-between',
      }}>
        <span>recent events ({events.length}) · newest first</span>
        <span style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: SRC_COLOR.iris }}>● iris</span>
          <SourceCell source="modal" />
          <SourceCell source="wandb" />
        </span>
      </div>
      <div style={{
        // No maxHeight cap — let the list expand so the user doesn't have to
        // scroll a nested container to see history. The page itself scrolls.
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: '0.75rem',
      }}>
        {events.map((e, i) => (
          <div
            key={i}
            style={{
              padding: '0.3rem 0.8rem',
              borderBottom: i < events.length - 1 ? '1px solid #232323' : 'none',
              backgroundColor: e.cls ? CLS_BG[e.cls] : 'transparent',
              display: 'grid',
              gridTemplateColumns: '180px 70px 1fr',
              gap: '0.6rem',
              alignItems: 'baseline',
            }}
          >
            <span style={{ color: '#888' }}>{formatTs(e.ts_ms)}</span>
            <span><SourceCell source={e.source} /></span>
            <span>
              <div>
                <span>{e.label}</span>
                {e.detail && (
                  <span style={{ color: '#888', marginLeft: '0.6rem' }}>
                    {e.detail}
                  </span>
                )}
              </div>
              {e.subline && (
                <div style={{
                  color: '#888',
                  marginTop: '0.15rem',
                  paddingLeft: '1.2rem',
                  // Box-drawing prefix mirrors the user-requested format
                  // (└── …). Rendered as a separate flex element so the
                  // text wraps cleanly under it.
                  position: 'relative',
                }}>
                  <span style={{
                    position: 'absolute',
                    left: 0,
                    color: '#555',
                  }}>└─</span>
                  <span style={{
                    color: e.subline.cls === 'error' ? '#ef9a9a'
                      : e.subline.cls === 'warn' ? '#f3c172'
                        : e.subline.cls === 'ok' ? '#9fdfa5'
                          : '#aaa',
                  }}>
                    {e.subline.text}
                  </span>
                </div>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
