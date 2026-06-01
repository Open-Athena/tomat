// Chronological listing of what's actually happened to a run.
// Sourced from real infra data only: iris-attempts sidecar (per-task
// start/finish + death reason), wandb history (`lifecycle/trainer_started`,
// `cluster/preemptions`), and Modal app state. No time-based heuristics —
// only events that were actually reported by an upstream system.

import { useMemo } from 'react'
import type { IrisAttempts, ModalApp } from './api'
import type { RunHistory } from './parquet'

interface Event {
  ts_ms: number
  source: 'iris' | 'wandb' | 'modal'
  label: string
  detail?: string
  cls?: 'info' | 'warn' | 'error' | 'ok'
}

const SRC_COLOR: Record<Event['source'], string> = {
  iris: '#8b9bff',     // soft blue
  wandb: '#facc15',    // yellow — distinct from the orange trainer_started vlines
  modal: '#22c55e',    // green
}

// Source-column rendering: brand icons for wandb / modal, text for iris
// (no first-party logo, and the colored-text reads cleanly anyway).
//
// The wandb.svg ships as the full horizontal lockup ("icon + Weights &
// Biases + tagline", 1360×269). We crop to just the leftmost ~14×14 square
// via a width-constrained overflow:hidden wrapper, so only the W-mark
// shows.
function SourceCell({ source }: { source: Event['source'] }) {
  if (source === 'wandb') {
    return (
      <span
        title="wandb"
        style={{
          display: 'inline-block',
          width: 18, height: 18,
          overflow: 'hidden',
          verticalAlign: 'middle',
        }}
      >
        <img
          src="/wandb.svg"
          alt="wandb"
          style={{ height: 18, width: 'auto', display: 'block' }}
        />
      </span>
    )
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

export function RecentEvents({ attempts, modalApp, history }: {
  attempts: IrisAttempts | null
  modalApp: ModalApp | null
  history: RunHistory | null
}) {
  const events = useMemo<Event[]>(() => {
    const out: Event[] = []

    // iris attempts: each attempt contributes a "started" and (if finished)
    // a "finished" event. The death reason is in `error`.
    if (attempts) {
      // job-level: submitted, started, finished
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
      // per-task attempts: only the finished ones (started gets too noisy)
      for (const task of attempts.tasks) {
        for (const a of task.attempts) {
          if (a.finished_at_ms == null) continue
          const taskN = task.task_id.split('/').pop()
          const isCascade = (a.error || '').includes('Coscheduled sibling')
          const isPreempt = a.state === 'preempted' && !a.is_worker_failure
          const cls: Event['cls'] =
            isPreempt ? 'warn' : isCascade ? 'warn' : a.state === 'failed' ? 'error' : 'info'
          out.push({
            ts_ms: a.finished_at_ms, source: 'iris',
            label: `task ${taskN} ${a.state}${isCascade ? ' (cascade)' : ''}`,
            detail: (a.error || '').slice(0, 80),
            cls,
          })
        }
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
      for (const fc of Object.values(modalApp.function_calls)) {
        // We don't have per-input timestamps from Modal — surface the
        // statuses with the app's created_at as a fallback (so they
        // appear in chronological order with the app event).
        for (const inp of fc.inputs) {
          if (inp.status === 'success' || inp.status === 'failure') {
            // No ts for these — fold under the modal app's lifetime.
            // Skip rather than fake a timestamp.
            continue
          }
        }
        // Surface ERROR on the fc itself if any
        if (fc.error) {
          out.push({
            ts_ms: modalApp.created_at_ms ?? Date.now(),
            source: 'modal',
            label: `fc ${fc.function_call_id.slice(3, 10)}… ERROR`,
            detail: fc.error.slice(0, 100),
            cls: 'error',
          })
        }
      }
    }

    // wandb history: trainer_started, sigterm, cluster_preempt — the
    // existing vlines on the plot already render these, but listing
    // them in the events log gives times + cumulative counts.
    if (history) {
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
      for (const ts of trainerStarts) {
        out.push({ ts_ms: ts, source: 'wandb', label: 'trainer_started', cls: 'info' })
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
              <span>{e.label}</span>
              {e.detail && (
                <span style={{ color: '#888', marginLeft: '0.6rem' }}>
                  {e.detail}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
