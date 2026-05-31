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
import { Tooltip } from '../Tooltip'
import { evalPhase, type EvalJob, type IrisJob, type RunManifest } from './api'
import { lineageFor } from './lineage'
import type { RunHistory } from './parquet'
import {
  formatFlops,
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

export function IrisBadge({ job, incomplete }: { job: IrisJob; incomplete?: boolean }) {
  // `incomplete`: iris says SUCCEEDED but the run stopped well short of its
  // step target — almost always a preemption whose clean SIGTERM exit (0)
  // iris mis-buckets as success. Show it as its own burnt-orange state so the
  // card doesn't read as a healthy finish (iris won't re-enqueue it).
  const showIncomplete = incomplete && job.state === 'SUCCEEDED'
  const style = showIncomplete
    ? { bg: '#b4632a', fg: '#fff' }
    : IRIS_STATE_STYLES[job.state] ?? { bg: '#888', fg: '#fff' }
  const tail = job.preempts > 0 || job.failures > 0
    ? ` (p=${job.preempts}, f=${job.failures})` : ''
  return (
    <Tooltip content={showIncomplete
      ? `iris reported SUCCEEDED, but the run ended early — likely preempted with a clean SIGTERM exit. `
        + `iris will not re-enqueue it. preempts=${job.preempts} failures=${job.failures}`
      : (job.error || `iris state=${job.state} preempts=${job.preempts} failures=${job.failures}`)}>
      <span
        style={{
          backgroundColor: style.bg, color: style.fg,
          padding: '1px 6px', borderRadius: 3,
          fontSize: '0.75rem', fontFamily: 'monospace',
        }}
      >
        {showIncomplete ? 'INCOMPLETE' : job.state}{tail}
      </span>
    </Tooltip>
  )
}

// Two dots per card — iris job state + wandb run state. Replaces the old
// italic distinction: colour shows each source's health, so a healthy run
// reads as two green dots and an iris/wandb disagreement is visible at a
// glance. A hollow dot = that source has nothing (e.g. no iris job).
export const DOT = {
  green: '#22c55e', amber: '#f59e0b', blue: '#3b82f6',
  orange: '#b4632a', red: '#ef4444', grey: '#6a737d',
}

function irisDotColor(job: IrisJob, incomplete: boolean): string {
  switch (job.state) {
    case 'RUNNING': return DOT.green
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

function StatusDots({ job, manifest, incomplete, lastLogTs }: {
  job: IrisJob | null
  manifest: RunManifest | null
  incomplete: boolean
  lastLogTs: number | null
}) {
  const logAgeSec = lastLogTs != null ? Math.max(0, Date.now() / 1000 - lastLogTs) : null
  const loggedStr = lastLogTs != null ? `last logged ${secsAgo(lastLogTs)}` : 'never logged'
  return (
    <span style={{ display: 'inline-flex', gap: 3, alignItems: 'center' }}>
      {job
        ? <Dot color={irisDotColor(job, incomplete)} title={`iris: ${job.state}`} />
        : <Dot hollow title="iris: no job (e.g. a Modal run)" />}
      {manifest
        ? <Dot
            color={wandbDotColor(manifest.run.state, logAgeSec)}
            title={`wandb: ${manifest.run.state} · ${loggedStr}`}
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
  const { id, manifest, job, history, evalJobs, err } = data
  const incomplete = isIncomplete(data)
  // wandb's run state can sit at "running" long after a Modal job has died
  // (it never flushed a terminal state). Treat a "running" run that hasn't
  // logged in >10min as stale — so the badge greys instead of reading green.
  const lastLogTs = manifest?.history.ts_max ?? null
  const wandbStale = manifest?.run.state === 'running'
    && lastLogTs != null && (Date.now() / 1000 - lastLogTs) >= 600

  const meta = parseRunName(id)
  const hwColor = meta.hardwareKind ? HW_COLORS[meta.hardwareKind] : '#888'
  const wbUrl = manifest?.run.url
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
  const gsRaw = manifest?.summary['global_step']
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
  const trainLoss = manifest?.summary['train/loss']
  const evalLoss = manifest?.summary['eval/loss']
  const mfu = manifest?.summary['throughput/mfu']
  // MT/MV — latest mat-NMAE on the {train,val}_200 mat snapshots. Stored already
  // as a percentage (cont7k-ext logs 1.73 = 1.73%), so display as-is — no ×100.
  const mtNmae = manifest?.summary['eval/mat_nmae/train_200/mean']
  const mvNmae = manifest?.summary['eval/mat_nmae/val_200/mean']
  const nEpochs = nEpochsOf(manifest)
  const nFlops = nFlopsOf(manifest)

  const sparkPts = useMemo(() => history ? recentStepPoints(history) : [], [history])
  const lastTs = sparkPts.length > 0 ? sparkPts[sparkPts.length - 1].x : null
  const freshSec = lastTs != null ? Math.max(0, Date.now() / 1000 - lastTs) : null

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
            manifest={manifest}
            incomplete={incomplete}
            lastLogTs={lastLogTs}
          />
          {job && <IrisBadge job={job} incomplete={incomplete} />}
          {!job && manifest && (
            // No iris job (e.g. a Modal run) → fall back to wandb's run state.
            // The status dots mark the source; a "running" run gone stale
            // (no logs in >10min) shows greyed as STALE rather than green.
            <Tooltip content={wandbStale
              ? `wandb says running, but last logged ${secsAgo(lastLogTs!)} — likely dead`
              : `wandb state (no iris job): ${manifest.run.state}`}>
              <span
                style={{
                  backgroundColor: wandbStale
                    ? '#6a737d' : (WANDB_STATE_BG[manifest.run.state] ?? '#555'),
                  color: '#fff', padding: '1px 6px', borderRadius: 3,
                  fontSize: '0.75rem', fontFamily: 'monospace',
                }}>
                {wandbStale ? 'STALE' : manifest.run.state.toUpperCase()}
              </span>
            </Tooltip>
          )}
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
                style={{ fontSize: '0.75rem', color: '#888' }}>
                wandb ↗
              </a>
            </Tooltip>
          )}
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
              created {timeAgo(manifest.run.created_at)}
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
              {manifest.run.entity && manifest.run.project && (
                <> · {manifest.run.entity}/{manifest.run.project}</>
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
          {nEpochs != null && <> · {nEpochs.toFixed(2)} ep</>}
          {nFlops != null && <> · {formatFlops(nFlops)} FLOP</>}
        </div>
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
  const gs = c.manifest?.summary['global_step']
  const lastGlobalStep = typeof gs === 'number' ? gs : null
  const target = numTrainStepsOf(c.manifest)
  return lastGlobalStep != null && target != null && lastGlobalStep + 1 < target
}

