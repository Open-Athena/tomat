import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { Tooltip } from '../Tooltip'
import { evalJobsByRun, evalPhase, fetchEval, fetchIrisState, fetchManifest, fetchRunsSnapshot, irisJobIdForRun, parquetUrl } from './api'
import type { EvalJob, EvalPoint, IrisJob, RunManifest } from './api'
import { fetchRunHistory } from './parquet'
import type { RunHistory } from './parquet'
import { WallclockPlot } from './WallclockPlot'
import { RunsTimelinePlot, colorForIndex, shortLabel, useNameFilter, compileMultiTermFilter, runHaystack, TAG_FILTER_KEY } from './RunsTimelinePlot'
import { parseTagFilters, runPassesTagFilters, serializeTagFilters, tagsFor, type TagFilters } from './tags'
import type { RunTimelineSeries } from './RunsTimelinePlot'
import { useTraceHighlight } from 'pltly/react'

/** Small SVG line chart (no plotly). `pts` is xs/ys in raw values; we scale. */
function Sparkline({
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

const LOOKBACK_HOURS = 6
const LOOKBACK_SEC = LOOKBACK_HOURS * 3600

/**
 * Compute steps completed within a wallclock window ending at the latest log.
 * Returns `null` if we don't have enough samples — common right after a fresh
 * resume, when the parquet's latest row was logged < windowSec ago.
 *
 * Step = `global_step` (Levanter's actual training step), not `_step` (wandb's
 * log-call counter) — the latter inflates 5–10× per row and isn't what users
 * mean when they ask "how many steps in the last hour?"
 */
function stepsInWindow(history: RunHistory, windowSec: number): number | null {
  const tsCol = history.timestamps
  const gsCol = history.cols.get('global_step') ?? []
  let latestTs: number | null = null
  let latestStep: number | null = null
  for (let i = 0; i < history.rowCount; i++) {
    const t = tsCol[i], s = gsCol[i]
    if (t == null || s == null) continue
    if (latestTs == null || t > latestTs) { latestTs = t; latestStep = s }
  }
  if (latestTs == null || latestStep == null) return null
  const cutoff = latestTs - windowSec
  // Earliest step at-or-before the cutoff (so we measure exact window).
  let baseStep: number | null = null
  let baseTs: number | null = null
  for (let i = 0; i < history.rowCount; i++) {
    const t = tsCol[i], s = gsCol[i]
    if (t == null || s == null) continue
    if (t <= cutoff && (baseTs == null || t > baseTs)) { baseTs = t; baseStep = s }
  }
  if (baseStep == null) return null
  return latestStep - baseStep
}

/** Filter history to last N seconds + downsample (max ~120 points for sparkline). */
function recentStepPoints(history: RunHistory): { x: number; y: number }[] {
  const tsCol = history.timestamps, stepCol = history.steps
  const now = Date.now() / 1000
  const cutoff = now - LOOKBACK_SEC
  const all: { x: number; y: number }[] = []
  for (let i = 0; i < history.rowCount; i++) {
    const t = tsCol[i], s = stepCol[i]
    if (t != null && s != null && t >= cutoff) {
      all.push({ x: t, y: s })
    }
  }
  // Sort by timestamp — parquet row order isn't guaranteed strictly
  // chronological (wandb syncs in chunks, interleaves system + training
  // metrics), so connecting points in row order produces a zigzag.
  all.sort((a, b) => a.x - b.x)
  if (all.length <= 120) return all
  // Downsample by stride
  const stride = Math.ceil(all.length / 120)
  const out: { x: number; y: number }[] = []
  for (let i = 0; i < all.length; i += stride) out.push(all[i])
  if (out[out.length - 1] !== all[all.length - 1]) out.push(all[all.length - 1])
  return out
}

/** "Xs ago" / "Xm ago" / "Xh ago" / "Xd ago"; nullable for unknown. */
function secsAgo(unixSec: number): string {
  const ago = Date.now() / 1000 - unixSec
  if (ago < 60) return `${Math.round(ago)}s ago`
  if (ago < 3600) return `${Math.round(ago / 60)}m ago`
  if (ago < 86400) return `${(ago / 3600).toFixed(1)}h ago`
  return `${(ago / 86400).toFixed(1)}d ago`
}

/** Freshness color: green < 2 min, yellow < 30 min, red older. */
function freshnessColor(secAgo: number): string {
  if (secAgo < 120) return '#22863a'
  if (secAgo < 1800) return '#d4a017'
  return '#cb2431'
}

// Auto-refresh cadences (ms) for the react-query polling on the runs
// dashboard. The `tomat-iris-cron` VM re-syncs R2 roughly every minute;
// the snapshot endpoint is edge-cached for ~30s on the Worker side so
// polling it faster than that just hits the same CF edge cache.
//
// The aggregated snapshot poll replaced the old per-run manifest fanout:
// instead of 43 × 1/min manifest GETs (43 req/min, with a 404 tail for
// runs-without-synced-manifests), we make ONE snapshot poll that
// multiplexes all manifests server-side.
//
// Per-run history (raw.parquet) polling is still split into `active`
// (iris reports RUNNING/PENDING/BUILDING) and `idle` (everything else)
// because parquets are big (~2 MB each) and unchanged for finished runs.
const REFETCH_MS = {
  snapshot: 30_000,
  iris: 30_000,
  // RunDetail polls a single manifest (not a fanout), so it goes direct
  // rather than via the aggregated snapshot endpoint.
  manifest: 60_000,
  history: { active: 180_000, idle: 600_000 },
  // On error (404, network), back off to keep TSQ from hammering R2 for a
  // never-synced manifest / a parquet that doesn't exist yet.
  errorBackoff: 300_000,
}

interface Props {
  parts: string[]
}

const navigate = (path: string) => {
  window.location.hash = `#/${path}`
}

const isModifiedClick = (e: React.MouseEvent) =>
  e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0

export function RunsPage({ parts }: Props) {
  const runId = parts[0]
  return runId ? <RunDetail runId={runId} /> : <RunsIndex />
}

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

function IrisBadge({ job, incomplete }: { job: IrisJob; incomplete?: boolean }) {
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

// ── status dots ────────────────────────────────────────────────────────────
// Two dots per card — iris job state + wandb run state. Replaces the old
// italic distinction: colour shows each source's health, so a healthy run
// reads as two green dots and an iris/wandb disagreement is visible at a
// glance. A hollow dot = that source has nothing (e.g. no iris job).
const DOT = {
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

// ── m-eval jobs ─────────────────────────────────────────────────────────────
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

// Parse run name into structured fields. Tolerant of unknown patterns: anything
// we can't recognize lands in `extra`.
interface RunMeta {
  model: string | null      // "200M", "1B"
  batchSize: number | null
  lossType: string | null   // "emd-do", "ce"
  targetSteps: string | null // "10k", "500"
  hardware: string | null   // "v5p-16", "v6e-16", "H100×8"
  hardwareKind: 'tpu-v5p' | 'tpu-v6e' | 'modal-gpu' | null
  suffix: string | null     // "z3", "cont7k-ext", etc.
}

function parseRunName(name: string): RunMeta {
  // train-full-<dataset>-<model>-bs<n>-<loss>-do-<steps>-<hw>-shuf1k-<suffix>
  const m = name.match(
    /^train-(?:full|cont)-(?:[a-z0-9-]+?-)?(\d+[MBK])-bs(\d+)-([a-z-]+?)-(?:do-)?(\d+k?|\d+)-([a-z0-9]+)(?:-shuf\d+k)?(?:-(.+))?$/i,
  )
  if (!m) {
    return {
      model: null, batchSize: null, lossType: null, targetSteps: null,
      hardware: null, hardwareKind: null, suffix: name,
    }
  }
  const [, model, bs, loss, steps, hwRaw, suffix] = m
  let hw: string | null = null
  let kind: RunMeta['hardwareKind'] = null
  const hwLower = hwRaw.toLowerCase()
  if (hwLower.startsWith('v5p')) { hw = `v5p-${hwLower.slice(3)}`; kind = 'tpu-v5p' }
  else if (hwLower.startsWith('v6e')) { hw = `v6e-${hwLower.slice(3)}`; kind = 'tpu-v6e' }
  else if (hwLower.startsWith('tpu')) { hw = `v6e-${hwLower.slice(3)}`; kind = 'tpu-v6e' }
  // slice(5) skips e.g. "h100x" (5 chars) — slice(4) leaked the 'x' through
  // giving "H100×x8" instead of "H100×8".
  else if (hwLower.startsWith('h100x')) { hw = `H100×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  else if (hwLower.startsWith('h200x')) { hw = `H200×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  else if (hwLower.startsWith('b200x')) { hw = `B200×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  return {
    model, batchSize: parseInt(bs, 10),
    lossType: loss === 'emd-do' ? 'EMD density-only'
      : loss === 'l1-do' ? 'L1 density-only'
      : loss === 'ce' ? 'CE'
      : loss,
    targetSteps: steps,
    hardware: hw, hardwareKind: kind,
    suffix: suffix ?? null,
  }
}

const HW_COLORS: Record<NonNullable<RunMeta['hardwareKind']>, string> = {
  'tpu-v5p': '#7c4dff',
  'tpu-v6e': '#00838f',
  'modal-gpu': '#f57c00',
}

function timeAgo(iso: string): string {
  const t = new Date(iso).getTime()
  if (!isFinite(t)) return ''
  const sec = (Date.now() - t) / 1000
  if (sec < 60) return `${Math.round(sec)}s ago`
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`
  if (sec < 86400) return `${(sec / 3600).toFixed(1)}h ago`
  return `${(sec / 86400).toFixed(1)}d ago`
}

interface RunCardData {
  id: string
  manifest: RunManifest | null
  job: IrisJob | null
  history: RunHistory | null
  evalJobs: EvalJob[]
  color: string
  err: string | null
}

interface CardProps {
  data: RunCardData
  activeRunId: string | null
  pinnedRunId: string | null
  onHover: (id: string | null) => void
  onClick: (id: string) => void
  onScrollTargetRef: (id: string, el: HTMLDivElement | null) => void
}

function RunCard({ data, activeRunId, pinnedRunId, onHover, onClick, onScrollTargetRef }: CardProps) {
  const { id, manifest, job, history, evalJobs, color, err } = data
  // `color` matches the run's timeline-plot trace. Used as a left accent bar
  // (and active-state border) so the card is visually tied to its plot line.
  const isActive = activeRunId === id
  const isPinned = pinnedRunId === id
  const otherActive = activeRunId != null && !isActive
  const incomplete = isIncomplete(data)
  // wandb's run state can sit at "running" long after a Modal job has died
  // (it never flushed a terminal state). Treat a "running" run that hasn't
  // logged in >10min as stale — so the badge greys instead of reading green.
  const lastLogTs = manifest?.history.ts_max ?? null
  const wandbStale = manifest?.run.state === 'running'
    && lastLogTs != null && (Date.now() / 1000 - lastLogTs) >= 600
  const cardRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    onScrollTargetRef(id, cardRef.current)
    return () => onScrollTargetRef(id, null)
  }, [id, onScrollTargetRef])

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
  // Steps completed. Levanter's `global_step` is 0-indexed — a finished
  // N-step run ends at `global_step = N-1` — so "steps done" is that + 1.
  // Showing `global_step` raw is the confusing "79,999 / 80k" off-by-one.
  // (Read from the run summary, not `history.step_max`, which is wandb's
  // `_step` log-counter — a different quantity.)
  const gsRaw = manifest?.summary['global_step']
  const lastGlobalStep = typeof gsRaw === 'number' ? gsRaw : null
  const stepsDone = lastGlobalStep != null
    ? (targetSteps != null
        ? Math.min(lastGlobalStep + 1, targetSteps)
        : lastGlobalStep + 1)
    : null
  const progressPct = stepsDone != null && targetSteps != null
    ? Math.min(100, (stepsDone / targetSteps) * 100)
    : null
  const trainLoss = manifest?.summary['train/loss']
  const evalLoss = manifest?.summary['eval/loss']
  const mfu = manifest?.summary['throughput/mfu']
  // MT/MV — latest mat-NMAE on the {train,val}_200 snapshots. Stored already
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
    <div
      ref={cardRef}
      data-runcard={id}
      onMouseEnter={() => onHover(id)}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        // Links/buttons inside the card keep their own behaviour; clicking
        // anywhere else pins this run (plot solos + rescales to it).
        if ((e.target as HTMLElement).closest('a, button')) return
        onClick(id)
      }}
      style={{
        position: 'relative',
        border: isActive ? `1px solid ${color}` : '1px solid #2a2a2a',
        borderRadius: 6,
        padding: '0.8rem 1rem',
        marginBottom: '0.6rem',
        backgroundColor: '#181818',
        cursor: 'pointer',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        gap: '0.5rem 1rem',
        overflow: 'hidden',
        transition: 'border-color 100ms, box-shadow 100ms',
        boxShadow: isPinned ? `0 0 0 2px ${color}`
          : isActive ? `0 0 0 1px ${color}` : 'none',
      }}>
      {/* Left identity accent — a real element, not a `borderLeft`. A `border`
          shorthand + a `borderLeft` longhand in one style object don't mix:
          React re-applies the shorthand when `isActive` toggles (resetting all
          4 sides to 1px) but skips the unchanged longhand, so the 4px accent
          vanished after hovering. A div is immune. Greys out when another run
          is highlighted, so faded cards de-emphasize fully. */}
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 4,
        backgroundColor: otherActive ? '#555' : color, zIndex: 3,
        transition: 'background-color 100ms',
      }} />
      {/* Dim overlay when another run is highlighted — inset past the accent
          (and below it in z-order) so the identity color stays full-strength
          even while the card is faded. */}
      <div style={{
        position: 'absolute', top: 0, right: 0, bottom: 0, left: 4,
        backgroundColor: 'rgba(24,24,24,0.6)', pointerEvents: 'none', zIndex: 2,
        opacity: otherActive ? 1 : 0, transition: 'opacity 100ms',
      }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {isPinned && (
            <Tooltip content="pinned — the plot is soloed + rescaled to this run; click the card again to unpin">
              <span>📌</span>
            </Tooltip>
          )}
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
          {wbUrl && (
            <Tooltip content="open this run in wandb">
              <a href={wbUrl} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.75rem', color: '#888' }}>
                wandb ↗
              </a>
            </Tooltip>
          )}
          <EvalChip jobs={evalJobs} />
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

function parseTargetSteps(s: string): number {
  // "10k" → 10000, "500" → 500
  const m = s.match(/^(\d+)(k|K)?$/)
  if (!m) return Number.POSITIVE_INFINITY
  return parseInt(m[1], 10) * (m[2] ? 1000 : 1)
}

/** Compact step count: 80000 → "80k", 1234 → "1,234". */
function formatStepCount(n: number): string {
  if (n >= 1000 && n % 1000 === 0) return `${n / 1000}k`
  if (n >= 10000) return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`
  return n.toLocaleString()
}

// Genuinely executing right now: iris RUNNING, or a job-less run (e.g. Modal)
// that wandb still reports as running.
function isRunning(c: RunCardData): boolean {
  if (c.job?.state === 'RUNNING') return true
  if (!c.job && c.manifest?.run.state === 'running') return true
  return false
}

// Enqueued / being scheduled — iris has the job but it isn't on a device yet.
// Distinct from an inactive/finished run: a queued run will start on its own.
function isQueued(c: RunCardData): boolean {
  return c.job?.state === 'PENDING' || c.job?.state === 'BUILDING'
}

/** Authoritative step target: `trainer.num_train_steps` from the run config. */
function numTrainStepsOf(manifest: RunManifest | null): number | null {
  const v = (manifest?.run.config as Record<string, unknown> | undefined)?.trainer
  if (v && typeof v === 'object' && 'num_train_steps' in v) {
    const n = (v as { num_train_steps?: unknown }).num_train_steps
    if (typeof n === 'number' && n > 0) return n
  }
  return null
}

// ── n_epochs / n_flops ──────────────────────────────────────────────────────
// Sequences per epoch (Levanter cache `total_num_rows`), keyed by data label —
// the `cache/<label>/` segment of `config.data.cache_dir`. From each cache's
// `train/shard_ledger.json`. TODO: fold into the manifest so a new data label
// doesn't need a code change here.
const EPOCH_SEQUENCES: Record<string, number> = {
  'train-full-v3': 4954176,
}

/** Data label (tokenization) of a run — the `cache/<label>/` path segment. */
function dataLabelOf(manifest: RunManifest | null): string | null {
  const data = (manifest?.run.config as Record<string, unknown> | undefined)?.data
  const cd = data && typeof data === 'object'
    ? (data as { cache_dir?: unknown }).cache_dir : undefined
  if (typeof cd !== 'string') return null
  const m = cd.match(/\/cache\/([^/]+)\/?$/)
  return m ? m[1] : null
}

/** Tokens trained on so far (`throughput/total_tokens` from the run summary). */
function totalTokensOf(manifest: RunManifest | null): number | null {
  const t = manifest?.summary['throughput/total_tokens']
  return typeof t === 'number' && t > 0 ? t : null
}

/** Forward+backward FLOPs — the standard 6·N·D estimate (N params, D tokens). */
function nFlopsOf(manifest: RunManifest | null): number | null {
  const n = manifest?.summary['parameter_count']
  const tok = totalTokensOf(manifest)
  if (typeof n !== 'number' || tok == null) return null
  return 6 * n * tok
}

/** Passes over the training set = tokens / (epoch_sequences · seq_len).
 *  Null when the data label's epoch size isn't in EPOCH_SEQUENCES. */
function nEpochsOf(manifest: RunManifest | null): number | null {
  const tok = totalTokensOf(manifest)
  const label = dataLabelOf(manifest)
  const seqLen = (manifest?.run.config as Record<string, unknown> | undefined)?.train_seq_len
  if (tok == null || label == null || typeof seqLen !== 'number' || seqLen <= 0) return null
  const epochSeqs = EPOCH_SEQUENCES[label]
  if (!epochSeqs) return null
  return tok / (epochSeqs * seqLen)
}

/** FLOP count → "1.1e20". */
function formatFlops(f: number): string {
  const exp = Math.floor(Math.log10(f))
  return `${(f / 10 ** exp).toFixed(1)}e${exp}`
}

/** Tokenizer generation, from the authoritative wandb project name
 *  (`tomat-…-P14` = v2 patch tokenizer, `…-P19` = v3). The run *name* is not
 *  reliable — some `train-full-v3-…` runs trained on v2 data (project P14). */
function tokenizerGen(manifest: RunManifest | null): 'v2' | 'v3' | null {
  const p = manifest?.run.project ?? ''
  if (/P14/.test(p)) return 'v2'
  if (/P19/.test(p)) return 'v3'
  return null
}

/** v2 (P14) runs are obsolete + a recurring source of `train-full-v3-…`
 *  mislabel confusion — hide them from the dashboard entirely. */
function isV2Run(c: RunCardData): boolean {
  return tokenizerGen(c.manifest) === 'v2'
}

// iris reports SUCCEEDED whenever the job process exits 0 — including a run
// that was preempted and shut down cleanly on SIGTERM. So a "SUCCEEDED" run
// that never reached its step target is really incomplete + inactive (iris
// will not re-enqueue it). Flag that so it doesn't read as a healthy finish.
function isIncomplete(c: RunCardData): boolean {
  if (c.job?.state !== 'SUCCEEDED') return false
  // `global_step` is 0-indexed (a finished N-step run ends at N-1), so steps
  // completed = global_step + 1; "incomplete" = that falls short of target.
  const gs = c.manifest?.summary['global_step']
  const lastGlobalStep = typeof gs === 'number' ? gs : null
  const target = numTrainStepsOf(c.manifest)
  return lastGlobalStep != null && target != null && lastGlobalStep + 1 < target
}

// Hide noisy/stale runs from the dashboard. Bare-names list for now — graduate
// to a sidecar JSON in R2 or to a UI toggle if this grows.
const EXCLUDED_RUNS: ReadonlySet<string> = new Set([
  // Cascade-bug victims from 2026-05-14 — all died <step 200 before --max-retries was wired.
  'train-full-v3-200M-bs1792-emd-do-15k-v5p32-shuf1k-zf-r2',
  'train-full-v3-200M-bs512-emd-do-15k-v5p16-shuf1k-zf-r2',
  'train-full-v3-200M-bs256-emd-do-15k-v5p16-shuf1k-zf-r2',
  // Pyspy profiling smoke (500 steps).
  'train-full-v3-200M-bs128-emd-do-500-v5p16-shuf1k-pyspy-3',
])

// Order: running first, then by "most recently active" (`history.ts_max`).
// Falls back to `created_at` when ts_max is missing. This puts cont7k-ext
// at the top after it last logged today, even though it was created May 11.
function orderRuns(cards: RunCardData[]): RunCardData[] {
  const activityTs = (c: RunCardData): number => {
    const tsMax = c.manifest?.history.ts_max
    if (typeof tsMax === 'number') return tsMax * 1000
    const ca = c.manifest?.run.created_at
    return ca ? Date.parse(ca) : 0
  }
  // Sort buckets: running first, then queued, then everything else; within a
  // bucket by most-recent activity.
  const rank = (c: RunCardData): number =>
    isRunning(c) ? 0 : isQueued(c) ? 1 : 2
  return [...cards].sort((a, b) => {
    const dr = rank(a) - rank(b)
    if (dr !== 0) return dr
    return activityTs(b) - activityTs(a)
  })
}

function RunsIndex() {
  // Per-card DOM refs so we can detect off-screen + scroll into view.
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const setCardRef = useCallback((id: string, el: HTMLDivElement | null) => {
    if (el) cardRefs.current.set(id, el)
    else cardRefs.current.delete(id)
  }, [])
  // Whether the currently-active card is inside the viewport. Drives the
  // sticky-bottom banner (shown when active but off-screen).
  // (Effect that watches `activeRunId` is registered below, after that
  // value is derived from the trace-highlight state.)
  const [activeInView, setActiveInView] = useState(true)

  // Live data via react-query: ONE aggregated snapshot poll (runs list + every
  // manifest + iris state) replaces what used to be N+2 polls (runs + iris + N
  // per-run manifests). The Worker fans out to R2 server-side and edge-caches
  // for ~30s. Per-run parquet histories are still polled individually because
  // each is ~2 MB and we don't want to multiplex MB-scale binary in one body.
  const snapshotQ = useQuery({
    queryKey: ['runs-snapshot'],
    queryFn: fetchRunsSnapshot,
    refetchInterval: REFETCH_MS.snapshot,
  })

  // Runs in the index that have a non-null manifest, in stable order. Runs
  // present in the R2 index but with `manifest: null` (just created, never
  // synced) are dropped from the dashboard — they'd just render as empty
  // cards with no metrics, and they'd also generate "loading…" forever.
  const visible = useMemo(() => {
    const data = snapshotQ.data
    if (!data) return []
    return Object.keys(data.runs)
      .filter((id) => !EXCLUDED_RUNS.has(id) && data.runs[id] != null)
      .sort()
  }, [snapshotQ.data])

  const iris = snapshotQ.data?.iris ?? null

  // Active-run set drives the per-run history poll interval. While iris hasn't
  // loaded yet, treat every run as active (fail-open to fast polling for
  // the first ~30s rather than freezing every run at 10-min cadence).
  const activeRunIds = useMemo(() => {
    if (!iris) return null
    const s = new Set<string>()
    for (const [jobName, job] of Object.entries(iris.jobs)) {
      if (job.state === 'RUNNING' || job.state === 'PENDING' || job.state === 'BUILDING') {
        // Job names are `/ryan/<run-label>`; strip the user prefix.
        const m = jobName.match(/^\/[^/]+\/(.+)$/)
        if (m) s.add(m[1])
      }
    }
    return s
  }, [iris])
  const intervalFor = (
    runId: string,
    cadence: { active: number; idle: number },
  ) => (q: { state: { error: unknown | null } }) => {
    if (q.state.error) return REFETCH_MS.errorBackoff
    if (!activeRunIds || activeRunIds.has(runId)) return cadence.active
    return cadence.idle
  }

  const historyQs = useQueries({
    queries: visible.map((id) => ({
      queryKey: ['history', id],
      queryFn: () => fetchRunHistory(parquetUrl(id)),
      refetchInterval: intervalFor(id, REFETCH_MS.history),
      retry: 1,
    })),
  })

  // m-eval jobs grouped by the run they evaluate (parsed from iris job names).
  const evalByRun = useMemo(() => evalJobsByRun(iris ?? undefined), [iris])
  const err = snapshotQ.error ? String(snapshotQ.error) : null
  // One card per visible run; manifests come straight from the snapshot,
  // histories from their per-run parquet queries. Failed history fetch is
  // non-fatal (the card just lacks its sparkline/plot).
  const cards: RunCardData[] | null = snapshotQ.data
    ? visible.map((id, idx) => ({
        id,
        manifest: snapshotQ.data.runs[id],
        job: iris?.jobs[irisJobIdForRun(id)] ?? null,
        history: historyQs[idx]?.data ?? null,
        evalJobs: evalByRun.get(id) ?? [],
        color: colorForIndex(idx),
        err: null,
      }))
    : null

  // Drop v2 (P14) runs — obsolete tokenizer, and a few are mislabeled
  // `train-full-v3-…` (trained on v2 data) which is pure confusion on the board.
  const ordered = cards ? orderRuns(cards.filter((c) => !isV2Run(c))) : null
  const runningCount = ordered?.filter(isRunning).length ?? 0
  const queuedCount = ordered?.filter(isQueued).length ?? 0
  const incompleteCount = ordered?.filter(isIncomplete).length ?? 0

  // Build timeline series from any cards that have history loaded.
  // Order matters for color stability — use index from `ordered`.
  const timelineSeries: RunTimelineSeries[] = (ordered ?? [])
    .filter((c) => c.history != null)
    .map((c) => ({
      id: c.id,
      history: c.history!,
      label: shortLabel(c.id),
      color: c.color,
    }))

  // Single source of truth for trace + card highlight. pltly's hook gives us
  // hover/pin state + handlers; the Plot consumes `highlight` directly, and
  // cards read `activeTrace` (and call `setHoverTrace` / `togglePin`).
  const traceLabels = timelineSeries.map((r) => r.label)
  // debounceMs: when the cursor crosses the gap between two cards, the leave
  // fires `setHoverTrace(null)` — debouncing it means a `setHoverTrace(next)`
  // arriving within the window cancels the null, so the highlight doesn't
  // flicker off-and-on between rows.
  const highlight = useTraceHighlight(traceLabels, { debounceMs: 120 })
  // label → id for converting active-trace back to runId for card matching.
  const labelToId = useMemo(
    () => new Map(timelineSeries.map((r) => [r.label, r.id])),
    // Re-derive when the set of trace labels changes (new run synced in).
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [traceLabels.join('|')],
  )
  const idToLabel = useMemo(
    () => new Map(timelineSeries.map((r) => [r.id, r.label])),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [traceLabels.join('|')],
  )
  const activeRunId = highlight.activeTrace ? labelToId.get(highlight.activeTrace) ?? null : null

  // `plotHoverRunId` is set ONLY when the cursor is over the timeline plot and
  // we compute a closest trace. Distinct from `activeRunId` (which is also set
  // by *card* hover) — using `activeRunId` to drive the float-to-top below
  // creates a feedback loop: hovering card N promotes it to slot 1, the cursor
  // is now over card N+1 (a different run), which promotes *that* one to
  // slot 1, …flicker. Card hover should highlight the legend/plot but NOT
  // reorder the card list.
  const [plotHoverLabel, setPlotHoverLabel] = useState<string | null>(null)
  const plotHoverRunId = plotHoverLabel ? labelToId.get(plotHoverLabel) ?? null : null

  // IntersectionObserver on the active card — drives the sticky-bottom banner.
  // Must live below `activeRunId` (TDZ): the dep array is evaluated during
  // render, so it can't appear before the declaration.
  useEffect(() => {
    if (!activeRunId) { setActiveInView(true); return }
    const el = cardRefs.current.get(activeRunId)
    if (!el) { setActiveInView(true); return }
    const io = new IntersectionObserver(
      ([entry]) => setActiveInView(entry.isIntersecting),
      { threshold: 0.2 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [activeRunId])

  const onCardHover = useCallback(
    (runId: string | null) => {
      highlight.setHoverTrace(runId ? idToLabel.get(runId) ?? null : null)
    },
    [highlight, idToLabel],
  )

  // Click a card to pin it: the timeline plot solos to that run and Plotly
  // autoranges to its extent (small/short runs are invisible at the shared
  // 0–80k scale). Click again — or another card — to unpin / switch.
  const onCardClick = useCallback(
    (runId: string) => {
      const label = idToLabel.get(runId)
      if (label) highlight.togglePin(label)
    },
    [highlight, idToLabel],
  )
  const pinnedRunId = highlight.pinnedTrace
    ? labelToId.get(highlight.pinnedTrace) ?? null
    : null

  // Per-run rich haystack (label + tags + YYMMDD created/last-activity +
  // hardware tokens + lineage). Built once for all cards, then keyed by both
  // run id (for card-match) and short-label (passed to the plot via prop).
  // `ordered` carries the same manifest/job/history triple the haystack needs,
  // so we build straight off of it.
  const runHaystacksById = useMemo(() => {
    const m = new Map<string, string>()
    if (!ordered) return m
    for (const c of ordered) {
      m.set(c.id, runHaystack(shortLabel(c.id), c.manifest, c.job, c.history))
    }
    return m
  }, [ordered])
  const runHaystacksByLabel = useMemo(() => {
    const m = new Map<string, string>()
    if (!ordered) return m
    for (const c of ordered) {
      m.set(shortLabel(c.id), runHaystacksById.get(c.id) ?? shortLabel(c.id))
    }
    return m
  }, [ordered, runHaystacksById])

  // Regex name-filter (synced with the plot's input via useNameFilter).
  // When active, matching cards float to the top of the list and non-matching
  // ones fade — same intent as the plot's filter, but non-destructive.
  // New whitespace=AND syntax (see filter.ts). Invalid filter → no matching
  // (treated as "no filter" so the dashboard stays usable while typing).
  const [nameFilter] = useNameFilter()
  const filterCompiled = useMemo(
    () => compileMultiTermFilter(nameFilter),
    [nameFilter],
  )
  const matchedIds = useMemo(() => {
    if (filterCompiled.empty || filterCompiled.error || !ordered) return null
    const s = new Set<string>()
    for (const c of ordered) {
      const h = runHaystacksById.get(c.id) ?? shortLabel(c.id)
      if (filterCompiled.matches(h)) s.add(c.id)
    }
    return s
  }, [filterCompiled, ordered, runHaystacksById])

  // Tag chip filter, lifted up from `RunsTimelinePlot` so the same selection
  // can hard-filter the card list (not just the plotted traces). Tri-state
  // per tag: each tag is off / `'in'` (run must have it) / `'out'` (run must
  // not have it). Initialized from `localStorage` once at mount, persisted
  // on every change. `parseTagFilters` handles the legacy single-state array
  // shape so old localStorage entries still load.
  const [tagFilters, setTagFilters] = useState<TagFilters>(() => {
    try {
      const raw = localStorage.getItem(TAG_FILTER_KEY)
      if (raw) return parseTagFilters(raw)
    } catch { /* ignore */ }
    return new Map()
  })
  const onTagFiltersChange = useCallback((next: TagFilters) => {
    setTagFilters(next)
    try {
      if (next.size === 0) localStorage.removeItem(TAG_FILTER_KEY)
      else localStorage.setItem(TAG_FILTER_KEY, serializeTagFilters(next))
    } catch { /* ignore */ }
  }, [])
  // Runs whose tag set satisfies the tri-state constraint. `null` when no
  // tags are active — semantics: "no tag constraint, show all". Distinct
  // from regex-`matchedIds`, which only sorts: tags hard-filter (matching
  // the plot's tag-chip behavior), so a run that fails the chip predicate
  // is hidden from the card list entirely.
  const tagMatchedIds = useMemo(() => {
    if (tagFilters.size === 0 || !ordered) return null
    const s = new Set<string>()
    for (const c of ordered) {
      if (runPassesTagFilters(tagsFor(shortLabel(c.id)), tagFilters)) s.add(c.id)
    }
    return s
  }, [tagFilters, ordered])

  // Card order: pinned first, then the plot-hover card (if any, and not the
  // already-pinned run), then regex-matches, then everything else (each group
  // preserving activity-order from `ordered`). The hover slot floats up the
  // card whose timeline trace the cursor is closest to — sourced from
  // `plotHoverRunId` (set ONLY by the plot's closest-trace hover handler).
  // Using `activeRunId` here would create a flicker loop on card hover.
  //
  // Tag filter is a *hard* filter (drops non-matching rows entirely). Regex
  // filter is a *soft* filter (sorts to top + fades non-matches). The plot
  // already constrains pinned/hovered to tag-matching runs, so those slots
  // don't need explicit tag filtering here.
  const displayed = useMemo(() => {
    if (!ordered) return ordered
    const tagFiltered = tagMatchedIds
      ? ordered.filter((c) => tagMatchedIds.has(c.id))
      : ordered
    const pinned = pinnedRunId ? tagFiltered.find((c) => c.id === pinnedRunId) : null
    const hovered = plotHoverRunId && plotHoverRunId !== pinnedRunId
      ? tagFiltered.find((c) => c.id === plotHoverRunId)
      : null
    const taken = new Set<string>()
    if (pinned) taken.add(pinned.id)
    if (hovered) taken.add(hovered.id)
    const rest = tagFiltered.filter((c) => !taken.has(c.id))
    const top: typeof tagFiltered = []
    if (pinned) top.push(pinned)
    if (hovered) top.push(hovered)
    if (!matchedIds || matchedIds.size === 0) {
      return [...top, ...rest]
    }
    const matched = rest.filter((c) => matchedIds.has(c.id))
    const others = rest.filter((c) => !matchedIds.has(c.id))
    return [...top, ...matched, ...others]
  }, [ordered, pinnedRunId, plotHoverRunId, matchedIds, tagMatchedIds])

  // On (re)pin, scroll the now-top pinned card into view.
  useEffect(() => {
    if (!pinnedRunId) return
    cardRefs.current.get(pinnedRunId)?.scrollIntoView({
      behavior: 'smooth', block: 'nearest',
    })
  }, [pinnedRunId])

  // Click anywhere that isn't a card / legend item / link / button — page
  // margins, header, gaps, plot background — to unpin. A document listener
  // (only while pinned) so it catches clicks outside the centered container.
  useEffect(() => {
    if (!highlight.isPinned) return
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null
      if (t?.closest('[data-runcard], .pltly-legend-item, a, button')) return
      highlight.clearPin()
    }
    document.addEventListener('click', onDocClick)
    return () => document.removeEventListener('click', onDocClick)
  }, [highlight])

  return (
    <div
      style={{ maxWidth: 1100, margin: '2rem auto', padding: '0 1rem' }}
      // Safety net: clear any highlight when the cursor leaves the whole view.
      // A card's own onMouseLeave can be missed if the list reorders under the
      // cursor during an auto-refresh — without this the fade could stick.
      onMouseLeave={() => highlight.setHoverTrace(null)}
    >
      <h1>tomat runs</h1>
      <p style={{ color: '#888', fontSize: '0.85rem' }}>
        {ordered && (
          <>
            <b>{runningCount}</b> running
            {queuedCount > 0 && <> · <b>{queuedCount}</b> queued</>}
            {incompleteCount > 0 && <> · <b>{incompleteCount}</b> incomplete</>}
            {' · '}<b>{ordered.length}</b> total ·{' '}
          </>
        )}
        synced from wandb via{' '}
        <code>tomat runs sync</code>. iris state via{' '}
        <code>tomat iris sync</code>
        {iris && <> (snapshot {timeAgo(iris.synced_at)})</>}
        . See{' '}
        <a href="https://github.com/Open-Athena/tomat/blob/main/specs/23-runs-dashboard.md">
          spec
        </a>.
      </p>
      {err && <p style={{ color: 'crimson' }}>error: {err}</p>}
      {!ordered && !err && <p>loading…</p>}
      {ordered && ordered.length === 0 && <p>(none synced yet)</p>}
      {timelineSeries.length > 0 && (
        <div style={{ marginBottom: '1.2rem' }}>
          <RunsTimelinePlot
            runs={timelineSeries}
            highlight={highlight}
            runHaystacks={runHaystacksByLabel}
            onPlotHover={setPlotHoverLabel}
            tagFilters={tagFilters}
            onTagFiltersChange={onTagFiltersChange}
          />
        </div>
      )}
      {displayed && displayed.map((c) => {
        const faded = matchedIds && matchedIds.size > 0 && !matchedIds.has(c.id)
        return (
          <div key={c.id} style={faded ? { opacity: 0.35 } : undefined}>
            <RunCard
              data={c}
              activeRunId={activeRunId}
              pinnedRunId={pinnedRunId}
              onHover={onCardHover}
              onClick={onCardClick}
              onScrollTargetRef={setCardRef}
            />
          </div>
        )
      })}
      {/* Bottom breadcrumb when a *hovered* card is off-screen. Pinned runs
          don't need it — they're rendered at the top of the list. */}
      {activeRunId && !activeInView && !highlight.isPinned && (
        <StickyActiveBanner
          runId={activeRunId}
          card={ordered?.find((c) => c.id === activeRunId) ?? null}
          onScrollTo={() => {
            const el = cardRefs.current.get(activeRunId)
            el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }}
        />
      )}
    </div>
  )
}

function StickyActiveBanner({
  runId, card, onScrollTo,
}: {
  runId: string
  card: RunCardData | null
  onScrollTo: () => void
}) {
  const meta = parseRunName(runId)
  const hwColor = meta.hardwareKind ? HW_COLORS[meta.hardwareKind] : '#888'
  return (
    <div
      // stopPropagation so this doesn't reach the view-level "click empty
      // space to unpin" handler.
      onClick={(e) => { e.stopPropagation(); onScrollTo() }}
      style={{
        position: 'fixed',
        bottom: 12, left: 12, right: 12,
        margin: '0 auto', maxWidth: 1050,
        backgroundColor: 'rgba(24,24,24,0.96)',
        border: '1px solid #4a8aff',
        borderRadius: 6,
        padding: '0.55rem 0.9rem',
        fontSize: '0.8rem',
        display: 'flex', alignItems: 'center', gap: '0.6rem',
        cursor: 'pointer',
        backdropFilter: 'blur(4px)',
        boxShadow: '0 4px 18px rgba(0,0,0,0.5)',
      }}>
      <span style={{ color: hwColor, fontWeight: 600 }}>
        {meta.hardware ?? '?'}
      </span>
      <span style={{ fontFamily: 'monospace' }}>{runId}</span>
      <span style={{ color: '#888', fontSize: '0.72rem', marginLeft: 'auto' }}>
        {card?.job && <>iris: {card.job.state} </>}
        click to scroll to its card
      </span>
    </div>
  )
}

// ── run-detail: m-eval jobs table ───────────────────────────────────────────
const EVAL_PHASE_COLOR: Record<string, string> = {
  flight: DOT.amber, done: DOT.blue, failed: DOT.red,
}

/** iris-dashboard URL for a job id (`/ryan/<job>`) — the marin-eng share form. */
function irisJobUrl(jobId: string): string {
  return `https://iris.oa.dev/#/job/${encodeURIComponent(jobId)}`
}

// One row per checkpoint step, a cell per mat-set — each cell shows the eval
// job's iris state (linked to its iris-dashboard page) and, once the result
// JSON has synced, the per-step mat-NMAE / mat-NEMD. NMAE/NEMD come from the
// canonical GCS eval results (`tomat evals sync` → R2 `eval.json`), NOT the
// harvested wandb points (those collapse in runs-sync's parquet merge).
function EvalJobsTable({ jobs, evalByStep }: {
  jobs: EvalJob[]
  evalByStep: Map<number, Record<string, EvalPoint>>
}) {
  const steps = [...new Set([
    ...jobs.map((j) => j.step),
    ...evalByStep.keys(),
  ])].sort((a, b) => a - b)
  const sets = ['val_200', 'train_200']
  const cell: React.CSSProperties = {
    padding: '3px 14px 3px 0', textAlign: 'left',
  }
  const pct = (v: number | null | undefined) =>
    typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : null
  return (
    <div style={{ marginBottom: '1.4rem' }}>
      <h2 style={{ fontSize: '1rem', marginBottom: '0.3rem' }}>
        m-evals ({steps.length} step{steps.length === 1 ? '' : 's'})
      </h2>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem',
                      fontFamily: 'monospace' }}>
        <thead>
          <tr style={{ color: '#888' }}>
            <th style={cell}>step</th>
            {sets.map((s) => <th key={s} style={cell}>{s} — state · NMAE · NEMD</th>)}
          </tr>
        </thead>
        <tbody>
          {steps.map((step) => (
            <tr key={step}>
              <td style={cell}>{step.toLocaleString()}</td>
              {sets.map((set) => {
                const ej = jobs.find((j) => j.step === step && j.matSet === set)
                const pt = evalByStep.get(step)?.[set]
                if (!ej && !pt) return <td key={set} style={{ ...cell, color: '#555' }}>–</td>
                const nmae = pct(pt?.nmae_mean)
                const nemd = pct(pt?.nemd_mean)
                return (
                  <td key={set} style={cell}>
                    {ej && (
                      <a href={irisJobUrl(ej.jobId)} target="_blank" rel="noreferrer"
                         style={{ color: EVAL_PHASE_COLOR[evalPhase(ej.job)] }}>
                        {ej.job.state.toLowerCase()}
                      </a>
                    )}
                    {(nmae || nemd) && (
                      <span style={{ color: '#888' }}>
                        {ej ? '  ' : ''}
                        {nmae && `NMAE ${nmae}`}
                        {nmae && nemd && ' · '}
                        {nemd && `NEMD ${nemd}`}
                      </span>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RunDetail({ runId }: { runId: string }) {
  // Same query keys as RunsIndex, so arriving from the index renders the
  // cached manifest/history instantly while a fresh fetch runs underneath.
  // Detail view is single-run focused, so always use the active cadence
  // (no need to gate on iris state — the user is staring at this one).
  const errBackoff = (q: { state: { error: unknown | null } }) =>
    q.state.error ? REFETCH_MS.errorBackoff : null
  const manifestQ = useQuery({
    queryKey: ['manifest', runId],
    queryFn: () => fetchManifest(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.manifest,
    retry: 1,
  })
  const historyQ = useQuery({
    queryKey: ['history', runId],
    queryFn: () => fetchRunHistory(parquetUrl(runId)),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.history.active,
    retry: 1,
  })
  const irisQ = useQuery({
    queryKey: ['iris'], queryFn: fetchIrisState, refetchInterval: REFETCH_MS.iris,
  })
  // Canonical per-step mat-NMAE/NEMD series — `tomat evals sync` → R2
  // `eval.json`. (The parquet's `eval/mat_nmae/*` columns are the harvested
  // wandb points, which collapse to a single value in runs-sync's merge.)
  const evalQ = useQuery({
    queryKey: ['eval', runId],
    queryFn: () => fetchEval(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.history.active,
    retry: 1,
  })
  const manifest = manifestQ.data ?? null
  const history = historyQ.data ?? null
  const evalJobs = useMemo(
    () => evalJobsByRun(irisQ.data ?? undefined).get(runId) ?? [],
    [irisQ.data, runId],
  )
  // step → { val_200, train_200 } eval point, from the synced eval.json.
  const evalByStep = useMemo(() => {
    const m = new Map<number, Record<string, EvalPoint>>()
    const sets = evalQ.data?.sets
    if (!sets) return m
    for (const [set, pts] of Object.entries(sets)) {
      for (const pt of pts) {
        const rec = m.get(pt.step) ?? {}
        rec[set] = pt
        m.set(pt.step, rec)
      }
    }
    return m
  }, [evalQ.data])
  const errObj = manifestQ.error || historyQ.error
  const err = errObj ? String(errObj) : null
  const nEpochs = nEpochsOf(manifest)
  const nFlops = nFlopsOf(manifest)

  return (
    <div style={{ maxWidth: 1600, margin: '2rem auto', padding: '0 1rem' }}>
      <p>
        <a
          href="#/runs"
          onClick={(e) => {
            if (isModifiedClick(e)) return
            e.preventDefault()
            navigate('runs')
          }}
        >
          ← runs
        </a>
      </p>
      <h1 style={{ fontSize: '1.2rem', fontFamily: 'monospace' }}>{runId}</h1>
      {err && <p style={{ color: 'crimson' }}>error: {err}</p>}
      {manifest && (
        <div style={{ color: '#888', fontSize: '0.9rem', marginBottom: '1rem' }}>
          state: {manifest.run.state} ·{' '}
          history: {manifest.history.rows} rows, steps [
          {manifest.history.step_min ?? '-'}, {manifest.history.step_max ?? '-'}] ·{' '}
          synced: {manifest.synced_at}
          {nEpochs != null && <> · {nEpochs.toFixed(2)} epochs</>}
          {nFlops != null && <> · {formatFlops(nFlops)} FLOPs</>}
          {' · '}
          <a href={manifest.run.url} target="_blank" rel="noreferrer">wandb ↗</a>
        </div>
      )}
      {(evalJobs.length > 0 || evalByStep.size > 0) && (
        <EvalJobsTable jobs={evalJobs} evalByStep={evalByStep} />
      )}
      {!history && !err && <p>loading parquet…</p>}
      {history && (
        <WallclockPlot history={history} evalSeries={evalQ.data ?? null} runId={runId} />
      )}
    </div>
  )
}
