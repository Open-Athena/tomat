import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { enumParam, useUrlState } from 'use-prms'
import { Tooltip } from '../Tooltip'
import { evalJobsByRun, fetchCronHeartbeat, fetchEval, fetchEvalsIndex, fetchIrisAttempts, fetchIrisState, fetchManifest, fetchModalState, fetchPendingFires, fetchRunCost, fetchRunsSnapshot, irisJobIdForRun, isModalRun, modalAppForRun, modalFcForPending, pendingFcForRun, pendingFcIdForRun, parquetUrl } from './api'
import type { EvalPoint, ModalApp, ModalFunctionCall, PendingFire, RunEval } from './api'
import { concatHistories, fetchRunHistory, truncateHistoryAtStep, type RunHistory } from './parquet'
import { WallclockPlot } from './WallclockPlot'
import { RecentEvents } from './RecentEvents'
import { EvalsPanel } from './EvalsPanel'
import { MEvalTable, type MEvalMetric } from './MEvalTable'

// Shared NMAE/NEMD toggle for the run-detail M-eval section — drives both
// the MEvalTable cells AND the WallclockPlot's MT/MV panel + legend so the
// user's choice affects both views at once. `?mevm=nmae|nemd` in the URL
// (defaults to NMAE).
const M_EVAL_METRICS = ['nmae', 'nemd'] as const
function useMEvalMetric() {
  return useUrlState('mevm', enumParam<MEvalMetric>('nmae', M_EVAL_METRICS))
}
import { RunsTimelinePlot, colorForIndex, useNameFilter, useTagFilters, useAncestorsToggle, compileMultiTermFilter, runHaystack, shortLabel } from './RunsTimelinePlot'
import { runPassesTagFilters, tagsFor } from './tags'
import { ancestorsOf, childrenOf, lineageFor, registerLineageFromManifest } from './lineage'
import type { RunTimelineSeries } from './RunsTimelinePlot'
import { useTraceHighlight } from 'pltly/react'
import {
  HW_COLORS,
  parseRunName,
  timeAgo,
  tokenizerGen,
} from './runMeta'
import { DOT, isIncomplete, RunHeaderRich, type RunCardData } from './RunHeaderRich'
import { LineageTable, type LineageRow } from './LineageTable'

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

interface CardProps {
  data: RunCardData
  activeRunId: string | null
  pinnedRunId: string | null
  /** wandb URL for this card's parent run (looked up by RunsIndex from
   *  the visible manifest map). When the parent is in the visible card list,
   *  clicking the parent chip scrolls to it; otherwise it opens this URL. */
  parentWandbUrl?: string | null
  /** Parent's trace color (when the parent is in the visible card list) —
   *  rendered as a small swatch on the ParentChip so the eye can match a
   *  child card to its parent's plot trace. */
  parentColor?: string | null
  onHover: (id: string | null) => void
  onClick: (id: string) => void
  onScrollTargetRef: (id: string, el: HTMLDivElement | null) => void
  /** Try to scroll the parent's card into view (returns true if found). */
  onScrollToParent: (parentId: string) => boolean
}

function RunCard({ data, activeRunId, pinnedRunId, parentWandbUrl, parentColor, onHover, onClick, onScrollTargetRef, onScrollToParent }: CardProps) {
  const { id, color } = data
  // `color` matches the run's timeline-plot trace. Used as a left accent bar
  // (and active-state border) so the card is visually tied to its plot line.
  const isActive = activeRunId === id
  const isPinned = pinnedRunId === id
  const otherActive = activeRunId != null && !isActive
  const cardRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    onScrollTargetRef(id, cardRef.current)
    return () => onScrollTargetRef(id, null)
  }, [id, onScrollTargetRef])

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
      {isPinned && (
        // Card-only chrome: the pin icon (anchored top-left, past the accent).
        // RunHeaderRich is shared with the detail page and doesn't carry pin
        // state, so the icon lives out here in the card body.
        <div style={{
          position: 'absolute', left: 10, top: 8, zIndex: 4,
          fontSize: '0.9rem',
        }}>
          <Tooltip content="pinned — the plot is soloed + rescaled to this run; click the card again to unpin">
            <span>📌</span>
          </Tooltip>
        </div>
      )}
      <div style={{ marginLeft: isPinned ? 22 : 0 }}>
        <RunHeaderRich
          data={data}
          parentWandbUrl={parentWandbUrl}
          parentColor={parentColor}
          onScrollToParent={onScrollToParent}
        />
      </div>
    </div>
  )
}

// ── pending-fire placeholder card ──────────────────────────────────────────
// Renders the `pending-fires.json` entries — Modal training fires that were
// `fn.spawn(...)`-ed but haven't opened their wandb session yet (cache build /
// JIT compile / ckpt load takes 15-30 min before the first metric lands).
//
// State for these cards is sourced ENTIRELY from Modal's lifecycle — the fc
// probe (when present in the modal-sync snapshot) plus the app-level state.
// We intentionally do NOT touch wandb's run.state for these — see memory
// `feedback_no_wandb_for_state`. The state label maps:
//   - fc probe not yet visible (sync hasn't run with --fc-id) → SPAWNED
//   - fc probe present + status=pending  → BUILDING (container starting)
//   - fc probe present + has task_id, status=pending → RUNNING (container up,
//     no wandb yet — JAX init / first iter / cache extract)
//   - fc probe present + status=success → FINISHED (will be GC'd soon)
//   - fc probe present + status in (failure, terminated, …) → FAILED
//
// All other Modal states pass through (unknown_*).
type PendingState = 'SPAWNED' | 'BUILDING' | 'RUNNING' | 'FINISHED'
                  | 'FAILED' | 'UNKNOWN'

const PENDING_STATE_STYLES: Record<PendingState, { bg: string; fg: string }> = {
  SPAWNED:  { bg: '#0366d6', fg: '#fff' },
  BUILDING: { bg: '#b08800', fg: '#fff' },
  RUNNING:  { bg: '#22863a', fg: '#fff' },
  FINISHED: { bg: '#22863a', fg: '#fff' },
  FAILED:   { bg: '#cb2431', fg: '#fff' },
  UNKNOWN:  { bg: '#6a737d', fg: '#fff' },
}

function pendingFireState(fc: ModalFunctionCall | null): PendingState {
  if (!fc) return 'SPAWNED'
  if (fc.error) return 'FAILED'
  if (fc.inputs.length === 0) return 'SPAWNED'
  const inp = fc.inputs[0]
  switch (inp.status) {
    case 'success': return 'FINISHED'
    case 'failure':
    case 'terminated':
    case 'timeout':
    case 'init_failure':
    case 'internal_failure':
    case 'idle_timeout':
      return 'FAILED'
    case 'pending':
      // No task_id assigned yet → container hasn't started.
      // task_id present → container up, training side may or may not have
      // started touching wandb. We label this RUNNING for the placeholder
      // (BUILDING would understate the work being done by the GPU).
      return inp.task_id ? 'RUNNING' : 'BUILDING'
    default: return 'UNKNOWN'
  }
}

function PendingStateBadge({ state, fc, app }: {
  state: PendingState
  fc: ModalFunctionCall | null
  app: ModalApp | null
}) {
  const style = PENDING_STATE_STYLES[state]
  const tipLines: string[] = [`modal-derived state: ${state}`]
  if (app) tipLines.push(`app=${app.description} app_state=${app.state}`)
  if (fc) {
    tipLines.push(`fc=${fc.function_call_id}`)
    if (fc.error) tipLines.push(`fc error: ${fc.error}`)
    for (const inp of fc.inputs) {
      tipLines.push(`input=${inp.status}${inp.task_id ? ` task=${inp.task_id.slice(0, 10)}…` : ''}`)
    }
  } else {
    tipLines.push('no fc probe yet (run `tomat modal sync --fc-id <fc>`)')
  }
  return (
    <Tooltip content={tipLines.join(' · ')}>
      <span style={{
        backgroundColor: style.bg, color: style.fg,
        padding: '1px 6px', borderRadius: 3,
        fontSize: '0.75rem', fontFamily: 'monospace',
      }}>
        {state}
      </span>
    </Tooltip>
  )
}

interface PendingFireCardProps {
  fire: PendingFire
  app: ModalApp | null
  fc: ModalFunctionCall | null
}

function PendingFireCard({ fire, app, fc }: PendingFireCardProps) {
  const state = pendingFireState(fc)
  const spawnedAgo = (() => {
    const dt = (Date.now() - fire.spawned_at_ms) / 1000
    if (dt < 60) return `${dt.toFixed(0)}s ago`
    if (dt < 3600) return `${(dt / 60).toFixed(0)}m ago`
    if (dt < 86400) return `${(dt / 3600).toFixed(1)}h ago`
    return `${(dt / 86400).toFixed(1)}d ago`
  })()
  const appUrl = `https://modal.com/apps/open-athena/main/deployed/${fire.app_name}?activeTab=logs`
  // Compose the loss-config chip text — `loss_type` plus `kl_sigma` (when
  // relevant) plus the maskgit_prior.
  const lossChip = fire.kl_sigma != null
    ? `${fire.loss_type} σ=${fire.kl_sigma}`
    : fire.loss_type
  return (
    <div style={{
      position: 'relative',
      border: '1px dashed #4a4a4a',
      borderRadius: 6,
      padding: '0.6rem 1rem 0.6rem 1.2rem',
      marginBottom: '0.6rem',
      backgroundColor: '#161616',
    }}>
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 4,
        backgroundColor: '#0366d6', zIndex: 1,
      }} />
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr auto', gap: '0.4rem 1rem',
      }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <PendingStateBadge state={state} fc={fc} app={app} />
            <span style={{ fontFamily: 'monospace', fontSize: '0.9rem', color: '#ddd' }}>
              {fire.run_id}
            </span>
            <Tooltip content={`open Modal app logs (${fire.app_name})\nfc: ${fire.fc_id}`}>
              <a href={appUrl} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.75rem', color: '#888',
                  display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                <img src="/modal.png" alt="modal"
                  style={{ height: 18, width: 'auto', verticalAlign: 'middle' }} />
                ↗
              </a>
            </Tooltip>
            <Tooltip content="placeholder card — pending wandb session, sourced from `pending-fires.json`">
              <span style={{
                fontSize: '0.66rem', fontFamily: 'monospace', color: '#888',
                border: '1px solid #444', borderRadius: 8, padding: '0 6px',
              }}>
                placeholder
              </span>
            </Tooltip>
          </div>
          <div style={{ marginTop: '0.35rem', fontSize: '0.8rem', color: '#ccc',
                        display: 'flex', flexWrap: 'wrap', gap: '0.4rem 0.9rem' }}>
            <span style={{ color: HW_COLORS['modal-gpu'], fontWeight: 600 }}>
              {fire.hardware}
            </span>
            <span>{fire.model_preset}</span>
            <span>BS={fire.batch_size}</span>
            <span>{lossChip}</span>
            <span>prior={fire.maskgit_prior}</span>
            <Tooltip content="num_train_steps the spawn requested">
              <span>target {fire.intended_steps.toLocaleString()}</span>
            </Tooltip>
          </div>
          <div style={{ marginTop: '0.3rem', fontSize: '0.75rem', color: '#888' }}>
            spawned {spawnedAgo} · {fire.wandb_project} · awaiting wandb session
            {fire.tags.length > 0 && <> · tags: {fire.tags.join(', ')}</>}
          </div>
        </div>
        <div style={{ minWidth: 120, fontSize: '0.75rem', color: '#888',
                      textAlign: 'right', fontFamily: 'monospace' }}>
          <div>fc {fire.fc_id.slice(0, 14)}…</div>
          <div style={{ marginTop: 4 }}>fn {fire.function_name}</div>
        </div>
      </div>
    </div>
  )
}

// Genuinely executing right now: iris RUNNING, or a job-less run (e.g. Modal)
// that wandb still reports as running AND has logged recently.
//
// Without the recency gate, wandb-zombies (runs that died without a clean
// shutdown — common for preempt-killed jobs) keep `manifest.run.state =
// 'running'` indefinitely, and every dashboard sort that buckets "running
// first" floats them ahead of genuinely-active runs. 10 minutes is well
// past the wandb heartbeat cadence (~30s for our trainers) so a still-alive
// run will always be inside the window.
// Bumped 10 → 60 min: when a Modal job is in cache-build / JAX compile /
// ckpt load, wandb stays silent for 30-60 min while the container is
// alive. The shorter 10-min window incorrectly downgrades these runs
// to UNKNOWN. See feedback_no_wandb_for_state — Modal/iris are the
// ground truth, wandb's silence is just a logging artifact.
const RUNNING_RECENCY_MS = 60 * 60 * 1000
function isRunning(c: RunCardData): boolean {
  if (c.job?.state === 'RUNNING') return true
  // For Modal runs the source of truth is the Modal lifecycle, not wandb's
  // run.state — per `feedback_no_wandb_for_state`, wandb stamps "crashed" on
  // any client disconnect (including intentional kills) and stays "running"
  // for zombies. A Modal run is genuinely running iff the matched app is
  // `deployed` and has running tasks. See ModalBadge for the same rule.
  if (!c.job && isModalRun(c.id) && c.modalApp) {
    return c.modalApp.state === 'deployed' && c.modalApp.n_running_tasks > 0
  }
  if (!c.job && !isModalRun(c.id) && c.manifest?.run?.state === 'running') {
    // Non-Modal job-less runs (no iris, no -modal- in the name) — fall back
    // to wandb-state with the recency gate, as before.
    const tsMax = c.manifest?.history?.ts_max
    if (typeof tsMax === 'number'
        && Date.now() - tsMax * 1000 < RUNNING_RECENCY_MS) {
      return true
    }
  }
  return false
}

// Enqueued / being scheduled — iris has the job but it isn't on a device yet.
// Distinct from an inactive/finished run: a queued run will start on its own.
function isQueued(c: RunCardData): boolean {
  return c.job?.state === 'PENDING' || c.job?.state === 'BUILDING'
}

/** v2 (P14) runs are obsolete + a recurring source of `train-full-v3-…`
 *  mislabel confusion — hide them from the dashboard entirely. */
function isV2Run(c: RunCardData): boolean {
  return tokenizerGen(c.manifest) === 'v2'
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
  // Misfire 2026-06-08: v4-epochwin resume v1 used default run_id template
  // `<results_label>-bs<batch_size>-seed<seed>` with batch_size=256, which
  // landed on a fresh wandb run `…-bs256-seed42` instead of resuming the
  // canonical `…-bs128-seed42`. Cancelled after ~360 steps; re-fired with
  // explicit `run_id="…-bs128-seed42"` (the actual v4-epochwin resume).
  'train-mg-modal-h200x8-tz-v4-epochwin-bs256-seed42',
])

/** Sort mode for the runs card list, URL-persisted via `?sort=`.
 *  - `active`: bucket-by-state (running → queued → other), within each by
 *    most-recently active. The default — keeps fresh activity at the top.
 *  - `updated`: pure last-updated desc (no bucket grouping).
 *  - `created`: by run creation time desc (newest first).
 *  - `name`: alphabetical asc.
 *  - `step`: by `last_train_step` desc (furthest-trained first). */
const SORT_MODES = ['active', 'updated', 'created', 'name', 'step'] as const
type SortMode = (typeof SORT_MODES)[number]

function activityTs(c: RunCardData): number {
  const tsMax = c.manifest?.history?.ts_max
  if (typeof tsMax === 'number') return tsMax * 1000
  const ca = c.manifest?.run?.created_at
  if (ca) return Date.parse(ca)
  // Manifest-less pending/building card: sort by iris submission time so
  // the most recent fire is at the top of the queued bucket.
  return c.job?.submitted_at_ms ?? 0
}

function createdTs(c: RunCardData): number {
  const ca = c.manifest?.run?.created_at
  if (ca) return Date.parse(ca)
  return c.job?.submitted_at_ms ?? 0
}

function lastStep(c: RunCardData): number {
  return c.manifest?.history?.last_train_step
    ?? c.manifest?.history?.step_max
    ?? -1
}

// Default order: running first, then queued (pending/building), then
// everything else; within each bucket by `activityTs` desc. Puts cont7k-ext
// at the top after it last logged today (even though created May 11), AND a
// fresh `tomat train` fire at the very top while it's still queueing.
function orderRuns(cards: RunCardData[], mode: SortMode = 'active'): RunCardData[] {
  if (mode === 'updated') {
    return [...cards].sort((a, b) => activityTs(b) - activityTs(a))
  }
  if (mode === 'created') {
    return [...cards].sort((a, b) => createdTs(b) - createdTs(a))
  }
  if (mode === 'name') {
    return [...cards].sort((a, b) => a.id.localeCompare(b.id))
  }
  if (mode === 'step') {
    return [...cards].sort((a, b) => lastStep(b) - lastStep(a))
  }
  // 'active' (default): bucket + activity-desc.
  const rank = (c: RunCardData): number =>
    isRunning(c) ? 0 : isQueued(c) ? 1 : 2
  return [...cards].sort((a, b) => {
    const dr = rank(a) - rank(b)
    if (dr !== 0) return dr
    return activityTs(b) - activityTs(a)
  })
}

/** iris job ids → run ids for `tomat train`-style training jobs.
 *
 * Convention: `iris job run --job-name <label>` from `tomat train` produces
 * `/ryan/train-<...>`. We exclude m-eval jobs (`/ryan/tomat-eval-*`) and
 * other tomat sidecar jobs that aren't training runs.
 *
 * Returns the run-id (label) for matches, null otherwise.
 */
function trainingRunIdFromIrisJob(jobId: string): string | null {
  // `/ryan/train-mg-tz-11` → `train-mg-tz-11`
  const m = jobId.match(/^\/[^/]+\/(train-.+)$/)
  if (!m) return null
  const id = m[1]
  // Defensive: never want to confuse a child task or an eval job for a run.
  if (id.includes('/')) return null
  return id
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

  // GCE-VM cron heartbeat — written every 2 min by `runs-sync-active.sh` →
  // R2. Surfaced as a small "cron 1m ago" chip in the header so we notice
  // when the VM stops syncing (badges/sparklines will drift but won't
  // visibly error). Refresh roughly once a minute; chip cadence is bounded
  // by the cron's own 2-min cadence regardless.
  const cronHeartbeatQ = useQuery({
    queryKey: ['cron-heartbeat'],
    queryFn: fetchCronHeartbeat,
    refetchInterval: 60_000,
  })

  // Runs in the index that have a non-null manifest, in stable order. Runs
  // present in the R2 index but with `manifest: null` (just created, never
  // synced) are dropped from the dashboard — they'd just render as empty
  // cards with no metrics, and they'd also generate "loading…" forever.
  const visible = useMemo(() => {
    const data = snapshotQ.data
    if (!data) return []
    return Object.keys(data.runs)
      .filter((id) => {
        if (EXCLUDED_RUNS.has(id)) return false
        const entry = data.runs[id]
        // Drop null entries AND malformed-manifest entries (missing run /
        // history / summary). Worker-side cost-only sidecars sometimes ship
        // without a manifest; those have no cards to render.
        return entry != null && entry.run != null && entry.history != null && entry.summary != null
      })
      .sort()
  }, [snapshotQ.data])

  const iris = snapshotQ.data?.iris ?? null
  const modal = snapshotQ.data?.modal ?? null
  const pendingFires = snapshotQ.data?.pending_fires ?? null

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

  // Register wandb-config-driven lineage from every run in the snapshot.
  // Future fires populated by `tomat train --parent` (spec 61 P1) appear in
  // the dashboard without any RUN_LINEAGE registry edit.
  useEffect(() => {
    const runs = snapshotQ.data?.runs
    if (!runs) return
    for (const [id, m] of Object.entries(runs)) {
      const cfg = (m as { run?: { config?: Record<string, unknown> } })?.run?.config
      if (cfg) registerLineageFromManifest(id, cfg)
    }
  }, [snapshotQ.data])
  const err = snapshotQ.error ? String(snapshotQ.error) : null
  // One card per visible run; manifests come straight from the snapshot,
  // histories from their per-run parquet queries. Failed history fetch is
  // non-fatal (the card just lacks its sparkline/plot).
  const cards: RunCardData[] | null = snapshotQ.data
    ? visible.map((id, idx) => {
        const manifest = snapshotQ.data.runs[id]
        // `modalAppForRun` matches the (single) deployed `tomat-train-smoke`
        // app to ANY `-modal-` run name — there's no proper run→fc mapping in
        // the snapshot. Without an extra freshness gate, all prior Modal runs
        // (v1/v2/v3 etc.) inherit the currently-deployed app's running-tasks
        // count and read as RUNNING even when they finished days ago. Drop the
        // association when this run hasn't logged inside `RUNNING_RECENCY_MS`.
        const candidate = modalAppForRun(modal, id)
        const ts = manifest?.history?.ts_max
        const recent = typeof ts === 'number'
          && Date.now() - ts * 1000 < RUNNING_RECENCY_MS
        // BUILDING-state allowance: even when wandb has been silent
        // (cache build / JAX compile / ckpt load), keep the modalApp
        // association if the run itself is recent enough that the
        // deployed app is plausibly its current fc. This avoids the
        // CRASHED/STALE misread per `feedback_no_wandb_for_state`.
        const createdMs = manifest?.run?.created_at
          ? Date.parse(manifest.run.created_at) : NaN
        const runRecent = Number.isFinite(createdMs)
          && Date.now() - createdMs < 24 * 3600 * 1000
        const modalApp = candidate && (recent || runRecent) ? candidate : null
        return {
          id,
          manifest,
          job: iris?.jobs[irisJobIdForRun(id)] ?? null,
          modalApp,
          modalFc: pendingFcForRun(pendingFires, modal, id),
          modalFcId: pendingFcIdForRun(pendingFires, id),
          history: historyQs[idx]?.data ?? null,
          evalJobs: evalByRun.get(id) ?? [],
          color: colorForIndex(idx),
          err: null,
          cost: manifest?.cost ?? null,
        }
      })
    : null

  // Placeholder cards for `tomat train` fires that iris knows about but
  // wandb hasn't logged a metric for yet (pending → building → JIT-compile).
  // These would otherwise be silently dropped by the `runs[id] != null`
  // filter in `visible`. We render them with manifest=null + the iris job,
  // and RunHeaderRich shows the iris state badge + submission timing chips
  // while wandb-derived metrics gracefully degrade to "—".
  //
  // Cards are merged into `ordered` after the v2-filter; `orderRuns` already
  // sorts queued (PENDING/BUILDING) below running but above inactive, and
  // we extended its tie-break to fall back to `job.submitted_at_ms`, so a
  // fresh fire floats to the top of the queued bucket.
  const visibleSet = useMemo(() => new Set(visible), [visible])
  const pendingCards: RunCardData[] = useMemo(() => {
    if (!iris) return []
    const out: RunCardData[] = []
    for (const [jobId, job] of Object.entries(iris.jobs)) {
      const runId = trainingRunIdFromIrisJob(jobId)
      if (runId == null) continue
      if (visibleSet.has(runId)) continue
      if (EXCLUDED_RUNS.has(runId)) continue
      if (job.state !== 'PENDING' && job.state !== 'BUILDING' && job.state !== 'RUNNING') continue
      out.push({
        id: runId,
        manifest: null,
        job,
        modalApp: modalAppForRun(modal, runId),
        modalFc: pendingFcForRun(pendingFires, modal, runId),
        modalFcId: pendingFcIdForRun(pendingFires, runId),
        history: null,
        evalJobs: evalByRun.get(runId) ?? [],
        color: '#888',
        err: null,
      })
    }
    return out
  }, [iris, modal, visibleSet, evalByRun])

  // Pending-fire placeholder rows — Modal training fires recorded by
  // `tomat modal pending-fire add` immediately after `fn.spawn(...)`, before
  // the trainer container opens its wandb session. We pre-filter out fires
  // whose run_id is already covered by the manifest-derived `visible` set
  // (so a freshly-synced run doesn't double-render) or the iris-derived
  // `pendingCards` set (so an iris-pending TPU run with a matching fire id
  // isn't shadowed). The CLI's `gc` pass keeps R2 in sync over time, but
  // this client-side filter avoids a stale-window double-render.
  const pendingFireRows: { fire: PendingFire; app: ModalApp | null; fc: ModalFunctionCall | null }[] = useMemo(() => {
    const fires = pendingFires?.fires ?? []
    if (fires.length === 0) return []
    const irisPendingIds = new Set(pendingCards.map((c) => c.id))
    const out: { fire: PendingFire; app: ModalApp | null; fc: ModalFunctionCall | null }[] = []
    for (const fire of fires) {
      if (visibleSet.has(fire.run_id)) continue
      if (irisPendingIds.has(fire.run_id)) continue
      if (EXCLUDED_RUNS.has(fire.run_id)) continue
      const fc = modalFcForPending(modal, fire.fc_id)
      // Prefer the (single) deployed `tomat-train-smoke` app — the same
      // `modalAppForRun` heuristic used by manifest-derived Modal cards.
      const app = modalAppForRun(modal, fire.run_id)
      out.push({ fire, app, fc })
    }
    // Newest first within the placeholder bucket.
    out.sort((a, b) => b.fire.spawned_at_ms - a.fire.spawned_at_ms)
    return out
  }, [pendingFires, modal, visibleSet, pendingCards])

  // URL-persisted card sort mode (`?sort=`). Default 'active' = the existing
  // bucket-by-state + activity-desc behavior; others let the user re-key on
  // pure last-updated, created, name, or current step.
  const [sortMode, setSortMode] = useUrlState('sort', enumParam<SortMode>('active', SORT_MODES))

  // Drop v2 (P14) runs — obsolete tokenizer, and a few are mislabeled
  // `train-full-v3-…` (trained on v2 data) which is pure confusion on the board.
  const ordered = cards
    ? orderRuns([...cards, ...pendingCards].filter((c) => !isV2Run(c)), sortMode)
    : null
  const runningCount = ordered?.filter(isRunning).length ?? 0
  const queuedCount = ordered?.filter(isQueued).length ?? 0
  const incompleteCount = ordered?.filter(isIncomplete).length ?? 0

  // Step-range filter driven by the timeline plot's drag-to-zoom in `loss`
  // x-mode (axis = `global_step`). When the user zooms to e.g. [0, 3000], any
  // run whose entire step trajectory falls outside the visible range
  // contributes nothing to the plot — hide it from the legend AND from the
  // card list below, so the dashboard view stays internally consistent. The
  // plot calls `onStepRangeChange(range | null)` on relayout; null means
  // auto-range OR a non-`loss` x-mode (steps don't translate to wallclock
  // axes). See `RunsTimelinePlot.onStepRangeChange` for the contract.
  const [stepZoomRange, setStepZoomRange] = useState<[number, number] | null>(null)
  const inStepRangeIds = useMemo(() => {
    if (!stepZoomRange || !ordered) return null
    const [lo, hi] = stepZoomRange
    const s = new Set<string>()
    for (const c of ordered) {
      // No manifest yet (placeholder PENDING/BUILDING card): keep visible —
      // there's no step data to disqualify it. last_train_step falls back to
      // step_max for older manifests that pre-date that field. Treat a
      // missing upper bound as Infinity (still-training), missing lower as 0.
      const h = c.manifest?.history
      const firstStep = h?.step_min ?? 0
      const lastStep = h?.last_train_step ?? h?.step_max ?? Infinity
      if (firstStep <= hi && lastStep >= lo) s.add(c.id)
    }
    return s
  }, [ordered, stepZoomRange])

  // Build timeline series from any cards that have history loaded. Order
  // matters for color stability — use index from `ordered`. When the user
  // has zoomed the x-axis in `loss` mode, drop runs whose step range is
  // entirely outside the visible window: their trace contributes nothing,
  // and keeping them in the legend (custom legend = `filteredRuns.length`)
  // is visually misleading.
  const timelineSeries: RunTimelineSeries[] = (ordered ?? [])
    .filter((c) => c.history != null)
    .filter((c) => inStepRangeIds == null || inStepRangeIds.has(c.id))
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

  // Scroll the parent's card into view if it's currently rendered. Returns
  // false when the parent isn't in the DOM — letting the chip handler fall
  // through to its wandb-URL fallback. (We don't pop tag/regex filters here;
  // the user can flip the `+ancestors` toggle for that.)
  const scrollToParent = useCallback((parentId: string): boolean => {
    const el = cardRefs.current.get(parentId)
    if (!el) return false
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    return true
  }, [])

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

  // Clear the pin if the pinned run drops out of the visible step-range zoom
  // (`inStepRangeIds` excludes it). Without this, pinning a run and then
  // zooming the plot to a window that doesn't intersect its step trajectory
  // leaves the plot soloed-to-empty (no traces) — confusing for the user.
  // Plot pins the cleared run in by re-clicking once the zoom is reset.
  useEffect(() => {
    if (!pinnedRunId) return
    if (inStepRangeIds && !inStepRangeIds.has(pinnedRunId)) {
      highlight.clearPin()
    }
  }, [pinnedRunId, inStepRangeIds, highlight])

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

  // Tag chip filter (tri-state, URL-synced). `useTagFilters` is the shared
  // source of truth: URL `?tags=` → state, default `{bunk: 'out'}` on first
  // visit, legacy localStorage migrated on first read. The plot consumes the
  // same hook via the controlled `tagFilters` / `onTagFiltersChange` props
  // below, so chip clicks update both the plot and the card list in lockstep.
  const [tagFilters, onTagFiltersChange] = useTagFilters()
  // "Include ancestors of matching runs" toggle (URL: ?anc=1). When on, the
  // ancestor closure of every regex- + tag-matched run is added back into the
  // visible set, so filtering to a sweep (e.g. `SS-sweep`) automatically pulls
  // the parent (cont33k) in as a baseline. Off by default.
  const [ancestorsOn, setAncestorsOn] = useAncestorsToggle()
  // Runs whose tag set satisfies the tri-state constraint (overrides +
  // per-tag defaults). Distinct from regex-`matchedIds`, which only sorts:
  // tags hard-filter, so a run that fails the chip predicate is hidden
  // from the card list entirely. No empty-Map fast-path — per-tag defaults
  // (`bunk = out`) still constrain even when the override Map is empty.
  const tagMatchedIds = useMemo(() => {
    if (!ordered) return null
    const s = new Set<string>()
    for (const c of ordered) {
      if (runPassesTagFilters(tagsFor(shortLabel(c.id)), tagFilters)) s.add(c.id)
    }
    return s
  }, [tagFilters, ordered])

  // When `ancestorsOn`, expand the tag- and regex-match sets by walking the
  // lineage map upward from each matching run. Ancestors join the visible set
  // even if they fail the tag predicate (the whole point of the toggle is to
  // pull the parent baseline back into view). For the regex matcher we add
  // ancestors to `matchedIds` so they DON'T fade — they're explicitly the
  // user-requested baselines, not "see also" noise.
  const tagMatchedIdsExpanded = useMemo(() => {
    if (!ordered || !tagMatchedIds || !ancestorsOn) return tagMatchedIds
    const s = new Set(tagMatchedIds)
    for (const id of tagMatchedIds) for (const a of ancestorsOf(id)) s.add(a)
    return s
  }, [tagMatchedIds, ordered, ancestorsOn])
  const matchedIdsExpanded = useMemo(() => {
    if (!ordered || !matchedIds || !ancestorsOn) return matchedIds
    const s = new Set(matchedIds)
    for (const id of matchedIds) for (const a of ancestorsOf(id)) s.add(a)
    return s
  }, [matchedIds, ordered, ancestorsOn])

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
    const tagFiltered = tagMatchedIdsExpanded
      ? ordered.filter((c) => tagMatchedIdsExpanded.has(c.id))
      : ordered
    // Step-range hard filter: runs whose step trajectory falls entirely
    // outside the loss-mode zoom range are dropped from the card list to
    // match what the plot/legend show. `inStepRangeIds == null` (no zoom set
    // or non-loss x-mode) passes through.
    const stepFiltered = inStepRangeIds
      ? tagFiltered.filter((c) => inStepRangeIds.has(c.id))
      : tagFiltered
    const pinned = pinnedRunId ? stepFiltered.find((c) => c.id === pinnedRunId) : null
    const hovered = plotHoverRunId && plotHoverRunId !== pinnedRunId
      ? stepFiltered.find((c) => c.id === plotHoverRunId)
      : null
    const taken = new Set<string>()
    if (pinned) taken.add(pinned.id)
    if (hovered) taken.add(hovered.id)
    const rest = stepFiltered.filter((c) => !taken.has(c.id))
    const top: typeof stepFiltered = []
    if (pinned) top.push(pinned)
    if (hovered) top.push(hovered)
    if (!matchedIdsExpanded || matchedIdsExpanded.size === 0) {
      return [...top, ...rest]
    }
    const matched = rest.filter((c) => matchedIdsExpanded.has(c.id))
    const others = rest.filter((c) => !matchedIdsExpanded.has(c.id))
    return [...top, ...matched, ...others]
  }, [ordered, pinnedRunId, plotHoverRunId, matchedIdsExpanded, tagMatchedIdsExpanded, inStepRangeIds])

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
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
        <h1 style={{ margin: 0 }}>tomat runs</h1>
        <nav style={{ display: 'flex', gap: '0.75rem', fontSize: '0.9rem' }}>
          <a href="#/posts">posts</a>
          <a href="#/files">files</a>
          <a href="#/deck">deck</a>
        </nav>
      </div>
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
        {cronHeartbeatQ.data && (
          <>
            {' · '}
            <Tooltip
              content={
                <div style={{ fontSize: '0.8rem' }}>
                  GCE-VM cron heartbeat — last <code>runs-sync-active</code> tick.
                  <br />
                  ran <b>{cronHeartbeatQ.data.last_run_count ?? 0}</b> sync(s)
                  {cronHeartbeatQ.data.last_failure_count != null && (
                    <>, <b>{cronHeartbeatQ.data.last_failure_count}</b> failure(s)</>
                  )}
                  {cronHeartbeatQ.data.host && <> on <code>{cronHeartbeatQ.data.host}</code></>}
                </div>
              }
            >
              <span
                style={{
                  color:
                    (Date.now() - new Date(cronHeartbeatQ.data.ts).getTime()) > 5 * 60_000
                      ? 'crimson'
                      : '#888',
                }}
              >
                cron {timeAgo(cronHeartbeatQ.data.ts)}
              </span>
            </Tooltip>
          </>
        )}
        . See{' '}
        <a href="https://github.com/Open-Athena/tomat/blob/main/specs/23-runs-dashboard.md">
          spec
        </a>.
      </p>
      {err && <p style={{ color: 'crimson' }}>error: {err}</p>}
      {!ordered && !err && <p>loading…</p>}
      {ordered && ordered.length === 0 && <p>(none synced yet)</p>}
      {ordered && ordered.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem', fontSize: '0.75rem' }}>
          <span style={{ color: '#888' }}>sort:</span>
          {SORT_MODES.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setSortMode(m)}
              style={{
                fontSize: '0.75rem',
                padding: '0.15rem 0.5rem',
                borderRadius: 4,
                border: `1px solid ${sortMode === m ? '#4a8aff' : '#444'}`,
                background: sortMode === m ? 'rgba(74,138,255,0.15)' : 'transparent',
                color: 'inherit',
                cursor: 'pointer',
              }}
            >
              {m === 'active' ? 'active+updated' : m === 'updated' ? 'updated' : m}
            </button>
          ))}
        </div>
      )}
      {timelineSeries.length > 0 && (
        <div style={{ marginBottom: '1.2rem' }}>
          <RunsTimelinePlot
            runs={timelineSeries}
            highlight={highlight}
            runHaystacks={runHaystacksByLabel}
            onPlotHover={setPlotHoverLabel}
            tagFilters={tagFilters}
            onTagFiltersChange={onTagFiltersChange}
            ancestorsOn={ancestorsOn}
            onAncestorsChange={setAncestorsOn}
            onStepRangeChange={setStepZoomRange}
          />
        </div>
      )}
      {pendingFireRows.length > 0 && (
        <div style={{
          marginBottom: '0.4rem', fontSize: '0.72rem', color: '#888',
        }}>
          {pendingFireRows.length} pending fire{pendingFireRows.length > 1 ? 's' : ''}
          {' '}— awaiting wandb session (see{' '}
          <code>tomat modal pending-fire list</code>)
        </div>
      )}
      {pendingFireRows.map(({ fire, app, fc }) => (
        <PendingFireCard key={fire.run_id} fire={fire} app={app} fc={fc} />
      ))}
      {displayed && displayed.map((c) => {
        const faded = matchedIdsExpanded && matchedIdsExpanded.size > 0 && !matchedIdsExpanded.has(c.id)
        // Look up the parent's wandb URL from the snapshot's manifest map
        // (cards-only — `cards` is the unfiltered list, so even filtered-out
        // parents still expose their URL as a click-fallback target).
        const lin = lineageFor(c.id)
        const parentCard = lin && cards
          ? cards.find((rc) => rc.id === lin.parent)
          : null
        const parentWandbUrl = parentCard?.manifest?.run?.url ?? null
        // Parent's trace color — when the parent is in `cards` (so the plot
        // assigned it a stable color), surface that on the ParentChip swatch.
        const parentColor = parentCard?.color ?? null
        return (
          <div key={c.id} style={faded ? { opacity: 0.35 } : undefined}>
            <RunCard
              data={c}
              activeRunId={activeRunId}
              pinnedRunId={pinnedRunId}
              parentWandbUrl={parentWandbUrl}
              parentColor={parentColor}
              onHover={onCardHover}
              onClick={onCardClick}
              onScrollTargetRef={setCardRef}
              onScrollToParent={scrollToParent}
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

// `?lineage=full|seg` — when the detail run has a recorded lineage parent
// (see `./lineage.ts:RUN_LINEAGE`), `full` (the default) glues each
// ancestor's history onto the front so the TL/VL/MFU plots render the
// full lineage from step 0 instead of just the current segment.
// `seg` shows only this run's segment (the old behaviour). Hidden
// silently when the run has no recorded ancestors.
const LINEAGE_MODES = ['full', 'seg'] as const
type LineageMode = (typeof LINEAGE_MODES)[number]
function useLineageMode(): readonly [LineageMode, (v: LineageMode) => void] {
  const [m, set] = useUrlState('lineage', enumParam<LineageMode>('full', LINEAGE_MODES))
  return [m, set] as const
}

/** Toggle the lineage-merge for the detail view (`?lineage=full|seg`).
 *  Rendered only when the current run has recorded ancestors. */
function LineageToggle({
  mode, setMode, ancestors, ancestorHistoriesLoading,
}: {
  mode: LineageMode
  setMode: (v: LineageMode) => void
  ancestors: string[]
  ancestorHistoriesLoading: boolean
}) {
  const tip = mode === 'full'
    ? `Plot is glued to ${ancestors.length} ancestor segment${ancestors.length === 1 ? '' : 's'}: `
      + ancestors.slice().reverse().join(' → ')
      + ` → (this run). Click to show only this segment.`
    : `Plot shows only this run's segment. Click to glue ${ancestors.length} `
      + `ancestor segment${ancestors.length === 1 ? '' : 's'} on the left.`
  const label = mode === 'full' ? '◀ lineage glued' : '◀ +lineage'
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      margin: '0.4rem 0 0.2rem 0',
    }}>
      <Tooltip content={tip}>
        <button
          type="button"
          onClick={() => setMode(mode === 'full' ? 'seg' : 'full')}
          style={{
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            color: mode === 'full' ? '#cfe1ff' : '#9aa6c2',
            background: mode === 'full'
              ? 'rgba(80, 130, 220, 0.25)'
              : 'rgba(120,140,200,0.10)',
            border: `1px solid ${mode === 'full'
              ? 'rgba(120, 170, 240, 0.65)'
              : 'rgba(120,140,200,0.30)'}`,
            borderRadius: 10,
            padding: '2px 10px',
            cursor: 'pointer',
          }}>
          {label}
        </button>
      </Tooltip>
      {ancestorHistoriesLoading && mode === 'full' && (
        <span style={{ fontSize: '0.7rem', color: '#888' }}>
          loading ancestor history…
        </span>
      )}
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
  // Shared-key snapshot poll so we can discover children of this run via
  // their wandb config's `TOMAT_PARENT_RUN_ID`. tanstack-query dedupes
  // with RunsIndex's identical queryKey, so this is free when navigating
  // from the index, and one extra fetch on direct-URL loads.
  const snapshotQ = useQuery({
    queryKey: ['runs-snapshot'],
    queryFn: fetchRunsSnapshot,
    refetchInterval: REFETCH_MS.snapshot,
  })
  const manifestQ = useQuery({
    queryKey: ['manifest', runId],
    queryFn: () => fetchManifest(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.manifest,
    retry: 1,
  })
  // Mirror RunsIndex's lineage-registration effect — needed so direct
  // URL loads of /runs/<id> populate LINEAGE_FROM_MANIFEST and thus see
  // children. Effect is a no-op when the snapshot was pre-registered by
  // RunsIndex.
  useEffect(() => {
    const runs = snapshotQ.data?.runs
    if (!runs) return
    for (const [id, m] of Object.entries(runs)) {
      const cfg = (m as { run?: { config?: Record<string, unknown> } })?.run?.config
      if (cfg) registerLineageFromManifest(id, cfg)
    }
  }, [snapshotQ.data])
  const historyQ = useQuery({
    queryKey: ['history', runId],
    queryFn: () => fetchRunHistory(parquetUrl(runId)),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.history.active,
    retry: 1,
  })
  // Lineage history glue: walk `RUN_LINEAGE` ancestors-of(`runId`) and fetch
  // each parent's parquet in parallel. The default mode (`full`) concatenates
  // them ancestor-first → child for the WallclockPlot below; `seg` shows only
  // this run's own segment. When the run has no recorded ancestors, the toggle
  // is hidden and we render this run's history directly (one query, no concat).
  const [lineageMode, setLineageMode] = useLineageMode()
  // NMAE / NEMD toggle shared between MEvalTable (table cells) and
  // WallclockPlot (MT/MV panel). URL-persisted via `?mevm=`.
  const [mevalMetric, setMevalMetric] = useMEvalMetric()
  const ancestors = useMemo(() => ancestorsOf(runId), [runId, snapshotQ.data])
  // Children — direct descendants whose `TOMAT_PARENT_RUN_ID === runId`. We
  // recompute when snapshotQ.data refreshes so newly-registered manifests
  // surface here without waiting for a route change.
  const children = useMemo(() => childrenOf(runId), [runId, snapshotQ.data])
  // `useQueries` returns a stable array indexed by ancestor; ancestors-first
  // (parent, grandparent, …) so concat below reverses to chronological order.
  const ancestorHistoryQs = useQueries({
    queries: ancestors.map((aid) => ({
      queryKey: ['history', aid],
      queryFn: () => fetchRunHistory(parquetUrl(aid)),
      // Ancestors are by-definition finished segments; their parquet doesn't
      // change. Idle cadence keeps polling cheap (single 2 MB fetch).
      refetchInterval: REFETCH_MS.history.idle,
      retry: 1,
    })),
  })
  const irisQ = useQuery({
    queryKey: ['iris'], queryFn: fetchIrisState, refetchInterval: REFETCH_MS.iris,
  })
  // Modal app + function-call snapshot. Same R2-sourced state as the
  // index page consumes via the aggregated snapshot — re-fetched standalone
  // here so the detail-page header can show the live Modal badge without
  // round-tripping through the snapshot's edge cache.
  const modalQ = useQuery({
    queryKey: ['modal'], queryFn: fetchModalState, refetchInterval: REFETCH_MS.iris,
  })
  // Pending-fires sidecar — needed to resolve this run's bound Modal fc-id
  // so the Modal-icon deep-links to the specific call (not the multiplexed
  // app log stream). Cheap small JSON; reuses the same cron + refetch
  // cadence as modal-state.
  const pendingFiresQ = useQuery({
    queryKey: ['pending-fires'], queryFn: fetchPendingFires,
    refetchInterval: REFETCH_MS.iris,
  })
  // Per-task attempt history sidecar — written by `tomat iris sync` alongside
  // iris-state.json. 404s are normal (e.g. before the cron has caught a new
  // training run) and resolve to null without poisoning the query.
  const attemptsQ = useQuery({
    queryKey: ['iris-attempts', runId],
    queryFn: () => fetchIrisAttempts(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.iris,
    retry: 1,
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
  // Same query for each ancestor — spec 61 P1 #334. Without this the
  // MEvalTable + WallclockPlot MT/MV panel on a child's page render empty
  // (or only the child's own /10k segment), because eval.json is per-run.
  // We merge ancestor `sets[*]` into the current run's evalSeries below.
  const ancestorEvalQs = useQueries({
    queries: ancestors.map((aid) => ({
      queryKey: ['eval', aid],
      queryFn: () => fetchEval(aid),
      refetchInterval: REFETCH_MS.history.idle,
      retry: 1,
    })),
  })
  // Eval-records index (spec 43): per-(step, set, mode[, task]) lifecycle
  // records. Drives the EvalsPanel matrix view between WallclockPlot and
  // RecentEvents. Phase A is read-only + backfill; Phase B/C wire live
  // pending/running states from `tomat evals fire`/`sync`.
  const evalsIndexQ = useQuery({
    queryKey: ['evals-index', runId],
    queryFn: () => fetchEvalsIndex(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.history.active,
    retry: 1,
  })
  // Spec 46 Phase A: run cost (MSRP). On the detail page we fetch the full
  // breakdown directly (the tooltip needs it anyway); 404s land as null when
  // the run hasn't been `tomat cost compute`-d.
  const costQ = useQuery({
    queryKey: ['cost', runId],
    queryFn: () => fetchRunCost(runId),
    refetchInterval: (q) => errBackoff(q) ?? REFETCH_MS.manifest,
    retry: 1,
  })
  const manifest = manifestQ.data ?? null
  // Data-driven lineage (spec 63): register parent from wandb config so
  // `lineageFor()` picks it up without a RUN_LINEAGE registry edit. Fires
  // populated by `tomat train --parent` (spec 61 P1) thus appear in the
  // dashboard automatically.
  useEffect(() => {
    if (manifest?.run?.config) {
      registerLineageFromManifest(runId, manifest.run.config as Record<string, unknown>)
    }
  }, [runId, manifest])
  const ownHistory = historyQ.data ?? null
  // Lineage-aware concatenated history: ancestor-first → … → this run.
  // Available when every ancestor history has loaded (`every(d != null)`).
  // While ancestor fetches are still in flight, fall back to the own history
  // so the plot doesn't blink empty waiting for a 2 MB parquet to arrive.
  const lineageHistory = useMemo<RunHistory | null>(() => {
    if (lineageMode === 'seg') return ownHistory
    if (ancestors.length === 0) return ownHistory
    if (!ownHistory) return null
    const ancestorHistories: RunHistory[] = []
    for (const q of ancestorHistoryQs) {
      const h = q.data
      if (!h) return ownHistory  // hold off on glue until every ancestor lands
      ancestorHistories.push(h)
    }
    // Truncate each ancestor at the step where its CHILD forked off. Without
    // this, the child run's page shows the parent's full post-fork training
    // (e.g. bin5 trained to step-105k after bin5-cont-from-80k-v6e branched
    // at step-80k — those steps are an alternate timeline, not part of the
    // child's lineage). `ancestors` is child→root; child-of-ancestors[i] is
    // either runId (for i==0) or ancestors[i-1] (for i>0). We look up the
    // child's lineage entry to get the fork step.
    const truncated: RunHistory[] = ancestorHistories.map((h, i) => {
      const childId = i === 0 ? runId : ancestors[i - 1]
      const forkStep = lineageFor(childId)?.parent_step
      return forkStep != null ? truncateHistoryAtStep(h, forkStep) : h
    })
    // `ancestors[0]` is the immediate parent; the array is child→root. Reverse
    // to root→…→parent so concat ends in chronological order, then append the
    // current run's segment.
    const ordered = truncated.reverse()
    ordered.push(ownHistory)
    return concatHistories(ordered)
  }, [lineageMode, ownHistory, runId, ancestors, ancestorHistoryQs])
  const history = lineageHistory
  // Ancestor lineage metadata for the WallclockPlot — root → parent order
  // (matches the prefix of `partRowCounts` on the concatenated history).
  // Each ancestor gets a distinct fallback color from the shared palette;
  // we don't have the runs-index `ordered` here (single-run page), so the
  // per-card colors aren't available — the contrast palette is the only
  // option, but it's stable and deterministic per (runId, ancestorIdx).
  const lineageInfo = useMemo(() => {
    if (lineageMode === 'seg') return null
    if (ancestors.length === 0) return null
    if (!history?.partRowCounts || history.partRowCounts.length < 2) return null
    // `ancestors` is child→root; reverse to root→…→parent so it matches the
    // concat order. partRowCounts has N+1 entries (N ancestors + 1 own); the
    // last is `ownHistory.rowCount`, so we drop it for the ancestor list.
    const rootFirst = [...ancestors].reverse()
    const counts = history.partRowCounts.slice(0, rootFirst.length)
    // Pick colors from a fixed contrast palette, offset by 2 so an ancestor's
    // color doesn't visually collide with TL (`#ef5350`, red) or VL
    // (`#ffa726`, amber) on the loss panel. Palette has 10 entries; we mod.
    const ancestorColor = (i: number) => colorForIndex(i + 2)
    return rootFirst.map((name, i) => ({
      name,
      rowCount: counts[i] ?? 0,
      color: ancestorColor(i),
    }))
  }, [lineageMode, ancestors, history])
  const evalJobs = useMemo(
    () => evalJobsByRun(irisQ.data ?? undefined).get(runId) ?? [],
    [irisQ.data, runId],
  )
  // Lineage-aware eval series: concatenate ancestor eval.json points
  // (truncated at each fork step, matching the lineageHistory pattern) onto
  // the current run's series. Without this, child run pages show no MT/MV
  // history because the autosync writes eval results under the run they
  // were fired from. spec 61 P1 #334.
  const evalSeries = useMemo<RunEval | null>(() => {
    const own = evalQ.data
    // Per-row `_writtenAt` proxy = the SOURCE run's `manifest.run.created_at`.
    // `formatStepDetail` reads this to decide whether to apply the legacy
    // info.step-OBO asterisk. Runs created after `LEGACY_STEP_NAMING_CUTOFF`
    // use exact step-N naming; their rows render without `*`.
    const snapRuns = (snapshotQ.data?.runs ?? {}) as Record<string, { run?: { created_at?: string } } | null>
    const writtenAtForRun = (rid: string): string | undefined =>
      snapRuns[rid]?.run?.created_at ?? (rid === runId ? manifest?.run?.created_at : undefined)
    const stamp = (pts: EvalPoint[], rid: string): EvalPoint[] => {
      const wa = writtenAtForRun(rid)
      if (wa === undefined) return pts
      return pts.map((p) => p._writtenAt === undefined ? { ...p, _writtenAt: wa } : p)
    }
    if (lineageMode === 'seg' || ancestors.length === 0) {
      if (!own?.sets) return own ?? null
      const stamped: Record<string, EvalPoint[]> = {}
      for (const [set, pts] of Object.entries(own.sets)) {
        stamped[set] = stamp(pts, runId)
      }
      return { ...own, sets: stamped }
    }
    // Hold off on glue while any ancestor eval.json is still loading. 404 is
    // a valid resolved state (`data === null` from fetchEval) — we just skip
    // those ancestors. We only block on still-pending fetches.
    if (ancestorEvalQs.some((q) => q.isLoading)) return own ?? null
    const merged: Record<string, EvalPoint[]> = {}
    // ancestors[i] is child→root; ancestors[i]'s child is runId (i==0) or
    // ancestors[i-1] (i>0). Truncate that ancestor's series at the fork step
    // so the child doesn't show its parent's post-fork eval points.
    ancestors.forEach((aid, i) => {
      const data = ancestorEvalQs[i]?.data
      if (!data?.sets) return
      const childId = i === 0 ? runId : ancestors[i - 1]
      const forkStep = lineageFor(childId)?.parent_step
      for (const [set, pts] of Object.entries(data.sets)) {
        const kept = forkStep == null
          ? pts
          : pts.filter((p) => p.step <= forkStep)
        if (!kept.length) continue
        const arr = merged[set] ?? []
        arr.push(...stamp(kept, aid))
        merged[set] = arr
      }
    })
    if (own?.sets) {
      for (const [set, pts] of Object.entries(own.sets)) {
        const arr = merged[set] ?? []
        arr.push(...stamp(pts, runId))
        merged[set] = arr
      }
    }
    for (const set of Object.keys(merged)) {
      merged[set].sort((a, b) => a.step - b.step)
    }
    if (!Object.keys(merged).length) return own ?? null
    return {
      schema_version: own?.schema_version ?? 1,
      synced_at: own?.synced_at ?? new Date().toISOString(),
      run: runId,
      sets: merged,
    }
  }, [evalQ.data, ancestorEvalQs, ancestors, lineageMode, runId, snapshotQ.data, manifest])
  // step → { val_200, train_200 } eval point, from the lineage-merged series.
  const evalByStep = useMemo(() => {
    const m = new Map<number, Record<string, EvalPoint>>()
    const sets = evalSeries?.sets
    if (!sets) return m
    for (const [set, pts] of Object.entries(sets)) {
      for (const pt of pts) {
        const rec = m.get(pt.step) ?? {}
        rec[set] = pt
        m.set(pt.step, rec)
      }
    }
    return m
  }, [evalSeries])
  const errObj = manifestQ.error || historyQ.error
  const err = errObj ? String(errObj) : null

  // Build a RunCardData for the shared header. iris job lookup mirrors the
  // index view's `cards: ... irisJobIdForRun(id)`. The color isn't shown on
  // the detail page's outer chrome (no accent bar), but the header doesn't
  // use it directly — leave it as a neutral colour for prop-shape parity.
  //
  // Apply the same `RUNNING_RECENCY_MS` freshness gate as the index page's
  // card construction: `modalAppForRun` matches a single deployed app to
  // any `-modal-` run, so without this gate every finished Modal-launched
  // run on its detail page inherits the live app and the badge reads
  // "DEPLOYED (1S)" even hours after the run completed.
  const detailModalCandidate = modalAppForRun(modalQ.data ?? null, runId)
  const detailTsMax = manifest?.history?.ts_max
  const detailModalRecent = typeof detailTsMax === 'number'
    && Date.now() - detailTsMax * 1000 < RUNNING_RECENCY_MS
  // Same BUILDING-state allowance as the index view's card construction.
  const detailCreatedMs = manifest?.run?.created_at
    ? Date.parse(manifest.run.created_at) : NaN
  const detailRunRecent = Number.isFinite(detailCreatedMs)
    && Date.now() - detailCreatedMs < 24 * 3600 * 1000
  const headerData: RunCardData = {
    id: runId,
    manifest,
    job: irisQ.data?.jobs[irisJobIdForRun(runId)] ?? null,
    modalApp: detailModalCandidate && (detailModalRecent || detailRunRecent) ? detailModalCandidate : null,
    modalFc: pendingFcForRun(pendingFiresQ.data ?? null, modalQ.data ?? null, runId),
    modalFcId: pendingFcIdForRun(pendingFiresQ.data ?? null, runId),
    history,
    evalJobs,
    color: '#888',
    err,
    attempts: attemptsQ.data ?? null,
    // Synthesize a RunCostSummary from the full RunCost record so the chip
    // can render without waiting on a second hover-fetch (the detail page
    // already has the full breakdown via `costQ`).
    cost: costQ.data
      ? {
          msrp_usd: costQ.data.msrp_usd,
          is_complete: costQ.data.is_complete,
          pricing_table_version: costQ.data.pricing_table_version,
          has_modal_pending: costQ.data.breakdown.some((b) => b.kind === 'modal'),
        }
      : null,
  }

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
      <div style={{
        // Match the card's chrome (border + padding + dark bg) so the detail
        // header reads as "the same block as the pinned card" — the user can
        // see at a glance that it's the same data, just at the top of a
        // different page.
        border: '1px solid #2a2a2a',
        borderRadius: 6,
        padding: '0.8rem 1rem',
        marginBottom: '1rem',
        backgroundColor: '#181818',
      }}>
        <RunHeaderRich data={headerData} linkRunName={false} showSegmentBreakdown hideJobLinks />
      </div>
      {(() => {
        // Lineage table: one row per wandb run in this experiment's chain
        // (root → … → current). The wandb/iris/modal title-bar icons that
        // used to live on the header now live here (per-row), so ancestors
        // get their links surfaced too instead of just the current run.
        // 1 wandb run ≈ 1 job under our forward-going invariant
        // (`tomat train --force` guards + new-fire-per-resume convention).
        // When that holds, this table doubles as the "job rollup" view;
        // when it doesn't, a v2 driven by fires-as-records will key on
        // fire-id from R2 instead.
        //
        // Step ranges use Levanter's `global_step` column (the training
        // counter) NOT wandb's `_step` (which restarts at 0 for each new
        // wandb run, so a resume's row would say `0-Nk` instead of
        // `parentStep-Nk`).
        const stepRangeFromHistory = (h: RunHistory | null): { start: number | null; end: number | null } => {
          if (!h) return { start: null, end: null }
          const gs = h.cols.get('global_step') ?? []
          let start: number | null = null
          let end: number | null = null
          for (let i = 0; i < gs.length; i++) {
            const v = gs[i]
            if (v == null) continue
            const n = Number(v)
            if (start == null) start = n
            end = n
          }
          return { start, end }
        }
        const rootFirst = [...ancestors].reverse()  // root → parent
        const ancestorRows: LineageRow[] = rootFirst.map((aid, i) => {
          const h = ancestorHistoryQs[ancestors.length - 1 - i]?.data ?? null
          const { start, end } = stepRangeFromHistory(h)
          return { runId: aid, startStep: start, endStep: end, current: false }
        })
        const { start: ownStart, end: ownEnd } = stepRangeFromHistory(ownHistory)
        // Children: any registered run whose parent === this run. Step ranges
        // are not fetched here (would 2× the history requests); the row shows
        // the run name + links and leaves step range null. Drill into the
        // child's own page for its full timeline.
        const childRows: LineageRow[] = children.map((cid) => ({
          runId: cid, startStep: null, endStep: null, current: false, kind: 'child',
        }))
        const allRows: LineageRow[] = [
          ...ancestorRows,
          { runId, startStep: ownStart, endStep: ownEnd, current: true },
          ...childRows,
        ]
        return <LineageTable rows={allRows} />
      })()}
      {ancestors.length > 0 && (
        <LineageToggle
          mode={lineageMode}
          setMode={setLineageMode}
          ancestors={ancestors}
          ancestorHistoriesLoading={ancestorHistoryQs.some((q) => q.isLoading)}
        />
      )}
      {/* Per-step m-eval table — compact summary of the eval.json points
          surfaced as `step | val_200 | train_200 | …-maskgit` with subtle
          colour-coding. Sits above the timeline so the most-important
          quality numbers are immediately visible. The NMAE/NEMD toggle on
          the table also flips the MT/MV panel on the WallclockPlot below
          (shared `?mevm` URL state, lifted here). Self-hides when
          `eval.json` returned nothing. */}
      <MEvalTable
        runId={runId}
        evalSeries={evalSeries}
        evalJobs={evalJobs}
        metric={mevalMetric}
        setMetric={setMevalMetric}
        runCreatedAt={manifest?.run?.created_at ?? null}
      />
      {!history && !err && <p>loading parquet…</p>}
      {history && (
        <WallclockPlot
          history={history}
          evalSeries={evalSeries}
          runId={runId}
          attempts={attemptsQ.data ?? null}
          manifest={manifest}
          lineageInfo={lineageInfo}
          mevalMetric={mevalMetric}
        />
      )}
      <EvalsPanel
        runId={runId}
        evalsIndex={evalsIndexQ.data ?? null}
        runCreatedAt={manifest?.run?.created_at ?? null}
      />
      <RecentEvents
        attempts={attemptsQ.data ?? null}
        modalApp={modalAppForRun(modalQ.data ?? null, runId)}
        history={history}
      />
    </div>
  )
}
