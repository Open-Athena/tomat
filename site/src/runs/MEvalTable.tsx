// Per-step mat-eval table for the run-detail page.
//
// One row per step that has eval results in `eval.json` (already fetched at
// the page level via `fetchEval(runId)`) OR an in-flight iris eval job for
// that step. Columns: `step`, `val_200`, `train_200`, `val_200-maskgit`,
// `train_200-maskgit`. Each set cell is `<NMAE> / <NEMD> / n=<N>` with
// subtle colour-coding by quality threshold; in-flight cells show a spinner
// + iris state + elapsed time; failed cells show a failure chip with the
// iris error in a tooltip; missing cells show a `[ fire ]` button so the
// user can request a new fire without leaving the page.
//
// Rows sorted by step descending (latest first). Hidden entirely when neither
// `eval.json` nor any in-flight eval job exists.
//
// Distinct from `EvalsPanel.tsx` (spec 43, lifecycle records): that view
// shows per-(step, set, mode, task) record states (pending/running/etc.)
// from the eval-records index; this view is the canonical per-step
// aggregate scalars from the eval-results JSON, augmented with the live
// iris-job state for in-flight + failed + missing cells.
//
// TODO(spec 51 Phase A): once per-mat prediction zarrs land on R2, add a
// `pred` column that opens elvis for the per-mat NMAE / GT-vs-pred view.
// `gs://marin-eu-west4/tomat/eval/results/<run>/<set><mode>/step-<N>.json`
// already carries `per_mat[].mp_id`, so the elvis link-out is just a
// (run_id, step, mp_id) tuple → an elvis URL.

import { useState, type CSSProperties, type ReactElement } from 'react'
import { useTheme } from 'pltly/react'
import { Tooltip } from '../Tooltip'
import {
  evalPhase, evalSetKey, fireEval,
  type EvalJob, type EvalMode, type EvalPoint, type RunEval,
} from './api'
import {
  nemdBucket, nmaeBucket, pointsByStepSet, stepsDescOf, type Bucket,
} from './MEvalTable.helpers'

// Re-export for downstream tests / consumers that want the bucket types.
export { nemdBucket, nmaeBucket, stepsDescOf } from './MEvalTable.helpers'
import { formatStepDetail } from '../lib/runNames'
export type { Bucket } from './MEvalTable.helpers'

/** Metric the M-eval section displays — NMAE (default) or NEMD. Both the
 *  plot (WallclockPlot's MT/MV panel) and this table read from the same
 *  shared state in RunsPage so the toggle drives both at once. */
export type MEvalMetric = 'nmae' | 'nemd'

interface Props {
  runId: string
  evalSeries: RunEval | null
  /** Iris eval jobs for this run (from `evalJobsByRun(iris).get(runId)`).
   *  Used to surface in-flight / failed cells AND to enrich the fire-button
   *  optimistic UI (so re-fired cells immediately show "queued"). */
  evalJobs: EvalJob[]
  metric: MEvalMetric
  setMetric: (m: MEvalMetric) => void
}

/** Column-key → display label. The four eval-set/mode combinations we
 *  surface in the table. Order is also the column display order. */
// Eval-mode labels (`K=1` / `K=12`) are renamed from the underlying setKey
// for honesty: `oneshot` is the K=1 limit of MaskGIT iterative decode (one
// bidir forward, no top-k schedule), `maskgit` is the K=12 schedule. The
// numeric labels make the cost/quality tradeoff explicit (K=12 is 12× per
// mat). Underlying setKeys stay `val_200`/`val_200-maskgit` so JSON
// reads + dump paths don't migrate.
interface ColSpec {
  key: string
  label: string
  matSet: string
  mode: EvalMode
}
// Per the project decision to drop the GIGO eval modes (memory
// [[k1-oneshot-on-mg-is-gigo]] — oneshot mode for MG models was
// double-OOD; the K=12 maskgit "re-forward" extra forward pass was also
// strictly OOD for absorbing-trained models), we only surface the new
// honest K=1 maskgit numbers — `--mode maskgit --mg-k-steps 1` with
// the re-forward removed (so argmax-per-iter `filled_bins` is the
// canonical prediction). Sync writes these to `<set>-maskgit-K1`
// (separate series from any historic K=12 in `<set>-maskgit`).
const COLUMNS: ColSpec[] = [
  { key: 'val_200-maskgit-K1',   label: 'val_200 · K=1',   matSet: 'val_200',   mode: 'maskgit' },
  { key: 'train_200-maskgit-K1', label: 'train_200 · K=1', matSet: 'train_200', mode: 'maskgit' },
]

const BUCKET_COLORS_LIGHT: Record<Bucket, string> = {
  good: '#1a7a39',   // muted green
  mid:  '#a47013',   // amber
  bad:  '#9b2a1f',   // muted red
  none: '#666',
}
const BUCKET_COLORS_DARK: Record<Bucket, string> = {
  good: '#7bd99d',
  mid:  '#f0c674',
  bad:  '#f49a92',
  none: '#888',
}

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '–'
  return `${(v * 100).toFixed(digits)}%`
}

/** "5m ago" / "2h ago" — same format used elsewhere in the dashboard. */
function fmtElapsedShort(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  if (s < 60) return `${s}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`
  return `${(s / 86400).toFixed(1)}d`
}

interface SetCellProps {
  pt: EvalPoint
  isDark: boolean
  metric: MEvalMetric
}

function SetCell({ pt, isDark, metric }: SetCellProps): ReactElement {
  const COLORS = isDark ? BUCKET_COLORS_DARK : BUCKET_COLORS_LIGHT
  // Single active metric per cell — the section header tells the reader
  // which it is, and the column labels stay short ("val_200 · K=1" rather
  // than "val_200 · K=1 NMAE"). Bucket thresholds + display digits are
  // metric-specific; everything else is symmetric.
  const meanFrac = metric === 'nmae' ? pt.nmae_mean : pt.nemd_mean
  const pct = typeof meanFrac === 'number' ? meanFrac * 100 : NaN
  const bucket = metric === 'nmae' ? nmaeBucket(pct) : nemdBucket(pct)
  const color = COLORS[bucket]
  return (
    <span>
      <span style={{ color }}>{fmtPct(meanFrac)}</span>
      {pt.n_mats != null && (
        <span style={{ color: '#777' }}> · n={pt.n_mats}</span>
      )}
    </span>
  )
}

/** Pure-CSS spinner — one element, animated via the keyframes injected once
 *  at module init (see `injectSpinnerKeyframes` below). */
function Spinner({ size = 10, color = '#d4a017' }: { size?: number; color?: string }): ReactElement {
  return (
    <span
      aria-label="loading"
      style={{
        display: 'inline-block',
        width: size, height: size,
        borderRadius: '50%',
        border: `2px solid ${color}`,
        borderTopColor: 'transparent',
        animation: 'meval-spin 0.9s linear infinite',
        verticalAlign: 'middle',
      }}
    />
  )
}

// One-shot stylesheet for the spinner animation + queued-cell pulse.
let stylesInjected = false
function injectStyles(): void {
  if (stylesInjected || typeof document === 'undefined') return
  stylesInjected = true
  const style = document.createElement('style')
  style.textContent = `
    @keyframes meval-spin { to { transform: rotate(360deg); } }
    @keyframes meval-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  `
  document.head.appendChild(style)
}

interface InFlightCellProps {
  job: EvalJob
  /** All jobs for this cell (one per task when `--num-tasks` > 1). */
  jobs: EvalJob[]
}

function InFlightCell({ job, jobs }: InFlightCellProps): ReactElement {
  const j = job.job
  const elapsedMs = j.started_at_ms != null
    ? Date.now() - j.started_at_ms
    : (j.submitted_at_ms != null ? Date.now() - j.submitted_at_ms : null)
  const stateLower = j.state.toLowerCase()
  const elapsedStr = elapsedMs != null ? fmtElapsedShort(elapsedMs) : null
  // Multi-task: show e.g. "running · 2/4". Single-task: show state alone.
  const multi = jobs.length > 1
  const succeededTasks = multi ? jobs.filter((ej) => ej.job.state === 'SUCCEEDED').length : 0
  const tooltipLines: string[] = [
    `${job.jobId}`,
    `state: ${j.state}`,
    elapsedStr ? `elapsed: ${elapsedStr}` : null,
    j.submitted_at_ms != null ? `submitted: ${new Date(j.submitted_at_ms).toLocaleString()}` : null,
    j.started_at_ms != null ? `started: ${new Date(j.started_at_ms).toLocaleString()}` : null,
    multi ? `${succeededTasks}/${jobs.length} tasks done` : null,
  ].filter(Boolean) as string[]
  return (
    <Tooltip content={tooltipLines.join(' · ')}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        <Spinner size={9} color="#d4a017" />
        <span style={{ color: '#d4a017' }}>{stateLower}</span>
        {multi && (
          <span style={{ color: '#888' }}> · {succeededTasks}/{jobs.length}</span>
        )}
        {elapsedStr && (
          <span style={{ color: '#666' }}> · {elapsedStr}</span>
        )}
      </span>
    </Tooltip>
  )
}

interface FailedCellProps {
  job: EvalJob
  jobs: EvalJob[]
  /** Renders a [retry] button alongside the failure chip. */
  onFire: () => void
  isFiring: boolean
  fireError: string | null
}

function FailedCell({ job, jobs, onFire, isFiring, fireError }: FailedCellProps): ReactElement {
  const j = job.job
  const tip = [
    job.jobId,
    `state: ${j.state}`,
    j.error ? `error: ${j.error}` : null,
    j.finished_at_ms != null ? `finished: ${new Date(j.finished_at_ms).toLocaleString()}` : null,
    jobs.length > 1 ? `${jobs.length} task(s) for this cell` : null,
    fireError ? `last fire-request error: ${fireError}` : null,
  ].filter(Boolean).join(' · ')
  return (
    <Tooltip content={tip}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        <span style={{ color: '#cb2431' }}>failed</span>
        <FireButton onClick={onFire} isFiring={isFiring} label="retry" />
      </span>
    </Tooltip>
  )
}

function FireButton({
  onClick, isFiring, label = 'fire',
}: { onClick: () => void; isFiring: boolean; label?: string }): ReactElement {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onClick() }}
      disabled={isFiring}
      style={{
        fontFamily: 'monospace', fontSize: '0.7rem',
        padding: '0px 5px',
        borderRadius: 3,
        border: '1px solid #4a8aff',
        background: 'rgba(74,138,255,0.08)',
        color: isFiring ? '#666' : '#9aa6c2',
        cursor: isFiring ? 'progress' : 'pointer',
      }}
    >
      [{isFiring ? '…' : label}]
    </button>
  )
}

function QueuedCell({ tooltip }: { tooltip: string }): ReactElement {
  return (
    <Tooltip content={tooltip}>
      <span style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        animation: 'meval-pulse 1.6s ease-in-out infinite',
      }}>
        <Spinner size={9} color="#4a8aff" />
        <span style={{ color: '#9aa6c2' }}>queued</span>
      </span>
    </Tooltip>
  )
}

/** Compact NMAE/NEMD segmented control — matches the x-axis chip rail at
 *  the top of WallclockPlot (same border + active-state colors) so the two
 *  controls read as siblings. */
function MetricChips({
  metric, setMetric, isDark,
}: {
  metric: MEvalMetric
  setMetric: (m: MEvalMetric) => void
  isDark: boolean
}): ReactElement {
  const btn = (m: MEvalMetric): CSSProperties => ({
    fontSize: '0.75rem',
    padding: '0.15rem 0.5rem',
    borderRadius: 4,
    border: `1px solid ${metric === m ? '#4a8aff' : (isDark ? '#444' : '#ccc')}`,
    background: metric === m ? 'rgba(74,138,255,0.15)' : 'transparent',
    color: 'inherit',
    cursor: 'pointer',
  })
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
      <span style={{ fontSize: '0.75rem', color: '#888' }}>metric:</span>
      {(['nmae', 'nemd'] as const).map((m) => (
        <button key={m} type="button" onClick={() => setMetric(m)} style={btn(m)}>
          {m.toUpperCase()}
        </button>
      ))}
    </span>
  )
}

/**
 * Group eval jobs by (step, set, mode) key matching column keys. Picks the
 * "most relevant" job per cell:
 *   - any in-flight task wins over terminal tasks (one in-flight task = cell
 *     is in flight, even if siblings finished)
 *   - among in-flight tasks, the most recently started
 *   - if no in-flight: most recent SUCCEEDED beats most recent FAILED
 *     (a result already lives in eval.json — cell will render the result, not
 *     the job, anyway — but the failed sibling shouldn't surface as the
 *     primary state)
 */
function jobsByStepCol(jobs: EvalJob[]): Map<string, EvalJob[]> {
  const byKey = new Map<string, EvalJob[]>()
  for (const j of jobs) {
    const setKey = evalSetKey(j.matSet, j.mode, j.variant)
    const k = `${j.step}|${setKey}`
    const arr = byKey.get(k) ?? []
    arr.push(j)
    byKey.set(k, arr)
  }
  return byKey
}

function representativeJob(jobs: EvalJob[]): EvalJob {
  // Prefer in-flight > succeeded > failed/killed. Within each bucket, prefer
  // the most-recently-touched (started_at, submitted_at).
  const order: Record<string, number> = { flight: 0, done: 1, failed: 2 }
  const rank = (ej: EvalJob): [number, number] => {
    const phase = evalPhase(ej.job)
    const tier = order[phase] ?? 3
    const ts = ej.job.started_at_ms ?? ej.job.submitted_at_ms ?? 0
    return [tier, -ts]
  }
  return [...jobs].sort((a, b) => {
    const [ra, rb] = [rank(a), rank(b)]
    return ra[0] - rb[0] || ra[1] - rb[1]
  })[0]
}

export function MEvalTable({
  runId, evalSeries, evalJobs, metric, setMetric,
}: Props): ReactElement | null {
  const { isDark } = useTheme()
  injectStyles()

  // Local UI state for fire-button optimistic / error rendering. Map key is
  // `${step}|${colKey}`.
  const [queued, setQueued] = useState<Set<string>>(new Set())
  const [firing, setFiring] = useState<Set<string>>(new Set())
  const [fireErrors, setFireErrors] = useState<Map<string, string>>(new Map())

  const evalSteps = stepsDescOf(evalSeries)
  const lookup = pointsByStepSet(evalSeries)
  const jobsByCol = jobsByStepCol(evalJobs)

  // Steps come from both eval.json (completed rows) and in-flight iris jobs
  // (so an in-flight K=12 fire at a step that has no completed rows still
  // renders as its own row). Sort descending.
  const allStepsSet = new Set<number>(evalSteps)
  for (const j of evalJobs) allStepsSet.add(j.step)
  const steps = [...allStepsSet].sort((a, b) => b - a)
  if (steps.length === 0) return null
  // Only render columns that have at least one data point OR an iris job
  // across all steps. Keeps the table narrow on teacher-only runs (no
  // maskgit-mode evals).
  const colsWithData = COLUMNS.filter((c) =>
    steps.some((s) => lookup.has(`${s}|${c.key}`) || jobsByCol.has(`${s}|${c.key}`)),
  )
  if (colsWithData.length === 0) return null

  // (The K=12 bulk-fire button is gone post-decision to drop K>1 evals —
  // see memory [[k1-oneshot-on-mg-is-gigo]] and the K=1-only columns above.
  // Missing cells still surface per-row via the `[fire]` button next to
  // empty cells; per-cell `fireOne` handles its own state.)

  const fireOne = async (step: number, col: ColSpec, reason: string): Promise<void> => {
    const key = `${step}|${col.key}`
    setFiring((prev) => new Set(prev).add(key))
    setFireErrors((prev) => {
      const m = new Map(prev)
      m.delete(key)
      return m
    })
    try {
      const resp = await fireEval({
        run_id: runId, step, mat_set: col.matSet, mode: col.mode, reason,
      })
      if (!resp.ok) {
        setFireErrors((prev) => {
          const m = new Map(prev)
          m.set(key, resp.error ?? 'fire request failed')
          return m
        })
      } else {
        setQueued((prev) => new Set(prev).add(key))
      }
    } catch (e) {
      setFireErrors((prev) => {
        const m = new Map(prev)
        m.set(key, String(e))
        return m
      })
    } finally {
      setFiring((prev) => {
        const s = new Set(prev)
        s.delete(key)
        return s
      })
    }
  }

  const headerStyle: React.CSSProperties = {
    padding: '4px 12px 4px 0',
    textAlign: 'left',
    color: '#888',
    fontWeight: 'normal',
    borderBottom: '1px solid #2a2a2a',
  }
  const cellStyle: React.CSSProperties = {
    padding: '4px 12px 4px 0',
    textAlign: 'left',
    verticalAlign: 'top',
    fontFamily: 'monospace',
    fontSize: '0.78rem',
  }

  return (
    <div style={{ marginTop: '0.8rem', marginBottom: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.8rem', marginBottom: '0.3rem',
                    flexWrap: 'wrap' }}>
        <h2 style={{ fontSize: '0.95rem', margin: 0 }}>
          Per-step m-eval ({steps.length} step{steps.length === 1 ? '' : 's'}) ·{' '}
          <span style={{ color: '#888', fontWeight: 'normal' }}>{metric.toUpperCase()}</span>
        </h2>
        <MetricChips metric={metric} setMetric={setMetric} isDark={isDark} />
        {/* Bulk-fire K=12 button removed — K>1 evals deprecated;
            per-cell `[fire]` button is still available for missing cells. */}
      </div>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem' }}>
        <thead>
          <tr>
            <th style={headerStyle}>step</th>
            {colsWithData.map((c) => (
              <th key={c.key} style={headerStyle}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {steps.map((step) => (
            <tr key={step}>
              <td style={cellStyle}>
                {(() => {
                  // `*` flags legacy info.step-OBO interpretation: pre-fix
                  // Levanter wrote `info.step = state.step − 1`, so periodic
                  // ckpts named `step-30000` actually = 30,001 completed
                  // steps, and force-save end-of-segment ckpts at `step-49999`
                  // = 50,000 completed steps (exact after +1). `*` is the
                  // periodic / off-by-one case; the force-save snap renders
                  // as round Nk with no asterisk (tooltip explains).
                  const d = formatStepDetail(step)
                  return d.isLegacy ? (
                    <Tooltip content={d.tooltip}>
                      <span>{d.display}</span>
                    </Tooltip>
                  ) : d.display
                })()}
              </td>
              {colsWithData.map((c) => {
                const key = `${step}|${c.key}`
                const pt = lookup.get(key)
                const jobs = jobsByCol.get(key)
                const isQueued = queued.has(key)
                const isFiring = firing.has(key)
                const fireError = fireErrors.get(key) ?? null
                let inner: ReactElement
                if (pt) {
                  inner = <SetCell pt={pt} isDark={isDark} metric={metric} />
                } else if (jobs && jobs.length > 0) {
                  // Pick the representative job (in-flight > done > failed,
                  // tiebreak by most-recent-touched).
                  const rep = representativeJob(jobs)
                  const phase = evalPhase(rep.job)
                  if (phase === 'flight') {
                    inner = <InFlightCell job={rep} jobs={jobs} />
                  } else if (phase === 'failed') {
                    inner = (
                      <FailedCell
                        job={rep} jobs={jobs}
                        onFire={() => fireOne(step, c, 'retry-from-dashboard')}
                        isFiring={isFiring}
                        fireError={fireError}
                      />
                    )
                  } else {
                    // SUCCEEDED but no eval.json point — shouldn't really
                    // happen (`tomat evals sync` would have ingested the
                    // result by now) but render plainly so the cell isn't
                    // a confusing dash.
                    inner = <span style={{ color: '#7bd99d' }}>succeeded</span>
                  }
                } else if (isQueued) {
                  inner = <QueuedCell tooltip={
                    `fire-request queued on R2 for ${runId} step ${step} ${c.matSet} (${c.mode}). `
                    + `The GCE-VM cron picks it up on its next pass.`
                  } />
                } else {
                  inner = (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ color: '#555' }}>–</span>
                      <FireButton
                        onClick={() => fireOne(step, c, 'single-from-dashboard')}
                        isFiring={isFiring}
                      />
                      {fireError && (
                        <Tooltip content={fireError}>
                          <span style={{ color: '#cb2431', fontSize: '0.7rem' }}>err</span>
                        </Tooltip>
                      )}
                    </span>
                  )
                }
                return (
                  <td key={c.key} style={cellStyle}>
                    {inner}
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
