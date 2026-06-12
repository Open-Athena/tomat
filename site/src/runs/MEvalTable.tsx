// Per-step mat-eval table for the run-detail page.
//
// One row per step that has eval results in `eval.json` (already fetched at
// the page level via `fetchEval(runId)`). Columns: `step`, `val_200`,
// `train_200`, `val_200-maskgit`, `train_200-maskgit`. Each set cell is
// `<NMAE> / <NEMD> / n=<N>` with subtle colour-coding by quality threshold.
//
// Rows sorted by step descending (latest first). Hidden entirely when
// `eval.json` returned nothing.
//
// Distinct from `EvalsPanel.tsx` (spec 43, lifecycle records): that view
// shows per-(step, set, mode, task) record states (pending/running/etc.)
// from the eval-records index; this view is the canonical per-step
// aggregate scalars from the eval-results JSON.
//
// TODO(spec 51 Phase A): once per-mat prediction zarrs land on R2, add a
// `pred` column that opens elvis for the per-mat NMAE / GT-vs-pred view.
// `gs://marin-eu-west4/tomat/eval/results/<run>/<set><mode>/step-<N>.json`
// already carries `per_mat[].mp_id`, so the elvis link-out is just a
// (run_id, step, mp_id) tuple → an elvis URL.

import type { CSSProperties, ReactElement } from 'react'
import { useTheme } from 'pltly/react'
import type { EvalPoint, RunEval } from './api'
import {
  nemdBucket, nmaeBucket, pointsByStepSet, stepsDescOf, type Bucket,
} from './MEvalTable.helpers'

// Re-export for downstream tests / consumers that want the bucket types.
export { nemdBucket, nmaeBucket, stepsDescOf } from './MEvalTable.helpers'
export type { Bucket } from './MEvalTable.helpers'

/** Metric the M-eval section displays — NMAE (default) or NEMD. Both the
 *  plot (WallclockPlot's MT/MV panel) and this table read from the same
 *  shared state in RunsPage so the toggle drives both at once. */
export type MEvalMetric = 'nmae' | 'nemd'

interface Props {
  evalSeries: RunEval | null
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
const COLUMNS: { key: string; label: string }[] = [
  { key: 'val_200',         label: 'val_200 · K=1' },
  { key: 'train_200',       label: 'train_200 · K=1' },
  { key: 'val_200-maskgit', label: 'val_200 · K=12' },
  { key: 'train_200-maskgit', label: 'train_200 · K=12' },
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

export function MEvalTable({ evalSeries, metric, setMetric }: Props): ReactElement | null {
  const { isDark } = useTheme()
  const steps = stepsDescOf(evalSeries)
  if (steps.length === 0) return null
  const lookup = pointsByStepSet(evalSeries)
  // Only render columns that have at least one point across all steps. Keeps
  // the table narrow on teacher-only runs (no maskgit-mode evals).
  const colsWithData = COLUMNS.filter((c) =>
    steps.some((s) => lookup.has(`${s}|${c.key}`)),
  )
  if (colsWithData.length === 0) return null

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
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.8rem', marginBottom: '0.3rem' }}>
        <h2 style={{ fontSize: '0.95rem', margin: 0 }}>
          Per-step m-eval ({steps.length} step{steps.length === 1 ? '' : 's'}) ·{' '}
          <span style={{ color: '#888', fontWeight: 'normal' }}>{metric.toUpperCase()}</span>
        </h2>
        <MetricChips metric={metric} setMetric={setMetric} isDark={isDark} />
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
              <td style={cellStyle}>{step.toLocaleString()}</td>
              {colsWithData.map((c) => {
                const pt = lookup.get(`${step}|${c.key}`)
                return (
                  <td key={c.key} style={cellStyle}>
                    {pt ? <SetCell pt={pt} isDark={isDark} metric={metric} /> : (
                      <span style={{ color: '#555' }}>–</span>
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
