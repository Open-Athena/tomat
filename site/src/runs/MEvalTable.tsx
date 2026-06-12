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

import type { ReactElement } from 'react'
import { useTheme } from 'pltly/react'
import type { EvalPoint, RunEval } from './api'
import {
  nemdBucket, nmaeBucket, pointsByStepSet, stepsDescOf, type Bucket,
} from './MEvalTable.helpers'

// Re-export for downstream tests / consumers that want the bucket types.
export { nemdBucket, nmaeBucket, stepsDescOf } from './MEvalTable.helpers'
export type { Bucket } from './MEvalTable.helpers'

interface Props {
  evalSeries: RunEval | null
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
}

function SetCell({ pt, isDark }: SetCellProps): ReactElement {
  const COLORS = isDark ? BUCKET_COLORS_DARK : BUCKET_COLORS_LIGHT
  const nmaePct = typeof pt.nmae_mean === 'number' ? pt.nmae_mean * 100 : NaN
  const nemdPct = typeof pt.nemd_mean === 'number' ? pt.nemd_mean * 100 : NaN
  const nmaeColor = COLORS[nmaeBucket(nmaePct)]
  const nemdColor = COLORS[nemdBucket(nemdPct)]
  return (
    <span>
      <span style={{ color: nmaeColor }}>{fmtPct(pt.nmae_mean)}</span>
      <span style={{ color: '#666' }}> / </span>
      <span style={{ color: nemdColor }}>{fmtPct(pt.nemd_mean)}</span>
      {pt.n_mats != null && (
        <span style={{ color: '#777' }}> · n={pt.n_mats}</span>
      )}
    </span>
  )
}

export function MEvalTable({ evalSeries }: Props): ReactElement | null {
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
      <h2 style={{ fontSize: '0.95rem', marginBottom: '0.3rem' }}>
        Per-step m-eval ({steps.length} step{steps.length === 1 ? '' : 's'})
      </h2>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem' }}>
        <thead>
          <tr>
            <th style={headerStyle}>step</th>
            {colsWithData.map((c) => (
              <th key={c.key} style={headerStyle}>
                {c.label} <span style={{ color: '#555' }}>NMAE / NEMD</span>
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
                    {pt ? <SetCell pt={pt} isDark={isDark} /> : (
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
