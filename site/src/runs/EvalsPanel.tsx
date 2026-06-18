// Per spec 43 §"Read paths" — eval matrix on the run-detail page.
//
// Rows = step (desc). Columns = `(set, mode)` tuple, only those present in
// the records (we don't pre-allocate columns for unfired combos). Cell content
// = state badge + result number (when succeeded). Click a cell → opens the
// GCS result-JSON link in a new tab (Phase A; richer Modal-call / iris-log
// drill-down lands in Phase C).
//
// Data source: the records-index from spec 43 §"Where the record lives". The
// dashboard pulls the index via `runs[i].evals` (inlined in the
// `/api/runs-snapshot.json` aggregate) and re-fetches the full per-record
// JSONs on demand for cells the user wants to inspect. For Phase A we only
// need the index (state + result-summary scalars travel with each index
// entry once we fold them in; for now the index just has key + state +
// fired_at, so we fetch each record for its `result` scalar).

import { useQueries } from '@tanstack/react-query'
import { useMemo } from 'react'
import { fetchEvalRecord } from './api'
import type { EvalIndexEntry, EvalRecord, EvalRecordState } from './api'
import { formatStepDetail } from '../lib/runNames'
import { Tooltip } from '../Tooltip'

const STATE_COLOR: Record<EvalRecordState, string> = {
  pending:   '#facc15', // yellow
  running:   '#60a5fa', // blue
  succeeded: '#22c55e', // green
  failed:    '#ef4444', // red
  killed:    '#9ca3af', // grey
}

const STATE_SYMBOL: Record<EvalRecordState, string> = {
  pending:   '⏳',
  running:   '⏳',
  succeeded: '✅',
  failed:    '✗',
  killed:    '⊘',
}

/** Parse a record key into its components. Keys are
 *  `<step>-<set>-<mode>` optionally with `-task<i>`. The set names can be
 *  `val_200`, `train_200`, etc.; the mode is one of `teacher` | `free` |
 *  `maskgit`. Returns null on a malformed key. */
function parseKey(key: string): { step: number; set: string; mode: string; task?: number } | null {
  // Greedy strip the trailing `-task<i>` if present.
  let task: number | undefined
  const taskMatch = key.match(/^(.+)-task(\d+)$/)
  let core = key
  if (taskMatch) {
    core = taskMatch[1]
    task = Number(taskMatch[2])
  }
  // Step is the first numeric prefix.
  const m = core.match(/^(\d+)-(.+)$/)
  if (!m) return null
  const step = Number(m[1])
  const rest = m[2]
  // Mode is the LAST `-`-separated token (so set names with underscores stay
  // intact — `val_200-maskgit` → set='val_200', mode='maskgit').
  const idx = rest.lastIndexOf('-')
  if (idx <= 0) return null
  const set = rest.slice(0, idx)
  const mode = rest.slice(idx + 1)
  return { step, set, mode, task }
}

/** A column header in the matrix. */
interface Column {
  set: string
  mode: string
  /** `set-mode` for stable React keys + sort. */
  id: string
}

interface CellEntries {
  /** All records that share (step, set, mode), keyed by task or undefined. */
  entries: EvalIndexEntry[]
}

function pct(v: number | null | undefined): string | null {
  if (typeof v !== 'number') return null
  // `eval_mat_nmae.py` always writes `nmae_mean = mean(mae)/mean(|rho|)` as
  // a fraction (NOT a percentage), regardless of mode. So 0.0019 means
  // 0.19% on a healthy teacher-mode eval; 3.86 means 386% on a collapsed
  // free-running eval (cf. memory `free-running-divergence.md`). Always
  // ×100 — don't try to disambiguate "fraction vs already-percent".
  return `${(v * 100).toFixed(2)}%`
}

/** Pick the "primary" result scalar for a `(set, mode)`'s display.
 *  Teacher mode → nmae_mean. Free → nmae_mean. MaskGIT → nmae_mean. We
 *  show median as a sub-text. */
function resultBlurb(rec: EvalRecord | null | undefined): { primary: string | null; secondary: string | null } {
  if (!rec?.result) return { primary: null, secondary: null }
  const primary = pct(rec.result.nmae_mean)
  const median = pct(rec.result.nmae_median)
  return {
    primary,
    secondary: median && median !== primary ? `med ${median}` : null,
  }
}

export function EvalsPanel({ runId, evalsIndex }: {
  runId: string
  evalsIndex: EvalIndexEntry[] | null
}) {
  const idx = evalsIndex ?? []

  // Group entries by (step, set, mode). Per-task records pool here; we still
  // surface task-count + per-task states on hover/click later.
  const { steps, columns, byCell, parsedByKey } = useMemo(() => {
    const stepsSet = new Set<number>()
    const colsMap = new Map<string, Column>()
    const cells = new Map<string, CellEntries>() // `${step}|${set}|${mode}`
    const parsed = new Map<string, ReturnType<typeof parseKey>>()
    for (const e of idx) {
      const p = parseKey(e.key)
      parsed.set(e.key, p)
      if (!p) continue
      stepsSet.add(p.step)
      const colId = `${p.set}-${p.mode}`
      if (!colsMap.has(colId)) colsMap.set(colId, { set: p.set, mode: p.mode, id: colId })
      const cellId = `${p.step}|${colId}`
      const cell = cells.get(cellId) ?? { entries: [] }
      cell.entries.push(e)
      cells.set(cellId, cell)
    }
    const steps = [...stepsSet].sort((a, b) => b - a)
    const columns = [...colsMap.values()].sort(
      (a, b) => a.set.localeCompare(b.set) || a.mode.localeCompare(b.mode),
    )
    return { steps, columns, byCell: cells, parsedByKey: parsed }
  }, [idx])

  // Fetch the full record JSON for each succeeded entry, so we can show the
  // NMAE number. We only fetch succeeded ones to avoid wasted GETs on
  // pending/failed (those don't have a `result` field set anyway). With
  // typical run counts of <100 cells, this is well under the browser's
  // parallel-fetch budget; if we ever overflow it, fold the result scalar
  // into the index.json so a single fetch carries everything.
  const recordQueries = useQueries({
    queries: idx
      .filter((e) => e.state === 'succeeded')
      .map((e) => ({
        queryKey: ['eval-record', runId, e.key],
        queryFn: () => fetchEvalRecord(runId, e.key),
        staleTime: 5 * 60_000,  // records are ≈immutable once succeeded
      })),
  })
  const recordByKey = useMemo(() => {
    const m = new Map<string, EvalRecord | null>()
    const succeeded = idx.filter((e) => e.state === 'succeeded')
    for (let i = 0; i < succeeded.length; i++) {
      m.set(succeeded[i].key, recordQueries[i]?.data ?? null)
    }
    return m
  }, [idx, recordQueries])

  if (idx.length === 0) {
    return null
  }

  const cellStyle: React.CSSProperties = {
    padding: '4px 12px 4px 0',
    textAlign: 'left',
    verticalAlign: 'top',
  }
  const headerStyle: React.CSSProperties = {
    ...cellStyle,
    color: '#888',
    fontWeight: 'normal',
    borderBottom: '1px solid #2a2a2a',
  }

  return (
    <div style={{ marginTop: '1.4rem', marginBottom: '1.4rem' }}>
      <h2 style={{ fontSize: '1rem', marginBottom: '0.4rem' }}>
        Evals ({idx.length} record{idx.length === 1 ? '' : 's'}, {steps.length} step{steps.length === 1 ? '' : 's'})
      </h2>
      <table style={{
        borderCollapse: 'collapse',
        fontSize: '0.8rem',
        fontFamily: 'monospace',
      }}>
        <thead>
          <tr>
            <th style={headerStyle}>step</th>
            {columns.map((c) => (
              <th key={c.id} style={headerStyle}>
                {c.set} <span style={{ color: '#666' }}>· {c.mode}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {steps.map((step) => (
            <tr key={step}>
              <td style={cellStyle}>
                {(() => {
                  // `*` flags legacy info.step-OBO interpretation (pre-fix
                  // Levanter; see `formatStepDetail` for the full semantics).
                  // The tooltip narrates raw vs completed steps.
                  const d = formatStepDetail(step)
                  return d.isLegacy ? (
                    <Tooltip content={d.tooltip}>
                      <span>{d.display}</span>
                    </Tooltip>
                  ) : d.display
                })()}
              </td>
              {columns.map((c) => {
                const cell = byCell.get(`${step}|${c.id}`)
                if (!cell) {
                  return <td key={c.id} style={{ ...cellStyle, color: '#555' }}>–</td>
                }
                // If multiple tasks, render each as a sub-row.
                const states = cell.entries.map((e) => e.state)
                // "Aggregate" state for the cell header: prefer succeeded if
                // any landed; else failed if any failed; else running; else pending.
                const aggState: EvalRecordState =
                  states.includes('succeeded') ? 'succeeded'
                  : states.includes('failed') ? 'failed'
                  : states.includes('running') ? 'running'
                  : states.includes('killed') ? 'killed'
                  : 'pending'
                // Single record (no fanout) — show inline. Multi-task: list
                // each task on its own line.
                const isMultiTask = cell.entries.length > 1 || cell.entries.some(
                  (e) => parsedByKey.get(e.key)?.task !== undefined,
                )
                return (
                  <td key={c.id} style={cellStyle}>
                    {!isMultiTask ? (
                      renderEntry(cell.entries[0], recordByKey.get(cell.entries[0].key))
                    ) : (
                      <div>
                        <span style={{ color: STATE_COLOR[aggState] }}>
                          {STATE_SYMBOL[aggState]} {aggState}
                        </span>
                        <div style={{ marginTop: 2, color: '#888' }}>
                          {cell.entries
                            .slice()
                            .sort((a, b) => {
                              const ta = parsedByKey.get(a.key)?.task ?? -1
                              const tb = parsedByKey.get(b.key)?.task ?? -1
                              return ta - tb
                            })
                            .map((e) => {
                              const taskIdx = parsedByKey.get(e.key)?.task
                              return (
                                <div key={e.key} style={{ marginLeft: 8 }}>
                                  <span style={{ color: '#666' }}>
                                    {taskIdx !== undefined ? `task${taskIdx}` : 'whole'}
                                  </span>{' '}
                                  {renderEntry(e, recordByKey.get(e.key))}
                                </div>
                              )
                            })}
                        </div>
                      </div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: '#555', fontSize: '0.7rem', marginTop: '0.4rem' }}>
        Click a cell to open its result JSON in GCS.
      </p>
    </div>
  )
}

function renderEntry(entry: EvalIndexEntry, record: EvalRecord | null | undefined) {
  const blurb = resultBlurb(record)
  const url = record?.gcs_result_path
    ? gcsUrl(record.gcs_result_path)
    : null
  const stateBadge = (
    <span style={{ color: STATE_COLOR[entry.state] }}>
      {STATE_SYMBOL[entry.state]} {entry.state}
    </span>
  )
  return (
    <span>
      {url ? (
        <a href={url} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
          {stateBadge}
        </a>
      ) : (
        stateBadge
      )}
      {blurb.primary && (
        <span style={{ marginLeft: 6 }}>{blurb.primary}</span>
      )}
      {blurb.secondary && (
        <span style={{ marginLeft: 4, color: '#777' }}>({blurb.secondary})</span>
      )}
    </span>
  )
}

/** `gs://bucket/path` → the GCS console URL for browsing the object. */
function gcsUrl(gcsPath: string): string {
  const m = gcsPath.match(/^gs:\/\/([^/]+)\/(.+)$/)
  if (!m) return gcsPath
  const [, bucket, path] = m
  return `https://storage.cloud.google.com/${bucket}/${path}`
}
