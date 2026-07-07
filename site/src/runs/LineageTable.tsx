// LineageTable — one row per wandb run in this experiment's lineage.
//
// Renders below the `LineageToggle` on the run detail page; expandable so
// it doesn't take vertical space when collapsed. Each row carries the
// canonical job links (wandb · iris/modal) that were previously crowding
// the title bar, plus the run's step range and a link to its own detail
// page. The current run's row is highlighted.
//
// Data model (matches the "tomat-level lineage" notion the user wants):
// one row = one wandb run. The "rollup of jobs per wandb run" (one fire
// per iris/modal submission) is a Phase 2 extension once we wire the
// fires-as-records manifests into the FE.

import { Tooltip } from '../Tooltip'
import { isModalRun, type WandbRunRef } from './api'
import { WandbIcon } from './WandbIcon'

const WANDB_PROJECT = 'open-athena/tomat-lmq-P19'

export interface LineageRow {
  runId: string
  /** First step on the parquet for this run; null if history hasn't loaded. */
  startStep: number | null
  /** Last step on the parquet for this run; null if history hasn't loaded. */
  endStep: number | null
  /** True for the row corresponding to the page's current run. */
  current: boolean
  /** Optional kind marker — drives the row's relationship label. Defaults:
   *  `current` if `current === true`, else `ancestor`. Pass `'child'` for
   *  rows that are direct children of the current run. `'sibling'` marks a
   *  crashed sibling gap-filler surfaced by RunsPage's ancestor discovery
   *  when the physical resume ckpt came from a crashed run that shares this
   *  run's logical parent (spec 61 §2.2 — the crashed run's parquet holds
   *  the intermediate steps between logical-parent and current). */
  kind?: 'ancestor' | 'current' | 'child' | 'sibling'
  /** Spec 61 §2.2: all wandb runs that contributed to this logical run,
   *  in `created_at` ascending order. Optional — when omitted or empty,
   *  the row falls back to a single icon linked by the plain runId. */
  wandbRefs?: WandbRunRef[]
}

function fmtStep(s: number | null): string {
  if (s == null) return '?'
  if (s >= 1000) return `${(s / 1000).toFixed(s % 1000 === 0 ? 0 : 1)}k`
  return String(s)
}

function StepRange({ start, end }: { start: number | null; end: number | null }) {
  return (
    <span style={{ fontFamily: 'monospace', color: '#bbb' }}>
      {fmtStep(start)}–{fmtStep(end)}
    </span>
  )
}

function WandbLink({ runId, refs }: { runId: string; refs?: WandbRunRef[] }) {
  // Spec 61 §2.2: when a manifest carries `wandb_run_ids[]` with 2+
  // entries (one per fire), render one chip per fire, mirroring
  // `RunHeaderRich`'s multi-chip pattern. When 0 or 1 entry, fall back
  // to the single legacy chip so older/single-fire runs still render.
  if (refs && refs.length > 1) {
    return (
      <span style={{ display: 'inline-flex', gap: 4 }}>
        {refs.map((ref, i) => {
          const href = `https://wandb.ai/${ref.entity}/${ref.project}/runs/${encodeURIComponent(ref.run_id)}`
          return (
            <Tooltip key={`${ref.entity}/${ref.project}/${ref.run_id}`}
              content={`open fire #${i + 1} (${ref.name}) in wandb`}>
              <a href={href} target="_blank" rel="noreferrer"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 1, color: '#888' }}>
                <WandbIcon />
                <span style={{ fontSize: '0.65rem' }}>↗{i + 1}</span>
              </a>
            </Tooltip>
          )
        })}
      </span>
    )
  }
  const ref = refs?.[0]
  const url = ref
    ? `https://wandb.ai/${ref.entity}/${ref.project}/runs/${encodeURIComponent(ref.run_id)}`
    : `https://wandb.ai/${WANDB_PROJECT}/runs/${runId}`
  return (
    <Tooltip content={`open ${runId} in wandb`}>
      <a href={url} target="_blank" rel="noreferrer"
        style={{ display: 'inline-flex', alignItems: 'center', gap: 2, color: '#888' }}>
        <WandbIcon />
        <span style={{ fontSize: '0.7rem' }}>↗</span>
      </a>
    </Tooltip>
  )
}

function IrisLink({ runId }: { runId: string }) {
  const irisJobId = `/ryan/${runId}`
  const url = `https://iris.oa.dev/#/job/${encodeURIComponent(irisJobId)}`
  return (
    <Tooltip content={`open iris job (${irisJobId})`}>
      <a href={url} target="_blank" rel="noreferrer"
        style={{ display: 'inline-flex', alignItems: 'center', gap: 2, color: '#888' }}>
        <img src="/iris.png" alt="iris"
          style={{ height: 18, width: 'auto', verticalAlign: 'middle' }} />
        <span style={{ fontSize: '0.7rem' }}>↗</span>
      </a>
    </Tooltip>
  )
}

function ModalLink({ runId }: { runId: string }) {
  // For older Modal runs we don't have app_id / fc_id at this layer (those
  // come from the live modalApp / pending-fires lookups on the current
  // run's row in the title bar). At the lineage-row level we link to the
  // canonical app overview by name — the user can drill in once landed.
  const url = `https://modal.com/apps/open-athena/main/deployed/tomat-train-smoke`
  return (
    <Tooltip content={`open Modal app (tomat-train-smoke)`}>
      <a href={url} target="_blank" rel="noreferrer"
        style={{ display: 'inline-flex', alignItems: 'center', gap: 2, color: '#888' }}>
        <img src="/modal.png" alt="modal"
          style={{ height: 18, width: 'auto', verticalAlign: 'middle' }} />
        <span style={{ fontSize: '0.7rem' }}>↗</span>
      </a>
    </Tooltip>
  )
}

export function LineageTable({
  rows, ancestorsLoading = false, snapshotLoading = false,
}: {
  rows: LineageRow[]
  /** Any ancestor's history parquet is still loading (step range shows `?`
   *  until it lands). Shown as a subtle caption in the table header. */
  ancestorsLoading?: boolean
  /** Snapshot itself is still loading, so ancestors haven't been registered
   *  yet — the table might be showing only the current row when it should
   *  show more. Visible caption so the user knows the row list isn't final. */
  snapshotLoading?: boolean
}) {
  const loadingText = snapshotLoading
    ? 'discovering ancestors…'
    : ancestorsLoading
      ? 'loading ancestor history…'
      : null
  return (
    <div style={{
      margin: '0.25rem 0 0.5rem 0',
      border: '1px solid #2a2a2a',
      borderRadius: 6,
      overflow: 'hidden',
      fontSize: '0.8rem',
    }}>
      {loadingText && (
        <div style={{
          padding: '3px 8px',
          background: '#1a1f2b',
          color: '#8fa4c8',
          fontSize: '0.7rem',
          borderBottom: '1px solid #232323',
          fontStyle: 'italic',
        }}>
          <span style={{ opacity: 0.8 }}>◐</span>{' '}
          {loadingText}
        </div>
      )}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: '#1d1d1d', color: '#888', fontSize: '0.7rem' }}>
            <th style={{ textAlign: 'left',  padding: '4px 8px', fontWeight: 500 }}>run</th>
            <th style={{ textAlign: 'left',  padding: '4px 8px', fontWeight: 500 }}>step range</th>
            <th style={{ textAlign: 'center', padding: '4px 8px', fontWeight: 500, width: 60 }}>wandb</th>
            <th style={{ textAlign: 'center', padding: '4px 8px', fontWeight: 500, width: 80 }}>iris/modal</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const isModal = isModalRun(row.runId)
            const isLast = i === rows.length - 1
            return (
              <tr key={row.runId}
                  style={{
                    background: row.current ? 'rgba(80, 130, 220, 0.10)' : undefined,
                    borderTop: i > 0 ? '1px solid #232323' : undefined,
                  }}>
                <td style={{ padding: '4px 8px', borderBottom: isLast ? undefined : '0' }}>
                  <a href={`#/runs/${row.runId}`}
                    style={{
                      fontFamily: 'monospace',
                      color: row.current ? '#cfe1ff' : '#9aa6c2',
                      textDecoration: 'none',
                    }}>
                    {row.runId}
                  </a>
                  {row.current && (
                    <span style={{ marginLeft: 6, fontSize: '0.7rem', color: '#7aa3ff' }}>
                      ◀ current
                    </span>
                  )}
                  {row.kind === 'child' && (
                    <span style={{ marginLeft: 6, fontSize: '0.7rem', color: '#7aa37a' }}>
                      ↳ child
                    </span>
                  )}
                  {row.kind === 'sibling' && (
                    <span style={{ marginLeft: 6, fontSize: '0.7rem', color: '#d4a374' }}
                      title="Crashed sibling fire — its parquet fills the gap between the logical parent and this run.">
                      ⚠ crashed sibling
                    </span>
                  )}
                </td>
                <td style={{ padding: '4px 8px' }}>
                  <StepRange start={row.startStep} end={row.endStep} />
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                  <WandbLink runId={row.runId} refs={row.wandbRefs} />
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                  {isModal ? <ModalLink runId={row.runId} /> : <IrisLink runId={row.runId} />}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
