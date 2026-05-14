import { useEffect, useState } from 'react'
import { fetchIrisState, fetchManifest, fetchRuns, irisJobIdForRun, parquetUrl } from './api'
import type { IrisJob, IrisState, RunManifest } from './api'
import { fetchRunHistory } from './parquet'
import type { RunHistory } from './parquet'
import { WallclockPlot } from './WallclockPlot'

interface Props {
  parts: string[]
}

const navigate = (path: string) => {
  window.location.hash = `#/${path}`
}

// Let cmd/ctrl/shift/middle-click fall through to the browser so it can open
// the link in a new tab / window.
const isModifiedClick = (e: React.MouseEvent) =>
  e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0

export function RunsPage({ parts }: Props) {
  const runId = parts[0]
  return runId ? <RunDetail runId={runId} /> : <RunsIndex />
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

function IrisBadge({ job }: { job: IrisJob }) {
  const style = IRIS_STATE_STYLES[job.state] ?? { bg: '#888', fg: '#fff' }
  const tail = job.preempts > 0 || job.failures > 0
    ? ` (p=${job.preempts}, f=${job.failures})` : ''
  return (
    <span
      title={job.error || `iris state=${job.state} preempts=${job.preempts} failures=${job.failures}`}
      style={{
        backgroundColor: style.bg, color: style.fg,
        padding: '1px 6px', borderRadius: 3,
        fontSize: '0.75rem', fontFamily: 'monospace',
      }}
    >
      {job.state}{tail}
    </span>
  )
}

function RunsIndex() {
  const [runs, setRuns] = useState<string[] | null>(null)
  const [iris, setIris] = useState<IrisState | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetchRuns()
      .then((r) => setRuns(r.runs))
      .catch((e) => setErr(String(e)))
    fetchIrisState()
      .then(setIris)
      .catch(() => {}) // soft-fail; iris-state is optional info
  }, [])

  return (
    <div style={{ maxWidth: 960, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>tomat runs</h1>
      <p style={{ color: '#666' }}>
        Synced from wandb (PrinceOA/tomat-lmq-P19) → R2 via{' '}
        <code>tomat runs sync</code>. iris state via{' '}
        <code>tomat iris sync</code>. See{' '}
        <a href="https://github.com/Open-Athena/tomat/blob/main/specs/23-runs-dashboard.md">
          spec
        </a>
        .
        {iris && <span style={{ marginLeft: 8 }}>
          iris snapshot: {new Date(iris.synced_at).toLocaleString()}
        </span>}
      </p>
      {err && <p style={{ color: 'crimson' }}>error: {err}</p>}
      {!runs && !err && <p>loading…</p>}
      {runs && runs.length === 0 && <p>(none synced yet)</p>}
      {runs && runs.length > 0 && (
        <ul style={{ listStyle: 'none', padding: 0 }}>
          {runs.map((id) => {
            const job = iris?.jobs[irisJobIdForRun(id)]
            return (
              <li key={id} style={{ marginBottom: '0.3rem' }}>
                {job && <IrisBadge job={job} />}{' '}
                <a
                  href={`#/runs/${id}`}
                  onClick={(e) => {
                    if (isModifiedClick(e)) return
                    e.preventDefault()
                    navigate(`runs/${id}`)
                  }}
                >
                  {id}
                </a>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function RunDetail({ runId }: { runId: string }) {
  const [manifest, setManifest] = useState<RunManifest | null>(null)
  const [history, setHistory] = useState<RunHistory | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    setManifest(null)
    setHistory(null)
    setErr(null)
    fetchManifest(runId)
      .then(setManifest)
      .catch((e) => setErr(String(e)))
    fetchRunHistory(parquetUrl(runId))
      .then(setHistory)
      .catch((e) => setErr(String(e)))
  }, [runId])

  return (
    <div style={{ maxWidth: 1200, margin: '2rem auto', padding: '0 1rem' }}>
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
        <div style={{ color: '#666', fontSize: '0.9rem', marginBottom: '1rem' }}>
          state: {manifest.run.state} ·{' '}
          history: {manifest.history.rows} rows, steps [
          {manifest.history.step_min ?? '-'}, {manifest.history.step_max ?? '-'}] ·{' '}
          synced: {manifest.synced_at}
          {' · '}
          <a href={manifest.run.url} target="_blank" rel="noreferrer">wandb ↗</a>
        </div>
      )}
      {!history && !err && <p>loading parquet…</p>}
      {history && <WallclockPlot history={history} runId={runId} />}
    </div>
  )
}
