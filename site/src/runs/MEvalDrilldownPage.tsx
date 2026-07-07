// MEvalDrilldownPage — spec 60 Phase A (minimal).
//
// Route: `#/run/<runId>/m-eval/<step>/<setKey>`
//
// Header: run name + step + setKey + summary stats pulled from the
// existing `eval.json`'s `sets[setKey]` entry for this step
// (nmae_mean, nmae_median, p1/p25/p75/p99, n_mats).
//
// Below: list of per-mat prediction zarrs found on R2 for this
// (run, step, setKey), each linking out to ELVis for GT-vs-pred diff.
// Spec 60 Phase A calls for a full per_mat[] table sorted by NMAE —
// but `per_mat[]` isn't currently mirrored into the R2 eval.json, so
// this first pass surfaces only what's readily available:
//   - Summary stats (from eval.json — the same numbers MEvalTable renders)
//   - Available pred zarrs (R2 `/api/files/list` → mp_id list)
//
// Follow-up (Phase B): thread `per_mat[]` through `tomat evals sync`
// into the R2 eval.json under a new per_mat_by_set key, then join with
// the zarr list to render one row per mat with NMAE + ELVis diff link.

import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { API_BASE, fetchEval, type EvalPoint } from './api'

interface FilesEntry { key: string; isDir: boolean }
interface FilesListResp { entries: FilesEntry[] }

async function listPredZarrs(
  runId: string, setKey: string, step: number,
): Promise<string[]> {
  const prefix = `tomat/eval/${runId}/predictions/${setKey}/step-${step}/`
  const url = `${API_BASE}/api/files/list?prefix=${encodeURIComponent(prefix)}`
  const r = await fetch(url)
  if (!r.ok) return []
  const j = (await r.json()) as FilesListResp
  const mp_ids: string[] = []
  for (const e of j.entries) {
    // Trim leading prefix and trailing `.zarr/`; strip any `--output-suffix`
    // like `-V` off the end of the mp_id (spec 51 mirror convention).
    if (!e.isDir) continue
    const tail = e.key.slice(prefix.length).replace(/\.zarr\/?$/, '')
    if (!tail || !tail.startsWith('mp-')) continue
    const bare = tail.replace(/-[A-Za-z]\w*$/, '')  // drop `-V` etc.
    mp_ids.push(bare)
  }
  return Array.from(new Set(mp_ids)).sort()
}

function r2RawUrl(r2_key: string): string {
  return `${API_BASE}/api/files/raw/${r2_key}`
}

const ELVIS_BASE = 'https://elvis.oa.dev'

function elvisDiffUrl(mp_id: string, gt_key: string, pred_key: string): string {
  const v0 = encodeURIComponent(r2RawUrl(gt_key))
  const v1 = encodeURIComponent(r2RawUrl(pred_key))
  return `${ELVIS_BASE}/?m=${encodeURIComponent(mp_id)}&s=d&v0=${v0}&v1=${v1}`
}

function elvisSingleUrl(mp_id: string, r2_key: string): string {
  const v1 = encodeURIComponent(r2RawUrl(r2_key))
  return `${ELVIS_BASE}/?m=${encodeURIComponent(mp_id)}&v1=${v1}`
}

/** From `val_200-…` or `train_200-…` → `validation` / `training` split
 *  (matches the R2 layout under `tomat/rho_gga_v3mr/<split>/<mp>.zarr`). */
function splitFromSetKey(setKey: string): 'validation' | 'training' {
  return setKey.startsWith('val') ? 'validation' : 'training'
}

function pct(x: number | null | undefined): string {
  if (x == null) return '–'
  return `${(x * 100).toFixed(2)}%`
}

export function MEvalDrilldownPage({
  runId, step, setKey,
}: {
  runId: string
  step: number
  setKey: string
}) {
  const evalQ = useQuery({
    queryKey: ['eval', runId],
    queryFn: () => fetchEval(runId),
    retry: 1,
  })
  const zarrsQ = useQuery({
    queryKey: ['pred-zarrs', runId, setKey, step],
    queryFn: () => listPredZarrs(runId, setKey, step),
    retry: 1,
  })

  const pt: EvalPoint | null = useMemo(() => {
    const points = evalQ.data?.sets?.[setKey] ?? []
    return points.find((p) => p.step === step) ?? null
  }, [evalQ.data, setKey, step])

  const split = splitFromSetKey(setKey)
  const gtKeyFor = (mp: string): string =>
    `tomat/rho_gga_v3mr/${split}/${mp}.zarr`
  const predKeyFor = (mp: string): string =>
    `tomat/eval/${runId}/predictions/${setKey}/step-${step}/${mp}.zarr`

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '1rem' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <a href={`#/runs/${encodeURIComponent(runId)}`}
          style={{ color: '#7aa3ff', textDecoration: 'none' }}>
          ← back to {runId}
        </a>
      </div>
      <h1 style={{ fontSize: '1.1rem', fontFamily: 'monospace', margin: 0 }}>
        {runId}
      </h1>
      <div style={{ color: '#9aa6c2', fontSize: '0.9rem', marginTop: 3 }}>
        step <b>{step.toLocaleString()}</b> · <code>{setKey}</code>
      </div>

      {evalQ.isLoading && <p style={{ color: '#888' }}>loading eval.json…</p>}
      {evalQ.error && (
        <p style={{ color: '#f49a92' }}>
          eval.json fetch failed: {String(evalQ.error)}
        </p>
      )}

      {pt && (
        <div style={{ marginTop: '0.8rem', padding: '0.6rem 0.8rem',
                      border: '1px solid #2a2a2a', borderRadius: 6 }}>
          <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: 4 }}>
            Summary · n_mats <b>{pt.n_mats ?? '?'}</b>
          </div>
          <div style={{ display: 'grid',
                        gridTemplateColumns: 'repeat(6, 1fr)',
                        fontFamily: 'monospace',
                        fontSize: '0.85rem', gap: '0.4rem 1rem' }}>
            <StatCol name="p1"     value={pct((pt as unknown as Record<string, number | null>).nmae_p1)} />
            <StatCol name="p25"    value={pct((pt as unknown as Record<string, number | null>).nmae_p25)} />
            <StatCol name="median" value={pct((pt as unknown as Record<string, number | null>).nmae_median)} />
            <StatCol name="mean"   value={pct(pt.nmae_mean)} bold />
            <StatCol name="p75"    value={pct((pt as unknown as Record<string, number | null>).nmae_p75)} />
            <StatCol name="p99"    value={pct((pt as unknown as Record<string, number | null>).nmae_p99)} />
          </div>
        </div>
      )}
      {evalQ.data && !pt && (
        <p style={{ color: '#d4a374', marginTop: '0.8rem' }}>
          no summary point for {setKey} @ step {step} — eval.json has
          other steps but not this one. Was the JSON synced?
        </p>
      )}

      <h2 style={{ fontSize: '0.95rem', marginTop: '1.5rem', marginBottom: '0.4rem' }}>
        Per-mat predictions{' '}
        <span style={{ color: '#888', fontWeight: 'normal', fontSize: '0.8rem' }}>
          (R2 pred zarrs · ELVis diff)
        </span>
      </h2>
      {zarrsQ.isLoading && <p style={{ color: '#888' }}>listing pred zarrs on R2…</p>}
      {zarrsQ.data && zarrsQ.data.length === 0 && (
        <p style={{ color: '#888', fontSize: '0.85rem' }}>
          No prediction zarrs mirrored to R2 for this (run, step, set).
          Fire with{' '}
          <code style={{ background: '#1a1f2b', padding: '1px 4px' }}>
            evals fire {runId} -s {step} -S {setKey.split('-')[0]} -V --output-suffix V
          </code>{' '}
          to produce them, then <code>tomat evals mirror-mat</code>.
        </p>
      )}
      {zarrsQ.data && zarrsQ.data.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse',
                        fontFamily: 'monospace', fontSize: '0.82rem' }}>
          <thead>
            <tr style={{ color: '#888', borderBottom: '1px solid #2a2a2a' }}>
              <th style={{ textAlign: 'left', padding: '3px 6px' }}>#</th>
              <th style={{ textAlign: 'left', padding: '3px 6px' }}>mp_id</th>
              <th style={{ textAlign: 'left', padding: '3px 6px' }}>ELVis</th>
            </tr>
          </thead>
          <tbody>
            {zarrsQ.data.map((mp, i) => (
              <tr key={mp} style={{ borderBottom: '1px solid #1a1a1a' }}>
                <td style={{ padding: '3px 6px', color: '#666' }}>{i + 1}</td>
                <td style={{ padding: '3px 6px' }}>
                  <a href={`#/mp/${mp}`}
                    style={{ color: '#cfe1ff', textDecoration: 'none' }}>
                    {mp}
                  </a>
                </td>
                <td style={{ padding: '3px 6px' }}>
                  <a href={elvisDiffUrl(mp, gtKeyFor(mp), predKeyFor(mp))}
                    target="_blank" rel="noreferrer"
                    style={{ color: '#7aa3ff', marginRight: 10 }}>
                    diff
                  </a>
                  <a href={elvisSingleUrl(mp, predKeyFor(mp))}
                    target="_blank" rel="noreferrer"
                    style={{ color: '#7aa3ff' }}>
                    pred
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p style={{ color: '#666', fontSize: '0.75rem', marginTop: '1.5rem' }}>
        Phase A: showing summary stats + available pred zarrs only. Full
        per-mat NMAE table needs `per_mat[]` mirrored into eval.json —
        see spec 60 §"Phase A — read-only render".
      </p>
    </div>
  )
}

function StatCol({ name, value, bold }: {
  name: string; value: string; bold?: boolean
}) {
  return (
    <div>
      <div style={{ color: '#888', fontSize: '0.7rem' }}>{name}</div>
      <div style={{ fontWeight: bold ? 600 : 400 }}>{value}</div>
    </div>
  )
}
