// Per-material grid index page (spec 58 Phase A).
//
// `#/mp/<mp_id>` renders a header (formula + atoms + lattice from the zarr
// `elvis` attrs) plus a table of every density grid we have for `<mp_id>`:
// one row per GT split (validation/train) + one row per (run, setmode, step)
// prediction checkpoint.
//
// Phase A scope:
// - List-only. No 2-way diff view, no patch overlay, no D1 — the CFW endpoint
//   stubs the index by walking R2 prefixes (`/api/mp/<mp_id>/grids`).
// - Each row's "Open in ELVis" link drops the user into the single-grid view
//   on `elvis.oa.dev` (`?m=<mp>&v1=<r2-url>`); Phase B will swap that for a
//   diff against a row chosen via column-A/B radio toggles.
// - NMAE column shows '—' for now; #289 follow-up wires it from eval.json.

import { useQuery } from '@tanstack/react-query'
import { API_BASE } from '../runs/api'
import { shortenRun, formatStep } from '../lib/runNames'

import { fetchGrids, type GridRow } from './api'

/** Subset of the zarr `elvis` attrs block we render in the header. The full
 *  schema also carries `stats`, `quantiles`, `shape`, etc.; we only pull the
 *  bits the table header needs (formula + lattice + atom list). */
interface ElvisAttrs {
  material_id: string
  role?: string
  lattice?: number[][]
  atoms?: { element: string; frac: number[] }[]
  stats?: { min: number; max: number; mean: number }
}

interface ZarrJson {
  attributes?: {
    elvis?: ElvisAttrs
    multiscales?: unknown[]
  }
}

const ELVIS_BASE = 'https://elvis.oa.dev'

/** Build the `/api/files/raw/<key>` URL that the worker exposes for in-browser
 *  zarr stores to fetch (FetchStore-friendly: zarrita appends `0/zarr.json`,
 *  `0/c/0/0/0` natively). The ELVis viewer accepts the same kind of URL. */
function r2RawUrl(r2_key: string): string {
  return `${API_BASE}/api/files/raw/${r2_key}`
}

/** Open-in-ELVis URL. When `gt_key` is passed, builds the 2-way diff URL
 *  (`?s=d&v0=<gt>&v1=<pred>`) so the viewer renders a side-by-side / overlay
 *  of GT vs. this row's prediction. Without `gt_key` (e.g. on the GT row
 *  itself), falls back to the single-grid render. */
function elvisUrlFor(mp_id: string, r2_key: string, gt_key?: string | null): string {
  const v1 = encodeURIComponent(r2RawUrl(r2_key))
  if (gt_key && gt_key !== r2_key) {
    const v0 = encodeURIComponent(r2RawUrl(gt_key))
    return `${ELVIS_BASE}/?m=${encodeURIComponent(mp_id)}&s=d&v0=${v0}&v1=${v1}`
  }
  return `${ELVIS_BASE}/?m=${encodeURIComponent(mp_id)}&v1=${v1}`
}

/** Fetch `<r2_key>/zarr.json` and pluck the `elvis` attrs block. The header
 *  is rendered from whichever row's attrs we happen to land on first — they
 *  agree on lattice + atoms across all rows for the same mat. */
async function fetchElvisAttrs(r2_key: string): Promise<ElvisAttrs | null> {
  const r = await fetch(`${API_BASE}/api/files/get?path=${encodeURIComponent(r2_key + '/zarr.json')}`)
  if (!r.ok) return null
  const json = (await r.json()) as ZarrJson
  return json.attributes?.elvis ?? null
}

/** Empirical formula from the atoms list: count elements, sort alphabetically,
 *  drop `1` subscripts. `[Fe×6, W×2] → 'Fe6W2'`. Materials Project uses the
 *  same convention for its `formula_pretty` field. */
function formulaFromAtoms(atoms: { element: string }[]): string {
  const counts = new Map<string, number>()
  for (const a of atoms) counts.set(a.element, (counts.get(a.element) ?? 0) + 1)
  const parts = [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  return parts.map(([el, n]) => (n === 1 ? el : `${el}${n}`)).join('')
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function formatWritten(iso: string): string {
  // YYYY-MM-DD HH:MM (UTC) — compact, sortable in the eye, no TZ surprise.
  return iso.slice(0, 16).replace('T', ' ')
}

/** Short label for the row's `set` column: `val_200-maskgit → vm·mg`,
 *  `train_200-maskgit → tm·mg`. For GT rows it's the split name as-is
 *  (`validation`, `train`). Falls back to the raw string for anything we
 *  don't recognize. */
function shortenSet(set: string | null): string {
  if (set == null) return '—'
  const m = set.match(/^(val|train)_\d+-(maskgit|oneshot|free)$/)
  if (m) {
    const splitShort = m[1] === 'val' ? 'v' : 't'
    const modeShort = m[2] === 'maskgit' ? 'mg' : m[2] === 'oneshot' ? 'os' : 'fr'
    return `${splitShort}·${modeShort}`
  }
  return set
}

/** Short label for the run column: `shortenRun` from the registry, or '—'
 *  for GT rows (no run). */
function shortenRunCell(run_id: string | null): string {
  if (run_id == null) return '—'
  return shortenRun(run_id)
}

/** Short label for the step column. `formatStep` handles the legacy OBO
 *  translation (`step-89999` → `90k` no asterisk = exactly 90k completed)
 *  and the legacy-periodic case (`step-90000` → `90k*` = 90,001 completed). */
function shortenStepCell(step: number | null): string {
  if (step == null) return '—'
  return formatStep(step)
}

export function MpPage({ mpId }: { mpId: string }) {
  const gridsQ = useQuery({
    queryKey: ['mp-grids', mpId],
    queryFn: () => fetchGrids(mpId),
  })

  // Pick any one grid's zarr.json to source the header info from — atoms +
  // lattice are mat-scoped, not run-scoped, so any row works. Prefer GT
  // (smallest read; always present if the mat is in the dataset at all).
  const headerSourceKey = gridsQ.data?.find((g) => g.role === 'gt')?.r2_key
    ?? gridsQ.data?.[0]?.r2_key
    ?? null

  const attrsQ = useQuery({
    queryKey: ['mp-elvis-attrs', headerSourceKey],
    queryFn: () => fetchElvisAttrs(headerSourceKey!),
    enabled: headerSourceKey != null,
  })

  return (
    <>
      <header>
        <h1>
          <a href="#/" style={{ color: 'inherit', textDecoration: 'none' }}>tomat 🍅</a>
          <span style={{ fontSize: '0.7em', opacity: 0.7 }}> · mp · {mpId}</span>
        </h1>
        <nav style={{ marginLeft: '1rem', display: 'flex', gap: '0.75rem', fontSize: '0.9rem' }}>
          <a href="#/runs">runs</a>
          <a href="#/files">files</a>
          <a href="#/posts">posts</a>
          <a href="#/deck">deck</a>
        </nav>
      </header>

      <MaterialHeader mpId={mpId} attrs={attrsQ.data ?? null} loading={attrsQ.isLoading} />

      <section style={{ marginTop: '1.5rem' }}>
        <h2 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Grids</h2>
        {gridsQ.isLoading && <p className="meta">Loading grid index…</p>}
        {gridsQ.error && <p className="meta" style={{ color: 'crimson' }}>Failed to load: {String(gridsQ.error)}</p>}
        {gridsQ.data && <GridTable mpId={mpId} rows={gridsQ.data} />}
      </section>
    </>
  )
}

function MaterialHeader({
  mpId,
  attrs,
  loading,
}: {
  mpId: string
  attrs: ElvisAttrs | null
  loading: boolean
}) {
  const formula = attrs?.atoms ? formulaFromAtoms(attrs.atoms) : null
  const lattice = attrs?.lattice
  return (
    <section style={{ marginTop: '0.5rem' }}>
      <h2 style={{ fontSize: '1.3rem', margin: 0 }}>
        {formula ?? mpId}
        {formula && <span style={{ fontSize: '0.7em', opacity: 0.7, marginLeft: '0.5rem' }}>({mpId})</span>}
      </h2>
      {loading && <p className="meta">Loading material attrs…</p>}
      {attrs && (
        <div style={{ display: 'flex', gap: '2rem', fontSize: '0.85rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
          <div>
            <strong>Atoms:</strong> {attrs.atoms?.length ?? 0}
            {attrs.atoms && attrs.atoms.length > 0 && (
              <span style={{ marginLeft: '0.5rem', opacity: 0.8 }}>
                ({[...new Set(attrs.atoms.map((a) => a.element))].join(', ')})
              </span>
            )}
          </div>
          {lattice && (
            <div>
              <strong>Lattice:</strong>{' '}
              <code style={{ fontSize: '0.85em' }}>
                {lattice.map((row) => `[${row.map((v) => v.toFixed(2)).join(', ')}]`).join(' ')}
              </code>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function GridTable({ mpId, rows }: { mpId: string; rows: GridRow[] }) {
  if (rows.length === 0) {
    return <p className="meta">No grids found for {mpId}.</p>
  }
  // The GT row is the diff anchor — pred rows link as `v0=<gt>&v1=<pred>&s=d`.
  // Prefer the validation GT (matches val_200 m-evals); fall back to any gt.
  const gtRow = rows.find((r) => r.role === 'gt' && r.set === 'validation')
    ?? rows.find((r) => r.role === 'gt')
  const gtKey = gtRow?.r2_key ?? null
  return (
    <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.9rem' }}>
      <thead>
        <tr style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>
          <th style={{ padding: '0.4rem 0.6rem' }}>Role</th>
          <th style={{ padding: '0.4rem 0.6rem' }}>Run</th>
          <th style={{ padding: '0.4rem 0.6rem' }}>Set</th>
          <th style={{ padding: '0.4rem 0.6rem' }}>Step</th>
          <th style={{ padding: '0.4rem 0.6rem' }}>Written</th>
          <th style={{ padding: '0.4rem 0.6rem', textAlign: 'right' }}>Size</th>
          <th style={{ padding: '0.4rem 0.6rem', textAlign: 'right' }}>NMAE</th>
          <th style={{ padding: '0.4rem 0.6rem' }}></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.r2_key} style={{ borderBottom: '1px solid #eee' }}>
            <td style={{ padding: '0.3rem 0.6rem' }}>
              <span
                style={{
                  background: row.role === 'gt' ? '#e0f0ff' : '#fff4e0',
                  color: '#333',
                  padding: '0.1rem 0.4rem',
                  borderRadius: 3,
                  fontSize: '0.8em',
                  fontWeight: 600,
                }}
              >
                {row.role}
              </span>
            </td>
            <td style={{ padding: '0.3rem 0.6rem' }} title={row.run_id ?? ''}>
              {shortenRunCell(row.run_id)}
            </td>
            <td style={{ padding: '0.3rem 0.6rem' }} title={row.set ?? ''}>
              {shortenSet(row.set)}
            </td>
            <td style={{ padding: '0.3rem 0.6rem' }} title={row.step != null ? String(row.step) : ''}>
              {shortenStepCell(row.step)}
            </td>
            <td style={{ padding: '0.3rem 0.6rem', fontSize: '0.85em', opacity: 0.85 }}>
              {formatWritten(row.written_at)}
            </td>
            <td style={{ padding: '0.3rem 0.6rem', textAlign: 'right', fontSize: '0.85em' }}>
              {formatBytes(row.size_bytes)}
            </td>
            <td style={{ padding: '0.3rem 0.6rem', textAlign: 'right' }}>
              {row.nmae == null ? '—' : `${(row.nmae * 100).toFixed(2)}%`}
            </td>
            <td style={{ padding: '0.3rem 0.6rem' }}>
              <a
                href={elvisUrlFor(mpId, row.r2_key, row.role === 'pred' ? gtKey : null)}
                target="_blank"
                rel="noreferrer"
              >
                Open in ELVis ↗
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
