import { useEffect, useMemo, useState } from 'react'
import { csvParse } from 'd3-dsv'
import { Plot, useTheme } from 'pltly/react'
import { themedHoverlabel } from './theme'

interface MatRow {
  mp_id: string
  nx: number
  ny: number
  nz: number
  n_atoms: number
}

const { max } = Math

export function MatsMetadataPlots({ url }: { url: string }) {
  const [rows, setRows] = useState<MatRow[] | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const { isDark } = useTheme()

  useEffect(() => {
    fetch(url)
      .then(r => r.ok ? r.text() : Promise.reject(new Error(`fetch ${url}: ${r.status}`)))
      .then(text => {
        const parsed = csvParse(text, d => ({
          mp_id: String(d.mp_id),
          nx: Number(d.nx),
          ny: Number(d.ny),
          nz: Number(d.nz),
          n_atoms: Number(d.n_atoms),
        }))
        setRows(parsed as MatRow[])
      })
      .catch(setError)
  }, [url])

  const derived = useMemo(() => {
    if (!rows) return null
    const n_atoms = rows.map(r => r.n_atoms)
    const max_dim = rows.map(r => max(r.nx, r.ny, r.nz))
    const n_voxels = rows.map(r => r.nx * r.ny * r.nz)
    return { n_atoms, max_dim, n_voxels, count: rows.length }
  }, [rows])

  if (error) return <p className="note">Error loading mats MD: {error.message}</p>
  if (!rows || !derived) return <p className="note">Loading materials metadata (~2 MB)…</p>

  const baseLayout = {
    autosize: true,
    height: 320,
    margin: { t: 40, r: 30, b: 48, l: 60 },
    hovermode: 'closest' as const,
    hoverlabel: themedHoverlabel(isDark),
    showlegend: false,
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Plot
        data={[{
          x: derived.n_atoms, type: 'histogram',
          xbins: { start: 0, end: max(...derived.n_atoms) + 2, size: 1 },
          marker: { color: '#3b6a9e' },
          hovertemplate: 'n_atoms = %{x}<br>count = %{y}<extra></extra>',
        }]}
        layout={{
          ...baseLayout,
          title: { text: `Atom count per material (N = ${derived.count.toLocaleString()})` },
          xaxis: { title: { text: 'atoms per material' }, fixedrange: false },
          yaxis: { title: { text: 'count (log)' }, type: 'log', fixedrange: false },
        }}
        style={{ width: '100%' }}
      />
      <Plot
        data={[{
          x: derived.max_dim, type: 'histogram',
          xbins: { start: 0, end: max(...derived.max_dim) + 8, size: 8 },
          marker: { color: '#2b8a3e' },
          hovertemplate: 'max(nx,ny,nz) = %{x}<br>count = %{y}<extra></extra>',
        }]}
        layout={{
          ...baseLayout,
          title: { text: 'Max grid dimension' },
          xaxis: { title: { text: 'max(nx, ny, nz) voxels' }, fixedrange: false },
          yaxis: { title: { text: 'count' }, fixedrange: false },
        }}
        style={{ width: '100%' }}
      />
      <Plot
        data={[{
          x: derived.n_voxels, type: 'histogram',
          xbins: { start: 0, end: 25_000_000, size: 250_000 },
          marker: { color: '#a63d40' },
          hovertemplate: 'n_voxels = %{x:,}<br>count = %{y}<extra></extra>',
        }]}
        layout={{
          ...baseLayout,
          title: { text: 'Total voxels per material' },
          xaxis: { title: { text: 'n_voxels = nx · ny · nz' }, fixedrange: false },
          yaxis: { title: { text: 'count (log)' }, type: 'log', fixedrange: false },
        }}
        style={{ width: '100%' }}
      />
      <Plot
        data={[{
          x: derived.n_atoms,
          y: derived.n_voxels,
          type: 'scattergl',
          mode: 'markers',
          marker: { color: '#7d3c98', size: 3, opacity: 0.35 },
          text: rows.map(r => r.mp_id),
          hovertemplate:
            '%{text}<br>n_atoms = %{x}<br>n_voxels = %{y:,}<extra></extra>',
        }]}
        layout={{
          ...baseLayout,
          title: { text: 'Atoms vs voxels (one point per material)' },
          xaxis: { title: { text: 'n_atoms' }, fixedrange: false },
          yaxis: { title: { text: 'n_voxels' }, fixedrange: false, type: 'log' },
        }}
        style={{ width: '100%' }}
      />
    </div>
  )
}
