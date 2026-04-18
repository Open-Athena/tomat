import { csvParse } from 'd3-dsv'
import { useEffect, useState } from 'react'
import type { SweepRow } from './types'

export function useSweep(url: string) {
  const [rows, setRows] = useState<SweepRow[] | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`)
        return r.text()
      })
      .then(text => {
        const parsed = csvParse(text, d => ({
          mp_id: d.mp_id!,
          category: d.category!,
          config: d.config!,
          tokens: Number(d.tokens),
          nmae: Number(d.nmae),
          chi_sq: Number(d.chi_sq),
          hellinger: Number(d.hellinger),
          jsd: Number(d.jsd),
          weighted_mae: Number(d.weighted_mae),
          seconds: Number(d.seconds),
          grid: d.grid!,
          mass_captured: d.mass_captured ? Number(d.mass_captured) : null,
          effective_threshold: d.effective_threshold ? Number(d.effective_threshold) : null,
        }))
        setRows(parsed as SweepRow[])
      })
      .catch(setError)
  }, [url])

  return { rows, error }
}
