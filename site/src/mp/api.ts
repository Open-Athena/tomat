// Shared mp-page API helpers — the /mp/<id> page (`MpPage.tsx`) and the
// joint-histogram page (`viz/JointHistPage.tsx`) both want the per-mat
// grid index and zarr-attrs helpers. Keeping them in one module so the
// two consumers don't drift.

import { API_BASE } from '../runs/api'

/** One row of the grid index — mirrors the CFW's `GridRow` shape. */
export interface GridRow {
  mp_id: string
  role: 'gt' | 'pred'
  run_id: string | null
  set: string | null
  step: number | null
  r2_key: string
  size_bytes: number
  written_at: string
  nmae: number | null
}

export async function fetchGrids(mp_id: string): Promise<GridRow[]> {
  const r = await fetch(`${API_BASE}/api/mp/${encodeURIComponent(mp_id)}/grids`)
  if (!r.ok) throw new Error(`fetchGrids(${mp_id}) ${r.status}`)
  return r.json()
}
