export interface SweepRow {
  mp_id: string
  category: string
  config: string
  tokens: number
  nmae: number
  chi_sq: number
  hellinger: number
  jsd: number
  weighted_mae: number
  seconds: number
  grid: string
  mass_captured: number | null
  effective_threshold: number | null
}

export type Metric = 'nmae' | 'chi_sq' | 'hellinger' | 'jsd' | 'weighted_mae'

export const METRIC_LABEL: Record<Metric, string> = {
  nmae: 'NMAE',
  chi_sq: 'χ²',
  hellinger: 'Hellinger',
  jsd: 'JSD',
  weighted_mae: 'Weighted MAE',
}

// Parse a config label like "fourier-lowg-5pct" into (scheme, fraction).
// Returns null for configs without a fraction (e.g. "direct").
export function parseConfig(config: string): { scheme: string; fraction: number } | null {
  const m = config.match(/^(.+?)-(\d+(?:\.\d+)?)pct$/)
  if (!m) return null
  return { scheme: m[1], fraction: Number(m[2]) / 100 }
}
