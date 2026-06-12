// Pure FLOP-formatting helper, factored out of `./flops.ts` so the unit tests
// can import without dragging in `use-prms` (the URL-state hook lives in
// `flops.ts`, where it composes with this formatter).

export type FlopUnit = 'EF' | 'PF' | 'TF' | 'sci'
export const FLOP_UNITS: readonly FlopUnit[] = ['EF', 'PF', 'TF', 'sci'] as const

const UNIT_EXP: Record<Exclude<FlopUnit, 'sci'>, number> = {
  EF: 18,
  PF: 15,
  TF: 12,
}

/** Format a FLOP count under the chosen unit.
 *
 *  - `EF`/`PF`/`TF`: scaled value + unit suffix, e.g.
 *    `format(2.4e20, 'EF') = "240 EF"`, `format(9.0e17, 'EF') = "0.90 EF"`,
 *    `format(3.1e18, 'EF') = "3.1 EF"`. Precision varies by magnitude so
 *    small values stay informative without trailing zeros on big ones.
 *  - `sci`: scientific notation matching the prior display, e.g. `1.1e20`. */
export function formatFlops(value: number, unit: FlopUnit = 'EF'): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  if (unit === 'sci') {
    const exp = Math.floor(Math.log10(value))
    return `${(value / 10 ** exp).toFixed(1)}e${exp}`
  }
  const scaled = value / 10 ** UNIT_EXP[unit]
  // Magnitude-based precision: <10 → 1 digit, ≥10 → 0 (with locale grouping
  // so "240 EF" stays uncluttered but "240,000 PF" reads cleanly when the
  // scaled value runs large).
  if (scaled < 10) {
    return `${scaled.toFixed(1)} ${unit}`
  }
  const rounded = Math.round(scaled)
  // toLocaleString defaults to comma grouping in en-US; works everywhere we
  // render strings into the DOM. No-op for values <1000.
  return `${rounded.toLocaleString('en-US')} ${unit}`
}

/** Scale factor that takes raw FLOPs to the user-chosen unit. For `sci`
 *  (legacy scientific-notation display) the scale stays 1 so axis ticks read
 *  in raw FLOPs (`.2e` formatted). For EF/PF/TF, divide by 10^{18,15,12} so
 *  plotly's tick labels read in human-friendly numbers. Drives the plotly
 *  trace x arrays in both `RunsTimelinePlot` and `WallclockPlot` so the
 *  axis ticks read in the unit chosen via the `FlopUnitChips` chip rail
 *  (bug 1 fix — was hardcoded `.2e` regardless of unit). */
export function flopUnitScale(unit: FlopUnit): number {
  if (unit === 'sci') return 1
  return 10 ** UNIT_EXP[unit]
}

/** Per-unit plotly `tickformat` for the FLOP-axis ticks. Pairs with
 *  `flopUnitScale` — the x values arrive in the chosen unit, so the format
 *  string is chosen to render those scaled magnitudes legibly:
 *  - `EF` lands in 0.1-1000 → fixed point with trailing-zero suppression
 *    (`.2~f` → "0.5" / "240")
 *  - `PF` goes into the thousands / hundreds-of-thousands → grouped fixed
 *    (`,.0f` → "240,000")
 *  - `TF` goes to hundreds-of-millions+ → scientific
 *  - `sci` always scientific (legacy display) */
export function flopTickformat(unit: FlopUnit): string {
  switch (unit) {
    case 'EF': return '.2~f'
    case 'PF': return ',.0f'
    case 'TF': return '.2e'
    case 'sci': return '.2e'
  }
}
