// FLOP formatting + the URL-backed unit preference (`?fopu=`).
//
// Lives in its own module so RunHeaderRich, WallclockPlot, RunsTimelinePlot,
// and the per-step m-eval table all share the same helper without circular
// imports through `runMeta`. `formatFlops` + its supporting constants are
// extracted into a sibling `flops.format.ts` so the unit tests can import
// the formatter without dragging in `use-prms` (which doesn't resolve under
// `node --test --experimental-strip-types` because workspace deps aren't on
// the standalone Node module path).
//
// Units:
//   - EF = exaflops  (10^18) — the user's default. 2.4e20 FLOP → 240 EF.
//   - PF = petaflops (10^15)
//   - TF = teraflops (10^12)
//   - sci = bare scientific notation, e.g. `2.4e20` (legacy display).

import type { CSSProperties, ReactElement } from 'react'
import type { Param } from 'use-prms'
import { enumParam, useUrlState } from 'use-prms'
import { Tooltip } from '../Tooltip'

export { FLOP_UNITS, formatFlops } from './flops.format'
export type { FlopUnit } from './flops.format'
import type { FlopUnit } from './flops.format'
import { FLOP_UNITS } from './flops.format'

/** `?fopu=EF|PF|TF|sci` — URL-backed FLOP-unit preference. Default `EF`. */
export const flopUnitParam: Param<FlopUnit> = enumParam<FlopUnit>('EF', FLOP_UNITS)

/** Hook owning the `?fopu=` URL state. Default `EF`. */
export function useFlopUnit(): readonly [FlopUnit, (v: FlopUnit) => void] {
  const [v, set] = useUrlState('fopu', flopUnitParam)
  return [v, set] as const
}

interface FlopUnitChipsProps {
  unit: FlopUnit
  setUnit: (u: FlopUnit) => void
  isDark: boolean
  fg: string
  muted: string
}

/** Compact chip rail for FLOP-unit display preference: `EF | PF | TF | sci`.
 *  Used by the cross-run timeline and per-run wallclock plots so the user
 *  can flip between scales in any view. */
export function FlopUnitChips({
  unit, setUnit, isDark, fg, muted,
}: FlopUnitChipsProps): ReactElement {
  const chipStyle = (on: boolean): CSSProperties => ({
    background: on ? (isDark ? '#2a2a2a' : '#e8e8e8') : 'transparent',
    border: `1px solid ${on ? (isDark ? '#444' : '#bbb') : 'transparent'}`,
    borderRadius: 4, cursor: 'pointer', padding: '2px 5px',
    fontSize: '0.72rem', fontFamily: 'inherit',
    color: on ? fg : muted,
  })
  const TIPS: Record<FlopUnit, string> = {
    EF: 'exaflops (10^18) — default. e.g. 2.4e20 FLOP → 240 EF',
    PF: 'petaflops (10^15) — e.g. 2.4e20 FLOP → 240,000 PF',
    TF: 'teraflops (10^12) — e.g. 2.4e20 FLOP → 240,000,000 TF',
    sci: 'scientific notation — e.g. 2.4e20 FLOP',
  }
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginLeft: 4 }}>
      <span style={{ fontSize: '0.7rem', color: muted }}>FLOP:</span>
      {FLOP_UNITS.map((u) => (
        <Tooltip key={u} content={TIPS[u]}>
          <button onClick={() => setUnit(u)} style={chipStyle(unit === u)}>
            {u}
          </button>
        </Tooltip>
      ))}
    </span>
  )
}
