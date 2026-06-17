// Local URL-param helpers that complement `use-prms`. Keep these tiny + pure
// so they're easy to test and reuse across pages.

import type { Param } from 'use-prms'

/** Bool param defaulting to TRUE: param absent → `true`; param present (any
 *  value) → `false`. Mirrors the convention from ELvis (`tomat-pred-zarrs-available.md`
 *  §URL param shortening): a single UPPERCASE letter name flags the inverted
 *  default-on semantics, e.g. `?M` = "no marginals", `?Z` = "no zarr". The
 *  encoder emits a valueless key so URLs stay terse (`…&M`, not `…&M=1`). */
export const boolTrueParam: Param<boolean> = {
  encode: (v) => (v ? undefined : ''),
  decode: (e) => e === undefined,
}
