// Smoothing mode + URL params for the run timeline / wallclock plots.
//
// EMA is implemented locally (pltly only ships rolling-window). The `rolling`
// path is handled in the plot components via pltly's `rolling()` (Welford's,
// O(n) two-pointer) — the per-sample-index σ from its `SmoothedMetric.stddev`
// gives the ±σ band that used to be a tomat-side rolling-std proxy. See
// `RunsTimelinePlot.tsx` / `WallclockPlot.tsx` for the pltly call sites.

import type { Param } from 'use-prms'

export type SmoothMode =
  | { kind: 'raw' }
  | { kind: 'ema'; alpha: number }
  | { kind: 'rolling'; window: number }

/** Treat NaN/Infinity as null so callers don't have to filter upstream. */
function clean(v: number | null | undefined): number | null {
  if (v == null) return null
  return Number.isFinite(v) ? (v as number) : null
}

/**
 * Exponential moving average: `y[t] = α·x[t] + (1-α)·y[t-1]`, `y[0] = x[0]`.
 * Null inputs pass through as null and don't update the running average — the
 * next valid sample resumes from the last good `y`.
 */
export function ema(values: (number | null)[], alpha: number): (number | null)[] {
  const a = Math.max(0, Math.min(1, alpha))
  const out: (number | null)[] = new Array(values.length)
  let prev: number | null = null
  for (let i = 0; i < values.length; i++) {
    const x = clean(values[i])
    if (x == null) { out[i] = null; continue }
    prev = prev == null ? x : a * x + (1 - a) * prev
    out[i] = prev
  }
  return out
}

/** Stable key for memoizing smoothing output per (series, mode). */
export function smoothKey(mode: SmoothMode): string {
  if (mode.kind === 'raw') return 'raw'
  if (mode.kind === 'ema') return `ema:${mode.alpha}`
  return `rolling:${mode.window}`
}

/**
 * Apply the smoothing mode to a y-array. Only handles `raw` (passthrough) and
 * `ema`; `rolling` is the responsibility of the consumer (which routes it
 * through pltly's `rolling()` to get mean + within-window σ in one pass).
 * Calling this with `rolling` is a programmer error — we throw so the misuse
 * is caught at first render, not silently ignored.
 */
export function applySmoothing(
  values: (number | null)[], mode: SmoothMode,
): (number | null)[] {
  if (mode.kind === 'raw') return values
  if (mode.kind === 'ema') return ema(values, mode.alpha)
  throw new Error('applySmoothing: rolling mode handled by pltly aggregation, not this helper')
}

// ── URL encoding (`?smooth=ema:0.1` / `?smooth=rolling:50`) ──
// Co-located with the mode type so `RunsTimelinePlot` + `WallclockPlot` share
// one parser. Decode is lenient (`raw`/empty → raw); encode omits raw so it
// stays out of the canonical URL.

const RAW: SmoothMode = { kind: 'raw' }

export const smoothParam: Param<SmoothMode> = {
  encode: (v) => {
    if (v.kind === 'raw') return undefined
    if (v.kind === 'ema') return `ema:${v.alpha}`
    return `rolling:${v.window}`
  },
  decode: (s) => {
    if (!s || s === 'raw') return RAW
    const [kind, rest] = s.split(':')
    if (kind === 'ema') {
      const a = Number(rest)
      if (!Number.isFinite(a) || a <= 0 || a > 1) return RAW
      return { kind: 'ema', alpha: a }
    }
    if (kind === 'rolling') {
      const n = Math.round(Number(rest))
      if (!Number.isFinite(n) || n < 1) return RAW
      return { kind: 'rolling', window: n }
    }
    return RAW
  },
}

export const DEFAULT_EMA_ALPHA = 0.1
export const DEFAULT_ROLLING_WINDOW = 50

/** Boolean `?bands=1` toggle (mirrors the `ancParam` `=1` convention so links
 *  read consistently with the rest of the dashboard). */
export const bandsParam: Param<boolean> = {
  encode: (v) => (v ? '1' : undefined),
  decode: (s) => s === '1' || s === 'true' || s === '',
}
