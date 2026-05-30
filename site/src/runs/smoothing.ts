// Pure-JS smoothing helpers for the run timeline / wallclock plots.
//
// All helpers preserve length and pass `null` through (and treat `NaN` /
// `Infinity` as null) so the existing trace pipelines — which carry `null` as
// "missing sample" — stay aligned with their `x` arrays. Centered windows are
// used for rolling stats so the smoothed curve doesn't lag the raw signal.
//
// No deps; keep tiny + readable. Tests via the dashboard rendering, not unit
// tests (the trace-builder smokes these on every render).

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

/** Centered half-window for a window of `n` samples. Even `n` rounds up. */
function halfWin(n: number): number {
  return Math.max(0, Math.floor((Math.max(1, n) - 1) / 2))
}

/**
 * Centered rolling mean over `window` points. Nulls don't contribute; the
 * window slides over the index (so endpoints have smaller effective windows).
 * Centered (not trailing) so the smoothed curve aligns with the raw signal in
 * time — trailing means the smoothed line lags ~window/2 steps behind, which
 * misleads on training curves where direction matters.
 */
export function rollingMean(
  values: (number | null)[], window: number,
): (number | null)[] {
  const h = halfWin(window)
  const out: (number | null)[] = new Array(values.length)
  for (let i = 0; i < values.length; i++) {
    if (clean(values[i]) == null) { out[i] = null; continue }
    let sum = 0
    let n = 0
    const lo = Math.max(0, i - h)
    const hi = Math.min(values.length - 1, i + h)
    for (let j = lo; j <= hi; j++) {
      const v = clean(values[j])
      if (v == null) continue
      sum += v; n++
    }
    out[i] = n > 0 ? sum / n : null
  }
  return out
}

/**
 * Centered rolling sample standard deviation, companion to `rollingMean`.
 * Sample std (n-1 divisor) — matches what most readers expect when they see
 * "±σ"; population std would underreport noise on small windows. Returns 0 at
 * indices with <2 valid neighbours (the band collapses to the line there).
 */
export function rollingStd(
  values: (number | null)[], window: number,
): (number | null)[] {
  const h = halfWin(window)
  const out: (number | null)[] = new Array(values.length)
  for (let i = 0; i < values.length; i++) {
    if (clean(values[i]) == null) { out[i] = null; continue }
    let sum = 0
    let sqsum = 0
    let n = 0
    const lo = Math.max(0, i - h)
    const hi = Math.min(values.length - 1, i + h)
    for (let j = lo; j <= hi; j++) {
      const v = clean(values[j])
      if (v == null) continue
      sum += v; sqsum += v * v; n++
    }
    if (n < 2) { out[i] = 0; continue }
    const mean = sum / n
    const variance = Math.max(0, (sqsum - n * mean * mean) / (n - 1))
    out[i] = Math.sqrt(variance)
  }
  return out
}

/** Stable key for memoizing smoothing output per (series, mode). */
export function smoothKey(mode: SmoothMode): string {
  if (mode.kind === 'raw') return 'raw'
  if (mode.kind === 'ema') return `ema:${mode.alpha}`
  return `rolling:${mode.window}`
}

/** Apply the smoothing mode to a y-array. `raw` returns the input unchanged. */
export function applySmoothing(
  values: (number | null)[], mode: SmoothMode,
): (number | null)[] {
  if (mode.kind === 'raw') return values
  if (mode.kind === 'ema') return ema(values, mode.alpha)
  return rollingMean(values, mode.window)
}

/** Companion std for the smoothing mode. EMA uses a rolling window of size
 *  `~ 1/α` (the EMA's effective memory) so the σ band tracks the same scale of
 *  fluctuation the EMA is smoothing over. */
export function smoothingStd(
  values: (number | null)[], mode: SmoothMode,
): (number | null)[] | null {
  if (mode.kind === 'raw') return null
  const window = mode.kind === 'rolling'
    ? mode.window
    : Math.max(2, Math.round(1 / Math.max(1e-3, mode.alpha)))
  return rollingStd(values, window)
}

// ── URL encoding (`?smooth=ema:0.1` / `?smooth=rolling:50`) ──
// Co-located with the mode type so `RunsTimelinePlot` + `WallclockPlot` share
// one parser. Decode is lenient (`raw`/empty → raw); encode omits raw so it
// stays out of the canonical URL.

import type { Param } from 'use-prms'

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
