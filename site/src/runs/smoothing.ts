// Smoothing mode + URL params for the run timeline / wallclock plots.
//
// `WallclockPlot.tsx` calls `computeSmoothedSeries` (below) — which keeps
// rolling sample-index-based to dodge the "wallclock-mode ±50 ms window"
// bug. `RunsTimelinePlot.tsx` still routes its own rolling through pltly's
// `rolling()` (Welford's, O(n) two-pointer) with `getX: i => i`, which is
// equivalent to what we do here. EMA is implemented inline (pltly only
// ships rolling-window).

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

/**
 * Centered sample-index rolling mean + within-window σ over a y-series.
 * Window of `window` samples means `±window/2` on either side of `i`,
 * using a pltly-`rolling()`-compatible `<=` boundary at both ends.
 *
 * Pure helper, no x-axis dependency. Used by `computeSmoothedSeries` for
 * the `rolling` branch and unit-tested directly. Nulls / non-finite values
 * are skipped from the running sums (matching pltly's behaviour). The mean
 * is `NaN` exactly when no valid samples fall in the window (which never
 * happens for a finite-y series and a positive window, since `i` itself is
 * always in the window).
 *
 * Implemented as O(n·window) for clarity — n is bounded by parquet history
 * size (≤ ~50k typically), so a sliding-pointer O(n) two-pointer pass would
 * be optimisation premature here. If this ever shows up in profiles, swap
 * in Welford's online algorithm with a two-pointer window.
 */
function rollingMeanStd(
  ys: (number | null)[], window: number,
): { mean: number[]; std: number[] } {
  const n = ys.length
  const half = window / 2
  const mean = new Array<number>(n)
  const std = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    // pltly's centered window uses `<= windowEnd` and `< windowStart`,
    // so it includes `windowSize+1` integer positions when the window aligns
    // on integer x-coords. Match that exactly so we stay drop-in compatible
    // with the rolling.test.js expectations.
    const lo = Math.max(0, Math.ceil(i - half))
    const hi = Math.min(n - 1, Math.floor(i + half))
    let count = 0
    let sum = 0
    let sumSq = 0
    for (let j = lo; j <= hi; j++) {
      const v = ys[j]
      if (v === null || !Number.isFinite(v)) continue
      count++
      sum += v
      sumSq += v * v
    }
    if (count === 0) { mean[i] = NaN; std[i] = NaN; continue }
    const m = sum / count
    mean[i] = m
    // E[X²] - E[X]², clamped to 0 for floating-point safety.
    std[i] = Math.sqrt(Math.max(0, sumSq / count - m * m))
  }
  return { mean, std }
}

/**
 * Smoothed mean (+ rolling-only σ) for a y-series, sample-index windowed.
 *
 *   • `raw`     → `{mean: ys, std: null}` (passthrough).
 *   • `ema`     → `{mean: ema(ys, α), std: null}`.
 *   • `rolling` → centered SAMPLE-INDEX window of `mode.window` samples
 *                 (i.e. `±window/2` on either side of `i`). Mean + within-
 *                 window σ via `rollingMeanStd` above.
 *
 * Sample-index — not x-axis-units — windowing is intentional: in the
 * wallclock x-mode an x-units window of N=100 means ±50 ms which captures
 * a single sample (ourselves) per cadence-~3 s training rows, so `rolling:100`
 * silently passes through unchanged. Switching to sample-index makes the URL
 * value mean the same thing in every x-mode AND matches the convention used
 * by `RunsTimelinePlot.tsx`.
 *
 * The smoothing is purely a function of `ys` and `mode` — it does NOT depend
 * on the x-coordinate at all. Sparse traces (only a handful of samples) with
 * a large window collapse to ≈the mean of all points; callers that don't want
 * that behaviour (e.g. MT/MV eval traces with 5-30 timepoints) should detect
 * that and pass `raw` to bypass smoothing instead.
 *
 * Returns arrays of the same length as `ys`. `NaN` mean/std are normalized
 * to `null` so plotly draws gaps over missing points instead of error-bombing
 * on a non-finite log-axis value.
 */
export function computeSmoothedSeries(
  ys: (number | null)[], mode: SmoothMode,
): { mean: (number | null)[]; std: (number | null)[] | null } {
  if (mode.kind === 'raw') return { mean: ys, std: null }
  if (mode.kind === 'ema') return { mean: ema(ys, mode.alpha), std: null }
  const { mean, std } = rollingMeanStd(ys, mode.window)
  return {
    mean: mean.map((m) => (Number.isNaN(m) ? null : m)),
    std: std.map((s) => (Number.isNaN(s) ? null : s)),
  }
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
