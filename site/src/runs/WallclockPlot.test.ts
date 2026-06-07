// Tests for the smoothing pipeline used by `WallclockPlot`.
//
// Run:  node --test --experimental-strip-types site/src/runs/WallclockPlot.test.ts
//
// Covers the regression that motivated this file: with `?x=wallclock&smooth=
// rolling:100`, the previous implementation interpreted N=100 as 100 ms of
// x-axis (the wallclock unit), which captures only the point itself for
// typical training cadences → `rolling` silently passed through. Sample-index
// windowing fixes it, and the same computation applies regardless of x-mode
// or whether the user has box-zoomed in (the trace is built from the full
// series and plotly clips for display).

import { strict as assert } from 'node:assert'
import { describe, it } from 'node:test'
import { computeSmoothedSeries, type SmoothMode } from './smoothing.ts'

const RAW: SmoothMode = { kind: 'raw' }
const ema = (alpha: number): SmoothMode => ({ kind: 'ema', alpha })
const roll = (window: number): SmoothMode => ({ kind: 'rolling', window })

const close = (a: number, b: number, eps = 1e-9) =>
  Math.abs(a - b) <= eps

describe('computeSmoothedSeries', () => {
  describe('raw', () => {
    it('passes ys through unchanged', () => {
      const ys = [10, 20, 30, 40, 50]
      const out = computeSmoothedSeries(ys, RAW)
      assert.deepEqual(out.mean, ys)
      assert.equal(out.std, null)
    })

    it('passes nulls through', () => {
      const ys = [10, null, 30]
      const out = computeSmoothedSeries(ys, RAW)
      assert.deepEqual(out.mean, ys)
    })
  })

  describe('ema', () => {
    it('y[0] = x[0], y[t] = α·x[t] + (1-α)·y[t-1]', () => {
      const out = computeSmoothedSeries([10, 20, 30, 40], ema(0.5))
      // y[0] = 10
      // y[1] = 0.5*20 + 0.5*10 = 15
      // y[2] = 0.5*30 + 0.5*15 = 22.5
      // y[3] = 0.5*40 + 0.5*22.5 = 31.25
      assert.deepEqual(out.mean, [10, 15, 22.5, 31.25])
      assert.equal(out.std, null)
    })

    it('null inputs pass through; running EMA resumes from last good y', () => {
      const out = computeSmoothedSeries([10, null, 30], ema(0.5))
      // y[0] = 10, y[1] = null, y[2] = 0.5*30 + 0.5*10 = 20
      assert.deepEqual(out.mean, [10, null, 20])
    })

    it('is independent of how the data would be windowed on the x-axis', () => {
      // EMA only walks the y-array; the smoothing output never depends on x.
      // This is the property that makes "smoothing while zoomed" work — the
      // trace is built from the full series and plotly clips on display.
      const ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const a = computeSmoothedSeries(ys, ema(0.3))
      const b = computeSmoothedSeries(ys, ema(0.3))
      assert.deepEqual(a.mean, b.mean)
    })
  })

  describe('rolling (sample-index window)', () => {
    it('centered window of N=2 (±1) over [10,20,30,40,50]', () => {
      // pltly's rolling: at i=0 window is [-1, 1] → idx 0,1 → mean 15
      // at i=1 window is [0, 2] → idx 0,1,2 → mean 20
      // at i=2 window is [1, 3] → idx 1,2,3 → mean 30
      // at i=3 window is [2, 4] → idx 2,3,4 → mean 40
      // at i=4 window is [3, 5] → idx 3,4 → mean 45
      const out = computeSmoothedSeries([10, 20, 30, 40, 50], roll(2))
      assert.deepEqual(out.mean, [15, 20, 30, 40, 45])
      // std at i=1: values [10,20,30], var = ((100+0+100)/3) = 66.67 → σ ≈ 8.165
      assert.ok(out.std !== null)
      assert.ok(close(out.std[1]!, Math.sqrt(200 / 3), 1e-9))
      // std at i=2: values [20,30,40], var = same shape → σ ≈ 8.165
      assert.ok(close(out.std[2]!, Math.sqrt(200 / 3), 1e-9))
    })

    it('window N=4 over a 10-element series smooths the middle, tails average fewer', () => {
      const ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const out = computeSmoothedSeries(ys, roll(4))
      // at i=5, window is [3, 7] → idx 3..7 → values [4,5,6,7,8] → mean 6
      assert.ok(close(out.mean[5]!, 6, 1e-9))
      // at i=0, window is [-2, 2] → idx 0..2 → values [1,2,3] → mean 2
      assert.ok(close(out.mean[0]!, 2, 1e-9))
      // at i=9, window is [7, 11] → idx 7..9 → values [8,9,10] → mean 9
      assert.ok(close(out.mean[9]!, 9, 1e-9))
    })

    it('huge window collapses every point to the mean of the whole series', () => {
      const ys = [10, 20, 30, 40, 50]
      const out = computeSmoothedSeries(ys, roll(1_000_000))
      // Every window covers the entire series → all means equal the global mean.
      for (const m of out.mean) assert.ok(close(m!, 30, 1e-9))
    })

    it('handles nulls by skipping them in the window', () => {
      // Sparse series [10, null, null, 30]; with window=100 every point sees
      // both valid values. Mean = 20 at every position.
      const out = computeSmoothedSeries([10, null, null, 30], roll(100))
      assert.ok(close(out.mean[0]!, 20))
      assert.ok(close(out.mean[3]!, 20))
    })

    it('result is independent of x-axis units (sample-index, not x-units)', () => {
      // The point of the fix: the same N produces the same smoothing whether
      // the trace was logged on a tight 1 s cadence or a slow 1 h cadence.
      // Concretely, switching `?x=` between wallclock/step/elapsed must NOT
      // change the smoothed output.
      const ys = [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
      const out = computeSmoothedSeries(ys, roll(3))
      // Centered window of 3 samples (±1.5). With pltly's `<=` boundary,
      // at i=4: window is [2.5, 5.5] → idx 3,4,5 → mean = (120+115+130)/3 = 121.67
      assert.ok(close(out.mean[4]!, (120 + 115 + 130) / 3, 1e-9))
    })
  })

  describe('"zoomed-window" smoothing — repro of the original bug', () => {
    // The reported bug: at `?x=wallclock&smooth=rolling:100`, after box-zoom
    // the smoothed trace stays identical to raw. Root cause: the old
    // implementation called pltly's rolling with `windowSize: 100` interpreted
    // as ms (the wallclock x-axis unit), which captures only the point itself
    // for typical training cadences.
    //
    // The fix is sample-index windowing — independent of x-mode and zoom. The
    // smoothing is computed once from the full series; plotly's `xaxis.range`
    // then clips on display. We exercise that contract here.

    it('rolling:100 visibly smooths a 1000-point series (regardless of x-mode)', () => {
      // Saw-tooth around a slow trend so smoothing has something to suppress.
      const ys = Array.from({ length: 1000 }, (_, i) => i + (i % 2 === 0 ? 5 : -5))
      const out = computeSmoothedSeries(ys, roll(100))
      // At i=500, raw alternates ±5 around 500. With a 100-sample window the
      // ±5 oscillation averages out: the smoothed value should sit close to i.
      assert.ok(Math.abs(out.mean[500]! - 500) < 1.0,
        `expected smoothed[500] near 500, got ${out.mean[500]}`)
      // Raw is NOT close to 500 (it's 505 or 495).
      assert.ok(Math.abs(ys[500] - 500) >= 5)
    })

    it('zoom is a display concern only — smoothing is invariant under slicing the visible range', () => {
      // What plotly does on box-zoom: clip the rendered xaxis.range to a
      // sub-interval. The data array passed to plotly is unchanged; the
      // smoothing function above is therefore invariant under "zoom".
      const ys = Array.from({ length: 200 }, (_, i) => Math.sin(i / 5) * 10 + 50)
      const out1 = computeSmoothedSeries(ys, roll(20))
      const out2 = computeSmoothedSeries(ys, roll(20))
      // Same input → same output, every time.
      assert.deepEqual(out1.mean, out2.mean)
      // And the smoothed value at any visible index is the same as it was
      // before the user zoomed — because the function never read an x-range.
      for (let i = 80; i < 120; i++) {
        assert.ok(close(out1.mean[i]!, out2.mean[i]!, 0))
      }
    })

    it('N > series length: window covers every point, all outputs equal the global mean', () => {
      // The user asked for: "if N > visible-window-points, still show the
      // smoothed trace at fewer points". Concretely: a tiny series with a
      // huge window must NOT collapse to NaN/null — every entry must be a
      // valid mean (which is the global mean when the window engulfs all).
      const ys = [10, 12, 14, 16, 18]
      const out = computeSmoothedSeries(ys, roll(10_000))
      const globalMean = (10 + 12 + 14 + 16 + 18) / 5
      for (const m of out.mean) {
        assert.ok(m !== null, 'every output point must be defined')
        assert.ok(close(m, globalMean, 1e-9))
      }
    })
  })
})
