// Tests for MEvalTable's sorting + colour-coding logic.
//
// Run:  node --test --experimental-strip-types site/src/runs/MEvalTable.test.ts
//
// We import the pure helpers (`stepsDescOf`, `nmaeBucket`, `nemdBucket`) and
// exercise them directly — no React rendering. The presentational React
// component is exercised in-browser via the dev server.

import { strict as assert } from 'node:assert'
import { describe, it } from 'node:test'
import {
  evalPhase,
  IRIS_LIVE_JOB_STATES, IRIS_TERMINAL_JOB_STATES,
  nemdBucket, nmaeBucket, parseEvalJobId, stepsDescOf,
  type Bucket, type EvalPhase,
} from './MEvalTable.helpers.ts'

const point = (step: number) => ({
  step,
  n_mats: 10,
  nmae_mean: null, nmae_p1: null, nmae_p25: null, nmae_median: null,
  nmae_p75: null, nmae_p99: null,
  nemd_mean: null, nemd_p1: null, nemd_p25: null, nemd_median: null,
  nemd_p75: null, nemd_p99: null,
})

describe('stepsDescOf', () => {
  it('returns [] for null input', () => {
    assert.deepEqual(stepsDescOf(null), [])
  })

  it('returns [] for empty sets dict', () => {
    assert.deepEqual(
      stepsDescOf({ schema_version: 1, synced_at: '', run: 'x', sets: {} }),
      [],
    )
  })

  it('returns descending step list across all sets', () => {
    const series = {
      schema_version: 1,
      synced_at: '',
      run: 'x',
      sets: {
        val_200:        [point(100), point(500), point(2000)],
        train_200:      [point(500), point(1500)],
        'val_200-maskgit': [point(2000), point(3000)],
      },
    }
    assert.deepEqual(stepsDescOf(series), [3000, 2000, 1500, 500, 100])
  })

  it('deduplicates step values across sets', () => {
    const series = {
      schema_version: 1,
      synced_at: '',
      run: 'x',
      sets: {
        val_200:   [point(100), point(500)],
        train_200: [point(100), point(500)],
      },
    }
    assert.deepEqual(stepsDescOf(series), [500, 100])
  })

  it('handles a single set with one point', () => {
    const series = {
      schema_version: 1,
      synced_at: '',
      run: 'x',
      sets: { val_200: [point(42)] },
    }
    assert.deepEqual(stepsDescOf(series), [42])
  })
})

describe('nmaeBucket', () => {
  // Thresholds: <1% green, 1-5% amber, >=5% red.
  const cases: [number, Bucket][] = [
    [0,    'good'],
    [0.5,  'good'],
    [0.99, 'good'],
    [1,    'mid'],   // boundary belongs to mid (>=1% is amber)
    [3.5,  'mid'],
    [4.99, 'mid'],
    [5,    'bad'],   // boundary belongs to bad (>=5% is red)
    [42,   'bad'],
    [386,  'bad'],   // mode-collapsed runs (cf. memory `free-running-divergence`)
  ]
  for (const [pct, want] of cases) {
    it(`${pct}% → ${want}`, () => {
      assert.equal(nmaeBucket(pct), want)
    })
  }

  it('NaN → none (not coloured)', () => {
    assert.equal(nmaeBucket(NaN), 'none')
  })

  it('negative → none (defensive: no valid NMAE is < 0)', () => {
    assert.equal(nmaeBucket(-1), 'none')
  })
})

describe('nemdBucket', () => {
  // Thresholds: <0.5% green, 0.5-3% amber, >=3% red.
  const cases: [number, Bucket][] = [
    [0,    'good'],
    [0.1,  'good'],
    [0.49, 'good'],
    [0.5,  'mid'],
    [2.99, 'mid'],
    [3,    'bad'],
    [100,  'bad'],
  ]
  for (const [pct, want] of cases) {
    it(`${pct}% → ${want}`, () => {
      assert.equal(nemdBucket(pct), want)
    })
  }

  it('NaN → none', () => {
    assert.equal(nemdBucket(NaN), 'none')
  })
})

describe('parseEvalJobId', () => {
  it('splits the mode infix out of the run label (oneshot / maskgit / free)', () => {
    // Oneshot: no mode infix, bare `tomat-eval-<run>-…`.
    assert.deepEqual(
      parseEvalJobId('/ryan/tomat-eval-train-mg-foo-val_200-step-40000'),
      { mode: 'oneshot', runLabel: 'train-mg-foo', matSet: 'val_200', step: 40000, taskIdx: null },
    )
    // K=12: `tomat-eval-maskgit-<run>-…`. The pre-fix regex used a greedy
    // `(.+)` and bound runLabel to `maskgit-train-mg-foo`, never matching
    // any actual run on the dashboard.
    assert.deepEqual(
      parseEvalJobId('/ryan/tomat-eval-maskgit-train-mg-foo-val_200-step-40000'),
      { mode: 'maskgit', runLabel: 'train-mg-foo', matSet: 'val_200', step: 40000, taskIdx: null },
    )
    // Free: `tomat-eval-free-<run>-…`.
    assert.deepEqual(
      parseEvalJobId('/ryan/tomat-eval-free-train-mg-foo-train_200-step-1000'),
      { mode: 'free', runLabel: 'train-mg-foo', matSet: 'train_200', step: 1000, taskIdx: null },
    )
  })

  it('captures the `-task<i>` fan-out suffix', () => {
    assert.deepEqual(
      parseEvalJobId('/ryan/tomat-eval-maskgit-train-foo-val_200-step-50000-task0'),
      { mode: 'maskgit', runLabel: 'train-foo', matSet: 'val_200', step: 50000, taskIdx: 0 },
    )
    assert.deepEqual(
      parseEvalJobId('/ryan/tomat-eval-train-foo-train_200-step-12345-task7'),
      { mode: 'oneshot', runLabel: 'train-foo', matSet: 'train_200', step: 12345, taskIdx: 7 },
    )
  })

  it('does not match training jobs (regex anchored to tomat-eval-)', () => {
    assert.equal(parseEvalJobId('/ryan/train-mg-foo'), null)
    assert.equal(parseEvalJobId('/ryan/tomat-train-smoke'), null)
    assert.equal(parseEvalJobId('not-an-iris-job-id'), null)
  })

  it('does not match other -<thing>- infixes (only maskgit / free)', () => {
    // `tomat-eval-junk-train-foo-…` → `junk` isn't a mode infix, so the
    // optional group bypasses → runLabel = `junk-train-foo`, mode = oneshot.
    const r = parseEvalJobId('/ryan/tomat-eval-junk-train-foo-val_200-step-100')
    assert.deepEqual(r, {
      mode: 'oneshot', runLabel: 'junk-train-foo', matSet: 'val_200', step: 100, taskIdx: null,
    })
  })
})

describe('evalPhase', () => {
  // Regression: prior implementation's `default → 'flight'` bucketed
  // `KILLED` (and any other state iris would add) as in-flight forever.
  // The bin5 run card showed `⏳ 3 m-evals` indefinitely because three
  // worker_lost_spec'd jobs sat in `KILLED` after their workers died.
  // Mirror of iris's `TERMINAL_JOB_STATES` (`marin/lib/iris/src/iris/
  // cluster/types.py`): SUCCEEDED → 'done', FAILED/KILLED/WORKER_FAILED/
  // UNSCHEDULABLE → 'failed', PENDING/BUILDING/RUNNING → 'flight'.
  const cases: [string, EvalPhase][] = [
    // — Live (non-terminal) iris job states. These must be 'flight'.
    ['PENDING',       'flight'],
    ['BUILDING',      'flight'],
    ['RUNNING',       'flight'],
    // Transient states that may appear in iris snapshots; bucket as flight
    // so we don't flap to 'failed' during normal scheduling.
    ['SUBMITTED',     'flight'],
    ['QUEUED',        'flight'],
    ['ASSIGNED',      'flight'],
    ['UNKNOWN',       'flight'],
    // — Terminal states.
    ['SUCCEEDED',     'done'],
    ['FAILED',        'failed'],
    ['KILLED',        'failed'],   // ← the bin5 case
    ['WORKER_FAILED', 'failed'],
    ['UNSCHEDULABLE', 'failed'],
    ['CANCELLED',     'failed'],   // historical alias from earlier iris versions
  ]
  for (const [state, want] of cases) {
    it(`${state} → ${want}`, () => {
      assert.equal(evalPhase({ state }), want)
    })
  }

  it('unknown / mistyped states bucket as failed (not flight)', () => {
    // The whole point of the fix: a state the FE doesn't recognise is
    // almost certainly stale, not in flight. Surface it as a failure so
    // the run card stops claiming a hourglass forever.
    assert.equal(evalPhase({ state: 'NOT_A_REAL_STATE' }), 'failed')
    assert.equal(evalPhase({ state: '' }), 'failed')
    assert.equal(evalPhase({ state: 'killed' /* lowercase typo */ }), 'failed')
  })

  it('IRIS_LIVE_JOB_STATES + IRIS_TERMINAL_JOB_STATES cover every state the helper buckets', () => {
    // Belt-and-braces: every name listed in the two canonical sets is
    // handled by the bucketing logic (the inverse — that no listed name
    // falls through to the unknown bucket — is implicit in this assertion
    // because the unknown bucket returns 'failed' and we already match
    // SUCCEEDED → 'done' / live → 'flight'.)
    for (const s of IRIS_LIVE_JOB_STATES) {
      assert.equal(evalPhase({ state: s }), 'flight', `live state ${s} should be flight`)
    }
    for (const s of IRIS_TERMINAL_JOB_STATES) {
      const want: EvalPhase = s === 'SUCCEEDED' ? 'done' : 'failed'
      assert.equal(evalPhase({ state: s }), want, `terminal state ${s} should be ${want}`)
    }
  })

  it('bin5 reproducer: KILLED jobs do not contribute to "in-flight" count', () => {
    // The 3 bin5 m-evals from 2026-06-16 that all worker_lost_spec'd. The
    // chip aggregator reads `evalPhase(job)` and counts 'flight' separately
    // from 'failed'; before the fix all three counted as flight, so the
    // card showed `⏳ 3 m-evals` forever. After the fix the count is 0.
    const bin5Jobs = [
      { state: 'KILLED' },  // train_200 step-60000
      { state: 'KILLED' },  // train_200 step-89999
      { state: 'KILLED' },  // val_200 step-49999
      { state: 'FAILED' },  // train_200 step-40000 (SIGSEGV)
      { state: 'SUCCEEDED' },  // val_200 step-89999, etc.
    ]
    const phases = bin5Jobs.map(j => evalPhase(j))
    const counts = { flight: 0, failed: 0, done: 0 }
    for (const p of phases) counts[p]++
    assert.deepEqual(counts, { flight: 0, failed: 4, done: 1 })
  })
})
