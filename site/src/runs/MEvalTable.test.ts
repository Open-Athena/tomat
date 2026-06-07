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
  nemdBucket, nmaeBucket, stepsDescOf, type Bucket,
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
