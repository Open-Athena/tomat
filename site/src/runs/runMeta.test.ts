// Tests for `epochOfStep` + `epochBreakdownAtStep` — the segment-aware epoch
// math that backs the run-detail header chip's `epoch X.YY` display.
//
// Run:  node --test --experimental-strip-types site/src/runs/runMeta.test.ts
//
// Strategy: drive `epochOfStep` via the public `SEGMENTS` table in
// `lineage.ts`. We register a synthetic run name in `SEGMENTS` for each
// scenario (and clean it up after) so tests don't depend on prod entries.
// The fallback path (no segments declared) reads `train_batch_size` + the
// data label from a hand-built minimal `RunManifest`.

import { strict as assert } from 'node:assert'
import { afterEach, describe, it } from 'node:test'
import type { RunManifest } from './api.ts'
import { SEGMENTS } from './lineage.ts'
import { epochBreakdownAtStep, epochOfStep, EPOCH_SEQUENCES } from './runMeta.ts'

// Minimal manifest factory — only the fields `epochOfStep` reads. `name` is
// the SEGMENTS key; `train_batch_size` + `cache_dir` drive the fallback path.
function mkManifest({
  name, batchSize, dataLabel,
}: {
  name: string
  batchSize?: number
  dataLabel?: string
}): RunManifest {
  return {
    schema_version: 1,
    synced_at: '',
    run: {
      id: name, name, project: 'p', entity: 'e', state: 'finished',
      url: '', created_at: '', tags: [], group: null,
      config: {
        ...(batchSize != null ? { trainer: { train_batch_size: batchSize } } : {}),
        ...(dataLabel != null ? { data: { cache_dir: `/x/cache/${dataLabel}/` } } : {}),
      },
    },
    summary: {},
    history: { rows: 0, step_min: 0, step_max: 0, ts_min: 0, ts_max: 0 },
  }
}

// Test runs we'll register/unregister in SEGMENTS. Use distinctive names so
// they never collide with real prod entries.
const TEST_SINGLE_SEG = 'test:epoch:single-seg'
const TEST_TWO_SEG = 'test:epoch:two-seg'
const TEST_UNKNOWN_LABEL = 'test:epoch:unknown-label'
const TEST_FALLBACK_NAME = 'test:epoch:fallback'

afterEach(() => {
  // Strip every test-only entry so tests stay isolated.
  for (const k of [TEST_SINGLE_SEG, TEST_TWO_SEG, TEST_UNKNOWN_LABEL]) {
    delete SEGMENTS[k]
  }
})

describe('epochOfStep — fallback (no SEGMENTS entry)', () => {
  it('uses single-(BS, label) math from the manifest', () => {
    // train-full-v3 has 4,954,176 rows; BS=128, step=1000 → 1000*128 / 4954176
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
    })
    const expected = (1000 * 128) / EPOCH_SEQUENCES['train-full-v3']
    assert.equal(epochOfStep(1000, m), expected)
  })

  it('returns null when train_batch_size is missing', () => {
    const m = mkManifest({ name: TEST_FALLBACK_NAME, dataLabel: 'train-full-v3' })
    assert.equal(epochOfStep(1000, m), null)
  })

  it('returns null when the data label is unknown', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'no-such-label',
    })
    assert.equal(epochOfStep(1000, m), null)
  })

  it('returns null on a null manifest', () => {
    assert.equal(epochOfStep(1000, null), null)
  })
})

describe('epochOfStep — segmented runs', () => {
  it('sums the per-segment epoch contributions', () => {
    // Segment A: 0..100, BS=128, train-full-v3   → 100*128 / 4954176
    // Segment B: 100..200, BS=256, train-full-v3-shard1 → 100*256 / 4964352
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    const expected =
      (100 * 128) / EPOCH_SEQUENCES['train-full-v3']
      + (100 * 256) / EPOCH_SEQUENCES['train-full-v3-shard1']
    assert.equal(epochOfStep(200, m), expected)
  })

  it('clamps a step partway through a segment to that step', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    // step=150 → full first seg + half of second's "100..150" → 50 steps at BS=256.
    const expected =
      (100 * 128) / EPOCH_SEQUENCES['train-full-v3']
      + (50 * 256) / EPOCH_SEQUENCES['train-full-v3-shard1']
    assert.equal(epochOfStep(150, m), expected)
  })

  it('reproduces the v4-epochwin numbers from the spec (1.97 + 0.76 ≈ 2.73)', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,     endStep: 76186,    dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 76186, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    const epAtBoundary = epochOfStep(76186, m)
    assert.ok(epAtBoundary != null)
    // 1.97 epochs at the segment boundary (spec).
    assert.equal(epAtBoundary.toFixed(2), '1.97')
    // 2.73 epochs at step 90982 (spec).
    const epAtCur = epochOfStep(90982, m)
    assert.ok(epAtCur != null)
    assert.equal(epAtCur.toFixed(2), '2.73')
  })

  it('returns null when a declared segment references an unknown label', () => {
    SEGMENTS[TEST_UNKNOWN_LABEL] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3', batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'not-a-label',   batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_UNKNOWN_LABEL })
    // step lands inside the unknown-label segment → null (don't guess).
    assert.equal(epochOfStep(150, m), null)
  })

  it('a single-segment SEGMENTS entry behaves identically to the fallback', () => {
    SEGMENTS[TEST_SINGLE_SEG] = [
      { startStep: 0, endStep: Infinity, dataLabel: 'train-full-v3', batchSize: 128 },
    ]
    const m = mkManifest({ name: TEST_SINGLE_SEG })
    const expected = (1000 * 128) / EPOCH_SEQUENCES['train-full-v3']
    assert.equal(epochOfStep(1000, m), expected)
  })
})

describe('epochBreakdownAtStep', () => {
  it('returns null for runs with no segments declared', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
    })
    assert.equal(epochBreakdownAtStep(1000, m), null)
  })

  it('returns one entry per touched segment, clamped at step', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    const breakdown = epochBreakdownAtStep(150, m)
    assert.ok(breakdown != null)
    assert.equal(breakdown.length, 2)
    assert.equal(breakdown[0].endStep, 100)
    assert.equal(breakdown[1].endStep, 150)
    assert.equal(breakdown[0].epochs, (100 * 128) / EPOCH_SEQUENCES['train-full-v3'])
    assert.equal(breakdown[1].epochs, (50 * 256) / EPOCH_SEQUENCES['train-full-v3-shard1'])
  })

  it('omits segments that haven\'t started yet at `step`', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    const breakdown = epochBreakdownAtStep(50, m)
    assert.ok(breakdown != null)
    assert.equal(breakdown.length, 1)
    assert.equal(breakdown[0].endStep, 50)
  })
})
