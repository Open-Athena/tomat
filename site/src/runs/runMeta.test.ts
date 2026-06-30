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
import { RUN_LINEAGE, SEGMENTS } from './lineage.ts'
import {
  costBreakdownAtStep,
  epochBreakdownAtStep,
  epochOfStep,
  EPOCH_SEQUENCES,
  perLabelEpochsAtStep,
  segmentBreakdownAtStep,
  totalFlopsAtStep,
  totalTokensAtStep,
} from './runMeta.ts'

// Minimal manifest factory — only the fields `epochOfStep` reads. `name` is
// the SEGMENTS key; `train_batch_size` + `cache_dir` drive the fallback path.
function mkManifest({
  name, batchSize, dataLabel, seqLen, paramCount,
}: {
  name: string
  batchSize?: number
  dataLabel?: string
  seqLen?: number
  paramCount?: number
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
        ...(seqLen != null ? { train_seq_len: seqLen } : {}),
      },
    },
    summary: paramCount != null ? { parameter_count: paramCount } : {},
    history: { rows: 0, step_min: 0, step_max: 0, ts_min: 0, ts_max: 0 },
  }
}

// Test runs we'll register/unregister in SEGMENTS. Use distinctive names so
// they never collide with real prod entries.
const TEST_SINGLE_SEG = 'test:epoch:single-seg'
const TEST_TWO_SEG = 'test:epoch:two-seg'
const TEST_UNKNOWN_LABEL = 'test:epoch:unknown-label'
const TEST_FALLBACK_NAME = 'test:epoch:fallback'
const TEST_PARENT = 'test:lineage:parent'
const TEST_CHILD = 'test:lineage:child'
const TEST_GRANDPARENT = 'test:lineage:grandparent'

afterEach(() => {
  // Strip every test-only entry so tests stay isolated.
  for (const k of [
    TEST_SINGLE_SEG, TEST_TWO_SEG, TEST_UNKNOWN_LABEL,
    TEST_PARENT, TEST_CHILD, TEST_GRANDPARENT,
  ]) {
    delete SEGMENTS[k]
    delete RUN_LINEAGE[k]
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

describe('totalTokensAtStep', () => {
  it('fallback path: step · batch_size · seq_len from the manifest', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
      seqLen: 8192,
    })
    assert.equal(totalTokensAtStep(1000, m), 1000 * 128 * 8192)
  })

  it('returns null when seq_len is missing', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
    })
    assert.equal(totalTokensAtStep(1000, m), null)
  })

  it('segmented: sums per-segment (Δstep · bs · seq_len)', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG, seqLen: 8192 })
    // step 200 = full first seg (100 · 128 · 8192) + full 100 of second (100 · 256 · 8192)
    const expected = 100 * 128 * 8192 + 100 * 256 * 8192
    assert.equal(totalTokensAtStep(200, m), expected)
  })

  it('segmented: clamps end of last segment to `step`', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG, seqLen: 8192 })
    // step 150 = full first seg + 50 steps of second
    const expected = 100 * 128 * 8192 + 50 * 256 * 8192
    assert.equal(totalTokensAtStep(150, m), expected)
  })
})

describe('totalFlopsAtStep', () => {
  it('fallback: 6 · parameter_count · totalTokens', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
      seqLen: 8192, paramCount: 200_000_000,
    })
    const tokens = 1000 * 128 * 8192
    assert.equal(totalFlopsAtStep(1000, m), 6 * 200_000_000 * tokens)
  })

  it('returns null when parameter_count is missing', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
      seqLen: 8192,
    })
    assert.equal(totalFlopsAtStep(1000, m), null)
  })

  it('segmented: 6 · N · (segment-aware tokens)', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({
      name: TEST_TWO_SEG, seqLen: 8192, paramCount: 200_000_000,
    })
    const tokens = 100 * 128 * 8192 + 50 * 256 * 8192
    assert.equal(totalFlopsAtStep(150, m), 6 * 200_000_000 * tokens)
  })
})

describe('segmentBreakdownAtStep', () => {
  it('returns one entry per touched segment with steps / tokens / flops / epochs', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({
      name: TEST_TWO_SEG, seqLen: 8192, paramCount: 200_000_000,
    })
    const out = segmentBreakdownAtStep(150, m)
    assert.ok(out != null)
    assert.equal(out.length, 2)
    assert.equal(out[0].steps, 100)
    assert.equal(out[0].sequences, 100 * 128)
    assert.equal(out[0].tokens, 100 * 128 * 8192)
    assert.equal(out[0].flops, 6 * 200_000_000 * 100 * 128 * 8192)
    assert.equal(out[0].epochs, (100 * 128) / EPOCH_SEQUENCES['train-full-v3'])
    assert.equal(out[1].steps, 50)
    assert.equal(out[1].sequences, 50 * 256)
    assert.equal(out[1].tokens, 50 * 256 * 8192)
    assert.equal(out[1].flops, 6 * 200_000_000 * 50 * 256 * 8192)
    assert.equal(out[1].epochs, (50 * 256) / EPOCH_SEQUENCES['train-full-v3-shard1'])
  })

  it('returns null when no segments are declared', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
      seqLen: 8192,
    })
    assert.equal(segmentBreakdownAtStep(1000, m), null)
  })
})

describe('perLabelEpochsAtStep', () => {
  it('returns null when step ≤ 0', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
    })
    assert.equal(perLabelEpochsAtStep(0, m), null)
    assert.equal(perLabelEpochsAtStep(-1, m), null)
  })

  it('no lineage, no SEGMENTS → single-label fallback from manifest', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
    })
    const out = perLabelEpochsAtStep(1000, m)
    assert.ok(out != null)
    const expected = (1000 * 128) / EPOCH_SEQUENCES['train-full-v3']
    assert.deepEqual(out.byLabel, { 'train-full-v3': expected })
    assert.equal(out.total, expected)
    assert.deepEqual(out.unresolved, [])
  })

  it('child of a SEGMENTS-declared parent: integrates parent up to parent_step, child from parent_step', () => {
    // Parent has two segments (TS1 then TS0123 union), cuts over at step 100.
    SEGMENTS[TEST_PARENT] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3-shard1', batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3+train-full-v3-shard1+train-full-v3-shard2+train-full-v3-shard3', batchSize: 128 },
    ]
    // Child resumed from parent at step 200, continues on the TS0123 union.
    RUN_LINEAGE[TEST_CHILD] = { parent: TEST_PARENT, parent_step: 200 }
    const childManifest = mkManifest({
      name: TEST_CHILD, batchSize: 128,
      dataLabel: 'train-full-v3+train-full-v3-shard1+train-full-v3-shard2+train-full-v3-shard3',
    })
    // step=250 → parent integrated [0,200], child integrated [200,250].
    const out = perLabelEpochsAtStep(250, childManifest)
    assert.ok(out != null)
    const shard1Eps = (100 * 128) / EPOCH_SEQUENCES['train-full-v3-shard1']
    const unionLabel = 'train-full-v3+train-full-v3-shard1+train-full-v3-shard2+train-full-v3-shard3'
    // Two separate contributions: parent [100,200] and child [200,250], same label.
    // Match the impl's summation order to avoid spurious floating-point drift.
    const unionEps =
      (100 * 128) / EPOCH_SEQUENCES[unionLabel]
      + (50 * 128) / EPOCH_SEQUENCES[unionLabel]
    assert.equal(out.byLabel['train-full-v3-shard1'], shard1Eps)
    assert.equal(out.byLabel[unionLabel], unionEps)
    assert.equal(out.total, shard1Eps + unionEps)
    assert.deepEqual(out.unresolved, [])
  })

  it('child without SEGMENTS but with parent: own contribution starts at parent_step', () => {
    // Parent has SEGMENTS (single seg). Child uses manifest fallback.
    SEGMENTS[TEST_PARENT] = [
      { startStep: 0, endStep: Infinity, dataLabel: 'train-full-v3', batchSize: 128 },
    ]
    RUN_LINEAGE[TEST_CHILD] = { parent: TEST_PARENT, parent_step: 1000 }
    const childManifest = mkManifest({
      name: TEST_CHILD, batchSize: 128, dataLabel: 'train-full-v3',
    })
    // step=1500 → parent integrated [0,1000], child integrated [1000,1500].
    const out = perLabelEpochsAtStep(1500, childManifest)
    assert.ok(out != null)
    // Both contribute to the same label → single key in byLabel.
    const expected = (1500 * 128) / EPOCH_SEQUENCES['train-full-v3']
    assert.deepEqual(out.byLabel, { 'train-full-v3': expected })
  })

  it('walks multi-level lineage chains (grandparent → parent → child)', () => {
    SEGMENTS[TEST_GRANDPARENT] = [
      { startStep: 0, endStep: Infinity, dataLabel: 'train-full-v3', batchSize: 128 },
    ]
    SEGMENTS[TEST_PARENT] = [
      { startStep: 0, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 128 },
    ]
    RUN_LINEAGE[TEST_PARENT] = { parent: TEST_GRANDPARENT, parent_step: 100 }
    RUN_LINEAGE[TEST_CHILD] = { parent: TEST_PARENT, parent_step: 200 }
    const childManifest = mkManifest({
      name: TEST_CHILD, batchSize: 128, dataLabel: 'train-full-v3-shard2',
    })
    // grandparent contributes [0,100] @ train-full-v3,
    // parent contributes [100,200] @ shard1,
    // child contributes [200,300] @ shard2.
    const out = perLabelEpochsAtStep(300, childManifest)
    assert.ok(out != null)
    assert.equal(out.byLabel['train-full-v3'],         (100 * 128) / EPOCH_SEQUENCES['train-full-v3'])
    assert.equal(out.byLabel['train-full-v3-shard1'],  (100 * 128) / EPOCH_SEQUENCES['train-full-v3-shard1'])
    assert.equal(out.byLabel['train-full-v3-shard2'],  (100 * 128) / EPOCH_SEQUENCES['train-full-v3-shard2'])
    assert.equal(Object.keys(out.byLabel).length, 3)
  })

  it('records unresolved ancestor when parent_step is missing', () => {
    SEGMENTS[TEST_PARENT] = [
      { startStep: 0, endStep: Infinity, dataLabel: 'train-full-v3', batchSize: 128 },
    ]
    // No parent_step recorded — we don't know how much of parent ran.
    RUN_LINEAGE[TEST_CHILD] = { parent: TEST_PARENT }
    const childManifest = mkManifest({
      name: TEST_CHILD, batchSize: 128, dataLabel: 'train-full-v3',
    })
    const out = perLabelEpochsAtStep(500, childManifest)
    assert.ok(out != null)
    assert.deepEqual(out.unresolved, [TEST_PARENT])
    // Child's own contribution still counted (parent_step=undefined → 0,
    // so child contributes [0, step]).
    assert.equal(out.byLabel['train-full-v3'], (500 * 128) / EPOCH_SEQUENCES['train-full-v3'])
  })

  it('records unresolved ancestor when ancestor has no SEGMENTS entry', () => {
    RUN_LINEAGE[TEST_CHILD] = { parent: TEST_PARENT, parent_step: 100 }
    const childManifest = mkManifest({
      name: TEST_CHILD, batchSize: 128, dataLabel: 'train-full-v3',
    })
    const out = perLabelEpochsAtStep(150, childManifest)
    assert.ok(out != null)
    assert.deepEqual(out.unresolved, [TEST_PARENT])
    // Child's own [100,150] contribution counted on its declared label.
    assert.equal(out.byLabel['train-full-v3'], (50 * 128) / EPOCH_SEQUENCES['train-full-v3'])
  })

  it('reproduces the v4-epochwin spec example (1.97 + 0.76 ≈ 2.73 by label)', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,     endStep: 76186,    dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 76186, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({ name: TEST_TWO_SEG })
    const out = perLabelEpochsAtStep(90982, m)
    assert.ok(out != null)
    assert.equal(out.byLabel['train-full-v3'].toFixed(2),        '1.97')
    assert.equal(out.byLabel['train-full-v3-shard1'].toFixed(2), '0.76')
    assert.equal(out.total.toFixed(2), '2.73')
  })

  it('legacy own-SEGMENTS encodes parent+child (v4-epochwin pattern): doesn\'t clamp to parent_step', () => {
    // v4-epochwin's SEGMENTS predate the lineage walk and encode the FULL
    // parent+child trajectory (seg #1 starts at 0, covers parent-era TS0).
    // Even when a parent is declared via RUN_LINEAGE, we must NOT clamp
    // the own-SEGMENTS to parent_step (would drop the parent-era slice).
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,     endStep: 76186,    dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 76186, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    // Parent has no SEGMENTS — its contribution is conceptually inside seg #1.
    RUN_LINEAGE[TEST_TWO_SEG] = { parent: TEST_PARENT, parent_step: 40000 }
    const m = mkManifest({ name: TEST_TWO_SEG })
    const out = perLabelEpochsAtStep(90982, m)
    assert.ok(out != null)
    // Same numbers as without the lineage entry — seg #1 still integrates [0, 76186].
    assert.equal(out.byLabel['train-full-v3'].toFixed(2),        '1.97')
    assert.equal(out.byLabel['train-full-v3-shard1'].toFixed(2), '0.76')
    // Parent ancestor flagged as unresolved (no SEGMENTS, no manifest).
    assert.deepEqual(out.unresolved, [TEST_PARENT])
  })
})

describe('costBreakdownAtStep', () => {
  it('splits total MSRP across segments proportional to tokens', () => {
    SEGMENTS[TEST_TWO_SEG] = [
      { startStep: 0,   endStep: 100,      dataLabel: 'train-full-v3',        batchSize: 128 },
      { startStep: 100, endStep: Infinity, dataLabel: 'train-full-v3-shard1', batchSize: 256 },
    ]
    const m = mkManifest({
      name: TEST_TWO_SEG, seqLen: 8192, paramCount: 200_000_000,
    })
    const totalCost = 300
    const out = costBreakdownAtStep(150, m, totalCost)
    assert.ok(out != null)
    assert.equal(out.length, 2)
    const tok0 = 100 * 128 * 8192
    const tok1 = 50 * 256 * 8192
    const total = tok0 + tok1
    assert.equal(out[0].msrp_usd, (tok0 / total) * totalCost)
    assert.equal(out[1].msrp_usd, (tok1 / total) * totalCost)
    // The shares must sum back to the total.
    assert.equal(out[0].msrp_usd + out[1].msrp_usd, totalCost)
  })

  it('returns null when no segments are declared', () => {
    const m = mkManifest({
      name: TEST_FALLBACK_NAME, batchSize: 128, dataLabel: 'train-full-v3',
      seqLen: 8192,
    })
    assert.equal(costBreakdownAtStep(1000, m, 100), null)
  })
})
