// Tests for the run-name short registry. Run via:
//   node --test --experimental-strip-types site/src/lib/runNames.test.ts
//
// The registry at `runs/registry.json` is the source of truth for vetted
// short names; functions here mirror `marin/run_names.py` exactly.

import { strict as assert } from 'node:assert'
import { describe, it } from 'node:test'

import {
  shortenRun,
  expandRun,
  shortenStep,
  expandStep,
  shortenSpec,
  expandSpec,
  snapStep,
  formatStep,
  formatStepDetail,
} from './runNames.ts'

describe('shortenRun', () => {
  it('returns registered short for curated canonical', () => {
    assert.equal(shortenRun('train-mg-kl-bin5-fs-tpu'), 'bin5')
    assert.equal(shortenRun('train-mg-4-cos-ce'), 'mg-4-cos-ce')
  })
  it('falls back to regex when not in registry', () => {
    // `train-mg-<seg>(-<seg>)?-…` with no registry entry.
    assert.equal(shortenRun('train-mg-uncurated-foo'), 'uncurated')
    assert.equal(shortenRun('train-mg-uncurated-foo-bar'), 'uncurated-foo')
  })
  it('truncates totally unknown shape to 32 chars', () => {
    const weird = 'totally-unparseable-' + 'x'.repeat(40)
    assert.equal(shortenRun(weird).length, 32)
  })
})

describe('expandRun', () => {
  it('round-trips registered shorts', () => {
    assert.equal(expandRun('bin5'), 'train-mg-kl-bin5-fs-tpu')
    assert.equal(expandRun('cont7k-ext'), 'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k-ext')
  })
  it('returns null for unregistered shorts', () => {
    assert.equal(expandRun('totally-fake'), null)
  })
})

describe('snapStep', () => {
  it('snaps near-round values to the round value', () => {
    assert.equal(snapStep(89999), 90000)
    assert.equal(snapStep(49999), 50000)
    assert.equal(snapStep(50001), 50000)
    assert.equal(snapStep(9999), 10000)
    assert.equal(snapStep(1_000_001), 1_000_000)
  })
  it('leaves non-round values unchanged', () => {
    assert.equal(snapStep(53000), 53000)
    assert.equal(snapStep(1234), 1234)
    assert.equal(snapStep(0), 0)
    assert.equal(snapStep(500), 500)
  })
})

describe('formatStep', () => {
  it('suffixes snapped values with *', () => {
    assert.equal(formatStep(50000), '50k')
    assert.equal(formatStep(49999), '50k*')
    assert.equal(formatStep(89999), '90k*')
  })
  it('renders mid-target snap-eligible values per tolerance', () => {
    assert.equal(formatStep(53000), '53k')
    // 52999 is within 0.1% of 53000 (Δ/raw ≈ 1.9e-5), so it snaps and the
    // display gets the * suffix.
    assert.equal(formatStep(52999), '53k*')
    assert.equal(formatStep(1_500_000), '1.5M')
  })
})

describe('formatStepDetail', () => {
  it('flags snapped values + builds an explanatory tooltip', () => {
    const d = formatStepDetail(89999)
    assert.equal(d.display, '90k*')
    assert.equal(d.isSnapped, true)
    assert.equal(d.raw, 89999)
    assert.equal(d.snapped, 90000)
    assert.ok(d.tooltip.includes('info.step'))
    assert.ok(d.tooltip.includes('0-indexed'))
    assert.ok(d.tooltip.includes('89,999'))
    assert.ok(d.tooltip.includes('90,000'))
  })
  it('renders clean values without * and a plain tooltip', () => {
    const d = formatStepDetail(50000)
    assert.equal(d.display, '50k')
    assert.equal(d.isSnapped, false)
    assert.equal(d.tooltip, 'step 50,000')
  })
})

describe('shortenStep / expandStep', () => {
  it('shortens common step counts', () => {
    assert.equal(shortenStep(30000), '30k')
    assert.equal(shortenStep(89999), '90k*')
    assert.equal(shortenStep(100000), '100k')
    assert.equal(shortenStep(1_500_000), '1.5M')
    assert.equal(shortenStep(500), '500')
  })
  it('expands step shorts (with or without ≈ prefix or * suffix)', () => {
    assert.equal(expandStep('30k'), 30000)
    assert.equal(expandStep('1.5M'), 1_500_000)
    assert.equal(expandStep('500'), 500)
    assert.equal(expandStep('≈90k'), 90000)  // legacy
    assert.equal(expandStep('90k*'), 90000)
    assert.equal(expandStep('not-a-step'), null)
  })
})

describe('shortenSpec / expandSpec', () => {
  it('shortens gt specs', () => {
    assert.equal(shortenSpec('gt:val'), 'GT (val)')
    assert.equal(shortenSpec('gt:train'), 'GT (train)')
  })
  it('shortens pred specs (maskgit + oneshot)', () => {
    assert.equal(
      shortenSpec('pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000'),
      'bin5 @ 30k',
    )
    assert.equal(
      shortenSpec('pred:train-mg-kl-bin5-fs-tpu:val_200-oneshot:30000'),
      'bin5 @ 30k-K1',
    )
  })
  it('expands gt shorts', () => {
    assert.equal(expandSpec('GT (val)'), 'gt:val')
    assert.equal(expandSpec('GT (train)'), 'gt:train')
  })
  it('round-trips registered pred specs', () => {
    const spec = 'pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000'
    assert.equal(expandSpec(shortenSpec(spec)), spec)
    const oneshot = 'pred:train-mg-kl-bin5-fs-tpu:val_200-oneshot:30000'
    assert.equal(expandSpec(shortenSpec(oneshot)), oneshot)
  })
  it('returns null when expanding an unregistered run-short', () => {
    assert.equal(expandSpec('totallyfake @ 30k'), null)
  })
})
