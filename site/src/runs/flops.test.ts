// Tests for `formatFlops`.
//
// Run:  node --test --experimental-strip-types site/src/runs/flops.test.ts
//
// The contract is the prose in the spec: `2.4e20 FLOP → 240 EF`, etc.

import { strict as assert } from 'node:assert'
import { describe, it } from 'node:test'
import { formatFlops } from './flops.format.ts'

describe('formatFlops', () => {
  describe('EF (default unit)', () => {
    it('formats 2.4e20 → "240 EF"', () => {
      assert.equal(formatFlops(2.4e20, 'EF'), '240 EF')
    })
    it('formats 9.0e17 → "0.9 EF"', () => {
      assert.equal(formatFlops(9.0e17, 'EF'), '0.9 EF')
    })
    it('formats 3.1e18 → "3.1 EF"', () => {
      assert.equal(formatFlops(3.1e18, 'EF'), '3.1 EF')
    })
    it('formats 1.0e18 → "1.0 EF"', () => {
      assert.equal(formatFlops(1.0e18, 'EF'), '1.0 EF')
    })
    it('formats 5.5e19 → "55 EF"', () => {
      assert.equal(formatFlops(5.5e19, 'EF'), '55 EF')
    })
    it('default unit is EF when omitted', () => {
      assert.equal(formatFlops(2.4e20), '240 EF')
    })
    it('huge value groups digits with commas', () => {
      // 1.234e22 EF = 12340 EF; rendered with locale grouping for readability.
      assert.equal(formatFlops(1.234e22, 'EF'), '12,340 EF')
    })
  })

  describe('PF', () => {
    it('1e15 → "1.0 PF"', () => {
      assert.equal(formatFlops(1e15, 'PF'), '1.0 PF')
    })
    it('2.4e20 → "240,000 PF"', () => {
      assert.equal(formatFlops(2.4e20, 'PF'), '240,000 PF')
    })
    it('5e14 → "0.5 PF"', () => {
      assert.equal(formatFlops(5e14, 'PF'), '0.5 PF')
    })
  })

  describe('TF', () => {
    it('1e12 → "1.0 TF"', () => {
      assert.equal(formatFlops(1e12, 'TF'), '1.0 TF')
    })
    it('2.4e20 → "240,000,000 TF"', () => {
      assert.equal(formatFlops(2.4e20, 'TF'), '240,000,000 TF')
    })
  })

  describe('sci', () => {
    it('2.4e20 → "2.4e20"', () => {
      assert.equal(formatFlops(2.4e20, 'sci'), '2.4e20')
    })
    it('1.0e18 → "1.0e18"', () => {
      assert.equal(formatFlops(1.0e18, 'sci'), '1.0e18')
    })
    it('matches the legacy display contract (no unit suffix)', () => {
      // The legacy `formatFlops` returned things like "1.1e20"; `sci` is the
      // back-compat option for users who still prefer that form.
      const out = formatFlops(1.1e20, 'sci')
      assert.ok(!out.includes('EF'), `expected no unit suffix, got "${out}"`)
      assert.equal(out, '1.1e20')
    })
  })

  describe('edge cases', () => {
    it('0 → "0"', () => {
      assert.equal(formatFlops(0, 'EF'), '0')
      assert.equal(formatFlops(0, 'sci'), '0')
    })
    it('negative → "0"', () => {
      // FLOP counts are non-negative; treat sub-zero as the sentinel.
      assert.equal(formatFlops(-1, 'EF'), '0')
    })
    it('NaN/Infinity → "0"', () => {
      assert.equal(formatFlops(NaN, 'EF'), '0')
      assert.equal(formatFlops(Infinity, 'EF'), '0')
      assert.equal(formatFlops(-Infinity, 'EF'), '0')
    })
  })
})
