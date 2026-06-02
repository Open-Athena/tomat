// Pure data / formatting helpers for the /runs dashboard. Shared between
// `RunsPage` (the index/card view) and `RunHeaderRich` (the per-run rich
// header rendered on both the card and the detail page).
//
// No JSX in here — everything is plain functions on RunManifest / RunHistory.
// Components that compose these helpers live in `RunHeaderRich.tsx`.

import type { RunManifest } from './api'
import type { RunHistory } from './parquet'

export const LOOKBACK_HOURS = 6
export const LOOKBACK_SEC = LOOKBACK_HOURS * 3600

/**
 * Compute steps completed within a wallclock window ending at the latest log.
 * Returns `null` if we don't have enough samples — common right after a fresh
 * resume, when the parquet's latest row was logged < windowSec ago.
 *
 * Step = `global_step` (Levanter's actual training step), not `_step` (wandb's
 * log-call counter) — the latter inflates 5–10× per row and isn't what users
 * mean when they ask "how many steps in the last hour?"
 *
 * Restricts to the latest monotonic-increasing segment so crash-loop runs
 * (whose parquets can contain interleaved rows from multiple attempts that
 * `tomat runs sync`'s last-reset-only trim doesn't fully clean up) don't show
 * a negative rate. Walks ts-desc from the latest row, accepting only rows
 * whose `global_step` is ≤ the running minimum — anything higher came from a
 * pre-restart attempt and is excluded.
 */
export function stepsInWindow(history: RunHistory, windowSec: number): number | null {
  const tsCol = history.timestamps
  const gsCol = history.cols.get('global_step') ?? []
  // Collect (ts, gs) for rows where both are set, sort ts-asc.
  const pts: { t: number; s: number }[] = []
  for (let i = 0; i < history.rowCount; i++) {
    const t = tsCol[i], s = gsCol[i]
    if (t == null || s == null) continue
    pts.push({ t, s })
  }
  if (pts.length === 0) return null
  pts.sort((a, b) => a.t - b.t)
  const latest = pts[pts.length - 1]
  // Walk ts-desc; keep rows whose gs ≤ running min. The running min starts at
  // latest.s and only decreases as we walk back, so any row with a higher gs
  // belongs to a pre-restart attempt (and gets dropped).
  let minStep = latest.s
  const seg: { t: number; s: number }[] = [latest]
  for (let i = pts.length - 2; i >= 0; i--) {
    const p = pts[i]
    if (p.s <= minStep) { seg.push(p); minStep = p.s }
  }
  if (seg.length < 2) return null
  // seg is ts-desc; the earliest accepted row is at the end.
  const cutoff = latest.t - windowSec
  // Latest row at-or-before the cutoff within the segment (= base for window).
  // If the segment doesn't extend back to the cutoff, fall back to the
  // earliest row in the segment — return whatever fits, rather than null.
  let base = seg[seg.length - 1]
  for (let i = 0; i < seg.length; i++) {
    const p = seg[i]
    if (p.t <= cutoff) { base = p; break }
  }
  return latest.s - base.s
}

/** Filter history to last N seconds + downsample (max ~120 points for sparkline). */
export function recentStepPoints(history: RunHistory): { x: number; y: number }[] {
  const tsCol = history.timestamps, stepCol = history.steps
  const now = Date.now() / 1000
  const cutoff = now - LOOKBACK_SEC
  const all: { x: number; y: number }[] = []
  for (let i = 0; i < history.rowCount; i++) {
    const t = tsCol[i], s = stepCol[i]
    if (t != null && s != null && t >= cutoff) {
      all.push({ x: t, y: s })
    }
  }
  // Sort by timestamp — parquet row order isn't guaranteed strictly
  // chronological (wandb syncs in chunks, interleaves system + training
  // metrics), so connecting points in row order produces a zigzag.
  all.sort((a, b) => a.x - b.x)
  if (all.length <= 120) return all
  // Downsample by stride
  const stride = Math.ceil(all.length / 120)
  const out: { x: number; y: number }[] = []
  for (let i = 0; i < all.length; i += stride) out.push(all[i])
  if (out[out.length - 1] !== all[all.length - 1]) out.push(all[all.length - 1])
  return out
}

/** "Xs ago" / "Xm ago" / "Xh ago" / "Xd ago"; for unix-second timestamps. */
export function secsAgo(unixSec: number): string {
  const ago = Date.now() / 1000 - unixSec
  if (ago < 60) return `${Math.round(ago)}s ago`
  if (ago < 3600) return `${Math.round(ago / 60)}m ago`
  if (ago < 86400) return `${(ago / 3600).toFixed(1)}h ago`
  return `${(ago / 86400).toFixed(1)}d ago`
}

/** Freshness color: green < 2 min, yellow < 30 min, red older. */
export function freshnessColor(secAgo: number): string {
  if (secAgo < 120) return '#22863a'
  if (secAgo < 1800) return '#d4a017'
  return '#cb2431'
}

/** "Xs ago" / "Xm ago" / "Xh ago" / "Xd ago"; for ISO-timestamp strings. */
export function timeAgo(iso: string): string {
  const t = new Date(iso).getTime()
  if (!isFinite(t)) return ''
  const sec = (Date.now() - t) / 1000
  if (sec < 60) return `${Math.round(sec)}s ago`
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`
  if (sec < 86400) return `${(sec / 3600).toFixed(1)}h ago`
  return `${(sec / 86400).toFixed(1)}d ago`
}

// Parse run name into structured fields. Tolerant of unknown patterns: anything
// we can't recognize lands in `extra`.
export interface RunMeta {
  model: string | null      // "200M", "1B"
  batchSize: number | null
  lossType: string | null   // "emd-do", "ce"
  targetSteps: string | null // "10k", "500"
  hardware: string | null   // "v5p-16", "v6e-16", "H100×8"
  hardwareKind: 'tpu-v5p' | 'tpu-v6e' | 'modal-gpu' | null
  suffix: string | null     // "z3", "cont7k-ext", etc.
}

export function parseRunName(name: string): RunMeta {
  // train-full-<dataset>-<model>-bs<n>-<loss>-do-<steps>-<hw>-shuf1k-<suffix>
  const m = name.match(
    /^train-(?:full|cont)-(?:[a-z0-9-]+?-)?(\d+[MBK])-bs(\d+)-([a-z-]+?)-(?:do-)?(\d+k?|\d+)-([a-z0-9]+)(?:-shuf\d+k)?(?:-(.+))?$/i,
  )
  if (!m) {
    return {
      model: null, batchSize: null, lossType: null, targetSteps: null,
      hardware: null, hardwareKind: null, suffix: name,
    }
  }
  const [, model, bs, loss, steps, hwRaw, suffix] = m
  let hw: string | null = null
  let kind: RunMeta['hardwareKind'] = null
  const hwLower = hwRaw.toLowerCase()
  if (hwLower.startsWith('v5p')) { hw = `v5p-${hwLower.slice(3)}`; kind = 'tpu-v5p' }
  else if (hwLower.startsWith('v6e')) { hw = `v6e-${hwLower.slice(3)}`; kind = 'tpu-v6e' }
  else if (hwLower.startsWith('tpu')) { hw = `v6e-${hwLower.slice(3)}`; kind = 'tpu-v6e' }
  // slice(5) skips e.g. "h100x" (5 chars) — slice(4) leaked the 'x' through
  // giving "H100×x8" instead of "H100×8".
  else if (hwLower.startsWith('h100x')) { hw = `H100×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  else if (hwLower.startsWith('h200x')) { hw = `H200×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  else if (hwLower.startsWith('b200x')) { hw = `B200×${hwLower.slice(5)}`; kind = 'modal-gpu' }
  return {
    model, batchSize: parseInt(bs, 10),
    lossType: loss === 'emd-do' ? 'EMD density-only'
      : loss === 'l1-do' ? 'L1 density-only'
      : loss === 'ce' ? 'CE'
      : loss,
    targetSteps: steps,
    hardware: hw, hardwareKind: kind,
    suffix: suffix ?? null,
  }
}

export const HW_COLORS: Record<NonNullable<RunMeta['hardwareKind']>, string> = {
  'tpu-v5p': '#7c4dff',
  'tpu-v6e': '#00838f',
  'modal-gpu': '#f57c00',
}

export function parseTargetSteps(s: string): number {
  // "10k" → 10000, "500" → 500
  const m = s.match(/^(\d+)(k|K)?$/)
  if (!m) return Number.POSITIVE_INFINITY
  return parseInt(m[1], 10) * (m[2] ? 1000 : 1)
}

/** Compact step count: 80000 → "80k", 1234 → "1,234". */
export function formatStepCount(n: number): string {
  if (n >= 1000 && n % 1000 === 0) return `${n / 1000}k`
  if (n >= 10000) return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`
  return n.toLocaleString()
}

/** Authoritative step target: `trainer.num_train_steps` from the run config. */
export function numTrainStepsOf(manifest: RunManifest | null): number | null {
  const v = (manifest?.run.config as Record<string, unknown> | undefined)?.trainer
  if (v && typeof v === 'object' && 'num_train_steps' in v) {
    const n = (v as { num_train_steps?: unknown }).num_train_steps
    if (typeof n === 'number' && n > 0) return n
  }
  return null
}

// ── n_epochs / n_flops ──────────────────────────────────────────────────────
// Sequences per epoch (Levanter cache `total_num_rows`), keyed by data label —
// the `cache/<label>/` segment of `config.data.cache_dir`. From each cache's
// `train/shard_ledger.json`. TODO: fold into the manifest so a new data label
// doesn't need a code change here.
export const EPOCH_SEQUENCES: Record<string, number> = {
  'train-full-v3': 4954176,
  // Globally pre-shuffled copy of `train-full-v3` (spec 50). Same row count,
  // same shard structure; just permuted. Pick this up via `TOMAT_LABEL` to
  // eliminate the 1024-step BlockShuffleConfig sawtooth (spec 44).
  'train-full-v3-shuffled': 4954176,
}

/** Data label (tokenization) of a run.
 *
 *  Two layouts in production:
 *   - iris/TPU: `cache_dir = ".../cache/<label>/"` — the canonical Levanter
 *     cache path; label is the trailing segment.
 *   - Modal: `cache_dir = "/vol/results/<run>/cache"` (the Modal-volume
 *     output cache, NOT named for the data label) AND `train_urls` like
 *     `/vol/tokenized/<label>/worker-*​/*.parquet` — label is in the
 *     URL's `/tokenized/<label>/` segment.
 *
 *  Falls back from the canonical cache_dir parse to the train_urls parse so
 *  Modal-trained runs still surface `nEpochs` + the `?x=epoch` axis. */
export function dataLabelOf(manifest: RunManifest | null): string | null {
  const data = (manifest?.run.config as Record<string, unknown> | undefined)?.data
  if (!data || typeof data !== 'object') return null
  const cd = (data as { cache_dir?: unknown }).cache_dir
  if (typeof cd === 'string') {
    const m = cd.match(/\/cache\/([^/]+)\/?$/)
    if (m) return m[1]
  }
  // Modal fallback: `components.tomat.source.train_urls[0]` contains the
  // pre-tokenization parquet path, which DOES embed the label.
  const comps = (data as { components?: { tomat?: { source?: { train_urls?: unknown } } } }).components
  const urls = comps?.tomat?.source?.train_urls
  if (Array.isArray(urls) && urls.length > 0 && typeof urls[0] === 'string') {
    const m = (urls[0] as string).match(/\/tokenized\/([^/]+)\//)
    if (m) return m[1]
  }
  return null
}

/** Tokens trained on so far (`throughput/total_tokens` from the run summary). */
export function totalTokensOf(manifest: RunManifest | null): number | null {
  const t = manifest?.summary['throughput/total_tokens']
  return typeof t === 'number' && t > 0 ? t : null
}

/** Forward+backward FLOPs — the standard 6·N·D estimate (N params, D tokens). */
export function nFlopsOf(manifest: RunManifest | null): number | null {
  const n = manifest?.summary['parameter_count']
  const tok = totalTokensOf(manifest)
  if (typeof n !== 'number' || tok == null) return null
  return 6 * n * tok
}

/** Training sequence length (`train_seq_len` from the run config). */
export function seqLenOf(manifest: RunManifest | null): number | null {
  const v = (manifest?.run.config as Record<string, unknown> | undefined)?.train_seq_len
  return typeof v === 'number' && v > 0 ? v : null
}

/** Train batch size (`trainer.train_batch_size`). */
export function batchSizeOf(manifest: RunManifest | null): number | null {
  const t = (manifest?.run.config as Record<string, unknown> | undefined)?.trainer
  if (t && typeof t === 'object' && 'train_batch_size' in t) {
    const n = (t as { train_batch_size?: unknown }).train_batch_size
    if (typeof n === 'number' && n > 0) return n
  }
  return null
}

/** Sequences per epoch for the run's data label, from `EPOCH_SEQUENCES`. */
export function epochSequencesOf(manifest: RunManifest | null): number | null {
  const label = dataLabelOf(manifest)
  if (label == null) return null
  return EPOCH_SEQUENCES[label] ?? null
}

/** Passes over the training set = tokens / (epoch_sequences · seq_len).
 *  Null when the data label's epoch size isn't in EPOCH_SEQUENCES. */
export function nEpochsOf(manifest: RunManifest | null): number | null {
  const tok = totalTokensOf(manifest)
  const seqLen = seqLenOf(manifest)
  const epochSeqs = epochSequencesOf(manifest)
  if (tok == null || seqLen == null || epochSeqs == null) return null
  return tok / (epochSeqs * seqLen)
}

/** Fractional epoch count at training `step`: `step · batch_size / epoch_seqs`.
 *  Pure rescaling of `step` (same monotonicity / linearity, just a different
 *  unit). Algebraically equivalent to `nEpochsOf` evaluated at `step` (the
 *  chip on the run card / detail page) — using `train_batch_size` instead of
 *  `total_tokens` so we can compute the epoch coordinate for ANY step, not
 *  just the latest one in the summary.
 *
 *  Returns null when the manifest doesn't carry the data needed —
 *  `train_batch_size` missing, data label unknown, or no `EPOCH_SEQUENCES`
 *  entry for that label. */
export function epochOfStep(step: number, manifest: RunManifest | null): number | null {
  const bs = batchSizeOf(manifest)
  const epochSeqs = epochSequencesOf(manifest)
  if (bs == null || epochSeqs == null) return null
  return (step * bs) / epochSeqs
}

/** FLOP count → "1.1e20". */
export function formatFlops(f: number): string {
  const exp = Math.floor(Math.log10(f))
  return `${(f / 10 ** exp).toFixed(1)}e${exp}`
}

/** Tokenizer generation, from the authoritative wandb project name
 *  (`tomat-…-P14` = v2 patch tokenizer, `…-P19` = v3). The run *name* is not
 *  reliable — some `train-full-v3-…` runs trained on v2 data (project P14). */
export function tokenizerGen(manifest: RunManifest | null): 'v2' | 'v3' | null {
  const p = manifest?.run.project ?? ''
  if (/P14/.test(p)) return 'v2'
  if (/P19/.test(p)) return 'v3'
  return null
}
