// Multi-term regex filter for the /runs dashboard.
//
// Previously the input was a SINGLE case-insensitive regex matched against
// `[shortLabel, ...tags].join(' ')`. That's expressive for one axis but awkward
// for compound queries — "TPU AND scratch", "SS-sweep AND created-on-260527" —
// because a literal space in a regex still means "space character".
//
// New syntax (whitespace = AND, `|` keeps regex-OR semantics within a term):
//
//   noprm                          → existing single-term behaviour
//   noprm paired-base              → BOTH must match (AND across terms)
//   noprm|paired-base              → one term, regex OR (matches either)
//   SS-sweep 260527                → SS-sweep cells AND created 2026-05-27
//   TPU resume|scratch             → TPU runs AND (resume OR scratch)
//
// Per-term semantics: each whitespace-separated token compiles to a
// case-insensitive RegExp. Any term failing to compile → whole filter
// invalid (same red-border UX as the old single-regex error path; the
// dashboard falls through to "no filtering applied").
//
// The haystack each filter runs against is built per-run by `runHaystack`
// below — it folds shortLabel, curated tags, dates, hardware, and lineage
// keywords into one string, so a regex like `TPU 260527` can hit hardware
// and date in one query without the caller having to special-case fields.
//
// Why no test runner is wired up: site/ has no vitest/jest config. The
// expected-behaviour spec lives in the EXAMPLES block at the bottom of this
// file as a static const that's read by `tsc` (so it can't drift undetected
// from the public API).

import type { IrisJob, RunManifest } from './api'
import type { RunHistory } from './parquet'
import { tagsFor } from './tags'

// ── compileMultiTermFilter ──────────────────────────────────────────────────

export interface MultiTermFilter {
  /** True iff every term's regex matches the haystack. */
  matches: (haystack: string) => boolean
  /** True iff any term failed to compile (filter is invalid). */
  error: boolean
  /** True iff the trimmed input was empty (no filtering should apply). */
  empty: boolean
}

/** Compile a multi-term whitespace-separated filter input into an
 *  AND-of-regex matcher. Each term is case-insensitive. An invalid regex in
 *  ANY term marks the entire filter `error: true` and falls through to a
 *  no-op `matches` (caller treats `error` as "no filtering applied").
 *
 *  Empty terms from leading/trailing/double whitespace are dropped silently:
 *  `"  foo   bar  "` → ["foo", "bar"], which keeps the matcher tolerant of
 *  the user mid-typing without flickering the red-border state. */
export function compileMultiTermFilter(input: string): MultiTermFilter {
  const trimmed = input.trim()
  if (trimmed === '') {
    return { matches: () => true, error: false, empty: true }
  }
  // `.split(/\s+/)` after trim → no empty leading/trailing tokens; internal
  // runs of whitespace collapse. (Splitting before trim would leak ''s.)
  const terms = trimmed.split(/\s+/).filter((t) => t.length > 0)
  if (terms.length === 0) {
    return { matches: () => true, error: false, empty: true }
  }
  const regexes: RegExp[] = []
  for (const t of terms) {
    try { regexes.push(new RegExp(t, 'i')) }
    catch { return { matches: () => false, error: true, empty: false } }
  }
  return {
    matches: (h: string) => {
      for (const re of regexes) if (!re.test(h)) return false
      return true
    },
    error: false,
    empty: false,
  }
}

// ── runHaystack: assemble per-run searchable text ──────────────────────────

/** 6-char YYMMDD for a Unix-seconds timestamp, in UTC.
 *
 *  UTC (not local TZ) so the same query string filters consistently for any
 *  viewer. The dashboard's other timestamp uses (sparkline, "Xm ago") are
 *  relative durations, immune to this choice; this is the only place we
 *  expose a specific calendar date as a token. */
function yymmdd(sec: number): string {
  const d = new Date(sec * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  // %y%m%d in UTC; e.g. 2026-05-27 → "260527"
  return `${String(d.getUTCFullYear() % 100).padStart(2, '0')}`
    + `${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}`
}

/** Same for an ISO datetime string (`manifest.run.created_at`). Returns ''
 *  on parse failure so the caller can `.filter(Boolean)` cheaply. */
function yymmddIso(iso: string | undefined | null): string {
  if (!iso) return ''
  const t = Date.parse(iso)
  if (!isFinite(t)) return ''
  return yymmdd(t / 1000)
}

/** Last-activity timestamp from history (max `_timestamp`). Null if the
 *  parquet hasn't loaded yet OR is empty. */
function lastActivitySec(history: RunHistory | null): number | null {
  if (!history) return null
  let maxT: number | null = null
  for (const t of history.timestamps) {
    if (t == null) continue
    if (maxT == null || t > maxT) maxT = t
  }
  return maxT
}

/** TPU prefixes we recognize. Kept in sync with the iris cluster YAML and
 *  `parseRunName` in RunsPage. `tpu` is a legacy generic prefix used by
 *  some older runs (we map to v6e there too). */
const TPU_PREFIXES = ['v5p', 'v5e', 'v5', 'v6e', 'tpu']

/** Detect TPU/GPU/Modal-ness for a run. Sources, in order of authority:
 *    1. iris job's job-name slug (already authoritative for queued/running)
 *    2. parsed run name (`v5p-16`, `h100x8`, etc.)
 *    3. fallback heuristic: no iris job + a manifest → Modal/GPU origin
 *
 *  Returns a set of haystack tokens (`TPU`, `GPU`, `modal`, plus a literal
 *  hardware string like `v6e-16`, `h100x8`). Multiple synonyms ride along
 *  so any of `TPU` / `v6e` / `v6e-16` will hit. */
function hardwareTokens(runId: string, job: IrisJob | null): string[] {
  const out: string[] = []
  const lower = runId.toLowerCase()

  // Pull a hardware-y slug out of the run id. The name parser in RunsPage is
  // strict (only matches the canonical `train-…-<hw>-shuf…` schema), so for
  // freer-form names (`train-ss-cont80k-…`, `vqvae-…`) we also do a loose
  // regex for any vN[pe]?-N / hNNNxN / bNNNxN / tpuN substring.
  const hwRe = /\b(v5p|v5e|v5|v6e|tpu|h100x|h200x|b200x)(\d+)\b/i
  const m = hwRe.exec(lower)
  if (m) {
    const prefix = m[1].toLowerCase()
    const n = m[2]
    if (prefix === 'h100x' || prefix === 'h200x' || prefix === 'b200x') {
      // `h100x8` → modal/GPU
      out.push('GPU', 'modal', `${prefix}${n}`)
    } else {
      // `v6e16` / `v5p-16` / `tpu16` → TPU
      out.push('TPU', `${prefix}-${n}`, `${prefix}${n}`)
      if (prefix === 'tpu') out.push('v6e', `v6e-${n}`)  // legacy alias
    }
  }

  // If we got TPU from the name, we're done. Otherwise, if there's NO iris
  // job and the run has a manifest, it's almost certainly a Modal-origin run
  // (iris is the TPU scheduler; Modal runs never appear in iris's snapshot).
  // The caller passes `job=null` for Modal runs.
  if (out.length === 0 && !job) {
    out.push('GPU', 'modal')
  }

  // Also expose any iris-known TPU prefix directly in case the run name
  // doesn't have it (some `train-ss-*` runs encode hw in the suffix).
  // Currently the IrisJob payload has no `.tpu` field — we keep the hook
  // here so adding one later doesn't require touching the haystack code.
  if (job && TPU_PREFIXES.some((p) => lower.startsWith(p))) {
    out.push('TPU')
  }

  return out
}

/** `resume` if the trainer process has started >1 time on this run (i.e. it
 *  was preempted/restored at some point); `scratch` for a single start or
 *  an absent counter. `lifecycle/trainer_starts` is incremented on every
 *  `trainer.train()` entry in `train_tomat_tpu.py`. */
function lineageTokens(manifest: RunManifest | null): string[] {
  const v = manifest?.summary?.['lifecycle/trainer_starts']
  const starts = typeof v === 'number' ? v : 0
  return starts > 1 ? ['resume'] : ['scratch']
}

/** Build the searchable haystack string for a run.
 *
 *  Inputs are all optional (a freshly-synced run may have no manifest yet);
 *  any missing field just drops its tokens. Result is `' '.join(tokens)`
 *  with leading/trailing whitespace squeezed out — no `undefined` slips in.
 */
export function runHaystack(
  shortLabel: string,
  manifest: RunManifest | null,
  job: IrisJob | null,
  history: RunHistory | null,
): string {
  const tokens: string[] = [shortLabel, ...tagsFor(runIdFromShort(shortLabel, manifest))]

  // Dates. created + last-activity (if history available). Often the same
  // day, in which case the haystack just has the date twice — harmless.
  const createdYmd = yymmddIso(manifest?.run?.created_at)
  if (createdYmd) tokens.push(createdYmd)
  const lastSec = lastActivitySec(history)
  if (lastSec != null) tokens.push(yymmdd(lastSec))

  // Hardware. Falls back to "GPU"/"modal" for no-iris-job runs.
  for (const t of hardwareTokens(runIdFromShort(shortLabel, manifest), job)) {
    tokens.push(t)
  }

  // Lineage.
  for (const t of lineageTokens(manifest)) tokens.push(t)

  return tokens.join(' ')
}

/** Recover the full run id from a shortLabel when the manifest carries it.
 *  Some haystack derivations (tagsFor, hardware-name detection) want the
 *  full name; we'd rather not require the caller to pass it separately. */
function runIdFromShort(shortLabel: string, manifest: RunManifest | null): string {
  // Manifest has the canonical name; fall back to the short label so the
  // function still works pre-manifest-load.
  return manifest?.run?.name ?? shortLabel
}

// ── EXAMPLES — manual spec of expected behaviour ───────────────────────────
//
// No test runner is configured for site/. Below is the "spec by example"
// that this module satisfies. Each entry is `[input, expected]` where the
// pseudo-haystack is constructed from the components shown. The actual unit
// tests would do `expect(compileMultiTermFilter(input).matches(buildHaystack(...))).toBe(expected)`.

/** Static spec for documentation + future automated tests. Read by `tsc`,
 *  so the field names and types are checked even without a test runner.
 *
 *  To run by hand in the browser console:
 *    import('./filter').then(({ FILTER_EXAMPLES, compileMultiTermFilter }) => {
 *      for (const e of FILTER_EXAMPLES) {
 *        const ok = compileMultiTermFilter(e.input).matches(e.haystack) === e.match
 *        console.log(ok ? '✓' : '✗', e.desc, '·', JSON.stringify(e.input))
 *      }
 *    }) */
export const FILTER_EXAMPLES: ReadonlyArray<{
  desc: string
  input: string
  haystack: string
  match: boolean
}> = [
  {
    desc: 'single term (regex) — backward compat',
    input: 'noprm',
    haystack: 'train-full-v3-200M-bs128-ce-10k-v5p16-noprm AR CE noprm-experiment 260527 TPU v5p-16 v5p16 scratch',
    match: true,
  },
  {
    desc: 'two terms — AND across haystack regions',
    input: 'noprm paired-base',
    haystack: 'paired-base AR CE noprm-experiment 260527 TPU v5p-16',
    match: true,
  },
  {
    desc: 'regex OR within a term',
    input: 'noprm|paired-base',
    haystack: 'paired-base AR CE 260527 TPU v5p-16',
    match: true,
  },
  {
    desc: 'date filter combined with tag chip',
    input: 'SS-sweep 260527',
    haystack: 'train-ss-cont80k-emax025-1 AR EMD SS SS-sweep 260527 TPU v5p-16 resume',
    match: true,
  },
  {
    desc: 'TPU hardware + lineage AND',
    input: 'TPU resume|scratch',
    haystack: 'train-foo TPU v6e-16 resume',
    match: true,
  },
  {
    desc: 'AND fails when one term misses',
    input: 'TPU 260527',
    haystack: 'train-foo GPU modal h100x8 260301 scratch',
    match: false,
  },
  {
    desc: 'leading/trailing/double spaces ignored',
    input: '  noprm   paired-base  ',
    haystack: 'paired-base AR CE noprm-experiment',
    match: true,
  },
]
