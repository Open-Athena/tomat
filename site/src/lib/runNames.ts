// Curatable short-name registry for tomat run names + source specs.
//
// The registry lives at `runs/registry.json` at the repo root and is the
// single source of truth for vetted short names. Functions here mirror
// `marin/run_names.py` — keep the two in sync.
//
// All step inputs here are 0-based `step_idx` values (Levanter `info.step`
// convention; see `docs/step-conventions.md` for the project's full
// `step_idx` / `step_n` / `target_steps` / `step_display` taxonomy).
//
// `shortenRun` falls back to the same regex extraction used by the CLI's
// `_viz_short_run` when the registry has no entry; uncurated runs still
// get a reasonable short name. `expandRun` is exact-match only — the
// regex fallback isn't invertible.
//
// `tsconfig.json` sets `"resolveJsonModule": true`, so the JSON import
// below is type-safe. The path reaches up out of `site/` to the repo
// root because the registry is shared between the CLI and the FE.

import registry from '../../../runs/registry.json' with { type: 'json' }

interface RegistryEntry {
  short: string
  canonical: string
  notes?: string
}

interface Registry {
  version: number
  runs: RegistryEntry[]
}

const REG = registry as Registry

const BY_CANONICAL = new Map<string, RegistryEntry>(REG.runs.map((e) => [e.canonical, e]))
const BY_SHORT = new Map<string, RegistryEntry>(REG.runs.map((e) => [e.short, e]))

// ── Run names ───────────────────────────────────────────────────────────────

// Mirrors `_VIZ_RUN_SHORTNAME_RE` in `tomat/tomat`. Strips an optional `kl-`
// prefix so `train-mg-kl-bin5-fs-tpu` regex-shortens to `bin5-fs`. The
// registry overrides this to `bin5`.
const RUN_SHORTNAME_RE = /^train-mg-(?:kl-)?([a-z0-9]+(?:-[a-z0-9]+)?)-/
const RUN_GENERIC_RE = /^train-(?:mg-)?([a-z0-9]+(?:-[a-z0-9]+)?)-/

/** Return the registered short name for `canonical`, else a regex-derived
 *  fallback. Truncates to 32 chars when nothing matches, to keep axis labels
 *  manageable. */
export function shortenRun(canonical: string): string {
  const entry = BY_CANONICAL.get(canonical)
  if (entry !== undefined) return entry.short
  let m = canonical.match(RUN_SHORTNAME_RE)
  if (m) return m[1]
  m = canonical.match(RUN_GENERIC_RE)
  if (m) return m[1]
  return canonical.length <= 32 ? canonical : canonical.slice(0, 32)
}

/** Return the canonical run name for `short`, or `null` if not registered.
 *  Regex fallback isn't invertible — only exact registry hits resolve. */
export function expandRun(short: string): string | null {
  const entry = BY_SHORT.get(short)
  return entry !== undefined ? entry.canonical : null
}

// ── Step counts ─────────────────────────────────────────────────────────────

/** Snap a raw step to the nearest round value within `tolerance` (default 0.1%).
 *
 *  `rawStep` is a 0-based `step_idx` from a Levanter `info.step` / GCS
 *  checkpoint name (see `docs/step-conventions.md` for the project's
 *  `step_idx` / `step_n` / `target_steps` / `step_display` distinction).
 *
 *  Why this exists: Levanter's force-saved final ckpt at end-of-training lands
 *  at `step-(N-1)` because `info.step = state.step - 1` is 0-indexed and the
 *  end-of-run hook fires with `force=True`. (Periodic ckpts during training
 *  land at clean `step-{N,2N,…}` because the modulo test runs against the same
 *  0-indexed `info.step`, just happens to land on round values.) This helper
 *  lets the FE / CLI render `step-49999` as `50k` without special-casing
 *  whether a given step value is a periodic or end-of-run ckpt.
 *
 *  Examples: `89999 → 90000`, `49999 → 50000`, `50001 → 50000`,
 *  `53000 → 53000` (already round), `1234 → 1234` (no round Nk within tol).
 */
export function snapStep(rawStep: number, tolerance: number = 0.001): number {
  if (rawStep <= 0) return rawStep
  for (const base of [1_000_000, 100_000, 10_000, 1_000]) {
    const rounded = Math.round(rawStep / base) * base
    if (rounded > 0 && Math.abs(rounded - rawStep) / rawStep <= tolerance) {
      return rounded
    }
  }
  return rawStep
}

/** Format a 0-based `step_idx` compactly: 30000 → `30k`, 89999 → `90k*`, 1500000 → `1.5M`.
 *
 *  `step` is a 0-based Levanter `info.step` value (i.e. lifted from a GCS
 *  `step-{step_idx}` checkpoint name). See `docs/step-conventions.md`.
 *
 *  Applies `snapStep` first so legacy `89999`-style raw values surface as
 *  `90k`; when a snap fires the result is suffixed with `*` so the rendering
 *  is transparent (universal footnote glyph, paired with a tooltip explaining
 *  the underlying raw value). Pass `opts.snap = false` to bypass (rarely
 *  needed; the marker is also suppressed in that case since no snap happened).
 */
export function shortenStep(step: number, opts?: { snap?: boolean }): string {
  const s = opts?.snap === false ? step : snapStep(step)
  let pretty: string
  if (s >= 1_000_000) {
    const v = s / 1_000_000
    pretty = v < 10 ? `${v.toFixed(1)}M` : `${Math.round(v)}M`
  } else if (s >= 1000) {
    pretty = `${Math.round(s / 1000)}k`
  } else {
    pretty = String(s)
  }
  return s === step ? pretty : `${pretty}*`
}

/** Pretty-print a 0-based `step_idx` for tables / tooltips. Delegates to
 *  `formatStepDetail(step).display` so the two functions are guaranteed
 *  consistent — see that function for the asterisk semantics and the
 *  Levanter `info.step` OBO history.
 *
 *  `step` is a 0-based Levanter `info.step` value. See `docs/step-conventions.md`.
 */
export function formatStep(step: number, opts?: { writtenAt?: string | Date }): string {
  return formatStepDetail(step, opts).display
}

/** Format a round step with `k`/`M` suffix where exact, else comma-separate.
 *  Used by `formatStep` so the snapped display matches the registry/CLI's
 *  short-form (`50k`, `1.5M`) rather than `50,000`. */
function prettifyRound(n: number): string {
  if (n >= 1_000_000 && n % 1_000_000 === 0) return `${n / 1_000_000}M`
  if (n >= 1_000_000 && n % 100_000 === 0) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000 && n % 1_000 === 0) return `${n / 1_000}k`
  if (n >= 1_000 && n % 100 === 0) return `${(n / 1_000).toFixed(1)}k`
  return n.toLocaleString()
}

/** UTC instant when the marin `rw/integration` HEAD `e20bdd1892ea` (which
 *  switches Levanter ckpt naming from `info.step` to `info.next_step`, so
 *  on-disk `step-N` matches N actual completed steps) was deployed to
 *  bin5's training venv. Ckpts written before this fall under the legacy
 *  OBO convention — `step-N` on disk = N+1 completed steps (periodic) or
 *  step-(N-1) for force-saves at end-of-segment.
 */
export const LEGACY_STEP_NAMING_CUTOFF = new Date('2026-06-17T03:30:00Z')

/** Was the ckpt this `written_at` was attached to written under the
 *  legacy info.step-naming convention? `undefined` falls back to legacy
 *  (conservative: nearly all current data predates the cutoff). */
function writtenAtIsLegacy(writtenAt?: string | Date): boolean {
  if (writtenAt === undefined) return true
  const d = typeof writtenAt === 'string' ? new Date(writtenAt) : writtenAt
  return d < LEGACY_STEP_NAMING_CUTOFF
}

/** Build a display string + tooltip for a 0-based `step_idx` accounting for
 *  Levanter's pre-fix info.step OBO convention.
 *
 *  Background: Before marin commit `e20bdd1892ea` (deployed 2026-06-17),
 *  Levanter wrote ckpts named with `info.step = state.step − 1`, so:
 *    - **Periodic** ckpts (e.g. save-every-10000): on-disk `step-30000`
 *      actually represents 30,001 completed steps.
 *    - **Force-save** end-of-segment ckpts: on-disk `step-49999` actually
 *      represents 50,000 completed steps (state.step was N when info.step
 *      = N−1 hit the disk).
 *
 *  Post-fix: on-disk `step-N` is exact — N completed steps.
 *
 *  Display semantics (`*` flags "this is an interpretation, hover for why"):
 *    - **Legacy force-save** (raw = round−1, e.g. `step-49999`): display
 *      the round form (`50k`), NO asterisk — it's the count of completed
 *      steps, exact after +1.
 *    - **Legacy periodic** (raw = round, e.g. `step-30000`): display the
 *      rounded raw + asterisk (`30k*`) — actual count is 30,001 but we
 *      keep the round label and flag the legacy interpretation.
 *    - **Post-fix** (raw = anything; on-disk is exact): plain `prettifyRound(raw)`.
 *
 *  `step` is a 0-based `step_idx` (Levanter `info.step` value). The
 *  `writtenAt` opt is the ISO timestamp on the carrying object (eval JSON,
 *  ckpt manifest, etc.); omit to default to "legacy" (the conservative
 *  assumption for now — all pre-2026-06-17 data).
 *
 *  Callers that already host a parent `Tooltip` should inline `tooltip` into
 *  their existing content (the explanation isn't gated on a separate hover).
 */
export function formatStepDetail(
  step: number,
  opts?: { writtenAt?: string | Date },
): {
  display: string
  isLegacy: boolean
  /** True iff the display was derived (snapped or +1'd from raw) rather
   *  than rendered as-is. Carries the *. */
  isMarked: boolean
  /** Raw on-disk `step-N` value (Levanter `info.step`). */
  rawStep: number
  /** Number of actually-completed steps (raw + OBO correction). */
  completedSteps: number
  tooltip: string
} {
  const isLegacy = writtenAtIsLegacy(opts?.writtenAt)
  if (!isLegacy) {
    // Post-fix: disk name is exact; no asterisk, no +1.
    return {
      display: prettifyRound(step),
      isLegacy: false,
      isMarked: false,
      rawStep: step,
      completedSteps: step,
      tooltip: `step ${step.toLocaleString()}`,
    }
  }
  // Legacy era: classify periodic vs force-save by structure of `raw`.
  // Force-saves leave on-disk `step-(N-1)` for a round target N; snapStep
  // detects "is this near a round Nk?", and a diff of exactly 1 (with raw
  // = snapped − 1) identifies the force-save case.
  const snapped = snapStep(step)
  const isForceSave = snapped !== step && snapped - step === 1
  if (isForceSave) {
    // Legacy force-save: disk = round−1; actual completed = round (exact after +1).
    const display = prettifyRound(snapped)
    const tooltip
      = `Legacy force-save: disk name step-${step} represents `
      + `${snapped.toLocaleString()} actual completed steps (exact after +1). `
      + `Pre-fix Levanter wrote info.step = state.step − 1, so end-of-segment `
      + `force-saves land at step-(N−1); fixed in marin e20bdd1892ea `
      + `(deployed 2026-06-17).`
    return {
      display,
      isLegacy: true,
      isMarked: false,
      rawStep: step,
      completedSteps: snapped,
      tooltip,
    }
  }
  // Legacy periodic (or non-snap-eligible legacy value): actual is raw + 1.
  // Display the rounded raw + `*` to flag the legacy interpretation.
  const completed = step + 1
  const display = `${prettifyRound(step)}*`
  const tooltip
    = `Legacy artifact: disk name step-${step} represents `
    + `${completed.toLocaleString()} actual completed steps due to Levanter `
    + `info.step OBO (fixed in marin commit e20bdd1892ea, deployed 2026-06-17). `
    + `Post-fix periodic ckpts save with exact-match disk names.`
  return {
    display,
    isLegacy: true,
    isMarked: true,
    rawStep: step,
    completedSteps: completed,
    tooltip,
  }
}

const STEP_RE = /^[≈]?(\d+(?:\.\d+)?)([kM])?\*?$/

/** Inverse of `shortenStep`. `30k` → 30000, `1.5M` → 1500000. `null` for
 *  unparseable inputs. Returns a 0-based `step_idx` (Levanter `info.step`
 *  convention; see `docs/step-conventions.md`). Lossy `shortenStep` cases
 *  round-trip to the rounded value, not the original. Accepts a leading `≈`
 *  (the snap marker emitted by `shortenStep`/`formatStep`) and discards it. */
export function expandStep(short: string): number | null {
  const m = short.trim().match(STEP_RE)
  if (!m) return null
  const val = parseFloat(m[1])
  const suffix = m[2]
  if (suffix === 'M') return Math.round(val * 1_000_000)
  if (suffix === 'k') return Math.round(val * 1000)
  return Math.round(val)
}

// ── Source specs ────────────────────────────────────────────────────────────
//
// Spec grammar (matches the viz CLI):
//   gt:<split>                              e.g. `gt:val`, `gt:train`
//   pred:<run>:<setmode>:<step>             e.g. `pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000`
//
// Short forms:
//   GT (val), GT (train)
//   <run-short> @ <step-short>              maskgit setmode
//   <run-short> @ <step-short>-K1           oneshot setmode

function splitShort(split: string): string {
  if (split.startsWith('val')) return 'val'
  if (split.startsWith('train')) return 'train'
  return split
}

/** Map a source spec to its short label. */
export function shortenSpec(spec: string): string {
  const parts = spec.split(':')
  if (parts[0] === 'gt' && parts.length === 2) {
    return `GT (${splitShort(parts[1])})`
  }
  if (parts[0] === 'pred' && parts.length === 4) {
    const [, run, setmode, step] = parts
    const stepI = parseInt(step, 10)
    const stepNum = Number.isNaN(stepI) ? 0 : stepI
    const runShort = shortenRun(run)
    const stepShort = shortenStep(stepNum)
    const suffix = setmode.includes('oneshot') ? '-K1' : ''
    return `${runShort} @ ${stepShort}${suffix}`
  }
  return spec
}

const SHORT_GT_RE = /^GT \((val|train)\)$/
const SHORT_PRED_RE = /^(?<run>.+?) @ (?<step>≈?\d+(?:\.\d+)?[kM]?)(?<k1>-K1)?$/

/** Inverse of `shortenSpec`. Returns `null` when the run-short isn't
 *  registered. Defaults setmode to `val_200-maskgit` (or `val_200-oneshot`
 *  when the short ends with `-K1`). */
export function expandSpec(short: string): string | null {
  let m = short.match(SHORT_GT_RE)
  if (m) return `gt:${m[1]}`
  m = short.match(SHORT_PRED_RE)
  if (m && m.groups) {
    const runShort = m.groups.run
    const stepShort = m.groups.step
    const isOneshot = m.groups.k1 !== undefined
    const canonical = expandRun(runShort)
    if (canonical === null) return null
    const stepI = expandStep(stepShort)
    if (stepI === null) return null
    const setmode = isOneshot ? 'val_200-oneshot' : 'val_200-maskgit'
    return `pred:${canonical}:${setmode}:${stepI}`
  }
  return null
}
