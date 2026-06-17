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

/** Pretty-print a 0-based `step_idx` for tables / tooltips, snapping near-round
 *  values to the round form. Snapped values are suffixed with `*` so the rendering
 *  is transparent: a reader sees `90k*` and knows the on-disk artifact is at a
 *  nearby raw value (e.g. `step-89999`). Non-snappable values render plain
 *  (e.g. `53,000`).
 *
 *  `step` is a 0-based Levanter `info.step` value. See `docs/step-conventions.md`.
 *
 *  Pair with `formatStepDetail` to wire a tooltip that explains the underlying
 *  raw value + Levanter's final-of-segment naming convention.
 */
export function formatStep(step: number): string {
  const snapped = snapStep(step)
  const pretty = prettifyRound(snapped)
  return snapped === step ? pretty : `${pretty}*`
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

/** Build the snapped display string + a tooltip that explains the snap.
 *
 *  `step` is a 0-based `step_idx` (Levanter `info.step` value; see
 *  `docs/step-conventions.md`). When the raw step was within the snap
 *  tolerance of a round Nk anchor, the display is prefixed with `≈` and the
 *  tooltip narrates the Levanter convention: end-of-segment ckpts save as
 *  `step-(N-1)` because Levanter's force-save hook uses
 *  `info.step = state.step − 1` (0-indexed) as the filename, while periodic
 *  ckpts (modulo against the same 0-indexed value) happen to land on clean Nk.
 *
 *  Callers that already host a parent `Tooltip` should inline `tooltip` into
 *  their existing content (the explanation isn't gated on a separate hover).
 */
export function formatStepDetail(step: number): {
  display: string
  isSnapped: boolean
  raw: number
  snapped: number
  tooltip: string
} {
  const snapped = snapStep(step)
  const isSnapped = snapped !== step
  const display = isSnapped ? `${prettifyRound(snapped)}*` : prettifyRound(snapped)
  const tooltip = isSnapped
    ? `raw step ${step.toLocaleString()} on disk; snapped to ${snapped.toLocaleString()}. `
      + `Levanter saves the final ckpt of each training segment using `
      + `info.step (0-indexed, = state.step − 1), so step-${step} is the `
      + `artifact name for what is logically the `
      + `${snapped.toLocaleString()}th step completed. Periodic ckpts within `
      + `the same run save at clean N.`
    : `step ${step.toLocaleString()}`
  return { display, isSnapped, raw: step, snapped, tooltip }
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
