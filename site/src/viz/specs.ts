// Source-spec parsing + label generation for joint-histogram viz.
//
// A "source" is one charge-density grid for one mat: either ground-truth
// (`gt:<set>`) or a model prediction (`pred:<run>:<setmode>:<step>`). The
// FE consumes specs from URL params (#/viz/joint-hist?x=…&y=…) and
// resolves each to a zarr URL on the runs-API CFW.
//
// Stub fallback for `expandSpec` / `shortenSpec` until #297's
// `@/lib/runNames` lands — the regex form covers our current canonical
// run names (`train-mg-kl-<arm>-fs-tpu` ± `step-N`).

import { API_BASE } from '../runs/api'
import { formatStep } from '../lib/runNames'

export type SourceSpec =
  | { kind: 'gt'; set: 'val' | 'train' }
  | {
      kind: 'pred'
      run: string
      setMode: string
      /** 0-indexed Levanter step from the checkpoint name (matches GCS path
       *  `step-{step_idx}`). See `docs/step-conventions.md`. */
      step_idx: number
    }

/** Parse a URL-form spec string into a structured `SourceSpec`.
 *  Accepts the canonical forms `gt:val`, `gt:train`,
 *  `pred:<run>:<setMode>:<step>`. Also accepts shorthand
 *  `<arm>@<stepShort>` via `expandSpec`. Returns null on malformed input.
 *  The trailing `<step>` in `pred:` specs is a 0-based `step_idx`. */
export function parseSpec(s: string): SourceSpec | null {
  if (s === 'gt:val') return { kind: 'gt', set: 'val' }
  if (s === 'gt:train') return { kind: 'gt', set: 'train' }
  const m = /^pred:([^:]+):([^:]+):(\d+)$/.exec(s)
  if (m) {
    return { kind: 'pred', run: m[1], setMode: m[2], step_idx: parseInt(m[3], 10) }
  }
  // Shorthand: try expanding into canonical, then re-parse.
  const expanded = expandSpec(s)
  if (expanded && expanded !== s) return parseSpec(expanded)
  return null
}

/** Human-readable short label for axis titles, hover, etc. */
export function shortenSpec(spec: string): string {
  if (spec === 'gt:val') return 'GT (val)'
  if (spec === 'gt:train') return 'GT (train)'
  const parsed = parseSpec(spec)
  if (!parsed) return spec
  if (parsed.kind === 'gt') return parsed.set === 'val' ? 'GT (val)' : 'GT (train)'
  // pred: strip the noisy training-fire prefix/suffix to surface the "arm".
  const armRaw = parsed.run
    .replace(/^train-mg-([a-z0-9-]+?)-fs-tpu$/, '$1')
    .replace(/^kl-/, '')
  // `formatStep` handles both the legacy OBO translation (`step-89999` →
  // exactly `90k` no asterisk, because it was a force-save of "90k steps
  // completed") AND the post-fix periodic stamping (where a `*` flags
  // the `N+1` actual-completed-steps case). `shortenStep` has inverted
  // semantics that pre-date the formatter rewrite — avoid for axis
  // titles.
  return `${armRaw} @ ${formatStep(parsed.step_idx)}`
}

/** Expand a shorthand like `bin5@30k` → `pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000`.
 *  Returns null when the input isn't a recognized shorthand (the caller
 *  should treat it as a canonical spec). The trailing integer in the
 *  canonical form is a 0-based `step_idx`. This is intentionally narrow —
 *  the real expansion table lands with #297. */
export function expandSpec(short: string): string | null {
  const m = /^([a-z0-9-]+)@(\d+(?:\.\d+)?)([kKmM])?$/.exec(short)
  if (!m) return null
  const arm = m[1]
  const num = parseFloat(m[2])
  const unit = (m[3] ?? '').toLowerCase()
  const step_idx = Math.round(num * (unit === 'm' ? 1e6 : unit === 'k' ? 1e3 : 1))
  // Default to the bin5-style MaskGIT eval set; honest disambiguation is
  // #297's job. The route still accepts canonical `pred:…:…:…` directly.
  return `pred:train-mg-kl-${arm}-fs-tpu:val_200-maskgit:${step_idx}`
}

/** Build the level-0 zarr URL for one source + mat. The CFW serves these at
 *  `/api/files/raw/<key>`, so zarrita's FetchStore (or our DIY loader) can
 *  append `0/zarr.json`, `0/c/0/0/0` natively. */
export function zarrUrlFor(spec: SourceSpec, mp: string): string {
  const base = `${API_BASE}/api/files/raw`
  if (spec.kind === 'gt') {
    const sub = spec.set === 'val' ? 'validation' : 'train'
    return `${base}/tomat/rho_gga_v3mr/${sub}/${mp}.zarr`
  }
  return `${base}/tomat/eval/predictions/${spec.run}/${spec.setMode}/step-${spec.step_idx}/${mp}.zarr`
}
