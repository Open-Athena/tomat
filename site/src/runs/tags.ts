// Curated tags per run name. Used by the /runs dashboard's tag-chip filter.
//
// IMPORTANT: in this project, run NAMES are unreliable. Every `*-emd-*` run
// trained before 2026-05-25 actually used standard CE on the next-token
// objective, because Levanter's `Qwen3LMHeadModel.init` discarded the
// `Qwen3DensityLMHeadModel` subclass (silent bug, see
// `specs/done/30-levanter-init-cls-postmortem.md`). The tag set below
// reflects what *actually* ran, not what the name implies.
//
// Conventions:
//   objective: 'AR' | 'MaskGIT'
//   loss:      'CE' | 'EMD' | 'CE+EMD'
//   status:    'production' | 'smoke' | 'collapsed' | 'bunk'
//   notable:   'cont33k' | 'pre-init-fix' | 'post-init-fix' (optional)
//
// A run can carry multiple tags. The filter does an AND across selected
// chips: a run is visible iff all selected chips are in its tag set.

export type RunTag = string

export const RUN_TAGS: Record<string, RunTag[]> = {
  // ---- Post-init-fix (2026-05-25+): tags reflect what actually runs. ----
  'train-mg-4-cos-ce': ['MaskGIT', 'CE', 'production', 'post-init-fix'],
  'train-ar-ce-emd-1': ['AR', 'CE+EMD', 'production', 'post-init-fix', 'eval-bidir-bug'],
  'train-mg-3-cos-emd': ['MaskGIT', 'EMD', 'collapsed', 'post-init-fix'],
  'train-ar-emd-real': ['AR', 'EMD', 'collapsed', 'post-init-fix', 'eval-bidir-bug'],
  'train-mg-fix-verify': ['MaskGIT', 'EMD', 'smoke', 'post-init-fix'],
  'train-mg-smk-A': ['smoke', 'pre-init-fix'],
  'train-mg-smk-B': ['smoke', 'pre-init-fix'],
  'train-mg-smk-C': ['smoke', 'pre-init-fix'],

  // ---- No-preamble experiment (2026-05-27): does cont33k use atoms at all? ----
  // Both are CE (TOMAT_DENSITY_L1_WEIGHT=0 → density path falls through to base
  // CE); only difference is preamble-zeroed vs preamble-intact, same seed.
  'train-full-v3-200M-bs128-ce-10k-v5p16-noprm':
    ['AR', 'CE', 'noprm-experiment', 'post-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-ce-10k-v5p16-paired-base':
    ['AR', 'CE', 'noprm-experiment', 'post-init-fix', 'eval-bidir-bug'],

  // ---- SS sweep (2026-05-27): scheduled-sampling fine-tunes off cont33k ----
  // All warm-started from cont33k step-79999; vary ε distribution + sampler.
  'train-ss-cont80k-emax025-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'eval-bidir-bug'],
  'train-ss-cont80k-emax050-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'eval-bidir-bug'],
  'train-ss-cont80k-emax075-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'eval-bidir-bug'],
  'train-ss-cont80k-emax100-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'eval-bidir-bug'],
  // BUNK: launcher set TOMAT_SS_EPS_MAX=1.0 but NOT TOMAT_SS_EPS_MIN=1.0,
  // so ε sampled U(0, 1.0) — i.e. trained bit-identically to emax100-1
  // (verified: TL matches to 6 decimal places across all 5000 fine-tune
  // steps). See spec 37 + the SS print-statement fix in train_tomat_tpu.py.
  'train-ss-cont80k-eps1const-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-ss-cont80k-hi-argmax-1':
    ['AR', 'EMD', 'SS', 'SS-sweep', 'post-init-fix', 'eval-bidir-bug'],

  // ---- VV: VQ-VAE tokenizer-only training (no atomic encoding, self-supervised) ----
  'vqvae-smoke-50k-02': ['VV', 'VV-smoke'],
  'vqvae-cb8k-01': ['VV', 'VV-sweep'],
  'vqvae-lat16-01': ['VV', 'VV-sweep'],

  // ---- Pre-init-fix: all trained as CE on the next-token objective ----
  // mg-* runs added a vocab+1 MASK token but never reached masked-loss code.
  'train-mg-1': ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-mg-2-cos-emd': ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-mg-2-hi-emd': ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-mg-2-uni-emd': ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],

  // Named "-emd-do-" but `density_only` + `emd` both routed through the base
  // class's CE — these are the real CE baselines.
  'train-full-v3-200M-bs128-emd-do-80k-v6e16-shuf1k-cont33k':
    ['AR', 'CE', 'pre-init-fix', 'cont33k', 'production', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-10k-v5p16-shuf1k-z3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs256-emd-do-15k-v5p16-shuf1k-z3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs512-emd-do-15k-v5p16-shuf1k-z3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs256-emd-do-10k-h100x8-r2-bs256-seed42':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs256-emd-do-10k-h100x8-bs256-seed42':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs256-emd-do-10k-h100x8-r2':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],

  // cont7k / cont8k WSD cooldown variants.
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont7k-ext':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont6kwsd':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k-cont8kwsd':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],

  // LR sweep series (lrsw-lr*).
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr3e4':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr3e4-r2':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr1e3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr2e3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr4e3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k-lrsw-lr8e3':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],

  // Other v3 and lmq-v2 runs.
  'train-full-v3-200M-bs128-emd-do-8k-v5p16-shuf1k':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-8k-tpu16':
    ['AR', 'CE', 'pre-init-fix', 'eval-bidir-bug'],
  'train-full-v3-200M-bs128-emd-do-500-v5p16-shuf1k-pyspy-3':
    ['AR', 'CE', 'pre-init-fix', 'smoke', 'eval-bidir-bug'],

  // Larger / failed-cascade era (kept tagged so they don't appear unlabeled).
  'train-full-v3-200M-bs256-emd-do-15k-v5p16-shuf1k-zf-r2':
    ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-full-v3-200M-bs512-emd-do-15k-v5p16-shuf1k-zf-r2':
    ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],
  'train-full-v3-200M-bs1792-emd-do-15k-v5p32-shuf1k-zf-r2':
    ['AR', 'CE', 'pre-init-fix', 'bunk', 'eval-bidir-bug'],

  // lmq-v2 era (older t10n) — orange in the dashboard.
  'tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32':
    ['AR', 'CE', 'pre-init-fix', '1B', 'eval-bidir-bug'],
  'train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32':
    ['AR', 'CE', 'pre-init-fix', '1B', 'v2', 'eval-bidir-bug'],
  'train-full-lmq-v2-200M-bs128-emd-do-16k-tpu16':
    ['AR', 'CE', 'pre-init-fix', 'v2', 'eval-bidir-bug'],
  'train-full-lmq-v2-200M-bs128-emd-do-8k-lat-tpu16':
    ['AR', 'CE', 'pre-init-fix', 'v2', 'eval-bidir-bug'],
}

/** All distinct tags present in the registry, in display order. */
export const ALL_TAGS: RunTag[] = (() => {
  // Stable, semantic ordering — objective → loss → status → other.
  const ordered: RunTag[] = [
    'AR', 'MaskGIT', 'SS', 'VV',
    'CE', 'EMD', 'CE+EMD',
    'production', 'collapsed', 'bunk', 'eval-bidir-bug', 'smoke',
    'SS-sweep', 'VV-sweep', 'VV-smoke', 'noprm-experiment',
    'cont33k', '1B', 'post-init-fix', 'pre-init-fix', 'v2',
  ]
  const seen = new Set<RunTag>()
  for (const tags of Object.values(RUN_TAGS)) for (const t of tags) seen.add(t)
  const out: RunTag[] = []
  for (const t of ordered) if (seen.has(t)) { out.push(t); seen.delete(t) }
  for (const t of Array.from(seen).sort()) out.push(t)  // fallback for unknowns
  return out
})()

/** Returns tags for a run name, or [] if untagged. */
export function tagsFor(runName: string): RunTag[] {
  return RUN_TAGS[runName] ?? []
}

/** Per-tag filter state.
 *  - `'in'`  ⇒ the run MUST have this tag.
 *  - `'out'` ⇒ the run MUST NOT have this tag.
 *  - `'off'` ⇒ no constraint (this entry exists in the Map only as an
 *               *override* of a non-`off` default — e.g. someone clicked
 *               `bunk` to switch it from its built-in `'out'` default
 *               back to "no filter").
 *
 *  The Map stores ONLY non-default states. A tag missing from the Map
 *  is implicitly at its default (see `TAG_DEFAULTS`), which is `'off'`
 *  for every tag except `bunk` (which defaults to `'out'`). URL
 *  serialization mirrors this — clean URLs only mention overrides.
 *
 *  Type re-exports from `use-prms` (which owns the URL `tagFilterParam`
 *  used by `useTagFilters` in `RunsTimelinePlot.tsx`): one source of
 *  truth for the Map shape across the project. */
import {
  cycleTagFilter as upCycleTagFilter,
  effectiveTagState as upEffectiveTagState,
  runPassesTagFilters as upRunPassesTagFilters,
  tagFilterParam,
  type TagDefaults,
  type TagFilters as UpTagFilters,
  type TagState,
} from 'use-prms'

export type TagFilterState = TagState
export type TagFilters = UpTagFilters<RunTag>

/** Per-tag default. Tags absent here implicitly default to `'off'`.
 *  Rationale for `bunk`: init-cls-bug-era runs (mg-1, mg-2-*) all
 *  trained as plain CE; they spam the CE / EMD chip filters and drown
 *  out post-fix runs that actually tested those losses. Hiding them
 *  by default matches what most viewers want; click the chip to opt
 *  back in. See `memory/levanter-init-cls-bug.md`. */
export const TAG_DEFAULTS: TagDefaults<RunTag> = {
  bunk: 'out',
}

/** The `use-prms` `Param<TagFilters<RunTag>>` driving `?tags=...`. URL
 *  encoding (overrides only — defaults stay out of the URL):
 *    bare `tag`   → in
 *    `-tag`       → out
 *    `~tag`       → off  (only ever appears when overriding a non-`off`
 *                          default — today: `bunk`, when the user
 *                          dismissed the pre-excluded chip)
 *  Tokens joined by spaces; `URLSearchParams` encodes space as `+`. So
 *  `{CE: 'in', bunk: 'off'}` round-trips as `?tags=CE+~bunk`. */
export const TAG_FILTER_PARAM = tagFilterParam<RunTag>({ defaults: TAG_DEFAULTS })

export function defaultStateFor(tag: RunTag): TagFilterState {
  return TAG_DEFAULTS[tag] ?? 'off'
}

/** The *effective* state for `tag`: the override if one exists,
 *  otherwise the per-tag default. */
export function effectiveTagState(filters: TagFilters, tag: RunTag): TagFilterState {
  return upEffectiveTagState(filters, tag, TAG_DEFAULTS)
}

/** Apply the (overrides + defaults) tag-filter to a run's tag list.
 *  Iterates the union of `filters` keys and `TAG_DEFAULTS` keys — both
 *  carry constraints. */
export function runPassesTagFilters(runTags: RunTag[], filters: TagFilters): boolean {
  return upRunPassesTagFilters(runTags, filters, TAG_DEFAULTS)
}

/** Cycle `tag`'s effective state via `in → out → off → in`. The new
 *  state is stored as an override only if it differs from the tag's
 *  default — otherwise the entry is removed (so the URL stays minimal).
 *  Returns a *new* Map. */
export function cycleTagFilter(filters: TagFilters, t: RunTag): TagFilters {
  return upCycleTagFilter(filters, t, TAG_DEFAULTS)
}

/** Parse the legacy localStorage payload into a `TagFilters` map. Accepts both
 *  the old single-state array shape (`["CE","bunk"]` → all `'in'`) and the
 *  object shape (`{"CE":"in","bunk":"out"}`), so a one-time migration from
 *  pre-`use-prms`-era state keeps working on first load. Only used by the
 *  legacy-LS migration path in `useTagFilters`; new writes go to the URL. */
export function parseTagFiltersLegacyLs(raw: string): TagFilters {
  const m: TagFilters = new Map()
  try {
    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed)) {
      // Legacy shape — every entry was an implicit `'in'`.
      for (const t of parsed as RunTag[]) m.set(t, 'in')
    } else if (parsed && typeof parsed === 'object') {
      for (const [k, v] of Object.entries(parsed as Record<string, string>)) {
        if (v === 'in' || v === 'out' || v === 'off') m.set(k, v)
      }
    }
  } catch { /* ignore — return empty */ }
  return m
}
