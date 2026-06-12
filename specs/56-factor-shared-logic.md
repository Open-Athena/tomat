# Spec 56: factor shared run-history logic to eliminate "same bug N places"

Action item from `feedback_factor_logic_not_copy_paste` (Ryan, 2026-06-12)
after the `_step` vs `_timestamp` ordering bug surfaced in four places.

Goal: every cluster of duplicated per-run-history logic gets one
canonical implementation. If a fix or schema bump needs the same edit
in two files, that's a defect: the helper hasn't been factored yet.

## Detected clusters of duplicated logic

### C1: `_trim_to_latest_trajectory` (Python)
- `tomat:740-767` — `rows` are `_step`-sorted by caller (line 1069), but
  the docstring at 743 STILL claims "must be sorted by `_timestamp`"
  (stale; missed in commit `33c688d`). FLAGGED-COSMETIC: doc/code drift.
- `scripts/runs-sync.py:229-255` — docstring says `_step` (correct).
- Bodies are byte-for-byte identical except (a) `err()` vs `print(…file=sys.stderr)`
  and (b) the log-line wording.
- **Right behaviour:** sort by `_step` ascending FIRST, then walk; find LAST
  index where `gs` decreased; drop everything before it.
- **Factor:** new `src/tomat/run_history.py` with `trim_to_latest_trajectory(rows, *, logger=None) -> list[dict]`
  that pre-asserts `_step`-monotonicity and trims. Both CLI and cron import
  it via `sys.path` insertion (cron stays standalone — `scripts/runs-sync.py`
  prepends `src/` to `sys.path`, no `tomat` package install needed; ship the
  module path in the cron-VM setup script).

### C2: Parquet schema + RUN_PARQUET_KEYS (Python)
- `tomat:81-101` (`_RUN_PARQUET_SCHEMA_VERSION`, `_RUN_PARQUET_KEYS`).
- `scripts/runs-sync.py:101-119` (`RUN_PARQUET_SCHEMA_VERSION`, `RUN_PARQUET_KEYS`).
- `tomat:874-911` (`_run_parquet_arrow_schema`).
- `scripts/runs-sync.py:137-174` (`run_parquet_schema`).
- The four blocks must move in lockstep — commit `a02cda7` already
  demonstrated drift cost (column added on one side without bump).
- **Factor:** same module as C1. Export `SCHEMA_VERSION`, `KEYS`, and
  `arrow_schema()` once.

### C3: `_BACKFILL_KEYS` summary backfill (Python)
- `tomat:1135-1151` — backfill summary from history tail.
- `scripts/runs-sync.py:320-332` — same list, same loop.
- Drift hazard is severe: missing a key means the dashboard card silently
  drops a field on a `failed`-state run (the bin5 case from spec
  `runs-sync trims pre-restart trajectories`).
- **Factor:** `BACKFILL_KEYS = (…)` constant + `backfill_summary_from_tail(summary, table_or_rows, n=200)`
  helper in `src/tomat/run_history.py`.

### C4: `_step` vs `_timestamp` row ordering (TS + Python)
- `tomat:1069` (sort by `_step` before trim) — fix `33c688d`.
- `scripts/runs-sync.py:428` (sort by `_step` before trim) — correct
  by design.
- `site/src/runs/WallclockPlot.tsx:499-551` (segments useMemo) — walk
  row-index, NOT `ordered` ts-sorted. Fix `95ee5de`.
- `site/src/runs/WallclockPlot.tsx:332-358` (`tsFlop`) — walk row
  order + running-max-flop, then post-sort by ts. Fix `076db56`.
- `site/src/runs/WallclockPlot.tsx:644-670` (`stepTrace` FLOP mode) —
  walks row order in FLOP mode only.
- `site/src/runs/RunsTimelinePlot.tsx:308-339` (`traceFor` FLOP mode) —
  walks row index for FLOP, ts-sorted for clock/rel/active. Fix `71c4dd7`.
- Four FE call sites + two Python sites; each independently learned the
  rule "wandb `_timestamp` is upload-time, NOT log-time on iris-TPU runs."
- **Factor (TS):** `site/src/runs/historyOrder.ts` with:
  - `rowOrder(history): number[]` — `[0..rowCount-1]` (canonical logical order).
  - `tsOrder(history): {ts, i}[]` — ts-ascending pairs (filtered for non-null ts).
  - `runningMaxAlongRow(values: (number|null)[]): number[]` — fold + monotone-clamp.
  - Helper `assertParquetIsStepAscending(history)` (dev-only invariant
    check; logs warn if violated).
- Add a top-of-file doc-comment that hard-codes "every per-row walk picks
  ONE of these two iteration orders — never roll your own."

### C5: Cost-compute helpers (Python)
- `src/tomat/cost.py` — full canonical impl (parses TPU/Modal, segments,
  rate tables, allocation-class detection, etc.).
- `scripts/runs-sync.py:467-650` — a parallel, partial reimplementation:
  `_parse_modal_gpu`, `_parse_tpu_variant`, `_parse_tpu_variant_from_manifest`,
  `_modal_segment_from_manifest`, `_tpu_segment_from_manifest`,
  `_compute_and_upload_cost`. Plus `TPU_RATES_USD_PER_CHIP_HR`,
  `MODAL_GPU_RATES_USD_PER_HR`, `MODAL_CPU_MEM_ADDER`,
  `DEVICE_KIND_TO_VARIANT` — all copy-pasted.
- ~180 lines of cron-side cost logic shadowing 300 lines of `tomat.cost`.
- The cron VM was kept standalone (no `tomat` pip install) per the
  module docstring; that was the original rationale for duplication.
  Today the cron VM has the repo cloned in `gce/` (commit `ae185e2`) —
  the constraint is gone.
- **Factor:** point cron at `from tomat.cost import …`. Drop the
  `scripts/runs-sync.py` copies. The CLI's `tomat:_compute_one_cost` is
  already a thin wrapper over `tomat.cost`; the cron call site mirrors it.

### C6: `WANDB_PROJECTS` list (Python)
- `tomat:52-66`.
- `scripts/runs-sync.py:48-55`.
- Per `wandb-team-layout` memory, drift here means runs sync silently
  ignores runs landing in a project you forgot to update on both sides.
- **Factor:** `src/tomat/run_history.py` (or a `wandb_projects.py`) +
  cron imports.

## Subtle near-duplicates (NOT bugs, but flag for cluster proximity)

- `runMeta.ts:33-67` `stepsInWindow` — ts-sorted walk, BUT explicitly
  builds a monotonic-segment via `s <= minStep` constraint. Robust to
  async-upload disorder. Don't "fix" it; do reuse the constraint pattern
  if a similar helper is needed elsewhere.
- `runMeta.ts:70-92` `recentStepPoints` — ts-sorted sparkline. Not
  segmented; visual zigzag possible on heavily async-uploaded runs.
  Acceptable cost for a sparkline; flag if ever promoted to a primary plot.
- `WallclockPlot.tsx:286-307` `ordered` + `tsGstep` — ts-sorted, but
  the comment at L298 notes "training is monotonic, effectively by gstep
  too." Correct for in-order runs; for async-upload runs the binary
  search (`gstepAtTs`, `tsAtGstep`) lands on slightly-stale results
  but doesn't poison the primary plots. Tolerable; the same `rowOrder`
  / `tsOrder` factoring still applies — let callers pick.

## CLI ↔ cron drift inventory

| `tomat` (CLI) | `scripts/runs-sync.py` (cron) | Shared today? | C# |
|---|---|---|---|
| `_RUN_PARQUET_KEYS` (82) | `RUN_PARQUET_KEYS` (102) | No | C2 |
| `_RUN_PARQUET_SCHEMA_VERSION` (81) | `RUN_PARQUET_SCHEMA_VERSION` (101) | No | C2 |
| `_run_parquet_arrow_schema` (874) | `run_parquet_schema` (137) | No | C2 |
| `_trim_to_latest_trajectory` (740) | `_trim_to_latest_trajectory` (229) | No | C1 |
| `_BACKFILL_KEYS` (1135) | `_BACKFILL_KEYS` (320) | No | C3 |
| `_compute_one_cost` (4834) | `_compute_and_upload_cost` (609) + helpers | No (CLI uses `tomat.cost`) | C5 |
| `WANDB_PROJECTS` (66) | `WANDB_PROJECTS` (48) | No | C6 |

Cron module-level docstring claims "standalone" — that justified C2/C3/C5
historically. With `gce/install.sh` now cloning the repo to the VM
(commit `ae185e2`), the constraint dissolved; refactor scope can grow.

## Recommended factoring order

1. **C1+C2+C3 together** — same module (`src/tomat/run_history.py`),
   smallest blast radius, every other refactor uses pieces of it. Both
   `tomat` CLI and `scripts/runs-sync.py` import from it; cron-VM setup
   adds the path. Add a unit test that runs `trim_to_latest_trajectory`
   on (a) clean rows, (b) one restart, (c) two restarts (keep only after
   the last), (d) empty rows. ~1 day; high value.
2. **C5** — drop `scripts/runs-sync.py` cost copies, import from
   `tomat.cost`. Validate cost.json shape didn't change via diff of one
   cron tick on a sample run. ~2 hours. Medium value (no current bug;
   high drift risk).
3. **C4** — `historyOrder.ts` + migrate the four FE call sites. Add a
   Vitest assertion that on a synthetic ts-disordered parquet, all four
   call sites land on the same answer. ~half-day; high value (this is
   the bug-of-record). Refactor in this worktree if time permits.
4. **C6** — `WANDB_PROJECTS` constant share. ~15 min, do alongside C1.

## What did NOT make the cut

- `marin/eval_mat_nmae.py` per-mat patch-array logic — distinct domain
  (eval-time padding/segmentation), not the same shape as the dashboard
  history walks.
- `flops.format.ts` / `flops.tsx` — already factored well; reference
  pattern for the FE pieces.
- `RUN_LINEAGE` (`lineage.ts`) — single source of truth on FE today;
  if backend ever needs the lineage map, the table moves to manifest
  metadata or a shared JSON, not duplicated.
- Smoothing / EMA / rolling helpers — already factored in
  `site/src/runs/smoothing.ts` and shared between `WallclockPlot` and
  `RunsTimelinePlot` via `applySmoothing`. Good template.

## Stale doc

`tomat:743` docstring says "rows must already be sorted by `_timestamp`"
— that contradicts the call site (1069 sorts by `_step`) and the
function's actual contract. Fix in the C1 refactor or sooner.
