# 46. Per-run cost estimation (MSRP) on the dashboard

## Motivation

We're firing multi-day Modal training runs and multi-hour TPU evals.
Right now there's no easy answer to "what did this run cost?" — the
dashboard surfaces `FLOP`, runtime, MFU, but no dollar number. The
user wants:

- A visible **`$X,XXX MSRP`** chip per run on the run-detail page +
  cards.
- Tooltip breakdown: `<hardware> · <wallclock> · <rate>` per
  contributing segment.
- Works for both Modal training runs and TPU iris jobs.

## Important framing

The number we display is **equivalent retail value (MSRP)**, NOT what
gets charged. Actual billing flows differently for both backends, but
that detail doesn't belong in a public-repo doc (see memory
`feedback_no_funding_in_public_repo`). Tooltips must use:

- "Estimated MSRP" / "equivalent retail value" — not "cost" or "spend".
- Note the table's edit date + reference the published price source
  (e.g. "Modal pricing page snapshot 2026-MM-DD", "GCP TPU pricing
  page snapshot 2026-MM-DD").

This is solely a "how much compute did this consume in retail dollars"
chip. Reviewers can interpret in whatever billing context applies.

## Data model

Per training run + per eval job, an estimate record:

```jsonc
{
  "msrp_usd": 728.40,
  "breakdown": [
    {
      "kind": "modal",
      "label": "H200×8",
      "wallclock_sec": 72100,
      "rate_per_hr_usd": 36.48,
      "msrp_usd": 730.61,
      "source": "fc-01KT35FV6C6G9CVWENRR4C51Z7"
    }
    // multiple entries when the run has multiple attempts / Modal calls
  ],
  "computed_at": "2026-06-02T03:00:00Z",
  "pricing_table_version": "2026-06-02",
  "is_complete": false   // true when the run has terminated
}
```

### Where the record lives

Per-run R2 sidecar: `runs/<run_label>/cost.json`. CFW serves it as
`GET /api/runs/<run_label>/cost.json`. The aggregated
`/api/runs-snapshot.json` inlines `runs[i].cost.msrp_usd` only (full
breakdown lazy-fetched on detail page hover).

## Modal pricing

Modal exposes per-call resource consumption via:

- `modal call info <fc_id>` — wallclock + GPU count
- Modal GraphQL `functionCallStats` query — finer-grained breakdown

Per-fc cost = `(wallclock_sec / 3600) × num_gpus × per_gpu_hr_usd`.

`per_gpu_hr_usd` table (snapshot 2026-06-02 from
https://modal.com/pricing):

| GPU   | $/GPU-hr |
|-------|----------|
| H100  | 3.95     |
| H200  | 4.56     |
| B200  | 6.25     |
| A100  | 2.78     |
| L40S  | 1.95     |

CPU + memory contributions are <5% for our GPU-heavy workloads;
include them as a flat 5% adder rather than tracking precisely.

## TPU MSRP

GCP publishes per-chip-hour rates by variant + allocation class. Table
(snapshot 2026-06-02 from https://cloud.google.com/tpu/pricing, USD,
europe-west4 region):

| variant | on-demand | preemptible |
|---------|-----------|-------------|
| v5p     | $4.20     | $1.68       |
| v6e     | $2.70     | $1.08       |

Per-attempt MSRP = `(wallclock_sec / 3600) × num_chips × rate`.

`num_chips` comes from the variant suffix (`v6e-16` → 16). Allocation
class is read from the iris job spec (the `tpu` arg parses as
`v6e-preemptible_16` etc. — see `iris-pool-naming` memory).

Sum across all per-task attempts (the `attempts` sidecar covers this).

## Lifecycle / who writes when

### Author: `tomat cost compute <run_label>` (new CLI subcommand)

A single command that:

1. Reads the run's manifest + attempts sidecar from R2.
2. For Modal runs: queries Modal for each call's stats (filter
   `function_calls` by `function_name=eval_checkpoint_h200x8` AND
   `train_bakeoff_h200x8` AND a label-matching argument).
3. For iris runs: walks the attempts sidecar, multiplies per-attempt
   wallclock × MSRP from the table.
4. Writes `runs/<run_label>/cost.json` to R2.

Idempotent — re-runs overwrite the sidecar with fresh data. Add an
`--all` flag to iterate every synced run.

### Cron hook: extend `scripts/cron_iris_sync_modal.py`

The existing per-minute cron already snapshots iris + Modal state.
Add a `tomat cost compute --all-running` call to keep cost.json
fresh-ish for in-flight runs. Idempotent + cheap (~1 Modal API call
per run).

## Read path: dashboard

`RunHeaderRich` (between `MV` and `FLOP` in the metrics row, or its
own line if compact):

```
… · MV 1.23% · $730 MSRP · 2.8e19 FLOP
```

Tooltip on hover:

```
Estimated MSRP
─────────────────────────────────
Modal H200×8 · 20.0h · $36.48/hr → $730.61
─────────────────────────────────
Total: $730 (computed 03m ago)
Snapshot pricing 2026-06-02 — published rates, not actual billing.
```

For runs with multiple segments (iris cascade restarts, Modal
respawns), show one row per non-trivial segment, sorted by
contribution desc.

## Phases

**Phase A — TPU first** (done): `tomat cost compute` CLI for iris
jobs. Sidecar + CFW endpoint + dashboard chip + tooltip. Modal returned
a `Modal MSRP TBD` placeholder.

**Phase B — Modal + auto-population** (done): Modal MSRP is derived
from wandb-tracked wallclock (`history.ts_max - ts_min`) × published
per-GPU-hour rates, with a flat 5% CPU/memory adder. The same wandb
fallback covers TPU runs whose attempts sidecar undercounts vs
`job_failure_count + job_preemption_count` (iris's bug-report drops
prior attempts). Both `tomat runs sync` (laptop) and
`scripts/runs-sync.py` (cron VM) auto-populate `cost.json` after every
manifest write — no manual `tomat cost compute` step required.

Trade-off vs spec's original Phase B (Modal billing API): the wandb
ts-span path requires zero Modal-API plumbing, charges all sessions
of the same wandb run id (so it handles cascade restarts correctly),
and matches empirical billing within ~1% for our H200×8 runs (the
flat adder is essentially a noise margin). The original "query
Modal's billing API" approach remains an option if higher-accuracy
attribution is ever needed; current impl is good enough for the
public-facing chip.

**Phase C — cron hook** (folded into Phase B): the per-manifest write
in the cron's `runs-sync.py` calls `_compute_and_upload_cost`
inline. The separate `cron_iris_sync_modal.py` does NOT need a
parallel hook — it only syncs iris-state, not per-run state.

**Phase D — refinements**: per-region pricing, distinguishing
preemptible vs reserved TPU at finer granularity (current wandb-span
fallback always uses preemptible), chips per run-card on the `/runs`
list view (already lands wherever `RunHeaderRich` is rendered, so
shipped with Phase B).

## Open questions

- **Multi-region rate variance**: v6e in `europe-west4` vs
  `us-east5` differs by ~2-3%. Single-region MSRP for v1; add a
  `region` field to the breakdown row for future refinement.
- **Eval-job cost as a child**: spec 43 already models eval jobs as
  run children. Each eval-job record could carry its own
  `msrp_usd`. Roll-up into parent's total. Phase C-ish.
- **Per-day burn chart**: out of scope here; a future
  cross-run dashboard view.
- **MSRP table refresh cadence**: prices change quarterly. Manual
  edits to the table with a date-stamped comment is fine; auto-pull
  from a published JSON feed is overkill for v1.

## Memories worth linking from impl PRs

- `feedback_no_funding_in_public_repo` — public framing rules.
- `iris-pool-naming` — TPU variant + allocation-class string parsing.
- Spec 43 — eval-job-as-child model; cost roll-up applies analogously.
