# Parent TL overlay on `/runs/<child>` WallclockPlot

**Status**: draft. Small follow-up; no dependencies blocking it.

**Triggering observation** (2026-06-25): when reviewing `bin5-cont-from-80k-v6e` (the
HW-switch hypothesis test fire), it took a CLI wandb pull to confirm the spike's magnitude
because the child run page doesn't render the parent's TL trajectory. The y-axis spike was
visible (~11.5 → 7.6 settled) but quantifying it as "+1 nat above parent" required pulling
the parent's TL at the same step out of band:

| run | TL just-pre-resume (parent@same-step) | TL at first post-resume step | settled TL |
|---|---|---|---|
| `cont-from-80k-v6e` | 6.71 (parent step-80k) | 8.16 | 7.55 |
| `cont-from-99k` | 6.54 (parent step-99k) | 8.16 | 7.56 |
| `cont-clean` | 6.38 (parent step-100k) | 8.13 | 7.51 |

Same +1.5 nat spike → +1.0 nat sustained gap from parent's trajectory, regardless of HW.
The visual story isn't legible from the child page alone — you'd think the child trained
fine and just settled at 7.55, missing that the parent was at 6.4-6.7 at the same step.

This spec wires a parent-TL overlay into `WallclockPlot` so the comparison is one-glance.

## Goals

- When a child has `parent_run_id` declared (via wandb config `TOMAT_PARENT_RUN_ID` or via
  the fire manifest's `parent_run_id` field), the child's WallclockPlot fetches and renders
  the parent's TL trajectory as a faded same-color trace.
- The parent's trace covers the x-range up to the child's resume step (+ a small lead-in, e.g.
  100 steps before, for visual context — but no overlap past the resume point).
- The parent's trace is fetched on-demand (lazy) — don't pay the fetch cost for runs the user
  doesn't drill into. Cache per parent-run-id.
- The header shows the parent as a link (e.g. `parent: train-mg-kl-bin5-fs-tpu ↗`) next to
  the existing run-state badge.
- The parent trace is toggleable via the existing legend (clicking it dims, just like the
  other traces).

## Non-goals

- No multi-parent / lineage chain rendering. If parent has its own parent, we still only
  fetch one hop up.
- No MT/MV/VL overlay from the parent. Just TL on the WallclockPlot.
- No restyle of the existing plot beyond the new trace. Smoothing / zoom / units stay as-is.
- Not extending to MEvalTable (that's a separate pending task, #334).

## Where the data lives

- **`parent_run_id`**: child run's wandb config has `TOMAT_PARENT_RUN_ID` (set by `tomat
  train --parent` in spec 61 Phase 1, already landed). The R2 fire manifest also carries
  `parent_run_id` at the top level (spec 61 §2.1 / fires-as-records). Either works; wandb is
  the simpler read because the `RunHeaderRich` component already has the wandb config in
  scope.
- **Parent's TL trajectory**: served by the existing `tomat/runs/<parent>/raw.parquet` route
  (the same route used by the WallclockPlot for the child). Cron syncs the parent's full
  history every minute (spec 55 / task #208).

## CLI surface

Already in place. `tomat train --parent <P>` writes `TOMAT_PARENT_RUN_ID` to wandb config;
the fire manifest carries `parent_run_id` in R2. No new CLI work.

## FE work

### 1. `RunHeaderRich`: surface parent link

Read `wandbConfig.TOMAT_PARENT_RUN_ID` (or fallback to the fire manifest's `parent_run_id`).
Render as a small inline link next to the existing target / synced-ago lines:

```tsx
{parentRunId && (
  <span>
    {' · '}
    parent: <Link to={`/runs/${parentRunId}`}>{shortName(parentRunId)}</Link>
  </span>
)}
```

### 2. `WallclockPlot`: fetch + overlay parent TL

- If `parentRunId` is present, kick off a TSQ query for the parent's `raw.parquet` (key it
  on `parentRunId`).
- Render the parent's TL as a separate trace, styled `dash: 'dot'` + `opacity: 0.35`,
  named `parent TL` in the legend.
- Trim the parent trace to `parent.x[i] ≤ childResumeStep + 100` (don't extend past where
  the child takes over — it's misleading, since the parent often kept training after the
  ckpt was forked).
- `childResumeStep` is inferable from the child's first step in `history.global_step`
  (since both children + same-label resumes save the global step counter).

### 3. Legend / hover

- Parent TL gets its own legend entry. Click-to-toggle works via existing legend code.
- Hover on the resume boundary should show both the parent's last TL and the child's first
  TL — the existing x-unified hover (post the recent fix) handles this for free once the
  parent trace is added; no extra work.

## Phasing

### Phase A — Surface parent link in header (15 min)
- Add the inline `parent: <Link/>` to `RunHeaderRich`. No data fetch; just config read.
- This alone makes parent lineage discoverable from the child page.

### Phase B — Overlay TL on WallclockPlot (1-2 hours)
- TSQ query for parent's raw.parquet (lazy, keyed on `parentRunId`).
- Render parent trace + trim past the resume boundary.
- Style: dotted, faded, distinct legend entry.

### Phase C — Optional: parent VL too (~30 min on top of B)
- Same pattern for parent VL trace, when available. Skip if parent doesn't have VL data.

## Open issues

1. **Resume step inference**: `childResumeStep = first(child.global_step)` is correct for
   `--from-ckpt` resumes (Levanter restores the counter). For warm-starts (fresh counter),
   the parent overlay would extend past the child's first step, which is the wrong story.
   Heuristic: if child.first_step is < 100, treat as warm-start and don't trim — just show
   the parent's full trajectory as context. (Warm-starts are rare; we hit them in 1B
   cont-from-4711 long ago. Not high priority to gate on.)
2. **Multiple `--parent` ancestors**: if cont-from-80k is itself a parent for another fire,
   the grandchild's overlay would only show cont-from-80k, not bin5. Acceptable — one hop is
   the readable level. Anyone needing more clicks through.
3. **Parent run is in a different wandb project**: rare today (everything's in
   `tomat-lmq-P19`) but possible. The parquet route is per-project-aware; need to surface
   the parent's project in the manifest or default-fallback to the child's project.

## Cross-references

- Task #176 (completed) — Parent/child lineage at /runs CARD level. This spec extends that
  to the detail page.
- Task #334 (pending) — MEvalTable inherits parent evals. Sibling concern; orthogonal impl
  but same parent-pulling discovery via `parent_run_id`.
- Spec 61 §2.1 — fire manifest schema (where `parent_run_id` is canonically stored in R2).
