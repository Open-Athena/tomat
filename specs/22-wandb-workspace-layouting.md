# 22 — Wandb workspace layouting

Programmatically manage the wandb workspace for `tomat-lmq-P14` (and
sibling projects, when we add them) so panel layouts are version-
controlled, reviewable, and reproducible across operators / machines.

## Motivation

For the immediate "I want grouped `eval/mat_nmae/{set}/{stat}` panels
on every run" goal, wandb's UI **Save personal workspace template**
(in the workspace `•••` menu) is sufficient — one click per
operator, applies project-wide. No code needed.

Reasons to build a scripted layer on top of that:

1. **Versioning.** A workspace layout is a piece of project
   configuration; checking it into git means layout changes are PR-
   reviewable rather than silent UI clicks. Recovers state if a
   workspace is accidentally reset / corrupted.
2. **Multi-operator parity.** Per-user templates are personal —
   teammates each have to click their own. A scripted layout pushed
   as a *named view* (visible to anyone in the workspace) gives one
   canonical layout we all reference.
3. **Multi-project rollout.** As we spin up sibling projects (per
   patch size, per codec generation, per architecture), we want the
   same panel set in each without re-clicking.
4. **CI-style invariants.** "Project X must have panels Y" can be
   asserted in a script that runs in CI; drift gets caught.

## Out of scope

- Reports (the curated, narrative-style read-only views). Those are
  better done in the UI for now; they're typically one-off
  artifacts, not template targets.
- Run-level metadata / config grouping (handled at training-script
  level via `WandbConfig.tags`, `group`, etc.).
- Sweeping panel definitions across runs of *different* projects with
  divergent metric schemas — start with one schema (`tomat-lmq-PN`)
  and generalize later.

## Stack

- **Library:** [`wandb-workspaces`](https://pypi.org/project/wandb-workspaces/)
  (separate from main `wandb` SDK; install via
  `pip install wandb[workspaces]`). Provides
  `wandb_workspaces.workspaces.Workspace`, `Section`,
  `wandb_workspaces.reports.v2.LinePlot`, etc.
- **Bug to work around:** `Workspace.from_url(...)` raises
  `Workspace `` not found in project …` if the project has only an
  auto-generated default workspace and no saved view. Workaround: do
  one UI Save-as-new-view first to materialize a queryable workspace,
  *or* construct a fresh `Workspace(...)` from scratch and `.save()`
  it without trying to load existing.

## Proposed shape

A new `tomat workspace` subgroup in the `tomat` CLI, with declarative
layout config in a Python module under `src/tomat/wandb_layout.py`
(or similar). Layout config is the source of truth; CLI commands push
that to wandb.

```bash
tomat workspace dump <project>          # fetch current saved workspace → file
tomat workspace push <project> [-v VIEW] # apply layout config as named view
tomat workspace diff <project>          # diff layout config vs live state
tomat workspace template <project>      # save as user's personal template
```

Layout module structure:

```python
# src/tomat/wandb_layout.py
EVAL_SECTION = Section(
    name="eval",
    panels=[
        LinePlot(title="eval/mat_nmae/val_200",
                 metric_regex=r"eval/mat_nmae/val_200/(p99|median|mean)"),
        LinePlot(title="eval/mat_nmae/train_200",
                 metric_regex=r"eval/mat_nmae/train_200/(p99|median|mean)"),
        # … any other eval panels we want canonical
    ],
)

TRAIN_SECTION = Section(name="train", panels=[...])

DEFAULT_LAYOUT = Workspace(sections=[EVAL_SECTION, TRAIN_SECTION, …])
```

`tomat workspace push tomat-lmq-P14 -v default` then materializes
this as a saved view in the project; `--template` instead saves it
as the user's personal template.

## Phases

### Phase A — single named view

Single layout (`DEFAULT_LAYOUT`), single project (`tomat-lmq-P14`),
single CLI command (`tomat workspace push`). Wires up the
`wandb-workspaces` dependency, validates that the regex grouping
trick works end-to-end, replaces the manual UI flow for our most-
used view.

Acceptance: running `tomat workspace push tomat-lmq-P14` from a
fresh checkout produces a workspace identical to the one I currently
have set up by hand.

### Phase B — diff + template

Add `tomat workspace diff` (compare live state vs config; surface
panels that exist in one but not the other) and `tomat workspace
template` (push as user's personal default rather than as a named
view). Diff is the CI-style invariant — flag drift, e.g. "someone
added a panel in the UI; either commit it to the layout config or
remove it".

### Phase C — multi-project + per-codec specialization

Generalize the layout module to take a project descriptor (codec
name, patch size, dataset) so the same Python config can target
sibling projects. Add a section-level "applies to projects matching
this filter" mechanism if some sections only make sense for some
projects.

## Open questions

1. Section-level `is_open` / collapse state: worth versioning, or
   leave to per-user UI preference? Lean toward versioning the open
   default and letting users override locally.
2. Panel ordering within a section: does the layout config preserve
   insertion order? (Almost certainly yes, but verify.)
3. Color assignment for the 3 traces in a grouped panel: wandb
   auto-assigns; if we want stable colors (e.g. p99 always red), the
   layout config needs explicit `line_colors`. Probably not worth it
   in Phase A.
4. Does `Workspace.save()` without an explicit name update the
   project's default workspace, or fail / no-op? If it updates the
   default, that's a footgun — easy to wipe a teammate's setup. Push
   should probably default to a named view, with `--template` being
   the explicit opt-in for default-workspace mutation.
5. Auth: does `wandb-workspaces` need anything beyond `WANDB_API_KEY`
   that the rest of `tomat` already uses? Spot-check during Phase A.

## Dependencies / risks

- `wandb-workspaces` is a separate optional install. Consider whether
  to add it to `pyproject.toml` as an optional extra (`[workspaces]`)
  vs. a required dep. Optional extra is cleaner — `tomat workspace`
  commands fail with a clear "install with `pip install wandb[workspaces]`"
  message; everything else keeps working.
- The library is relatively new / API can shift between releases.
  Pin a version in the project's lock file once Phase A is wired up.

## Priority

Low. The "Save personal workspace template" UI button covers ~80% of
the practical value (auto-apply this layout to all my runs in this
project) for one click. Phase A is worth doing the next time we
either change the layout meaningfully *or* onboard another operator
who wants the same setup; until then, the manual flow is fine.
