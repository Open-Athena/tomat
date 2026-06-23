# Resume invariants + run-record ownership

**Status**: draft — Phase 1 to land first; Phase 2 dependent on Phase 1 demonstrating the
invariant model is right.

**Triggering incidents** (2026-06-20 → 2026-06-22):

- `bin5-extend-10k.sh`: extension of bin5 from step-100000 with `TOMAT_LABEL=train-full-v3`
  (single-shard TS0) when bin5's last 33k steps had been on TS0123 union. Caused a sustained
  TL spike at the resume step.
- `bin5-sigma-cont.sh` (× 3 fires — cont-s3/s10/s20): same TS0-instead-of-TS0123 mistake,
  on top of an intentional σ change. All 3 also spiked at the resume step in a way *not*
  fully explained by σ math alone.
- I had a memory rule about "TS0123 on resume" but it lived in the agent's MEMORY.md
  and was not consulted before authoring 4 fire scripts in a row that violated it.

The user's diagnosis (correct): "memory feels too weak — we need scripts/CLIs that
**structurally** encourage right behavior, not memorized rules." This spec is the structural
fix.

## Pain points the spec addresses

1. **Eval inheritance is broken on child run pages.** A child run's `/runs/<child>` page only
   shows the child's own evals, even though the parent ckpts had their own m-evals fired.
2. **Resume drift silently corrupts runs.** Any HP delta vs. the parent — data label, LR,
   schedule, warmup, σ, loss type, batch size, t10n version — can break the resume in ways
   that don't get caught until the TL plot reveals a spike.
3. **The "what is a run" abstraction is muddled.** Wandb run ID, ckpt lineage, fire-id, and
   data-label cohort are tangled. Asking "the bin5 run" can mean five different things.
4. **Whack-a-mole fatigue.** The same class of bug recurs every fire because each fire is
   a fresh bash script with no structural check that it matches its parent.

## Goals

- A fire that resumes from a parent ckpt declares its parent **explicitly** and is **refused
  by the CLI** if any of {data label, LR, schedule, warmup, σ, loss type, codec version,
  batch size, model size, patch size} differs from the parent's last committed segment —
  unless the caller passes `--allow-config-change=<key>,<key>,…` to enumerate the intended
  deltas.
- The set of "intended deltas" lands in the child's wandb config + R2 manifest so the
  dashboard can annotate the resume boundary with "loss changed: σ=5→σ=10" or "data label
  changed: TS0123→TS0".
- A child run's m-eval table on the dashboard inherits its parent's eval rows (rendered
  as "from parent", faded) so the lineage of m-eval points is visible without clicking up.
- `TOMAT_WARMUP=0` becomes the default on `--resume` (not a memory rule).
- A clear story for what changes a config delta is *expected* to cause — vs. an
  unexplained TL shift that needs investigation.

## Non-goals

- Auto-detecting the **cause** of the bin5 resume spike. That's a separate investigation —
  this spec just ensures next time the spike happens, we know which HP changed (because
  the CLI either rejected the fire or explicitly logged the delta).
- Rewriting any in-flight training code. The guard sits at the CLI layer
  (`tomat train --resume`); the trainer itself doesn't change.
- Replacing wandb. The wandb run ID becomes one cross-ref in the manifest, not the
  canonical key — but wandb stays as the metric sink and run viewer.

## Phase 1 — invariants enforced in the fire path

### 1.1 `--parent` declaration

Every fire path becomes one of three:

```bash
tomat train --from-scratch --label <new-run-id> [flags…]
tomat train --resume --parent <parent-run-id> [--allow-config-change=…] [flags…]
tomat train --warm-start --parent <parent-run-id> --reinit-optimizer [flags…]
```

`--from-scratch` and `--resume` are mutually exclusive. `--resume` without `--parent` is
an error. `--warm-start` is the explicit "load weights, throw away optimizer state, fresh
schedule" path — config drift is allowed by default (it's a new run conceptually) but the
parent is still recorded for lineage.

The `--parent` flag flows into:
- The child's wandb config as `tomat.parent_run_id = <parent>`.
- The child's R2 manifest (`/api/runs/<child>/manifest.json`) as `lineage.parent` +
  `lineage.parent_last_step`.
- The fire script's iris job name (for grep-ability).

### 1.2 Resume-config guard

`tomat train --resume --parent P` does:

1. Pulls P's wandb config (canonical source) + R2 manifest (for `lineage.applied_deltas`).
2. Builds a child config from the CLI flags + env vars.
3. Computes a **diff** over a well-known list of "frozen on resume" keys:

   ```python
   FROZEN_ON_RESUME = {
       "data.cache_dir",         # incl. label / shard layout
       "data.shuffle.window_blocks",
       "data.tokenizer",         # codec version
       "data.components.*.cache_dir",
       "model.hidden_size",
       "model.num_layers",
       "model.num_heads",
       "model.intermediate_size",
       "trainer.train_batch_size",
       "trainer.train_seq_len",
       "optimizer.lr_schedule",
       "optimizer.learning_rate",
       "optimizer.warmup",
       "loss.type",              # e.g. ce, emd, kl_gauss, crps
       "loss.kl_sigma",
       "loss.density_only",
   }
   ```
4. For each key in the diff:
   - If listed in `--allow-config-change`, mark it as "intentional delta" and continue.
   - Otherwise, refuse the fire with a diff like:

     ```
     refusing to resume bin5@100000:
       data.cache_dir:
         parent: gs://marin-us-east5/tomat/cache/train-full-v3-shard{0,1,2,3}/
         child:  gs://marin-us-east5/tomat/cache/train-full-v3/
       loss.kl_sigma:
         parent: 5
         child:  10

     to override: --allow-config-change=data.cache_dir,loss.kl_sigma
     (these will be recorded as intentional deltas in the child manifest)
     ```
5. The intentional deltas land in the child's wandb config as
   `tomat.intended_resume_deltas: ["data.cache_dir", "loss.kl_sigma"]`.

This is the structural fix. The bin5+10k extension would have been refused because
`data.cache_dir` differed. The σ-cont fires would have been refused unless I passed
`--allow-config-change=loss.kl_sigma,data.cache_dir` — at which point the data-label change
would have been a conscious decision, not a copy-paste accident.

### 1.3 `TOMAT_WARMUP=0` default on resume

`tomat train --resume` defaults `TOMAT_WARMUP` to `0` unless explicitly overridden. The
warmup re-application question goes away.

`TOMAT_WARMUP` is added to the resume-config diff machinery as a soft check (warn, don't
refuse) — overriding it logs a warning that future analysis should account for it.

### 1.4 MEvalTable parent-eval inheritance

The run-detail page reads `lineage.parent` from the manifest. If set:
- Pull the parent's eval.json.
- Pre-pend its eval rows to the child's MEvalTable, rendered with a faded background and a
  small "(from parent)" badge in the step cell.
- Annotation line on the WallclockPlot marks the resume boundary + the intentional deltas
  (e.g. "σ: 5 → 10").

The phantom-row filter from cb37232 stays — only K=1 maskgit rows that match a displayed
column get pulled in.

### 1.5 Epoch counter respects parent's label history

Currently the `/runs/<run>` page computes epoch as `(child.tokens / child.label.size)`.
If the child resumed from a parent with a different data label, this is wrong — the model
saw more tokens than the child's own count, and the epoch number for the child's label is
also wrong because the parent contributed to it.

Walk the parent chain via `lineage.parent`. For each segment, accumulate tokens consumed
per data label. Display the epoch as a per-label tuple: `epoch: TS0=1.4, TS0123=0.8`.

### 1.6 Fire-script generator

A `tomat fires new --kind {fs,resume,warm-start} --parent P --tag …` command that emits a
new `scripts/fires/<name>.sh` with all the resume-required flags pre-filled from the
parent's manifest. Removes the hand-authoring step that introduced the TS0 mistake 4
times.

## Phase 2 — layered source of truth (sketch)

Phase 1 makes the wandb config the source of truth for the resume guard. **That's a
weakness.** Wandb config is overwritten by every resume — bin5's current wandb config
no longer reflects its TS0123 mid-life state because `bin5-extend-10k.sh` overwrote
it with TS0. The guard catches drift against the *last fire's overwrite*, not the
parent's prevailing state. We can't backfill the truth either: wandb config isn't
ours to edit.

Phase 2's reframe: **separate "what was fired" (immutable, our record) from "what
the canonical thread should look like" (editable, our curation) from "metrics" (wandb,
their record).** Three layers, no single one is the source of truth — each is
authoritative for its concern:

| Layer            | Where                                    | Authority over                                 | Editable? |
|------------------|------------------------------------------|------------------------------------------------|-----------|
| **Fire**         | `R2/openathena/tomat/fires/<fire-id>/`   | what was actually fired (env-vars, manifest)   | NO — immutable provenance |
| **Run**          | D1 (or SQLite during dev) `runs` table   | canonical "thread" for a logical training arc  | YES — we can backfill / correct |
| **Wandb**        | wandb.ai/open-athena/...                 | metric trajectories (TL, VL, m-evals, …)       | partial — config we don't own |
| **Manifest**     | `R2/.../runs/<run-id>/manifest.json`     | cached aggregate of the above for FE reads     | regenerated, not edited |

### 2.1 Fires: immutable provenance

Every fire — iris OR Modal — writes:

```
R2: openathena/tomat/fires/<fire-id>/
  manifest.json     # frozen at fire-spawn time; substrate-tagged (see schema below).
  raw.parquet       # history this fire emitted, suffix-appended as the fire runs
                    # (immutable once the fire reaches a terminal state)
  eval.json         # m-eval points fired against THIS fire's ckpts (not the run's
                    # other fires); merged at view-time
```

`fire-id` is e.g. `<UTC-isoformat>-<8-hex-shasum-of-manifest>`. Any fire entry point
(`tomat train`, `tomat evals fire`, `scripts/fires/*.sh` calling `tomat fires record`
post-submit) writes this directory at submission time so we have the manifest even
if the fire never starts. The trainer appends parquet rows as it goes (no overwrites).

**Schema** (the substrate determines which `<X>_id` field is populated; `tomat
fires record` validates exactly one):

```jsonc
{
  "schema_version": 1,
  "fire_id": "2026-06-23T17:27:07Z-9f3a1c4e",
  "run_id": "train-mg-kl-bin5-cont-from-80k-v6e",  // canonical run name
  "parent_run_id": "train-mg-kl-bin5-fs-tpu",      // editable; null for from-scratch
  "fired_at": "2026-06-23T17:27:07Z",
  "fired_by": "ryan",                              // getpass.getuser()
  "fired_host": "ryan-laptop",                     // platform.node()
  "fire_argv": ["scripts/fires/bin5-cont-from-80k-v6e.sh"],
  "git_sha": "f51b9ee...",                         // local repo HEAD at fire time
  "marin_pin": "abc1234...",                       // marin/ submodule SHA

  // Exactly one substrate block:
  "substrate": "iris",                             // | "modal"
  "iris": {
    "job_id": "/ryan/train-mg-kl-bin5-cont-from-80k-v6e",
    "tpu": "v6e-16", "zone": "us-east5-b", "priority": "interactive"
  },
  // OR:
  "modal": {
    "app_name": "tomat-train-h200x8",
    "function_call_id": "fc-...",
    "fn": "train_bakeoff_h200x8",
    "region": "us-east-1"
  },

  // wandb is ALWAYS captured (every fire emits to a wandb run, regardless of
  // execution substrate):
  "wandb": {"entity": "open-athena", "project": "tomat-lmq-P19",
            "run_id": "..."},

  // Intended deltas vs parent at fire time — used by the resume guard + the
  // dashboard's "what changed" diff. Optional for from-scratch fires.
  "intended_deltas": {
    "data.cache_dir": {"from": "train-full-v3", "to": "train-full-v3,...shard3"}
  },

  // Resume / horizon knobs (per-substrate cared about by the trainer; mirrored
  // here so the fire record can be read without a separate trainer.json):
  "horizon": {"resume_from_step": 80000, "target_steps": 82000}
}
```

A future Phase 2.x can add `substrate: "hybrid"` for fires that fan out across
iris + Modal in one logical step, but in practice every concrete fire to date is
one substrate only. Keep the schema simple until we have a real hybrid case.

### 2.2 Runs: editable canonical thread (1:N over wandb + iris + Modal)

**Core insight (Ryan 2026-06-23, expanded 2026-06-23 PM after Betsy
discussion):** a "run" in our model is *not* 1:1 with any single execution
substrate. It's a composition over **three** independent multi-valued
relations:

| substrate | what counts as "one" | example |
| --- | --- | --- |
| **wandb runs** | one `wandb_run_id` (per entity+project) | the original bin5 run-id; a child-resume's separate wandb id |
| **iris jobs** | one `/<user>/<label>` job path | `/ryan/train-mg-kl-bin5-fs-tpu`; each cont-clean is its own iris job |
| **Modal function calls** | one `function_call_id` for the long-running training/eval call | each `modal run train_bakeoff_h200x8.py` = one fc; modal-v4-epochwin's H200×8 training was one fc |

A single canonical run can compose any (N, M, P) where each is ≥ 0 and
their sum is ≥ 1. Examples in our current fleet:

- **bin5-fs-tpu:** N=1 wandb (mid-life flip rejoined the same id), M=1 iris
  job (~73 preempt-restarts within), P=0 Modal — pure iris track.
- **modal-v4-epochwin:** N=1 wandb, M=0 iris, P=1 Modal fc — pure Modal
  track. The dashboard today shows it with a Modal-app chip and *no* iris
  chip, which is correct in spirit but accidental (we only got one shot
  right, not N:M:P right).
- **modal-v4-cont-clean:** N=1 wandb, M=1 iris (v6e-16 us-east5-b
  continuation), **inherits P=1 Modal fc from its parent** through
  `parent_run_id`. The chip strip should reflect both substrates as the
  thread crosses them.

The current FE encodes 1:1:1 (single wandb link + single iris link + single
Modal link in the run-page header); that assumption is wrong on all three
axes and needs to be migrated out aggressively across the code. The `runs`
table is what makes the 1:N:M:P relation explicit:

```sql
-- D1 / SQLite schema (dev: SQLite under tmp/runs.db; prod: D1 binding on CFW)
CREATE TABLE runs (
  run_id              TEXT PRIMARY KEY,     -- "bin5", "bin5-cont-s10", etc.
                                            -- our canonical name, not wandb's.
  display_name        TEXT,                 -- nicer label for the dashboard
  parent_run_id       TEXT,                 -- editable; backfill-able for old runs
  fire_ids            TEXT NOT NULL,        -- JSON array, ordered chronologically
  wandb_run_ids       TEXT NOT NULL,        -- JSON array of (entity, project,
                                            -- wandb_run_id) tuples — N>=0 wandb
                                            -- runs that contributed to this
                                            -- thread (almost always >=1, but
                                            -- nothing demands it). Replaces the
                                            -- 1:1 wandb link in header.
  iris_job_ids        TEXT NOT NULL,        -- JSON array of /ryan/<label> iris
                                            -- job paths — M>=0 jobs (Modal-only
                                            -- runs have M=0).
                                            -- Replaces 1:1 iris link in header.
  modal_function_calls TEXT NOT NULL DEFAULT '[]',
                                            -- JSON array of {app_name,
                                            -- function_call_id, fn, started_at,
                                            -- finished_at, region} — P>=0 Modal
                                            -- fcs that contributed (TPU-only
                                            -- runs have P=0). Replaces 1:1
                                            -- modal-app heuristic.
  blacklisted_fires   TEXT NOT NULL DEFAULT '[]',  -- JSON array of fire-ids whose
                                            -- history rollup the view should skip
                                            -- (bad fires we want hidden, not deleted)
  segment_overrides   TEXT NOT NULL DEFAULT '{}',  -- JSON; manual annotations like
                                            -- "step-67000: data label switched to
                                            -- TS0123" — what annotations.ts holds today
  notes               TEXT,                 -- free-form for human context
  created_at, updated_at  TIMESTAMP
);
```

**Why function_call, not app, for Modal:** a Modal `App` is the deployed
unit (long-lived, shared across many invocations); a `function_call` is a
single `.remote()` / `.spawn()` invocation, and that's the granularity we
care about for "what trained this run". `modal-v4-epochwin` = one
`train_bakeoff_h200x8` function call; `modal-v4-cont-clean`'s parent
inheritance points to that same fc. App-level grouping happens at render
time (dedup on `app_name` if the chip strip needs it).

The 1:N:M:P migration touches every place the FE assumes a single execution
on any substrate:

| substrate | site | today's assumption | migration |
| --- | --- | --- | --- |
| iris | `RunHeaderRich.tsx:1386` (chip) | `\`/ryan/${data.id}\`` | render M chips, one per `iris_job_ids[]` |
| iris | `RunsPage.tsx` `irisJobIdForRun(id)` lookups | single job per run | take a `runs.iris_job_ids[]` and aggregate state across them |
| iris | iris-state badges, preempt counts | picks "the one" iris job | aggregate over M |
| iris | `cost compute` per-run | one job's chip-hours | sum across `iris_job_ids[]` |
| wandb | header wandb chip | single project/id | render N chips |
| wandb | WallclockPlot trajectory union | one parquet stream | concat metrics from all N (Phase 2 manifest aggregate handles this upstream) |
| Modal | `modalAppForRun(modal, id)` | single deployed app | render P chips, one per `modal_function_calls[]` |
| Modal | Modal "isRunning" badge | per-app | per-fc (already partially done — see memory `feedback_dashboard_modal_isrunning_per_fc`) |
| Modal | `pendingFcForRun` | single fc | walk `modal_function_calls[]` |
| sync  | `tomat runs sync` | one wandb run + one iris job per run | walk all N+M+P; dedup; merge into the `runs` row |

Migration strategy: **add the new columns alongside the existing 1:1 fields,
populate from inferred lineage during sync, double-render in the FE
("primary wandb: X · also: [Y, Z]") for a window, then cut over once every
runs row has its wandb_run_ids / iris_job_ids backfilled.**

Critical: the `runs` row is **editable** — we can set `parent_run_id = 'train-mg-kl-bin5-fs-tpu'`
on `cont-s10` even though that fire's wandb config doesn't carry it (because cont-s10
was fired before `tomat train --parent` existed). The FE reads from this table, not
from wandb config.

The `runs` table is also what closes the "memory feels too weak" gap: instead of
relying on `annotations.ts` comments + bash history to reconstruct "what bin5 actually
ran at step 67k", that knowledge lives in `runs.segment_overrides` as structured data,
queryable across the dashboard.

### 2.3 Manifest = cached aggregate

`R2/.../runs/<run-id>/manifest.json` becomes a derived artifact. `tomat runs sync`
regenerates it by:

1. Reading the `runs` table row for `<run-id>`.
2. For each `fire_id` in `fire_ids` (excluding `blacklisted_fires`), reading that
   fire's manifest + parquet.
3. Merging them (deduping on `_step`, preferring later fires on overlap).
4. Joining metric trajectories from wandb (the only thing wandb is still authoritative
   for).
5. Writing the aggregate to R2.

The FE reads from this aggregated manifest. No FE logic changes — the join just
happens upstream.

### 2.4 Blacklist mechanism

Bad fires (like the bin5+10k extension) get added to `runs.blacklisted_fires`.
The view's trajectory rollup skips them. The fire's `R2/fires/<fire-id>/` records
stay forever (provenance is precious; the bad fire happened and someone might want
to know why later). The ckpts they produced stay on GCS until GC'd by the standard
retention rule. Nothing destructive.

### 2.5 Migration path

Phase 2 is delivered in **dev mode first**, prod-second:

1. **SQLite under `tmp/runs.db`** while the schema settles. `tomat runs ls` / `tomat
   runs edit` / `tomat runs set-parent` subcommands let us populate + correct rows
   by hand.
2. **Manual backfill** of the editable rows for the dozen runs we actually care
   about (bin5 + family). Surfaces schema gaps quickly.
3. **Worker reads from new layout, falls back to old `runs/<run-id>/` layout** when
   no `runs` row exists. `tomat runs sync` writes both layouts during the transition.
4. **Promote to D1** once the schema is stable and we trust the manual backfill.
   `tomat runs` subcommands speak to D1 instead of SQLite. FE auto-cuts over.

### 2.6 Region-agnostic resumes

Currently each fire pins its `--zone` and its `TOMAT_BUCKET` (ckpt region) at fire
time. Pre-emption auto-resumes inherit the same spec, so a starving zone wedges
the run forever. Manual re-fires can pick any zone but pay cross-region ckpt-read
costs unless we mirror.

Phase 2 surfaces region preferences as **first-class run state**:

```sql
ALTER TABLE runs ADD COLUMN
  preferred_tpus    TEXT NOT NULL DEFAULT '[]', -- JSON, e.g.
                                                -- [{"tpu":"v5p-16","zones":["us-east5-a"]},
                                                --  {"tpu":"v6e-16","zones":["us-east5-b","eu-west4-a"]}]
  ckpt_regions      TEXT NOT NULL DEFAULT '[]'; -- JSON, e.g. ["us-east5","eu-west4"]
                                                -- (used by --zone widening + by the
                                                -- `tomat ckpt mirror` op when adding
                                                -- a new fallback region)
```

`tomat train --resume <run-id>` reads these and:
1. Widens its `--zone` to every zone in `preferred_tpus` whose TPU pool has capacity *and*
   whose region appears in `ckpt_regions`.
2. Sets `TOMAT_BUCKET` to whichever region the chosen zone is in.

Pre-emption auto-resumes inherit the widened `--zone`, so a starving zone hand-off to
its peer just works. New regions get added to `ckpt_regions` only after `tomat ckpt
mirror` finishes copying the ckpt there, so we never fire into a region that can't read.

### 2.7 Why this works where Phase 1 alone doesn't

The bin5-extend-10k mistake would be **caught** by Phase 1's guard *only if* bin5's
wandb config still reflected its TS0123 prevailing state at fire time. It didn't.

Under Phase 2:
- `bin5-extend-10k` would be a `fire-id` in `runs.bin5.fire_ids`, recorded
  immutably.
- The drift guard would diff against `runs.bin5.fire_ids[-1]`'s **immutable fire
  manifest** (which captures the env-vars of bin5's *last clean fire*, before
  bin5-extend overwrote anything).
- After we discovered the spike, `bin5-extend-10k` would land in
  `runs.bin5.blacklisted_fires`, and bin5's trajectory plot would drop it.
- The `runs.bin5.parent_run_id` for any future `cont-clean`-style fire would point
  at the *clean* fire, not the contaminated one.

The Phase 1 guard becomes a special case of the Phase 2 architecture: "diff against
the parent's last non-blacklisted fire-id's manifest" instead of "diff against the
parent's current wandb config." The CLI signature stays identical.

## Open questions (deferred)

- **Root cause of the bin5 resume spike.** The σ-monotonic gap is KL-Gauss target entropy
  math (lower σ → narrower target → lower entropy floor). But all 4 fires (including σ=5
  extension where σ didn't change) spiked at step 100000. Candidates: (a) the
  `train-full-v3` cache was re-tokenized with a different LMQ codec than what bin5 saw at
  steps 67k+; (b) warmup re-applied on resume (LR ramping from 0); (c) shuffle/dataloader
  state mismatch on cache_dir change. Needs a dedicated diagnostic — out of scope for this
  spec.
- **Should `--allow-config-change` accept globs?** E.g. `--allow-config-change='loss.*'`
  for whole-subsystem deltas. Yes for ergonomics, but only with a `--dry-run` flag that
  prints the resolved set before firing. Defer to Phase 1 implementation.
- **wandb config drift vs. R2 manifest drift.** Wandb config is authoritative at fire
  time, but Levanter rewrites it on each resume. The R2 manifest snapshots wandb config at
  sync time, so it lags. Decision: trust wandb at fire time, but include the manifest hash
  in the parent reference (`--parent P@<manifest-sha>`) so a stale manifest can't silently
  cause a guard miss. Defer to Phase 1 implementation.

## Rollout

1. **Land Phase 1.1 + 1.2** (`--parent` + resume guard) as a single PR; verify it would
   have rejected `bin5-extend-10k.sh`.
2. **Land Phase 1.3** (warmup default).
3. **Land Phase 1.4 + 1.5** (FE inheritance + epoch fix) in parallel — they're FE-only.
4. **Land Phase 1.6** (fire-script generator) after 1.1/1.2 have been used for ~2 weeks;
   form should follow function.
5. **Re-evaluate Phase 2** after Phase 1 has caught its first wrong fire in the wild.
