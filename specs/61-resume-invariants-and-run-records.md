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

## Phase 2 — fires-as-immutable-records (sketch)

This phase decouples wandb run-id from the canonical record. Out of scope to implement
until Phase 1 has shipped and we've used it for ~1 month.

### 2.1 Storage layout

```
R2: openathena/tomat/fires/<fire-id>/
  manifest.json     # frozen at fire-spawn time; never updated
  raw.parquet       # history written by tomat runs sync; immutable suffix-appendable
  eval.json         # m-eval points fired against this fire's ckpts
  ckpts/            # (or pointer to GCS)
```

`fire-id` is e.g. `<UTC-isoformat>-<8-hex-shasum-of-manifest>`. Immutable.

### 2.2 Run = view

A "run" becomes a view, not a record:

```
R2: openathena/tomat/views/<view-name>.json
  {
    "view": "bin5",
    "fires": ["2026-06-15T01:23:00Z-abc12345", "2026-06-22T14:18:50Z-def67890", …],
    "blacklisted": ["2026-06-22T14:18:50Z-def67890"]  // bad fires we want hidden from rollup
  }
```

The dashboard's `/runs/bin5` page assembles by reading the view, then reading each fire's
manifest + parquet, then doing the join client-side (or in the CFW).

### 2.3 Blacklist mechanism

Bad fires (like the bin5+10k extension) get added to `blacklisted`. The view's
trajectory rollup skips blacklisted fires. The ckpts they produced stay on GCS until
GC'd. Nothing gets deleted.

### 2.4 Migration

Worker reads from new layout, falls back to old `runs/<run-id>/` layout when no view
exists. `tomat runs sync` writes both layouts during the transition (3-6 months) so
nothing breaks if we revert.

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
