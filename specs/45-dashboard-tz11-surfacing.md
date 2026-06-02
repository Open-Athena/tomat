# Standing rule — surface per-attempt causation when iris parent-state lies

**Status:** Active invariant for the `/runs` dashboard.
**Triggered by:** `train-mg-tz-11` crash-loop, May 31 – Jun 1 (`specs/done/31-tz11-postmortem.md`).
**Touches:** `site/src/runs/RunHeaderRich.tsx` (`IrisBadge` / `isCrashLoop`), `site/src/runs/RecentEvents.tsx`, `site/src/runs/errorClassification.ts`, `scripts/iris_attempts_dump.py`, `tomat iris sync`.

## The bug-class this rule prevents

Iris reports two states that can independently lie about whether a run is making progress:

1. **Parent-job state.** `state == 'RUNNING'` means the gang of N tasks is *scheduled* — it does NOT mean any task is currently executing useful work. A crash-looping job ("each restart trains for 3 min then dies, iris rescheduled the gang 34×") will read `RUNNING` for the entire 36 h plateau.
2. **Per-task state histogram.** When iris is between restart cycles, every task reads `pending` (cooling off between bounces, or waiting in the queue) even though the parent is `RUNNING`. The contradiction (`parent=RUNNING ∧ tasks_all_pending`) is the signal.

The post-mortem documents 36 h of `train-mg-tz-11` reading `RUNNING (p=2, f=13)` on the dashboard while every post-12c6757 attempt was actually dying at step 2500 on a JAX-mesh `ValueError` at the first eval boundary. The user spent 21 h between "fix landed" and "noticed the new failure mode" because the dashboard hid the cascade.

## The standing rule

For any iris job where **`state == 'RUNNING'` ∧ `failures > 0` ∧ `task_state_counts.running == 0` ∧ `pending + building >= num_tasks`**, the dashboard MUST:

1. **Replace the badge label** with `CRASH-LOOP (f=N)` (red `#cb2431`), NOT `RUNNING (Np)` (green).
   - Tooltip explains the rule: "iris parent is RUNNING but 0 of N tasks have started; F task-level failures so far — restart cascade."
2. **Pair every `trainer_started` row in `RecentEvents`** with a sub-line describing how that attempt ended. The sub-line is rendered as `└─ <verb> Xm Ys later · step S · <classification>` where:
   - `verb ∈ {died, preempted, succeeded, completed}` from `classifyDeath`.
   - `classification` is from a curated regex set (`site/src/runs/errorClassification.ts`, mirrored server-side in `scripts/iris_attempts_dump.py`). Falls back to the cleaned first line of stderr for unrecognised modes.
   - `step S` is the highest `global_step` in the wandb history window between the attempt's `started_at_ms` and `finished_at_ms`.
   - For the currently-running attempt (no `finished_at_ms` yet): `└─ alive · started Xm ago · step S`.
3. **Group per-task attempts by `attempt_id`.** Iris re-uses the same attempt number across the gang's N tasks for a coordinated restart. The dashboard collapses N task-attempts → 1 row per cycle and picks a "trigger" task (first non-cascade error) to caption the cycle.

## What the dashboard MUST NOT do

- Do not regress to the green `RUNNING (Np)` badge for a crash-looping job. The whole point of the rule is to make the contradiction visible.
- Do not render only `trainer_started` rows without their terminating causation. A chronological list of restart timestamps with no "why did each one die?" defeats the purpose.
- Do not silence the rule for "old" runs. The check is on iris's live `task_state_counts` + `failures`; finished crash-looped runs naturally fall out of the rule (their state is `FAILED` / `KILLED`, not `RUNNING`).

## What the rule does NOT trigger on

- **All-healthy `RUNNING (Nr)`** (every task running). Renders as today — green `RUNNING` badge, no crash-loop styling.
- **`PENDING` / `BUILDING` at run start.** No `failures > 0` yet.
- **Single-failure recoveries.** A run that hit one preempt + one restart and is back to `running` reads as healthy: `running` count is `num_tasks`, not zero. The rule keys on `running == 0` ∧ pending-or-building = all.
- **Modal-hosted runs.** No iris job → `IrisBadge` doesn't render; `ModalBadge` carries its own logic.

## Backend wire format (schema v2)

The iris-attempts sidecar (`scripts/iris_attempts_dump.py` → `tomat/iris-attempts/<label>.json`) adds, per attempt:

- `error_first_line`: cleaned first line of stderr (drops iris's `Exit code: N. stderr:` wrapper).
- `error_classification`: one-line bucket label from the regex set.

And, at the top level:

- `attempts_summary`: a flat list of `{task_id, attempt_id, trainer_started_ts_ms, ended_ts_ms, state, exit_code, error_first_line, error_classification}` records, sorted by `trainer_started_ts_ms`. Easier wire format for downstream consumers (the dashboard still groups by `attempt_id` for its UI, but CLIs and ad-hoc jq scripts walk this directly).

The dashboard prefers server-provided fields when present and falls back to its own client-side regex for v1 sidecars.

## Test posture

The dashboard renders against R2-cached sidecars and a live iris-state snapshot. The visual test is:

1. Open `https://tomat.oa.dev/#/runs/train-mg-tz-11` (or the local dev server's equivalent during development).
2. **Badge** reads `CRASH-LOOP (f=13)` in red, not `RUNNING (4p)` in green.
3. **RecentEvents** newest 3 rows each have a `└─` sub-line with a non-empty classification (e.g. `JAX mesh ValueError (eval boundary)`) and a `step 2500` hint.
4. After the next `tomat iris sync` cron pass picks up v2 schema, the sub-lines should use `error_classification` from the sidecar (not the client-side fallback). Verify by grepping the R2 sidecar JSON for `"error_classification"`.

## Followups (not in this commit's scope)

- The wallclock-plot's `trainer_started` vlines could carry the same `classification` label as a hover tag (currently just the timestamp). Touch: `WallclockPlot.tsx`.
- Stuck-at-step derived metric: `max(step) for last N attempts` flagged when N >= 3 and Δstep < 200 across attempts. Surface on the card-summary line, not just the detail page.
- Silence the benign `Cannot find choice name for Qwen3MaskGITConfig` warning at trainer-startup, or relabel it so it doesn't read as the cascade trigger when grepping logs. Touches: levanter's `choice-name introspection` or our subclass registration.
