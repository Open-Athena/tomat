# Spec 38: FR eval parallelism via iris task fan-out

## Problem

Free-running (FR) mat-NMAE eval at the default `n_mats=200` is
architecturally impractical on a single iris task. Observed runtime:
**~80 min per material on TPU**, dominated by `P^3 ≈ 6859` decode
steps × ~0.6s/step forward pass. At n_mats=200 → ~270 hours per
(run, set, mode) cell. Last night's 10-cell SS-sweep FR fire would
take ~38 days to finish if let alone.

JIT compile is NOT the bottleneck — JAX's process-internal cache
hits across mats, so the "compiling bucket=N" log lines past mat 1
are misleading (paid only at first use). The cost is genuine forward
passes.

## What already exists

`marin/eval_mat_nmae.py` already supports fan-out via env vars:

```
TOMAT_EVAL_N_MATS       cap number of mats to eval (default 10)
TOMAT_EVAL_SKIP         skip first N mats (for fan-out across jobs; default 0)
```

The `evals fire` CLI in `tomat:1407-1408` hardcodes `TOMAT_EVAL_SKIP=0` and
`TOMAT_EVAL_N_MATS = n_mats` — one job per (run, set, mode). No fan-out.

## Proposal

Add `-T, --num-tasks INTEGER` to `tomat evals fire` (default 1). When > 1:

1. Submit N iris jobs. Each gets a distinct `TOMAT_EVAL_SKIP` and a
   smaller `TOMAT_EVAL_N_MATS` such that every mat is covered exactly once.
2. Append `-task{i}` to the job name and a per-task result-file suffix
   so the per-task JSONs don't collide in GCS.
3. `tomat evals sync` merges per-task JSONs into the canonical
   `step-{N}.json` (it already reads all JSONs under a run's
   `<set>-<mode>/` prefix → just needs to handle the new naming).

Speedup: N=8 tasks → ~34 h per cell at n=200, fits in an overnight.
N=20 → ~13 h, comfortable.

Caveat: each task pays the cold-start JIT compile (~5 min). At N=20
that's ~100 task-minutes of redundant compile — accept it; the
forward-pass-dominant total wins anyway.

## Result-file naming options

Option A — per-task suffix:
```
gs://.../val_200-free/step-84999-task-{i}.json
```
`evals sync` reads all matching prefixes and merges.

Option B — per-mat:
```
gs://.../val_200-free/step-84999/mp-{id}.json
```
More natural fanout (no aggregation needed at write time), but more
GCS objects (200 vs 1). Memory `data-locations.md` already shows
this kind of per-mat layout for some artifacts.

Recommend B — cleaner, parallel-safe, and makes per-mat trajectories
trivially available without re-running.

## Smaller change: just expose `--n-mats` + `--skip`

If the fan-out spec is too invasive, the minimal change is:
- Default `n_mats` drops from 200 to a more honest "give-me-a-headline"
  number (e.g. 20). Memory `bootstrap-cis-n-mat-partial-mask-diagnostic`
  (task #153 pending) is the right way to argue about variance bounds.
- Add `-K, --skip INTEGER` to surface `TOMAT_EVAL_SKIP`. Users can
  manually fan out by firing several jobs with different `-K`.

## What this unblocks

- SS-sweep FR comparisons (5 cells × 200 mats × ~80 min): currently
  ~38 days serial, ~2 days with N=20 fan-out.
- noprm vs paired-base NMAE eval — currently un-fired.
- Any future-architecture NMAE eval (e.g. F1, VV) inherits the cost.

## Out of scope

- Speeding up the per-mat cost itself. That requires either a
  smaller eval model, bigger batch B, or fewer decode steps —
  all separate concerns.
- Modal-side eval (no FR equivalent yet).
