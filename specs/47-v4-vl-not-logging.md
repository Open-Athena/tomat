# 47 — v4-cont-2 doesn't log `eval/loss` despite `steps_per_eval=1000`

## Symptom

`train-mg-modal-h200x8-tz-v4-bs128-seed42` (v4-cont-2 spawn, commit `4b80545`)
set `trainer.steps_per_eval=1000` to surface VL on the dashboard's
WallclockPlot. Wandb confirms:

- `config.trainer.steps_per_eval = 1000` ✓
- `config.data.num_validation_sequences = None` ✗
- No `eval/*` keys in summary, 0 rows for `eval/loss` in history.

## Diagnosis (hypothesis 1 confirmed)

The Modal training entry never carves out a validation split. In
`scripts/train_smoke_modal.py:545-551` `_train_bakeoff_impl` builds
`LmDataConfig` with only `train_urls` (via `UrlDatasetSourceConfig`) and
no `num_validation_sequences=…` kwarg, so `LmDataConfig.num_validation_sequences`
defaults to `None` (see `levanter/data/text/datasets.py:606`).

Levanter's `train_lm.main` then:
- calls `config.data.tagged_eval_sets(Pos)` → empty (no val component, no
  carve-out) (`train_lm.py:172`)
- hits `if len(tagged_eval_datasets) == 0:` → logs
  `"No evaluation datasets provided."` and **never installs the eval
  callback** (`train_lm.py:211-229`)
- the `every=steps_per_eval` hook from `cb_tagged_lm_evaluate` is unreachable,
  so `trainer.steps_per_eval=1000` is dead config.

The TPU entry (`marin/train_tomat_tpu.py:690`) does the carve-out:
```python
num_validation_sequences={"tomat": val_seqs} if val_seqs > 0 else None,
```
gated on `TOMAT_VAL_SEQS` env var (defaults to 0, tz-11 sets 256). Every
open-athena TPU run that logged `eval/*` has
`data.num_validation_sequences = {'tomat': 256}` in its wandb config;
every one missing `eval/*` has `None`.

## Root cause file:line

`scripts/train_smoke_modal.py:545-551` (and the second copy at
`:612-618` inside the MaskGIT branch) — `LmDataConfig(...)` missing
`num_validation_sequences=…`.

## Specific fix

Plumb a `val_seqs: int = 0` param through `_train_bakeoff_impl`,
`train_bakeoff_h200x8` (and h100x8/b200x8 for parity), and the
`main_bakeoff_*` entrypoints. Pass
`num_validation_sequences={"tomat": val_seqs} if val_seqs > 0 else None`
to **both** `LmDataConfig(…)` call sites (the pre-MG branch at :545 and
the MG-rebind at :612).

That alone is enough. The spawn-script then passes `val_seqs=256` to
match the TPU recipe.

## Lands in

Both files for the proper fix:
- `scripts/train_smoke_modal.py` — plumb the new param.
- spawn-script — pass `val_seqs=256`.

### Spawn-script-only path for v4-cont-3

Since v4-cont-2 is in flight and we don't want to disturb it, the
shortest path to surface VL on a v4-cont-3 resume (from the latest
ckpt) is: **(a)** add the `val_seqs` param to
`scripts/train_smoke_modal.py` (1-line kwarg on each `LmDataConfig`
+ thread through the wrapper) — unavoidable, the Modal entry has to
accept it — and **(b)** set `val_seqs=256` in
`tmp/spawn_mg_v4_cont_2.py` → save as `tmp/spawn_mg_v4_cont_3.py`.

### Caveat for resume

`num_validation_sequences` slices the train dataset (shuffled with a
fixed PRNGKey(0), last 256 sequences after shuffle become val — see
`_split_into_trainval_sets`, `levanter/data/text/datasets.py:519-539`).
This means **v4-cont-3 will train on a slightly different sequence
ordering than v4 / v4-cont-1 / v4-cont-2**:
- The 256 holdout sequences are removed from train.
- The remaining train-set order is unchanged (shuffle is deterministic,
  and `slice_dataset(0, length-256)` keeps the same prefix), so the
  resume reads from `step-N` of an ever-so-slightly-shorter dataset.

For a continuation run that's purely about getting VL telemetry, this is
acceptable: 256 / ~2.48M sequences is <0.011% of the dataset. The TF
loss curve will be visually indistinguishable from the no-carve-out
version, and now we get VL.

## Constraints check

- Read-only on infra: ✓ (no calls made).
- No spec push: will commit only.
