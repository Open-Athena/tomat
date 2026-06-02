# 49 — maskgit eval doesn't OOM, it times out (n_mats=1-8/50)

## Symptom

`tomat evals fire --backend modal --mode maskgit -n 50 …` for
`train-mg-modal-h200x8-tz-v4-bs128-seed42` at steps 5k/9969/15k/20k/25k ×
`val_200`/`train_200` is asked for 50 mats per eval, but the result JSONs
land with only 1-8 mats:

| step  | val_200-maskgit | train_200-maskgit |
| ----- | --------------- | ----------------- |
| 5000  | 5               | 1                 |
| 9969  | 8               | 1                 |
| 15000 | 4               | 2                 |
| 20000 | 4               | 2                 |
| 25000 | 4               | 2                 |

So the MT/MV numbers the dashboard surfaces for v4 are aggregates over
1-8 cherry-picked-by-truncation mats, not 50. They're a biased sample
(roughly the smaller mats — the bigger ones get the timeout) and not
representative.

## Root cause (high confidence)

NOT OOM. The Modal function call hits its 4-hour `timeout=14400` while
sequentially decoding mats one-by-one in the MaskGIT K-step loop.

Per-mat wallclock evidence from
`ap-9HyoUDPNYYTJlqlLPFBalZ` (the 25k val_200 eval) logs:

```
[eval-mat] mp-1920042 MASKGIT: 507 patches, K=12 iters, 7740.9s
[eval-mat] mp-1819387 MASKGIT: 324 patches, K=12 iters, 4700.5s
[eval-mat] mat=mp-1850304: grid=(192, 192, 192), atoms=48 …
[eval-mat] tiling: 1331 patches (full coverage), grid=(192, 192, 192)
```

- 507 patches × 12 K-iters → 7741s ≈ 1.27 s/forward (8-way H200 model
  parallel, batch=32 across patches).
- 324 patches × 12 K-iters → 4700s ≈ 1.21 s/forward.
- mp-1850304 has 1331 patches → would be ~16,000s (4.4 h) alone, more
  than the entire 4-h budget.

Throughput consistency across the two completed mats argues the
bottleneck is compute (forward-pass-count), not memory: an HBM OOM would
crash the function rather than complete two mats and start a third. No
`RESOURCE_EXHAUSTED` / `device out of memory` lines appear in the logs.

The `ap-fgdT2WfKX29wUATnpVd1Dn` (20k) app has already been retired by
Modal and its logs are no longer fetchable (`modal app logs` returns
empty); the in-flight ap-9HyoUDPNYYTJlqlLPFBalZ logs above are the
authoritative sample. Across all five steps the n_mats pattern is the
same — 5/8/4/4/4 val + 1/1/2/2/2 train — which is consistent with each
container timing out after completing as many of the first-50 mats as
fit in 4 h. val gets more done (avg 5) than train (avg 1.6) because the
train-split mats happen to be larger on average in the first slot of
this fire's mat list (mp-1850304's 192³ at slot 3 of train_200 is the
classic offender).

Why per-mat = 1-2 h: this is the cost of bidi MaskGIT decoding with
K=12 iterations over ~500 patches on H200×8 at the v4 200M model. The
decode loop in `eval_mat_nmae.py` (search for `mp.. MASKGIT:`) calls
the model forward K × n_patches times serially per mat — there's no
across-mat batching, no across-K-step parallelism.

## Recommended fix

Two independent levers, lowest-effort first:

### 1. Use the existing `--num-tasks N` fanout when firing (no code change)

`tomat evals fire` already takes `--num-tasks N` (see `_evals_fire_modal`
in `tomat`); spawns `N` parallel H200×8 calls per (step × set), each
with `eval_skip=task_i*per_task` and `n_mats=per_task`. Output JSONs
land as `step-N-task<i>.json` and sync to per-task records that the
dashboard already supports (see `_backfill_record_key` /
`EvalRecord.task_idx`).

For v4: `-n 50 --num-tasks 10 --per-task 5` would put 5 mats per
4-h container, completing all 50 in parallel containers in ~5-8 h
wallclock per step (matches the per-mat throughput numbers above).

This is the right immediate move and explains the cleanest path back to
honest 50-mat aggregates without touching `eval_mat_nmae.py`.

### 2. Per-mat skip-and-log on the timeout boundary (later)

The "skip-and-log" pattern deferred since the v6e-8 OOMs still has
value as a safety net for both real OOMs and individual oversize mats:

In `marin/eval_mat_nmae.py`, wrap the per-mat decode (the `for mp_id in
mp_ids:` body around line ~1700-1900) in a `try` / `except` catching
both `MemoryError` / `XlaRuntimeError` (HBM OOM on individual oversize
mats) AND a soft per-mat wallclock deadline (e.g. `signal.alarm(1800)`
or a checked-after-each-K-step `time.monotonic()` against a 1800s
per-mat budget). On either kind of skip:

- log the mat-id + size + reason to stderr,
- append a skip record to a `skipped` list inside the summary JSON,
- continue to the next mat.

This means a 1331-patch mat doesn't silently consume the whole 4-h
budget and leave the remaining 47 mats untouched.

## "How many mats actually fit on H200×8?"

Per the throughput above (~1.25 s/forward, K=12 iters):

- Median mat ~500 patches → 12 × 500 × 1.25s ≈ 1.25 h/mat → ~3
  mats per 4-h container.
- Big mats (>1000 patches, ~5-10% of MP) → ≥4 h/mat → 0 mats per
  4-h container before timeout.
- Small mats (<200 patches, ~25% of MP) → <30 min/mat → 8+ mats per
  container.

So with the current K=12 the cap is ~3 mats/container per 4-h budget.
To honestly get 50 mats per step in one pass would need either
`--num-tasks 17+` or a per-mat budget reduction (smaller K / batching
across mats / smaller patches).

The exact grid-size distribution across the 81k MP mats lives at
`gs://marin-eu-west4/tomat/eval/mat_sets/{val,train}_200.json` (the
fixed-id sets `--mat-set val_200`/`train_200` pull from) — out-of-scope
to cull here, but worth examining if we want the "1331-patch monster"
mats moved out of the canonical 200 (they bias the visible aggregates
toward incomplete runs and the median is over too few small mats).

## Status

Diagnosis only. No code change in this pass. Filed as spec, not bug;
implementation will land via `--num-tasks` invocations on the next
v4 / v5 eval round, with the per-mat skip-and-log pattern queued for
a follow-up PR.
