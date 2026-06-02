# `train-mg-tz-11` — 36h crash-loop post-mortem

**Date:** 2026-06-01
**Run:** `train-mg-tz-11` (v6e-16, MaskGIT-absorbing-CE, 200M, P=19, lmq codec)
**Wandb:** [open-athena/tomat-lmq-P19/train-mg-tz-11](https://wandb.ai/open-athena/tomat-lmq-P19/runs/train-mg-tz-11)
**Iris dashboard:** https://iris.oa.dev/#/job/%2Fryan%2Ftrain-mg-tz-11
**Last reported state:** `running  exit=0  failures=6  preemptions=1` — `step 2495 / 10000`, wandb `failed`, runtime 23.1 h.

The plot showed step rising 0 → ~2500 over the first few hours then plateauing for ~36 h while iris kept reporting `RUNNING`. Every restart trained ≤40 steps from the step-2460 checkpoint and crashed at the first eval boundary (step 2500) on a fresh manifestation of the same `Received incompatible devices` bug the JAX_PLATFORMS pin (12c6757) was supposed to kill.

## Timeline

Two distinct phases, split by the JAX_PLATFORMS=tpu fix (commit 12c6757, `2026-05-31 21:56 EDT` = `2026-06-01 01:56 UTC`).

### Pre-fix (run start → `2026-06-01 01:56 UTC`)

Per the IrisBadge counters at the time the fix landed: ~25 of the 31 cumulative starts and 5 of the 5 preemptions accrued in this window — the "train_step JIT compiles on CPU mesh, then TPU data arrives → `ValueError: Received incompatible devices`" failure mode documented in commit 12c6757's message. Iris's tail buffer no longer contains stderr from these attempts (the per-task log only retains roughly the last ~8 hours), so the count comes from the badge / wandb event history.

Failure mode (all attempts): **JAX-CPU-mesh-at-train-startup**. Trainer never reaches step 2461 — the very first jitted train step blows up because the post-restart worker bootstrap hadn't exported `PJRT_DEVICE=TPU` by the time our trainer imported JAX, so JAX initialized on CPU.

### Post-fix (`2026-06-01 01:56 UTC` onward — visible portion of iris log)

The JAX_PLATFORMS pin worked **for the training step**: every restart now reaches `First train step completed in ~26s (step 2460)`, runs ~40 train steps, then dies at step 2500 when the first eval hook fires. New failure mode: **JAX-CPU-mesh-at-eval**. Each cycle is ~2-3 minutes of useful work followed by crash + iris-bounce + cooldown.

Visible cycles in task-1 stderr tail (all `2026-06-01 UTC`, `Resuming training from step 2460` every time):

| trainer_started | First train step | eval crash | Δ start→crash |
|---|---|---|---|
| (clipped) | 15:40:18 | 15:42:19 | — |
| 15:43:14 | 15:43:52 | (next gap)| — |
| 21:13:53 | 21:14:32 | 21:16:35 | 2m42s |
| 21:17:27 | 21:18:05 | 21:20:04 | 2m37s |
| 21:32:18 | 21:33:10 | 21:35:13 | 2m55s |
| 21:35:51 | 21:36:43 | 21:38:40 | 2m49s |
| 22:44:48 | 22:45:30 | 22:47:29 | 2m41s |
| 22:48:19 | 22:48:59 | 22:51:02 | 2m43s |
| 23:51:32 | 23:52:13 | 23:54:13 | 2m41s |

8 train-then-crash cycles in the visible window. Note the **1-hour gaps** at `21:38 → 22:44` and `22:51 → 23:51`: iris's exponential-ish backoff on a job that keeps crash-looping. Between back-offs, the gang is bounced and re-scheduled within ~30s.

Eval crash traceback (identical every cycle):

```
levanter/eval.py:382  eval_callback → eval_model
levanter/eval.py:495  evaluate
levanter/data/loader.py:258  iter __next__
levanter/utils/background_iterable.py:75  reraise
levanter/data/loader.py:315  _produce_batches
levanter/data/loader.py:420  _batchify_local_data → make_global_array_for_leaf
jax/_src/array.py:768  make_array_from_callback → get_data
levanter/data/loader.py:377  get_local_batch → stack_tree
ValueError: Received incompatible devices for jitted computation. Got argument
individual_datums[0][1] of stack_tree with shape int32[1] and with device ids
[0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4] on platform TPU and
jit's context mesh with device ids [2048] on platform CPU
```

(Task 0 shows the same trace with `device ids [0]` instead of `[2048]` — that's the per-host CPU device id; both hosts hit the same bug independently and the gang dies as one.)

The benign `ValueError: Cannot find choice name for <class 'qwen3_density.Qwen3MaskGITConfig'>` printed at every restart immediately before "Resuming training from step 2460" is a separate cosmetic warning (levanter's choice-name introspection doesn't know about our subclass), **not** the cascade trigger. Training resumes successfully after it.

### Split summary

- **Pre-fix attempts:** ~23-25. All died **before** step 2461 with train-step CPU-mesh `ValueError`. Net step progress: 0 → ~2460 took until some attempt finally got both halves of the gang on TPU at the same time and ran far enough to checkpoint.
- **Post-fix attempts (visible):** 8. All reached step 2500 train, all died at eval. Net step progress: **0** (no checkpoint past 2460 was ever saved, because the crash is at the first eval boundary and `step_per_eval == steps_per_checkpoint == 500`).

The JAX_PLATFORMS pin **did fix the bug it targeted** (train-step CPU mesh). It revealed a second, narrower instance of the same class of bug on the eval path.

## Root-cause hypothesis (eval-time mesh)

The training-step JIT goes through `haliax.partitioning.named_jit(axis_resources=...)`, which carries the explicit `parameter_axis_mapping` and resolves the mesh via `haliax.partitioning.set_mesh` (entered by `Trainer.__enter__`, `levanter/trainer.py:371`). That's the path the JAX_PLATFORMS pin makes deterministic.

The data loader's per-batch staging uses a **different** JIT:

```python
# levanter/data/loader.py:564
@functools.partial(jax.jit, static_argnums=(0,))
def stack_tree(batch_name, individual_datums):
    ...
```

Bare `jax.jit` with no `mesh=` / no axis-resources. It picks up the **current thread-local mesh** at trace time. And it runs in a **background producer thread** spun up by `BackgroundIterable` (`levanter/utils/background_iterable.py:54`, a vanilla `threading.Thread(target=...)` with no context propagation):

```python
self.thread = threading.Thread(target=self._fill_queue_with_batches)
```

`jax.set_mesh` / `jax.sharding.use_mesh` (which `haliax.partitioning.set_mesh` delegates to — see `haliax/partitioning.py:138-156`) is **thread-local**. The producer thread inherits nothing.

Why does the *training* `stack_tree` not blow up, when the producer thread for training has no mesh context either?

Hypothesis: the train-step `stack_tree` trace gets cached on the first train step, when JAX's jit cache key matches whatever the global-cache fallback resolved to (likely TPU-as-default-because-`JAX_PLATFORMS=tpu` + no explicit mesh, sticking the output on TPU device 0 and then `make_array_from_callback`'s `NamedSharding(mesh, ...)` rematerializes across the full mesh). The eval `stack_tree` then mismatches because:

1. The eval `EvalBatch` size differs from the train batch size → different `static_argnums`-influenced cache key → a **new** trace.
2. By the time that new trace happens, iris/levanter has done more JAX-init bookkeeping that leaves a CPU mesh in some intermediate `ContextVar` that the train-time trace didn't observe — `jax.sharding.use_mesh` propagates through `ContextVar`, and `contextvars.copy_context` is **not** invoked when you spawn a vanilla `Thread`. So the eval-loader thread starts with default JAX state, and the first thing it touches that needs a mesh resolves to the host CPU device (the `[2048]` / `[0]` we see in the error).

The empirical proof of (1) + (2) is exactly the symptom: train works for the same `stack_tree` source code, eval doesn't, no other code changed.

## Fix candidates (ranked)

1. **Wrap `evaluate()`'s data-loader consumption in the train mesh context** (low cost, high confidence). Change `levanter/eval.py:482-485` to:

   ```python
   with haliax.partitioning.set_mesh(self.device_mesh):
       for batch, tags in tqdm(iterator, ...):
           state = self.accum_for_batch(...)
   ```

   But `set_mesh` is thread-local, and the data is produced on the bg thread, so this **only** helps if the producer thread inherits via `ContextVar` propagation. Real fix is on the producer thread, not the consumer.

2. **Pass `mesh=` (or wrap in `with set_mesh:`) inside `_produce_batches`** before any `jax.jit`-compiled function runs (`levanter/data/loader.py:308-`). Either:
   - Have `DataLoader` capture the calling thread's mesh in `__init__` and re-enter it inside `_produce_batches`, or
   - Annotate `stack_tree` with `jax.jit(stack_tree, ...)` invoked under an explicit `with jax.sharding.use_mesh(self.dl.mesh):` block before first call.

   This is the structurally-correct fix: the bg producer thread is the unit of work that needs a mesh, and currently it has none.

3. **Use `contextvars.copy_context()` when spawning the `BackgroundIterable` thread** (`levanter/utils/background_iterable.py:54`). One-line change:

   ```python
   ctx = contextvars.copy_context()
   self.thread = threading.Thread(target=lambda: ctx.run(self._fill_queue_with_batches))
   ```

   Most general fix — propagates the *entire* JAX context (mesh, axis mapping, defaults) into the bg thread. Lowest blast radius if anything else in JAX state matters. **Top recommendation.**

4. (Workaround, not a fix) `eval_batch_size = train_batch_size` to dodge the cache-miss recompile. Removes the symptom for this run but the next mesh shape change brings it back.

Ranking by **(effectiveness × likelihood-correct) / cost**: **3 > 2 > 1 > 4**. (3) is one line in a place where the existing comment already acknowledges thread-context weirdness ("I'm getting an error that the thread is `threading.current_thread()`, which seems impossible" — `background_iterable.py:101`).

## Lessons

- **Pull per-task stderr the first time a job cascades, not the Nth** — [feedback_pull_task_logs_on_first_cascade](file:/Users/ryan/.claude/projects/-Users-ryan-c-oa-tomat/memory/feedback_pull_task_logs_on_first_cascade.md) was written on exactly this run's pre-fix phase. Same lesson cost another ~12h on the post-fix phase: from `01:56 UTC` (fix landed) to ~`22:50 UTC` (real ValueError noticed in the logs) was ~21h of "iris is bouncing, will settle" while it was actually crash-looping on a brand-new variant of the same bug. Should have grepped task-1 stderr for `Traceback` the *moment* the first IrisBadge increment came in.

- **A single-task probe doesn't validate a distributed cascade-restart fix.** The pre-fix bug was caught and tz-10 (single-host probe) showed the JAX_PLATFORMS pin made train init deterministic. But the eval-loop manifestation requires the EvalBatch ≠ TrainBatch recompile + multi-host coscheduling. The first non-smoke run of the fix was the real test, and we treated `RUNNING` as success while the real test was failing.

- **The dashboard hid the failure mode.** Symptoms we now know to surface:
  - Step plateau + healthy IrisBadge counter is a *contradiction* — if `starts` is incrementing and `step` is not, the cycle is `start → train a bit → crash → bounce`, regardless of what the `running` state field claims. The `IrisBadge` panel added in 12c6757 has the data; the **plot's main-line "step vs wall-clock"** doesn't visually flag the divergence.
  - Suggested dashboard tweaks (followup): (a) annotate trainer_started vlines with the step at which the *previous* attempt crashed — a stack of vlines all at the same step makes "stuck at 2500" visible at a glance; (b) compute a `stuck_at_step` derived metric (`max(step) for last N attempts` — flag if N≥3 and all attempts crashed within Δstep < 200 of each other); (c) for any `failures > 0`, surface the **most recent per-task stderr tail** inline in `RecentEvents`, not just the iris-job lifecycle reasons.

- **The benign `Cannot find choice name for Qwen3MaskGITConfig` warning is noise** that shows up at every restart and looks scary in the log. Worth either silencing or labeling — it has nothing to do with the actual cascade trigger and ate review time.

- The fix landed in 12c6757 was real but **incomplete in the way that any "JAX_PLATFORMS pin" alone would be incomplete**: the pin makes the *default* device platform deterministic, but every code path that runs in a thread *without* an explicit mesh context still resolves to the default. Permanently fixing this class of bug needs the bg producer threads to inherit JAX context (candidate #3 above).

## Status

Not fixed. Run left in current state (crash-looping; iris reports `running` with `failures=6 preemptions=1` since the most recent fresh attempt window opened). Not killing or restarting per the user's instruction. Recommend candidate fix #3 (`contextvars.copy_context`) be tried next, as a single-line patch to `levanter/utils/background_iterable.py:54`.
