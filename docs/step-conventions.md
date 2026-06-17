# Step-variable naming convention

Decided per #298. Use these names consistently across Python + TypeScript so
the off-by-one bug class around training-step counters dies at the source.

## The four kinds of "step"

- **`step_idx`** — **0-based**, like Levanter's `info.step`. This is the value
  on disk: GCS checkpoint dirs are named `step-{step_idx}` (e.g. `step-29999`
  is the final ckpt of a 30,000-step training segment). Use this when reading
  ckpt names, parsing eval-record keys (`<step_idx>-<set>-<mode>`), or
  indexing into structures keyed by checkpoint identifier.

- **`step_n`** / **`step`** — **1-based** count of completed steps. This is
  Levanter's `state.step`. HF Trainer's `global_step`, PyTorch Lightning's
  `global_step`, and Keras's `step` all use the same 1-based convention. Use
  this when answering "how many steps has training completed?". Wandb summary
  fields like `summary.global_step` are this flavour.

- **`target_steps`** / **`num_train_steps`** — user-config target (total
  number of steps the run is requested to perform). Already standard
  throughout the codebase — leave alone.

- **`step_display`** / `formatStep()` — UI-formatted **string**. May include
  a `≈` prefix for snapped values (e.g. `≈90k` for a `step_idx=89999` ckpt
  that's logically the 90,000th step). See `marin/run_names.py:format_step`
  / `site/src/lib/runNames.ts:formatStep`.

## The Levanter quirk

Levanter saves checkpoints with `info.step` (0-based) as the directory name:

- **Periodic ckpts** during training land on clean `step-{N,2N,…}` (e.g.
  `step-10000`, `step-20000`). The modulo test runs against the 0-indexed
  `info.step` but happens to land on round values.
- **End-of-run forced ckpt**: with `num_train_steps=30000`, the final ckpt is
  `step-29999` because the force-save hook fires with `info.step =
  state.step - 1 = 29999`. This is what's logically the "30,000th step
  completed" — but on disk it's `step-29999`, not `step-30000`.

`format_step` / `formatStep` snap these final-of-segment values to their
nearest round Nk and prefix with `≈` so the display is transparent (`≈30k`
makes it clear the on-disk artifact is `step-29999`).

## Future work

If/when [Levanter is patched upstream](https://github.com/stanford-crfm/levanter)
to save final-of-run ckpts as `step-N` (instead of `step-(N-1)`), the snap
logic can be retired — but the `step_idx` / `step_n` distinction remains
useful for code clarity.
