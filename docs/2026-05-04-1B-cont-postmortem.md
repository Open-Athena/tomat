# 2026-05-04 — 1B cont-from-4711 wasted-run post-mortem

Status snapshot. The 1B 16k-step run did not finish cleanly; the
continuation attempt diverged at step ~2000 and never fully recovered.
Net cost is ~17 hr of v6e-32 wall-clock, plus the operator time to
diagnose. Two fixes have landed on `main`; future continuations should
not hit this.

## TL;DR

| Run | Steps | NMAE (val_200, mean) | Notes |
|-----|------:|---------------------:|-------|
| `train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32` (source) | crashed at 4733; last clean ckpt step-4711 | 2.33% @ step-4711 | Original 1B from-scratch |
| `tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32` (cont) | trained 11288 (cumulative ≈16k) | ~10–13% at end | Diverged at step ~2000, recovered to ~14% plateau, cooldown polished to 10–13% — still 5× worse than source step-4711 |

Two distinct engineering problems compounded:

1. **Iris job rebuild failure** killed the source run mid-training because the pinned marin dev wheel rotated overnight and a recovery rebuild couldn't fetch it.
2. **The "continuation"** was launched as a *new* iris job (different name → different output dir → Levanter's native auto-resume couldn't see the prior ckpts), and we reached for a model-only warm-start path instead. Warm-starting a fresh Adam optimizer on already-trained weights moved the model into a worse basin from which it never fully recovered.

## Timeline

- **Source 1B run** (`train-full-lmq-v2-1B-bs256-emd-do-16k-tpu32`) launched on v6e-32, target 16,000 steps. Native Levanter auto-resume survived several preemptions.
- **Step ~4723**: an iris recovery (after a preemption) attempted to rebuild the container. The build pulled a pinned `marin==…dev…` wheel from GitHub Pages and got HTTP 502 — the daily-rotated dev wheel had been deleted overnight (a known marin gotcha; fix is `uv lock --upgrade-package marin{,-…}`). Job died. Last clean checkpoint was step-4711.
- **Cont attempt** launched as a *new* iris job, `tomat-train-cont-from-4711-1B-bs256-emd-do-12k-tpu32`, target 11,289 steps (= 16k − 4711). It pointed at the source ckpt via `TOMAT_INIT_FROM_CHECKPOINT` and ran a model-only weight load (monkey-patched `load_checkpoint` to load `subpath="model"` only, leaving optimizer state at fresh init). LR schedule was constant + 2% warmup + 10% cooldown.
- **Step ~700–2400 of the cont run**: training-loss bounces between 0.9 and 1.6 (vs <0.6 in stable phase), pre-clip grad norm sustained at 5–15 (vs <2 in healthy training). Not a single-step spike — a sustained instability over ~1700 steps.
- **Step ~3000 onward**: model recovers to a plateau around 14% NMAE (val_200), then cooldown polishes it to 10–13%. The per-mat-set NMAE curves were harvested across 12 checkpoints × 2 mat sets and they all tell the same story: the cont run never recovers the source step-4711 quality.

## Why the resume diverged

Levanter has two distinct ckpt-loading paths, and we used the wrong one:

1. **Native auto-resume** — discovers `step-N` ckpts inside the *current run's own* output_dir and loads model + optimizer + step counter intact. This is what kept the source run alive across preemptions: same iris job name → same `checkpointer.base_path` → restart finds its own ckpts.
2. **`initialize_from_checkpoint_path`** — point at *some other* ckpt and warm-start. Calls `load_checkpoint(state, path)` with no `subpath`, deserializes the FULL `TrainerState`, then `replace(state, step=0)`.

The cont was launched with a *new* iris job name (`tomat-train-cont-from-4711-…`), which gave it a *new* `checkpointer.base_path`. Native auto-resume couldn't see the prior run's ckpts there. So we reached for path (2), with a custom monkey-patch that loaded only model weights (skipping optimizer state, because path (2)'s default optimizer-state behavior interacts badly with a re-targeted LR schedule).

The result: trained 1B weights × fresh-zero Adam moments × full-amplitude warmup-then-peak LR. With Adam β₂ = 0.95, the variance estimate has a time-constant of ~20 steps; the first ~50–100 steps of the cont run were effectively RMSProp with badly-conditioned variance estimates. Each step applied an update that perturbed already-trained weights in poorly-conditioned directions. Levanter's default `max_grad_norm = 1.0` clip-by-global-norm bounds *magnitude* but not *direction*, so the perturbations went through. By the time Adam's variance estimate had settled (~step 100–200) the model was already in a worse basin.

The correct move would have been to relaunch with the **same** iris job name. That would have given the cont run the same `checkpointer.base_path` as the source, which Levanter's auto-resume would have walked to find step-4711, loaded model + optimizer + step counter intact, and just continued mid-stable per the original 16k schedule. No monkey-patch, no fresh optimizer, no spike.

## Fixes landed (commits 5f914e0, 6c991e7)

1. **Drop the monkey-patch and the `TOMAT_INIT_FROM_CHECKPOINT` env var entirely.** No more model-only warm-start path. If a real warm-start use case (transfer learning, fine-tuning) becomes relevant, it gets re-added as an explicit `--warm-start` flag with safer LR defaults and a documented contract — not a default code path that's easy to reach for from the wrong use case.
2. **`tomat train --resume LABEL`** is the canonical way to continue an interrupted run. Same job name → same `checkpointer.base_path` → native auto-resume → optimizer + step counter intact. The CLI guards against accidentally starting fresh on top of an existing results dir, and against `--resume`'ing a label with no checkpoints.
3. **Independent partial-checkpoint fix** (`atexit` drain in `train_tomat_tpu.py`). Levanter's checkpointer schedules OCDBT commits asynchronously and writes `metadata.json` from a callback after the async drain. If the process exits before the final commit lands, the ckpt dir has the weight blob but no `metadata.json`, and downstream evals raise `FileNotFoundError: Missing paths: …`. Hit on three separate runs (lat-aware step-7999, 1B from-scratch step-4400, cont step-11288). Now: every Checkpointer the trainer creates is registered in a `WeakSet` and `wait_until_finished()`-ed at process exit, blocking until commits land.

## What this means for future runs

- After a preemption-induced crash, **reuse the same iris job name** to pick up where we left off. The CLI makes this an explicit one-flag operation: `tomat train --resume <same-label>`.
- After a *config* change (model size, batch size, LR schedule), it's a *new* run — use a fresh label and accept the new from-scratch cost. There is no shortcut path that lets you keep optimizer state but change the schedule, because the schedule reads from the optimizer's internal step counter.
- Marin dev wheels rotate daily. If a recovery build fails with a 502 fetching `marin==…dev…`, run `uv lock --upgrade-package marin{,-…}` before relaunch.
- The atexit drain fix means the final-step ckpt is now reliable for evals, so we don't have to fall back to the second-to-last ckpt anymore.
