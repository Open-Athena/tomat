# tomat — state of experiments (2026-05-11)

Snapshot. Living doc — `./tomat runs links` regenerates the runs table.
Previous snapshot: [`docs/2026-04-30-overview-snapshot.md`](docs/2026-04-30-overview-snapshot.md).

## TL;DR

- **New best 200M val NMAE: 1.73% / NEMD 1.76%** (cont7k-ext, step ~21000),
  down from 2.02 / 2.14 last week. Run continues past 1 epoch at constant
  LR; trajectory still descending non-monotonically. The "stop at
  Chinchilla" rule is doing nothing useful for this task — we're at
  ~5–6× past Chinchilla-optimal data budget and still gaining.
- **Reference points** (unchanged): ChargE3Net SOTA 0.523% (arxiv 2312.05388),
  electrAI ResUNet 0.86% (Hananeh, Meeting 26). LMQ-v2-16k codec floor 0.18%
  (P=14). We're at ~2× electrAI's NMAE with 17× the params — per-param gap
  is real; bet is that transformer scaling closes it.
- **NEMD instrumented**: now reporting per-mat normalized EMD alongside NMAE.
  NEMD is decoder-independent and directly comparable to point-estimate
  models (ElectrAI NMAE = NEMD algebraically for δ-distributions). See
  [`docs/lmq-vs-equal-mass.md`](docs/lmq-vs-equal-mass.md) for codec
  background.
- **Infra: cluster is rough this week.** v6e-32 large-slice preempts are
  killing 1B from-scratch runs at cache-build. v5p (both 16 and 32) hangs
  silently after JAX coordination init. Only v6e-16 is reliably healthy.
  See `gh/drafts/cache-build-brittleness/` for the marin issue we'll file.

## Run inventory (current; regenerated via `./tomat runs links`)

> See `tmp/runs-links-preview.md` for the most-recent markdown table with
> direct wandb + iris dashboard links. Below is the digest:

| run | model | TPU | state | step | val NMAE | val NEMD |
|---|---|---|---|---:|---:|---:|
| `cont7k-ext` (training in progress) | 200M | v6e-16 | running | 25k+ | **1.73 %**@21k | **1.76 %**@20k |
| `1B-bs256-26k-tpu32-shuf1k` | 1B | v6e-32 | killed (cache-build hang) | 0 | — | — |
| `1B-bs512-13k-tpu32-shuf1k` | 1B | v6e-32 | failed (Zephyr cache-build crashloop, 6h20m) | 0 | — | — |
| `1B-bs256-26k-v5p32-shuf1k` | 1B | v5p-32 | failed (JAX `RegisterTask` RPC timeout, 14m) | 0 | — | — |
| `200M-bs128-500-v5p16-shuf1k-smoke` | 200M | v5p-16 | running but hung at JAX init (same v5p pattern) | 0 | — | — |
| `cont6kwsd` / `cont8kwsd` (WSD ablations) | 200M | v6e-16 | finished | 8/10k | 2.88 % / 2.69 % | 3.63 / 3.19 |
| `shuf1k` (baseline 8k) | 200M | v6e-16 | finished | 7999 | 2.27 % | — (pre-NEMD) |

## Headline trajectory — cont7k-ext val NMAE / NEMD (M-train / M-val)

`cont7k-ext` is a +10k constant-LR continuation of `shuf1k` from step-7999,
then re-resumed for another +10k (target step 38k = ~2 epochs). Step
counts are total since-pretraining.

| step | val NMAE | val NEMD | train NMAE | train NEMD |
|---:|---:|---:|---:|---:|
| 7999 (start of cont) | 2.27 | 2.82 | 2.28 | 2.82 |
| 14000 | 2.44 | 2.65 | 2.65 | 2.85 |
| 17999 (prior "best") | 2.02 | 2.14 | 2.15 | 2.28 |
| 18000 | 1.97 | 2.10 | 2.11 | 2.24 |
| 19000 | 1.74 | 1.84 | 1.87 | 1.97 |
| 20000 | 1.86 | **1.76** | 1.83 | 1.91 |
| **21000** | **1.73** | 1.98 | 1.94 | 2.04 |
| 22000 | — (eval pending) | 1.80 | — | — |

Plot: [`plots/nmae-nemd-trajectory.png`](plots/nmae-nemd-trajectory.png)
(regenerated via `scripts/plot_nmae_nemd_trajectory.py`).

NMAE / NEMD trajectories are *non-monotonic* — single-step movements
~0.2 pp are within noise. Best-of-N over last few ckpts is the
right deployment criterion, not "latest".

## Technical findings

### MFU profile (200M v6e-16 / 1B v6e-32)

| category | 200M v6e-16 | 1B v6e-32 |
|---|---:|---:|
| attention | 47 % | 17 % |
| matmul (custom-call) | 21 % | 40 % |
| fusion | 28 % | 30 % |
| comm | 5.5 % | 2.1 % |
| **MFU** | **~9 %** | **~13 %** |

Reading: 200M is HBM-bandwidth-bound on v6e. Larger models shift toward
compute-bound, freeing more headroom for BS↑ / grad_ckpt off. Open:
v5p might actually beat v6e here once cluster is healthy — v5p has ~2×
HBM bandwidth, which is the binding constraint at 200M.

### NEMD vs NMAE

NMAE uses a median decoder (L₁-optimal point estimator). NEMD is the
per-voxel `E_v~P[|v − ρ_true|]`, decoder-independent and directly tied to
the EMD training loss. NMAE responded slowly to mid-cooldown improvements;
NEMD tracked them earlier. Both are now logged to wandb (eval/mat_nmae/*
and eval/mat_nemd/*). The cont6k/8kwsd cooldown bakeoff was a false
alarm — NEMD showed at most a 10–15 % regression while NMAE looked like
40 %, because median-decode amplifies distribution-shape changes.

### Levanter cache_dir is per-run by default — `--share-cache` available

`marin/train_tomat_tpu.py:240` was pointing cache_dir at
`{BUCKET}/results/{results_label}/cache`, so every from-scratch run rebuilt
the cache from the same source parquets. Under v6e-32 preempt pressure
this cost 8h+ of failed cache-build crashloops on the 1B runs. Patched:
`tomat train --share-cache` now points all runs over the same data label
at a shared `gs://.../cache/<data_label>/` dir. Seeded
`cache/train-full-v3/` from shuf1k's existing cache (47 GiB).

### Cluster state (2026-05-11)

- v6e-16: healthy; cont7k-ext makes steady progress.
- v6e-32: cache-build crashloops on from-scratch 1B runs (the failures
  above). Even with --share-cache, the 1B BS=256 v6e-32 silently stalled
  after JAX init (no log delivery, no ckpts) — separate bug.
- v5p-16 / v5p-32: both hang after JAX coordination init.
  `RegisterTask` RPC deadline-exceeded. Same signature on a 200M smoke as
  on a 1B training run, so it's not workload-dependent.
- v4: avoid; Tim reports chip-config (`jellyfish`) errors today.

Drafted GH issue for Marin team: `gh/drafts/cache-build-brittleness/`.

## In flight

1. `cont7k-ext` 200M v6e-16 → step 38k (~2 epochs, ~14h ETA from step 25k+).
2. `200M-bs128-500-v5p16-shuf1k-smoke` — hung at JAX init; will likely kill.
3. M=128 v3 tokenize: train + val complete on Modal volumes
   (`tomat-rho-gga-train` / `tomat-rho-gga`); needs sync to GCS before
   training can use it.

## Open questions

- Push `cont7k-ext` past 2 epochs? Repeated tokens start at ~step 19k.
  Will overfit kick in eventually?
- What is v5p actually doing for our workload, once it stops hanging?
  HBM bandwidth thesis predicts faster step time + higher MFU than v6e
  at 200M.
- 1B from scratch on v3 has yet to land — both BS=256 + BS=512 attempts
  failed before any training. The MFU bakeoff thesis ("1B is more
  compute-bound, less HBM-BW-bound, higher MFU") is supported by an
  earlier 1B-cont profile but not yet by a fresh run we can iterate on.
- M=128 dataset (just tokenized) enables 2-epoch training without
  repeating tokens — when do we use it?

## Related GH issues

- [#4 MFU](https://github.com/Open-Athena/tomat/issues/4) — ongoing topic;
  this OVERVIEW + runs table linked from there.
- [#3 sampling-weight study](https://github.com/Open-Athena/tomat/issues/3) —
  separate.
- [#1 v3 tokenizer](https://github.com/Open-Athena/tomat/issues/1),
  [#2 MPDB](https://github.com/Open-Athena/tomat/issues/2) — done, due to close.

## Tooling

- `./tomat runs links` — markdown table of all wandb runs with wandb +
  iris dashboard URLs + latest NMAE/NEMD. Slack-paste-ready.
- `./tomat runs nmae <substr>` — full NMAE+NEMD curve for a run.
- `./tomat iris ls -f <substr>` — list iris jobs matching a substring.
- `./tomat train [-r] [--share-cache] LABEL ...` — launch / resume training.
- `scripts/backfill_eval_to_wandb.py <labels…>` — push per-mat JSONs from
  GCS to wandb after re-evals.
- `scripts/plot_nmae_nemd_trajectory.py <labels…>` — regenerate trajectory
  plot.
