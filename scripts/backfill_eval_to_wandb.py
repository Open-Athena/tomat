#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["wandb", "click", "fsspec", "gcsfs"]
# ///
"""Backfill mat-NMAE + mat-NEMD eval metrics to wandb from per-mat JSONs in GCS.

For each run_label, scans `gs://marin-eu-west4/tomat/eval/results/<RL>/<split>/`
for `step-*.json` files, and pushes one wandb.log per step into the matching
training run (id == run_label).

Usage:
  scripts/backfill_eval_to_wandb.py <run_label> [<run_label> ...]
  scripts/backfill_eval_to_wandb.py --dry-run shuf1k cont7k cont7k-ext

`shuf1k` is shorthand for `train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k`;
arbitrary suffixes get appended with a hyphen.
"""
from __future__ import annotations

import json
import sys
from functools import partial

from click import argument, command, option

err = partial(print, file=sys.stderr)

BASE = "marin-eu-west4/tomat/eval/results"
RUN_PREFIX = "train-full-v3-200M-bs128-emd-do-8k-tpu16-shuf1k"


def expand_label(short: str) -> str:
    if short.startswith("train-"):
        return short
    if short == "shuf1k":
        return RUN_PREFIX
    return f"{RUN_PREFIX}-{short}"


@command()
@option('-e', '--entity', default='PrinceOA')
@option('-P', '--project', default='tomat-lmq-P19')
@option('-n', '--dry-run', is_flag=True, help='List points without logging.')
@argument('labels', nargs=-1, required=True)
def main(entity, project, dry_run, labels):
    import fsspec
    fs = fsspec.filesystem("gcs")

    runs: list[tuple[str, dict[str, dict[int, dict]]]] = []
    for short in labels:
        rl = expand_label(short)
        per_set: dict[str, dict[int, dict]] = {}
        for split in ("val_200", "train_200"):
            base = f"{BASE}/{rl}/{split}"
            if not fs.exists(base):
                err(f"[backfill] {rl} {split}: no results dir")
                continue
            steps: dict[int, dict] = {}
            for path in fs.ls(base):
                name = path.rsplit("/", 1)[-1]
                if not (name.startswith("step-") and name.endswith(".json")):
                    continue
                step = int(name[len("step-"):-len(".json")])
                try:
                    with fs.open(path, "r") as f:
                        d = json.load(f)
                except Exception as e:
                    err(f"[backfill] {rl} {split} step-{step}: parse fail {e}")
                    continue
                steps[step] = d
            per_set[split] = steps
            err(f"[backfill] {rl} {split}: {len(steps)} step JSONs")
        runs.append((rl, per_set))

    if dry_run:
        for rl, per_set in runs:
            for split, steps in per_set.items():
                for step in sorted(steps):
                    d = steps[step]
                    err(f"  {rl} {split} step={step}: "
                        f"nmae={d.get('nmae_mean')} nemd={d.get('nemd_mean')}")
        return

    import wandb
    for rl, per_set in runs:
        wandb.init(entity=entity, project=project, id=rl, resume='allow')
        wandb.define_metric('eval/mat_nmae/step', hidden=True)
        wandb.define_metric('eval/mat_nmae/*', step_metric='eval/mat_nmae/step')
        wandb.define_metric('eval/mat_nemd/*', step_metric='eval/mat_nmae/step')

        n = 0
        all_steps = sorted({s for steps in per_set.values() for s in steps})
        for step in all_steps:
            payload = {'eval/mat_nmae/step': step}
            for split, steps in per_set.items():
                ms = split  # 'val_200' / 'train_200' — keep raw key name
                if step not in steps:
                    continue
                d = steps[step]
                pct = lambda v: None if v is None else v * 100
                for key, name in (
                    ('nmae_mean', f'eval/mat_nmae/{ms}/mean'),
                    ('nmae_median', f'eval/mat_nmae/{ms}/median'),
                    ('nmae_p99', f'eval/mat_nmae/{ms}/p99'),
                    ('nemd_mean', f'eval/mat_nemd/{ms}/mean'),
                    ('nemd_median', f'eval/mat_nemd/{ms}/median'),
                    ('nemd_p99', f'eval/mat_nemd/{ms}/p99'),
                ):
                    v = pct(d.get(key))
                    if v is not None:
                        payload[name] = v
            wandb.log(payload)
            n += 1
        err(f"[backfill] {rl}: logged {n} steps")
        wandb.finish()


if __name__ == '__main__':
    main()
