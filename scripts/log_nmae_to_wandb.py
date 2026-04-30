#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["wandb", "click"]
# ///
"""Log mat-NMAE eval metrics back to a wandb run (resumed by id).

Use cases:
  - Watchdog harvest path: log a single (step, mat_set, mean/median/p99) point.
  - Backfill: read a JSONL curve and log every entry.

Default mode is single-point via args; backfill via --jsonl-file.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

from click import command, option, Path as ClickPath

err = partial(print, file=sys.stderr)


@command()
@option('-e', '--entity', default='PrinceOA', help='Wandb entity.')
@option('-f', '--jsonl-file', type=ClickPath(exists=True, dir_okay=False), help='Backfill from JSONL (one entry per line).')
@option('-i', '--run-id', required=True, help='Wandb run id (== levanter run name in our setup).')
@option('-m', '--mat-set', help='Mat-set label (val_200, train_200) for single-point mode.')
@option('-M', '--mean', type=float, help='Mean NMAE percentage for single-point mode.')
@option('-d', '--median', type=float, help='Median NMAE percentage for single-point mode.')
@option('-p', '--p99', type=float, help='p99 NMAE percentage for single-point mode.')
@option('-P', '--project', default='tomat-lmq-P14', help='Wandb project.')
@option('-s', '--step', type=int, help='Training step for single-point mode.')
def main(entity, jsonl_file, run_id, mat_set, mean, median, p99, project, step):
    import wandb

    wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume='allow',
    )
    # Use a custom step metric so we can log retroactively (after the
    # training run "finished" at its final step). Without this, wandb
    # rejects any wandb.log(..., step=N) where N < the run's current step.
    wandb.define_metric('eval/mat_nmae/step', hidden=True)
    wandb.define_metric('eval/mat_nmae/*', step_metric='eval/mat_nmae/step')

    def log_point(s, ms, mn, md, p9):
        wandb.log({
            'eval/mat_nmae/step': s,
            f'eval/mat_nmae/{ms}/mean': mn,
            f'eval/mat_nmae/{ms}/median': md,
            f'eval/mat_nmae/{ms}/p99': p9,
        })

    if jsonl_file:
        if not mat_set:
            err('--mat-set is required when using --jsonl-file (the JSONL has no mat-set marker).')
            sys.exit(2)
        n = 0
        for line in Path(jsonl_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            log_point(r['step'], mat_set, r['mean_nmae'], r['median_nmae'], r['p99_nmae'])
            n += 1
        err(f'Logged {n} NMAE points from {jsonl_file} ({mat_set}) to run {run_id}.')
    else:
        if step is None or mat_set is None or mean is None or median is None or p99 is None:
            err('Single-point mode requires --step --mat-set --mean --median --p99.')
            sys.exit(2)
        log_point(step, mat_set, mean, median, p99)
        err(f'Logged step={step} {mat_set}: mean={mean}% median={median}% p99={p99}% to run {run_id}.')

    wandb.finish()


if __name__ == '__main__':
    main()
