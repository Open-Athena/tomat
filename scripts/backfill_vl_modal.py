#!/usr/bin/env python
"""[Compat shim] backfill `eval/loss` for v4's saved ckpts via Modal H200×8.

The original v4-specific implementation has graduated to a CLI subcommand
(`tomat evals vl-backfill <run>`), with the library body living in
`src/tomat/vl_backfill.py`. This shim is kept so:

  * Existing tmp/ + spec/ references that link to this file still resolve.
  * `python scripts/backfill_vl_modal.py -s 10000` still fires the v4
    backfill against the same Modal app + GCS layout.

Prefer the CLI for new fires:

    tomat evals vl-backfill train-mg-modal-h200x8-tz-v4-bs128-seed42 \\
        --bucket gs://marin-eu-west4/tomat --smoke

Direct invocation here forwards to the same library function as the CLI
(`tomat.vl_backfill.app` + `backfill_one_h200x8`), so result JSONs still
land at `gs://<bucket>/tomat/eval/vl-backfill/<ckpt_leaf>/step-N.json`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

# Make `src/tomat/...` importable when this is invoked as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tomat.vl_backfill import app, backfill_one_h200x8  # noqa: F401  (re-export for `modal deploy`)


# Default v4 layout — every existing v4 ckpt lives at this path.
V4_BUCKET = "gs://marin-eu-west4/tomat"
V4_RUN_LABEL = "train-mg-modal-h200x8-tz-v4"
V4_CKPT_LEAF = "train-mg-modal-h200x8-tz-v4-bs128-seed42"
V4_PARQUET_LABEL = "train-full-v3"


def _default_steps() -> list[int]:
    return list(range(1000, 41000, 1000))


@click.command()
@click.option("-s", "--steps", default="",
              help="comma-sep step list (default: every 1k from 1000 to 40000)")
@click.option("-r", "--run-label", default=V4_RUN_LABEL,
              help="run dir under gs://.../results/")
@click.option("-l", "--ckpt-leaf", default=V4_CKPT_LEAF,
              help="ckpt subdir under .../checkpoints/")
@click.option("-p", "--parquet-label", default=V4_PARQUET_LABEL,
              help="parquet label under /vol/tokenized/")
@click.option("-m", "--model-preset", default="200M")
@click.option("-v", "--val-seqs", default=256, type=int)
@click.option("-b", "--eval-batch", default=64, type=int)
@click.option("-c", "--cache-dir", default=None,
              help="Override levanter cache dir (e.g. `/vol/results/<rl>/cache`)")
@click.option("--bucket", default=V4_BUCKET, show_default=True,
              help="GCS bucket where ckpts live")
def main(steps, run_label, ckpt_leaf, parquet_label, model_preset,
         val_seqs, eval_batch, cache_dir, bucket):
    """Forward to `tomat.vl_backfill` lib. Identical I/O surface to the
    pre-CLI script (kept for tmp/ + spec/ compat)."""
    step_list = (
        [int(s.strip()) for s in steps.split(",") if s.strip()]
        if steps else _default_steps()
    )
    print(f"[vl-backfill] firing {len(step_list)} step(s): {step_list[:5]}"
          f"{'…' if len(step_list) > 5 else ''}", file=sys.stderr)
    print(f"[vl-backfill] deploying app {app.name!r}...", file=sys.stderr)
    app.deploy()
    print(f"[vl-backfill] deployed; spawning {len(step_list)} call(s)...", file=sys.stderr)
    lmq_path = f"{bucket}/codecs/lmq-v2-16k.npz"
    calls: dict[int, str] = {}
    for st in step_list:
        call = backfill_one_h200x8.spawn(
            run_label=run_label,
            ckpt_leaf=ckpt_leaf,
            step=st,
            bucket=bucket,
            parquet_label=parquet_label,
            lmq_path=lmq_path,
            model_preset=model_preset,
            val_seqs=val_seqs,
            eval_batch=eval_batch,
            seq_len=8192,
            loss_type="ce",
            kl_sigma=None,
            kl_sigma_unit="bin",
            mask_prior="absorbing",
            cache_dir_override=cache_dir,
        )
        print(f"[vl-backfill spawned] step={st} call_id={call.object_id}",
              file=sys.stderr)
        calls[st] = call.object_id
    print(f"[vl-backfill] all {len(calls)} call(s) spawned. "
          f"`modal app logs {app.name}` to watch.", file=sys.stderr)
    print("\n[modal call IDs]")
    for st, cid in sorted(calls.items()):
        print(f"  step={st:<5}  {cid}")


if __name__ == "__main__":
    main()
