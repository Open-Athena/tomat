#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["wandb>=0.26", "click"]
# ///
"""Pull loss + throughput history for tomat runs into CSVs.

One CSV per run under `site/public/run-histories/<run_id>.csv`, plus a
summary CSV `site/public/runs-summary.csv` with one row per run.

Usage:
    scripts/pull_wandb_runs.py [--project tomat-two_token_9_12-P14]
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path
import sys

import click
import wandb

err = partial(print, file=sys.stderr)


@click.command()
@click.option("-e", "--entity", default="PrinceOA")
@click.option("-p", "--project", default="tomat-two_token_9_12-P14")
@click.option("-o", "--out-dir", type=click.Path(path_type=Path), default=Path("site/public/run-histories"))
@click.argument("run_ids", nargs=-1)
def main(entity: str, project: str, out_dir: Path, run_ids: tuple[str, ...]) -> None:
    api = wandb.Api()
    if not run_ids:
        run_ids = (
            "val-full-5k-bs32-bs32-seed42",
            "val-full-5k-bs32-2gpu-bs32-seed42",
            "val-full-5k-bs64-4gpu-bs64-seed42",
            "val-full-5k-bs128-8gpu-bs128-seed42",
            "val-full-tpu-bs128-seed42",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for rid in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{rid}")
        except wandb.errors.CommError as e:
            err(f"[skip] {rid}: {e}")
            continue

        hist = run.history(
            keys=["_step", "train/loss", "throughput/tokens_per_second", "throughput/mfu"],
            pandas=False,
        )
        rows = [r for r in hist if r.get("train/loss") is not None]
        csv_path = out_dir / f"{rid}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "tok_s", "mfu"])
            for r in rows:
                w.writerow([
                    r.get("_step"),
                    r.get("train/loss"),
                    r.get("throughput/tokens_per_second") or "",
                    r.get("throughput/mfu") or "",
                ])

        err(f"[ok] {rid}: {len(rows)} rows → {csv_path}")

        cfg = run.config or {}
        summary_metrics = run.summary._json_dict if run.summary else {}
        summary.append({
            "run_id": rid,
            "compute": _compute_label(rid, cfg),
            "batch_size": cfg.get("trainer", {}).get("train_batch_size"),
            "num_steps": cfg.get("trainer", {}).get("num_train_steps"),
            "final_step": summary_metrics.get("global_step"),
            "final_train_loss": summary_metrics.get("train/loss"),
            "mean_mfu": summary_metrics.get("throughput/mean_mfu"),
            "final_tok_s": summary_metrics.get("throughput/tokens_per_second"),
            "parameter_count": summary_metrics.get("parameter_count"),
            "total_gflops": summary_metrics.get("throughput/total_gflops"),
            "total_tokens": summary_metrics.get("throughput/total_tokens"),
            "state": run.state,
        })

    # summary CSV
    summary_path = out_dir.parent / "runs-summary.csv"
    with summary_path.open("w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for row in summary:
                w.writerow(row)
    err(f"[ok] summary → {summary_path} ({len(summary)} runs)")


def _compute_label(run_id: str, cfg: dict) -> str:
    # TPU sizes are embedded in the run_id as tpu, tpu4, tpu8, tpu16, etc.
    if "tpu16" in run_id:
        return "TPU v6e-16"
    if "tpu8" in run_id:
        return "TPU v6e-8"
    if "tpu" in run_id:
        return "TPU v6e-4"
    if "8gpu" in run_id:
        return "A100:8"
    if "4gpu" in run_id:
        return "A100:4"
    if "2gpu" in run_id:
        return "A100:2"
    return "A100:1"


if __name__ == "__main__":
    main()
