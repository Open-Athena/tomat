#!/usr/bin/env python
"""Pull v4-epochwin VL backfill JSON results and emit a CSV summary.

Reads `gs://marin-eu-west4/tomat/eval/vl-backfill/<leaf>/step-<N>.json`
for each leaf (parent + child) and prints a single CSV:

    step, eval_loss, n_seqs, run_label

Sorted by step ascending so the user can paste it into a notebook / pgfplots.

The parent (v4) and child (v4-epochwin) carve the same 256-sequence val
slice — see backfill_vl_modal's docstring. v4 ckpts were trained
WITHOUT a val carve-out so those 256 seqs were part of training; v4-cont/
epochwin held them out. The CSV makes this explicit via `held_out`.
"""
from __future__ import annotations

import json
import sys
from functools import partial

import click
import fsspec

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"
LEAFS = (
    # (leaf, held_out)
    ("train-mg-modal-h200x8-tz-v4-bs128-seed42", False),
    ("train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42", True),
)


def _list_steps(leaf: str) -> list[int]:
    """Steps with a landed JSON for `leaf`."""
    fs = fsspec.filesystem("gs")
    pat = f"{BUCKET}/eval/vl-backfill/{leaf}/step-*.json".replace("gs://", "")
    paths = fs.glob(pat)
    out: list[int] = []
    for p in paths:
        name = p.rsplit("/", 1)[-1]
        # strip "step-" and ".json"
        if name.startswith("step-") and name.endswith(".json"):
            try:
                out.append(int(name[len("step-"):-len(".json")]))
            except ValueError:
                err(f"[warn] unparseable step file: {p}")
    return sorted(out)


def _load_one(leaf: str, step: int) -> dict:
    url = f"{BUCKET}/eval/vl-backfill/{leaf}/step-{step}.json"
    with fsspec.open(url, "r") as f:
        return json.load(f)


@click.command()
@click.option("-l", "--leaf", default=None,
              help="Restrict to this ckpt-leaf (default: all known leafs)")
def main(leaf: str | None) -> None:
    leafs = [(leaf, False)] if leaf else list(LEAFS)
    rows: list[dict] = []
    for lf, held_out in leafs:
        steps = _list_steps(lf)
        if not steps:
            err(f"[vl-summary] no JSON results for leaf={lf}")
            continue
        for st in steps:
            try:
                d = _load_one(lf, st)
            except Exception as e:
                err(f"[warn] {lf}/step-{st}: {type(e).__name__}: {e}")
                continue
            # Two possible JSON shapes:
            #  (a) backfill_vl_modal: {eval_loss, n_seqs, ...}
            #  (b) preamble_vl_modal w/ --modes baseline:
            #      {results: {baseline: {mean, std, n_patches, ...}}, ...}
            # The semantic equivalence holds: both compute mean
            # bidir-absorbing-mask CE over the same patch population.
            if "eval_loss" in d:
                mean = d["eval_loss"]
                std = None
                n = d.get("n_seqs")
            elif "results" in d and "baseline" in d["results"]:
                r = d["results"]["baseline"]
                mean = r["mean"]
                std = r.get("std")
                n = r.get("n_patches")
            else:
                err(f"[warn] {lf}/step-{st}: unrecognized JSON shape — keys={list(d)[:5]}")
                continue
            rows.append({
                "step": st,
                "eval_loss": mean,
                "std": std,
                "n_seqs": n,
                "held_out": held_out,
                "run_label": lf,
            })
    rows.sort(key=lambda r: r["step"])
    print("step,eval_loss,std,n_seqs,held_out,run_label")
    for r in rows:
        std_str = f"{r['std']:.6f}" if r["std"] is not None else ""
        print(f"{r['step']},{r['eval_loss']:.6f},{std_str},{r['n_seqs']},"
              f"{int(r['held_out'])},{r['run_label']}")


if __name__ == "__main__":
    main()
