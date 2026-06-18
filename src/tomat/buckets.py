"""Canonical GCS bucket list (broken out from `vl_backfill.py` so importers
don't pull in `modal` transitively).

`vl_backfill.py` lives in the Modal-deployable code path (it does
`import modal` at module load). The standalone cron scripts on
`tomat-iris-cron` use an `iris-sync-venv` that doesn't have `modal`
installed, so `from tomat.vl_backfill import CANONICAL_BUCKETS` poisons
every `tomat runs sync` invocation with `ModuleNotFoundError: modal`.

This module carries ONLY the constant + the bucket-resolver helper.
Anything heavier (Modal app, image, secrets) stays in `vl_backfill.py`.
"""

from __future__ import annotations

import subprocess


CANONICAL_BUCKETS = (
    "gs://marin-eu-west4/tomat",
    "gs://marin-us-east5/tomat",
    "gs://marin-us-central1/tomat",
    "gs://marin-us-east1/tomat",
)


def resolve_run_bucket(run_label: str, ckpt_leaf: str | None = None) -> str | None:
    """Find which canonical GCS bucket a run's ckpts live in. Mirrors the
    `_resolve_run_bucket` helper in the `tomat` CLI; duplicated here so the
    library is importable without depending on the top-level script.
    """
    leaf = ckpt_leaf or run_label
    for bkt in CANONICAL_BUCKETS:
        path = f"{bkt}/results/{run_label}/checkpoints/{leaf}/"
        r = subprocess.run(
            ["gcloud", "storage", "ls", path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            return bkt
    return None
