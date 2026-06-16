#!/usr/bin/env python3
# NOTE: invoked via marin's venv python (see `_run_iris_attempts_dump` in
# tomat). Don't rely on `tomat/.venv` here — iris isn't installed there.
"""Dump per-task per-attempt history for one iris job, as JSON.

Reuses iris's own CLI machinery for controller-URL + token-provider setup
(register a custom subcommand on the existing `iris` group → invoke via
Click standalone_mode=False), then calls `gather_bug_report` directly and
emits a flat JSON payload to stdout.

Why a helper script: tomat's CLI doesn't have iris's Click context, and
re-implementing the tunnel-discovery / token-loading dance against the
iris RPC surface is fragile (memory `iris-auth-gotcha`). Subprocess this
script with the same `--cluster=marin` flag tomat already uses for
`iris job list --json`. The helper does NOT shell out to `iris …
bug-report`: that would force Markdown parsing (rejected per spec) and
re-establish a tunnel for every label, which is the slow part.

Usage:
    iris_attempts_dump.py --cluster=marin --tail=0 /ryan/train-mg-tz-11

Output:
    {
      "label": "train-mg-tz-11",
      "job_id": "/ryan/train-mg-tz-11",
      "synced_at": "2026-05-31T20:30:00Z",
      "job_state": "running",
      "job_failure_count": 19,
      "job_preemption_count": 3,
      "submitted_at": "...",
      "submitted_at_ms": 1715000000000,
      "started_at": "...",
      "started_at_ms": 1715000000001,
      "finished_at": "-",
      "finished_at_ms": null,
      "tasks": [
        {
          "task_id": "/ryan/train-mg-tz-11/0",
          "state": "pending",
          "attempts": [
            {"attempt_id": 1, "started_at": "...", "started_at_ms": ...,
             "finished_at": "...", "finished_at_ms": ...,
             "state": "preempted", "exit_code": 0, "error": "Coscheduled ...",
             "is_worker_failure": true, "worker_id": "..."}
          ]
        }, ...
      ]
    }
"""

from __future__ import annotations

import datetime
import json
import re

import click


# Pull in the existing iris CLI machinery: registering on `iris.cli.main.iris`
# (the Click group) inherits all of its `--cluster=marin` / `--controller-url`
# / `--config` setup automatically.
from iris.cli.main import iris  # noqa: E402
from iris.cli.bug_report import gather_bug_report  # noqa: E402

# `require_controller_url` moved from `iris.cli.main` into `iris.cli.connect`
# at some point in iris's history (sometime around the spec-45 timeframe).
# Marin's locally-vendored copy + our tomat venv still expose it from
# `iris.cli.main`; the upstream-fresh checkout splits it out. Try both so
# this script keeps working across versions without an iris bump.
try:
    from iris.cli.connect import require_controller_url  # noqa: E402
except ImportError:
    from iris.cli.main import require_controller_url  # noqa: E402

from iris.cluster.types import JobName  # noqa: E402


# Error-classification regex set — mirrors `site/src/runs/errorClassification.ts`.
# Surface one-line `error_classification` per attempt so the dashboard
# (and any CLI walking the attempts sidecar) gets the failure mode at a
# glance instead of having to grep a multi-line stderr blob. See
# `specs/45-dashboard-tz11-surfacing.md` for the standing-rule motivation.
#
# Keep this list short + curated. New entry threshold: a failure mode
# showing up >2× across runs we're triaging.
_ERROR_CLASSIFIERS: list[tuple[re.Pattern[str], str]] = [
    # tz-11 post-fix: JAX-mesh ValueError when the eval data-loader's bg
    # producer thread runs jit'd stack_tree without a mesh ContextVar.
    (
        re.compile(r"Received incompatible devices for jitted computation", re.I),
        "JAX mesh ValueError (eval boundary)",
    ),
    # tz-11 pre-fix: jax.distributed.initialize() called too late / not at
    # all (PJRT_DEVICE wasn't TPU at JAX-import time).
    (
        re.compile(
            r"jax\.distributed\.initialize\(\) must be called before any JAX calls",
            re.I,
        ),
        "jax.distributed not initialised",
    ),
    # tz-10 mode: asyncio socket teardown during worker bootstrap.
    (
        re.compile(r"asyncio socket\.send\(\) raised exception", re.I),
        "asyncio socket teardown (worker startup)",
    ),
    (
        re.compile(r"\b(?:OOM|OutOfMemory|Resource exhausted|RESOURCE_EXHAUSTED)\b"),
        "OOM",
    ),
    (re.compile(r"Coscheduled sibling", re.I), "cascade (sibling died)"),
    (re.compile(r"\bSIGTERM\b|Terminated by signal 15"), "SIGTERM"),
    (re.compile(r"\bSIGKILL\b|Killed by signal 9"), "SIGKILL"),
    (re.compile(r"XLA compilation failed|HLO module", re.I), "XLA compile failure"),
    # iris worker bounce — controller lost heartbeat from the worker.
    # Common during TPU preempt cleanup; conceptually a worker-loss
    # cascade trigger.
    (re.compile(r"worker ping threshold exceeded", re.I), "worker ping timeout"),
    # Tokenizer / vocab-size mismatches at eval restart (e.g. a checkpoint
    # vocab=18570 vs config vocab=18571). Frequent on tomat-eval-* jobs
    # against in-flight training checkpoints.
    (re.compile(r"Axis vocab has different sizes", re.I), "vocab-size mismatch"),
]

_EXIT_CODE_PREFIX = re.compile(r"^Exit code:\s*\d+\.\s*stderr:\s*")


def _error_first_line(err: str | None) -> str | None:
    """Strip iris's `Exit code: N. stderr: ` wrapper + return the first line.

    Returns None when `err` is empty or whitespace-only.
    """
    if not err:
        return None
    trimmed = err.strip()
    if not trimmed:
        return None
    cleaned = _EXIT_CODE_PREFIX.sub("", trimmed)
    return cleaned.splitlines()[0] if cleaned else None


def _classify_error(err: str | None) -> str | None:
    """Bucket an iris error string into a short human-readable label.

    Returns None when the message is empty; falls back to the first 80
    chars of the cleaned first line for unrecognised messages — the
    dashboard's sub-line never shows "<empty cause>".
    """
    if not err:
        return None
    trimmed = err.strip()
    if not trimmed:
        return None
    for pat, label in _ERROR_CLASSIFIERS:
        if pat.search(trimmed):
            return label
    first = _error_first_line(trimmed) or ""
    return (first[:77] + "…") if len(first) > 80 else first


def _iso_to_epoch_ms(iso: str) -> int | None:
    """`as_formatted_date` returns ISO-8601 UTC or '-' for missing.

    Convert to epoch-ms so the frontend can do arithmetic without an extra
    parser. Pass through '-' and obviously-empty strings as None.
    """
    if not iso or iso == "-":
        return None
    try:
        # Python's fromisoformat handles `+00:00` (rigging emits with tz).
        dt = datetime.datetime.fromisoformat(iso)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


@iris.command("attempts-dump", hidden=True)
@click.option(
    "--tail",
    type=int,
    default=0,
    help="Recent log lines per task (0 = skip log fetch, what we want)",
)
@click.argument("job_id")
@click.pass_context
def attempts_dump(ctx, tail: int, job_id: str):
    """Internal: gather per-attempt history for JOB_ID, emit JSON to stdout."""
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")
    name = (
        JobName.from_string(job_id)
        if not job_id.startswith("/")
        else JobName.from_wire(job_id)
    )
    report = gather_bug_report(
        controller_url, name, tail=tail, token_provider=token_provider
    )

    # Build the flat payload. We KEEP the iso strings (human-readable in dev /
    # `jq` walks) and ADD epoch_ms for the JS dashboard. The frontend doesn't
    # need worker/log details, just attempt timing + state + error.
    # job_id is "/<user>/<label>" for any runner — strip the user namespace.
    label = report.job_id.split("/", 2)[-1]
    tasks_out: list[dict] = []
    # `attempts_summary`: flat per-attempt-per-task records ordered by
    # `started_at_ms`. The dashboard groups by `attempt_id` to render
    # per-restart-cycle rows; the flat list is the easier wire format and
    # downstream CLIs (`tomat iris …`, ad-hoc jq) can walk it without
    # joining the per-task arrays. See specs/45-dashboard-tz11-surfacing.md.
    summary_records: list[dict] = []
    for t in report.tasks:
        atts_out: list[dict] = []
        for a in t.attempts:
            err_first = _error_first_line(a.error)
            err_class = _classify_error(a.error)
            att_rec = {
                "attempt_id": a.attempt_id,
                "worker_id": a.worker_id,
                "state": a.state,
                "exit_code": a.exit_code,
                "error": a.error,
                "error_first_line": err_first,
                "error_classification": err_class,
                "is_worker_failure": a.is_worker_failure,
                "started_at": a.started_at,
                "started_at_ms": _iso_to_epoch_ms(a.started_at),
                "finished_at": a.finished_at,
                "finished_at_ms": _iso_to_epoch_ms(a.finished_at),
            }
            atts_out.append(att_rec)
            summary_records.append(
                {
                    "task_id": t.task_id,
                    "attempt_id": a.attempt_id,
                    "trainer_started_ts_ms": _iso_to_epoch_ms(a.started_at),
                    "ended_ts_ms": _iso_to_epoch_ms(a.finished_at),
                    "state": a.state,
                    "exit_code": a.exit_code,
                    "error_first_line": err_first,
                    "error_classification": err_class,
                }
            )
        tasks_out.append(
            {
                "task_id": t.task_id,
                "state": t.state,
                "started_at": t.started_at,
                "started_at_ms": _iso_to_epoch_ms(t.started_at),
                "finished_at": t.finished_at,
                "finished_at_ms": _iso_to_epoch_ms(t.finished_at),
                "exit_code": t.exit_code,
                "error": t.error,
                "attempts": atts_out,
            }
        )

    summary_records.sort(
        key=lambda r: (r.get("trainer_started_ts_ms") or 0, r["task_id"])
    )

    payload = {
        "schema_version": 2,
        "label": label,
        "job_id": report.job_id,
        "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "job_state": report.state_name,
        "job_failure_count": report.failure_count,
        "job_preemption_count": report.preemption_count,
        "completed_count": report.completed_count,
        "submitted_at": report.submitted_at,
        "submitted_at_ms": _iso_to_epoch_ms(report.submitted_at),
        "started_at": report.started_at,
        "started_at_ms": _iso_to_epoch_ms(report.started_at),
        "finished_at": report.finished_at,
        "finished_at_ms": _iso_to_epoch_ms(report.finished_at),
        "tasks": tasks_out,
        "attempts_summary": summary_records,
    }
    click.echo(json.dumps(payload, indent=1, default=str))


if __name__ == "__main__":
    # Drop our script name so iris sees argv as if it were invoked directly.
    iris()
