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
import sys
from dataclasses import asdict

import click


# Pull in the existing iris CLI machinery: registering on `iris.cli.main.iris`
# (the Click group) inherits all of its `--cluster=marin` / `--controller-url`
# / `--config` setup automatically.
from iris.cli.main import iris  # noqa: E402
from iris.cli.bug_report import gather_bug_report  # noqa: E402
from iris.cli.connect import require_controller_url  # noqa: E402
from iris.cluster.types import JobName  # noqa: E402


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
@click.option("--tail", type=int, default=0,
              help="Recent log lines per task (0 = skip log fetch, what we want)")
@click.argument("job_id")
@click.pass_context
def attempts_dump(ctx, tail: int, job_id: str):
    """Internal: gather per-attempt history for JOB_ID, emit JSON to stdout."""
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider")
    name = JobName.from_string(job_id) if not job_id.startswith("/") else JobName.from_wire(job_id)
    report = gather_bug_report(controller_url, name, tail=tail, token_provider=token_provider)

    # Build the flat payload. We KEEP the iso strings (human-readable in dev /
    # `jq` walks) and ADD epoch_ms for the JS dashboard. The frontend doesn't
    # need worker/log details, just attempt timing + state + error.
    label = report.job_id.removeprefix("/ryan/")
    tasks_out: list[dict] = []
    for t in report.tasks:
        atts_out: list[dict] = []
        for a in t.attempts:
            atts_out.append({
                "attempt_id": a.attempt_id,
                "worker_id": a.worker_id,
                "state": a.state,
                "exit_code": a.exit_code,
                "error": a.error,
                "is_worker_failure": a.is_worker_failure,
                "started_at": a.started_at,
                "started_at_ms": _iso_to_epoch_ms(a.started_at),
                "finished_at": a.finished_at,
                "finished_at_ms": _iso_to_epoch_ms(a.finished_at),
            })
        tasks_out.append({
            "task_id": t.task_id,
            "state": t.state,
            "started_at": t.started_at,
            "started_at_ms": _iso_to_epoch_ms(t.started_at),
            "finished_at": t.finished_at,
            "finished_at_ms": _iso_to_epoch_ms(t.finished_at),
            "exit_code": t.exit_code,
            "error": t.error,
            "attempts": atts_out,
        })

    payload = {
        "schema_version": 1,
        "label": label,
        "job_id": report.job_id,
        "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
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
    }
    click.echo(json.dumps(payload, indent=1, default=str))


if __name__ == "__main__":
    # Drop our script name so iris sees argv as if it were invoked directly.
    iris()
