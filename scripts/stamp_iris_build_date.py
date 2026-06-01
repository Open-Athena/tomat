#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "utz"]
# ///
"""Stamp `BUILD_DATE` in `iris/_build_info.py` to today's date + clear pyc.

Workaround for marin-iris editable installs shipping `BUILD_DATE = ""` →
iris server rejects with "marin-iris client is too old (build 2026-04-22;
minimum YYYY-MM-DD)". Stamping fixes the freshness check.

`uv sync` re-installs marin-iris and wipes the stamp, so this script is
idempotent and meant to be re-run after every sync. Also clears
`iris/**/__pycache__` because a half-finished `uv sync --upgrade` can leave
stale .pyc next to fresh .py files, and Python's importer happily mixes
them — symptom: `ModuleNotFoundError: No module named
'iris.cluster.controller.autoscaler.planning'` even though `planning.py`
exists on disk.

See: $oa/marin/gh/drafts/iris-empty-build-date/ for the upstream issue.
"""
from datetime import date
from functools import partial
from pathlib import Path
import re
import shutil
import sys

from click import command, option
from utz.cli import flag

err = partial(print, file=sys.stderr)

CANDIDATES = [
    Path('/Users/ryan/c/oa/tomat/.venv/lib/python3.12/site-packages/iris/_build_info.py'),
    Path('/Users/ryan/c/oa/tomat/marin/.venv/lib/python3.12/site-packages/iris/_build_info.py'),
]


@command()
@option('-d', '--date', 'date_str', help='Date to stamp (default: today, YYYY-MM-DD)')
@flag('-n', '--dry-run', help='print actions without writing')
def main(date_str: str | None, dry_run: bool):
    """Stamp BUILD_DATE in any venv iris/_build_info.py files found."""
    target = date_str or date.today().isoformat()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', target):
        err(f'bad date: {target!r}; expected YYYY-MM-DD')
        sys.exit(1)
    stamped = 0
    for path in CANDIDATES:
        if not path.exists():
            err(f'  skip (missing): {path}')
            continue
        content = path.read_text()
        m = re.search(r'^BUILD_DATE\s*=\s*"([^"]*)"', content, flags=re.M)
        if not m:
            err(f'  skip (no BUILD_DATE line): {path}')
            continue
        cur = m.group(1)
        if cur == target:
            err(f'  unchanged ({cur}): {path}')
            continue
        new = re.sub(r'^BUILD_DATE\s*=\s*"[^"]*"',
                     f'BUILD_DATE = "{target}"', content, flags=re.M)
        if dry_run:
            err(f'  would stamp {cur or "<empty>"} → {target}: {path}')
        else:
            path.write_text(new)
            err(f'  stamped {cur or "<empty>"} → {target}: {path}')
        stamped += 1
    err(f'done ({stamped} file{"s" if stamped != 1 else ""})')

    # Clear iris __pycache__ trees so stale .pyc from prior installs don't
    # mask freshly-installed .py modules. Symptom: ImportError on a module
    # whose .py file is present on disk (e.g.
    # `iris.cluster.controller.autoscaler.planning`).
    iris_roots = {p.parent for p in CANDIDATES if p.exists()}
    cleared = 0
    for root in iris_roots:
        for cache in root.rglob('__pycache__'):
            if dry_run:
                err(f'  would rm pyc: {cache}')
            else:
                shutil.rmtree(cache, ignore_errors=True)
            cleared += 1
    if cleared:
        err(f'cleared {cleared} __pycache__ dir{"s" if cleared != 1 else ""}')


if __name__ == '__main__':
    main()
