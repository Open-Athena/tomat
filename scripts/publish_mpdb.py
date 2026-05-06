#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "utz"]
# ///
"""Publish ``data/mpdb.sqlite`` to R2 under a versioned key.

Layout: ``s3://<bucket>/mpdb/v<N>/mpdb.sqlite``

Bumps the version on schema changes. Reads the schema version from the
DB itself (``PRAGMA user_version``); falls back to ``--version`` flag.

Shell-out approach: invokes ``aws s3 cp`` with ``AWS_PROFILE=cfo``
(or whatever ``--profile`` is passed). No boto3 dependency.

Usage:
    scripts/publish_mpdb.py                       # default: read user_version
    scripts/publish_mpdb.py --version 2 --dry-run # explicit version, no-op
    scripts/publish_mpdb.py --bucket openathena --prefix mpdb
"""

from __future__ import annotations

import sqlite3
import sys
from functools import partial
from pathlib import Path

from utz.cli import arg, cmd, flag, opt
from utz.proc import run

err = partial(print, file=sys.stderr)

DEFAULT_DB = Path("data/mpdb.sqlite")
DEFAULT_BUCKET = "openathena"
DEFAULT_PREFIX = "mpdb"
DEFAULT_PROFILE = "cfo"


def read_schema_version(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


@cmd
@opt('-b', '--bucket', default=DEFAULT_BUCKET, help=f"R2 bucket (default {DEFAULT_BUCKET})")
@opt('-d', '--db', type=Path, default=DEFAULT_DB, help=f"DB path (default {DEFAULT_DB})")
@flag('-n', '--dry-run', help="Print the aws command without running it")
@opt('-p', '--profile', default=DEFAULT_PROFILE, help=f"AWS profile (default {DEFAULT_PROFILE})")
@opt('-P', '--prefix', default=DEFAULT_PREFIX, help=f"Key prefix under bucket (default {DEFAULT_PREFIX})")
@opt('-v', '--version', type=int, default=None, help="Override schema version (else read from PRAGMA user_version)")
def main(bucket: str, db: Path, dry_run: bool, profile: str, prefix: str, version: int | None):
    if not db.exists():
        err(f"[publish_mpdb] DB not found: {db}")
        sys.exit(1)

    v = version if version is not None else read_schema_version(db)
    if v == 0 and version is None:
        err(f"[publish_mpdb] PRAGMA user_version is 0; pass --version explicitly")
        sys.exit(1)

    key = f"{prefix}/v{v}/mpdb.sqlite"
    dst = f"s3://{bucket}/{key}"
    cmd_args = ["aws", "s3", "cp", str(db), dst, "--profile", profile]

    err(f"[publish_mpdb] {db} → {dst}")
    if dry_run:
        err(f"[publish_mpdb] (dry-run) {' '.join(cmd_args)}")
        return

    run(cmd_args)
    err(f"[publish_mpdb] uploaded {db.stat().st_size:,} bytes to {dst}")


if __name__ == "__main__":
    main()
