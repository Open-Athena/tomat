"""Canonical iris runner roster — the ONE place to add a new runner.

Iris namespaces every job as `/<user>/<job-name>`, derived from the
submitting machine's OS user (`getpass.getuser()`, per iris's
`resolve_job_user`). So the listing / sync / lookup paths have to fan across
every runner whose jobs we care about. This module is the single source of
truth for that roster, shared by the `tomat` CLI and the iris-state sync
crons (`scripts/iris-sync.py` on the GCE VM, `scripts/cron_iris_sync_modal.py`
on Modal).

Kept dependency-free (stdlib only, no `tomat`-package import) so the lean
sync environments can import it directly. The Modal cron ships it into the
image via `add_local_python_source("iris_runners")`.
"""

from __future__ import annotations

# Every runner whose jobs we list / sync / look up. ADD A NEW RUNNER HERE
# (pre-2026-06 runs all live under /ryan/).
KNOWN_USERS: tuple[str, ...] = ("ryan", "betsy")

# Conventional job-name prefixes used in tomat work. `kl` catches
# loss-name-prefixed labels (e.g. `kl-s05-tpu-cont-v3`); see TRAINING_JOB_RE
# in the `tomat` CLI. iris requires `--prefix` to be `/<user>/<jobname>`
# (2+ segments), which is why every user is paired with every prefix.
JOB_PREFIXES: tuple[str, ...] = ("tomat", "train", "eval", "kl")


def iris_prefixes(
    users: tuple[str, ...] = KNOWN_USERS,
    prefixes: tuple[str, ...] = JOB_PREFIXES,
) -> list[str]:
    """`/<user>/<prefix>` for every (user, prefix) pair — iris `--prefix` form."""
    return [f"/{u}/{p}" for u in users for p in prefixes]
