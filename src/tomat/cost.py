"""TPU MSRP pricing table + per-run MSRP estimation (Phase A of spec 46).

Computes "equivalent retail value" per training run: total chip-seconds
× published GCP TPU rate. Does **not** model actual billing — Marin's
academic-credit flow is opaque to us. See `feedback_no_funding_in_public_repo`
memory + spec 46's "Important framing" section.

Modal pricing is Phase B; this module's `compute_modal_cost()` currently
returns `None` so the dashboard can render a "Modal MSRP TBD" placeholder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Pricing-table snapshot date. Update this string + the rates below together
# when GCP / Modal publish new prices, so the dashboard tooltip's "snapshot
# pricing 2026-MM-DD" caption stays honest.
PRICING_TABLE_VERSION = "2026-06-02"

# Cost-record JSON schema. Bump on incompatible changes so the worker /
# dashboard can detect older sidecars.
COST_SCHEMA_VERSION = 1


# Per-variant on-demand + preemptible rates, USD/chip-hour. Snapshot from
# https://cloud.google.com/tpu/pricing (europe-west4 region). v5p + v6e are
# the only TPU families tomat uses; add new entries here as needed.
TPU_RATES_USD_PER_CHIP_HR: dict[str, dict[str, float]] = {
    "v5p": {"on-demand": 4.20, "preemptible": 1.68},
    "v6e": {"on-demand": 2.70, "preemptible": 1.08},
}


# Iris worker hostnames embed both the variant (`v5p`, `v6e`) and the
# allocation class (`preemptible`, `reserved`, `serving`) — see
# `iris-pool-naming` memory. Example:
#     marin-tpu-v6e-preemptible-16-us-east1-d-20260602-0300-...
# The same fragment also appears in the iris `tpu` arg (`v6e-preemptible_16`).
_WORKER_HOSTNAME_RE = re.compile(
    r"\bmarin-tpu-(?P<variant>v[56][pe])-(?P<klass>preemptible|reserved|serving)-(?P<chips>\d+)\b"
)


@dataclass(frozen=True)
class TpuVariant:
    """Parsed TPU variant from an iris job / run name.

    `variant`: family string (`v5p`, `v6e`).
    `num_chips`: chip count from the size suffix (`-16` → 16).
    `allocation_class`: `preemptible` (Marin's spot pool) / `reserved`
        / `serving` / `on-demand`. Derived from the worker-hostname infix
        when the attempts sidecar has any worker-id-bearing error line;
        falls back to `preemptible` (Marin's default) when we can't tell.
    """
    variant: str
    num_chips: int
    allocation_class: str

    @property
    def rate_usd_per_chip_hr(self) -> float | None:
        rates = TPU_RATES_USD_PER_CHIP_HR.get(self.variant)
        if rates is None:
            return None
        # Treat `reserved` / `serving` as on-demand equivalent (GCP publishes
        # them at the on-demand rate; the spec snapshot only carries the two
        # tiers). `on-demand` is the explicit key.
        if self.allocation_class in ("on-demand", "reserved", "serving"):
            return rates.get("on-demand")
        return rates.get("preemptible")

    def label(self) -> str:
        """Human-readable: `v6e-16 preemptible`. Used in the tooltip."""
        return f"{self.variant}-{self.num_chips} {self.allocation_class}"


def parse_tpu_hardware(label: str) -> tuple[str, int] | None:
    """Extract `(variant, num_chips)` from a run label's hw suffix.

    Mirrors the parsing in `site/src/runs/runMeta.ts::parseRunName` but
    re-implemented here so the Python CLI can run without a JS bridge.
    Returns None for non-TPU / unrecognised labels — Modal runs land here.
    """
    # Match the same hw suffixes the dashboard parses: `v5p16`, `v6e32`,
    # `tpu16` (old shorthand → v6e), bare `tpuN`. The suffix can be the
    # last hw chunk or sit before a trailing `-shuf1k-<tag>` etc.
    for m in re.finditer(r"(?<![a-z0-9])(v5p|v6e|tpu)(\d+)(?![a-z0-9])", label, re.I):
        family, chips = m.group(1).lower(), int(m.group(2))
        variant = "v6e" if family == "tpu" else family
        return variant, chips
    return None


def parse_tpu_hardware_from_attempts(attempts: dict | None) -> tuple[str, int] | None:
    """Recover `(variant, num_chips)` from a worker hostname in the attempts
    sidecar. Used as a fallback for run labels that don't embed an hw suffix
    (e.g. `train-mg-tz-11` — pre-spec-46 naming convention) but whose iris
    worker hostname encodes the slice we ended up running on.
    """
    if attempts is None:
        return None
    candidates: list[str] = []
    for t in attempts.get("tasks", []):
        if t.get("error"):
            candidates.append(t["error"])
        for a in t.get("attempts", []):
            if a.get("worker_id"):
                candidates.append(a["worker_id"])
            if a.get("error"):
                candidates.append(a["error"])
            if a.get("error_first_line"):
                candidates.append(a["error_first_line"])
    for s in candidates:
        m = _WORKER_HOSTNAME_RE.search(s)
        if m:
            return m.group("variant"), int(m.group("chips"))
    return None


def detect_allocation_class(attempts: dict | None) -> str:
    """Sniff the allocation class out of an attempts-sidecar's worker names.

    Iris worker hostnames embed `<variant>-<class>-<chips>` (see
    `iris-pool-naming` memory). Worker IDs themselves are sometimes empty
    (early attempts), but the `error` field of a failed attempt typically
    includes the full worker hostname (`Worker marin-tpu-v6e-preemptible-16-...
    failed: ...`). Walk both.

    Defaults to `preemptible` when no signal is present — Marin's spot pool
    is the overwhelmingly common case and over-attributing to preemptible
    biases the MSRP estimate DOWN, which is the safer direction for a
    public-facing chip.
    """
    if attempts is None:
        return "preemptible"
    candidates: list[str] = []
    for t in attempts.get("tasks", []):
        if t.get("error"):
            candidates.append(t["error"])
        for a in t.get("attempts", []):
            if a.get("worker_id"):
                candidates.append(a["worker_id"])
            if a.get("error"):
                candidates.append(a["error"])
            if a.get("error_first_line"):
                candidates.append(a["error_first_line"])
    for s in candidates:
        m = _WORKER_HOSTNAME_RE.search(s)
        if m:
            return m.group("klass")
    return "preemptible"


@dataclass
class CostSegment:
    """One contributor to a run's MSRP — one task-attempt window for TPU,
    one Modal function call for Modal (Phase B)."""
    kind: str                       # "tpu" (Phase A) / "modal" (Phase B)
    label: str                      # "v6e-16 preemptible", "H200×8", ...
    wallclock_sec: float
    rate_per_hr_usd: float | None   # None when rate is unknown (Phase B Modal)
    msrp_usd: float | None
    source: str                     # `task_id` (TPU) / `fc_id` (Modal)
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "wallclock_sec": round(self.wallclock_sec, 3),
            "rate_per_hr_usd": self.rate_per_hr_usd,
            "msrp_usd": (None if self.msrp_usd is None
                         else round(self.msrp_usd, 4)),
            "source": self.source,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }


def compute_tpu_segments(
    attempts: dict,
    variant: TpuVariant,
    *,
    now_ms: int | None = None,
) -> list[CostSegment]:
    """One CostSegment per *attempt cycle* (gang-scheduled across tasks).

    Iris co-schedules N tasks per attempt; each task is one host within the
    same TPU slice. Billing happens at the slice level — `num_chips × rate`
    is already the per-slice cost — so we collapse to one segment per
    `attempt_id`, using the earliest-start → latest-end window across the
    attempt's tasks.

    Unfinished attempts (any task still running) bill up to `now_ms` so the
    chip keeps ticking. Attempts whose tasks never started contribute nothing.
    """
    segs: list[CostSegment] = []
    rate = variant.rate_usd_per_chip_hr
    if rate is None:
        return segs
    label_str = variant.label()
    # Group per `attempt_id`. Within one attempt, tasks share a slice, so
    # we take the union of their windows (min start, max end across the
    # tasks that DO have an end). A common cascade pattern: task 0 has a
    # finished_at (worker_failed), tasks 1..N show as 'preempted' with
    # null finished_at — they actually ended when iris bounced the cycle.
    # So we use the latest known end across the attempt's tasks; only if
    # NO task in the attempt has an end do we charge to `now_ms`.
    by_attempt: dict[int, dict] = {}
    for t in attempts.get("tasks", []):
        task_id = t.get("task_id", "")
        for a in t.get("attempts", []):
            aid = a.get("attempt_id")
            if aid is None:
                continue
            start_ms = a.get("started_at_ms")
            if start_ms is None:
                continue
            end_ms = a.get("finished_at_ms")
            bucket = by_attempt.setdefault(aid, {
                "start_ms": start_ms,
                "end_ms": None,
                "task_ids": [],
            })
            if start_ms < bucket["start_ms"]:
                bucket["start_ms"] = start_ms
            if end_ms is not None:
                if bucket["end_ms"] is None or end_ms > bucket["end_ms"]:
                    bucket["end_ms"] = end_ms
            bucket["task_ids"].append(task_id)
    # Determine which attempt is the "live" one (largest attempt_id) — only
    # it should bill to now_ms if its tasks are unfinished. Earlier attempts
    # whose tasks have null finished_at are cascade-cleaned siblings; they
    # ended when the cycle bounced. We approximate that end as the next
    # attempt's start; only the largest attempt_id may bill to `now_ms`.
    sorted_aids = sorted(by_attempt)
    max_aid = sorted_aids[-1] if sorted_aids else None
    next_start_by_aid: dict[int, int] = {}
    for i, aid in enumerate(sorted_aids[:-1]):
        next_start_by_aid[aid] = by_attempt[sorted_aids[i + 1]]["start_ms"]
    for aid in sorted_aids:
        b = by_attempt[aid]
        start_ms = b["start_ms"]
        end_ms = b["end_ms"]
        if end_ms is None:
            # No task in this attempt reported a finished_at. Fall back to:
            # (1) the next attempt's start (cascade boundary) for non-latest,
            # (2) `now_ms` for the latest attempt (still in flight).
            if aid == max_aid:
                if now_ms is None:
                    continue
                end_ms = now_ms
            elif aid in next_start_by_aid:
                end_ms = next_start_by_aid[aid]
            else:
                continue
        wallclock_sec = max(0.0, (end_ms - start_ms) / 1000.0)
        if wallclock_sec < 1.0:
            # Drop trivial sub-second cycles — usually iris's immediate-bounce
            # records (worker died at start) that bloat the breakdown without
            # contributing real cost.
            continue
        msrp = (wallclock_sec / 3600.0) * variant.num_chips * rate
        segs.append(CostSegment(
            kind="tpu",
            label=label_str,
            wallclock_sec=wallclock_sec,
            rate_per_hr_usd=rate * variant.num_chips,
            msrp_usd=msrp,
            source=f"attempt {aid} ({len(b['task_ids'])} tasks)",
            started_at_ms=start_ms,
            finished_at_ms=end_ms,
        ))
    return segs


def compute_tpu_segments_from_job_window(
    job: dict,
    variant: TpuVariant,
    *,
    now_ms: int | None = None,
) -> list[CostSegment]:
    """Fallback when no attempts sidecar exists: charge for the full
    `submitted_at_ms → finished_at_ms` window of the parent iris job times
    `num_chips`.

    Less accurate than the attempts walk (counts queue-wait + provisioning
    as if the slice were billed), but better than nothing for older runs
    whose attempts sidecar predates the bug-report machinery.
    """
    rate = variant.rate_usd_per_chip_hr
    if rate is None:
        return []
    start_ms = job.get("started_at_ms") or job.get("submitted_at_ms")
    end_ms = job.get("finished_at_ms")
    if start_ms is None:
        return []
    if end_ms is None:
        if now_ms is None:
            return []
        end_ms = now_ms
    wallclock_sec = max(0.0, (end_ms - start_ms) / 1000.0)
    if wallclock_sec < 1.0:
        return []
    msrp = (wallclock_sec / 3600.0) * variant.num_chips * rate
    return [CostSegment(
        kind="tpu",
        label=variant.label(),
        wallclock_sec=wallclock_sec,
        rate_per_hr_usd=rate * variant.num_chips,
        msrp_usd=msrp,
        source="iris-state (no attempts sidecar)",
        started_at_ms=start_ms,
        finished_at_ms=end_ms,
    )]


def build_cost_record(
    segments: list[CostSegment],
    *,
    is_complete: bool,
    computed_at_iso: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON record written to `runs/<label>/cost.json`."""
    # Sort breakdown by contribution desc — the tooltip renders top-down,
    # so the biggest segment appears first.
    sorted_segs = sorted(
        segments,
        key=lambda s: (s.msrp_usd or 0.0),
        reverse=True,
    )
    total = sum((s.msrp_usd or 0.0) for s in sorted_segs)
    return {
        "schema_version": COST_SCHEMA_VERSION,
        "pricing_table_version": PRICING_TABLE_VERSION,
        "computed_at": computed_at_iso,
        "is_complete": is_complete,
        "msrp_usd": round(total, 2),
        "breakdown": [s.to_dict() for s in sorted_segs],
        "notes": notes or [],
    }
