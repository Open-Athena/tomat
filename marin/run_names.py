"""Curatable short-name registry for tomat run names + source specs.

The registry lives at `runs/registry.json` at the repo root and is the
single source of truth for *vetted* short names. Functions here mirror
`site/src/lib/runNames.ts` — keep the two in sync.

Public surface:
  - `shorten_run(canonical)` / `expand_run(short)`
  - `shorten_step(step_idx)` / `expand_step(short)` — `step_idx` is a 0-based
    Levanter `info.step` value; see `docs/step-conventions.md`.
  - `shorten_spec(spec)` / `expand_spec(short)`

`shorten_run` falls back to the same regex extraction used by the CLI's
`_viz_short_run` when the registry has no entry; uncurated runs still get
a reasonable short name. `expand_run` is exact-match only — the regex
fallback isn't invertible.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# ── Registry ────────────────────────────────────────────────────────────────

# `runs/registry.json` lives at the repo root; this file is at
# `<repo>/marin/run_names.py`, so the registry is one directory up.
_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "runs" / "registry.json"


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    """Parse `runs/registry.json` once; return `{canonical → entry, short → entry}` index."""
    raw = json.loads(_REGISTRY_PATH.read_text())
    by_canonical: dict[str, dict] = {}
    by_short: dict[str, dict] = {}
    for entry in raw.get("runs", []):
        canonical = entry["canonical"]
        short = entry["short"]
        if canonical in by_canonical:
            raise ValueError(f"duplicate canonical run name in registry: {canonical!r}")
        if short in by_short:
            raise ValueError(f"duplicate short name in registry: {short!r}")
        by_canonical[canonical] = entry
        by_short[short] = entry
    return {"by_canonical": by_canonical, "by_short": by_short}


# ── Run names ───────────────────────────────────────────────────────────────

# Mirrors `_VIZ_RUN_SHORTNAME_RE` in `tomat/tomat`. Strips an optional `kl-`
# prefix so `train-mg-kl-bin5-fs-tpu` → `bin5` (not `kl-bin5`).
_RUN_SHORTNAME_RE = re.compile(r"train-mg-(?:kl-)?([a-z0-9]+(?:-[a-z0-9]+)?)-")
_RUN_GENERIC_RE = re.compile(r"train-(?:mg-)?([a-z0-9]+(?:-[a-z0-9]+)?)-")


def shorten_run(canonical: str) -> str:
    """Return the registered short name for `canonical`, else a regex-derived fallback.

    If neither the registry nor the regex matches, the canonical name is
    truncated to 32 chars to keep axis labels manageable.
    """
    entry = _load_registry()["by_canonical"].get(canonical)
    if entry is not None:
        return entry["short"]
    m = _RUN_SHORTNAME_RE.match(canonical)
    if m:
        return m.group(1)
    m = _RUN_GENERIC_RE.match(canonical)
    if m:
        return m.group(1)
    return canonical[:32]


def expand_run(short: str) -> str | None:
    """Return the canonical run name for `short`, or `None` if not registered.

    Regex fallback isn't invertible — only exact registry hits resolve.
    """
    entry = _load_registry()["by_short"].get(short)
    return entry["canonical"] if entry is not None else None


# ── Step counts ─────────────────────────────────────────────────────────────


def snap_step(raw_step: int, tolerance: float = 0.001) -> int:
    """Snap a raw step to the nearest round value within `tolerance` (default 0.1%).

    `raw_step` is a 0-based `step_idx` from the GCS checkpoint name (Levanter
    `info.step` convention) — see `docs/step-conventions.md` for the project's
    `step_idx` / `step_n` / `target_steps` / `step_display` distinction.

    Why this exists: Levanter's force-saved final ckpt at end-of-training lands
    at `step-(N-1)` because `info.step = state.step - 1` is 0-indexed and the
    end-of-run hook fires with `force=True`. (Periodic ckpts during training
    land at clean `step-{N,2N,…}` because the modulo test runs against the same
    0-indexed `info.step`, just happens to land on round values.) This helper
    lets the CLI / FE render `step-49999` as `50k` without special-casing
    whether a given step value is a periodic or end-of-run ckpt.

    Examples: `89999 → 90000`, `49999 → 50000`, `50001 → 50000`,
    `53000 → 53000` (already round), `1234 → 1234` (no round Nk within tol).
    """
    if raw_step <= 0:
        return raw_step
    for base in (1_000_000, 100_000, 10_000, 1_000):
        rounded = round(raw_step / base) * base
        if rounded > 0 and abs(rounded - raw_step) / raw_step <= tolerance:
            return rounded
    return raw_step


def shorten_step(step_idx: int) -> str:
    """Format a 0-based `step_idx` compactly: 30000 → `30k`, 89999 → `≈90k`, 1500000 → `1.5M`.

    `step_idx` is a 0-based Levanter `info.step` value (i.e. lifted from a
    GCS `step-{step_idx}` checkpoint name). See `docs/step-conventions.md`.

    Snaps `step_idx` to the nearest round Nk first (so 89999 → 90k explicitly,
    not via rounding accident), then base-1000-formats with 1-decimal
    precision for non-round values. When the raw value was snapped (i.e.
    differed from the rendered round form), the result is prefixed with `≈`
    so the rendering is transparent — a reader of e.g. `≈90k` knows the
    on-disk artifact is at a nearby raw value (Levanter saves final-of-
    segment ckpts as `step-(N-1)`).
    """
    snapped = snap_step(step_idx)
    if snapped >= 10**6:
        v = snapped / 1e6
        pretty = f"{v:.1f}M" if v < 10 else f"{round(v)}M"
    elif snapped >= 1000:
        pretty = f"{round(snapped / 1000)}k"
    else:
        pretty = str(snapped)
    return pretty if snapped == step_idx else f"{pretty}*"


def format_step(step_idx: int) -> str:
    """Snap + pretty-print a 0-based `step_idx`, mirroring `formatStep` in `site/src/lib/runNames.ts`.

    `step_idx` is a 0-based Levanter `info.step` value. Identical to
    `shorten_step` for now — both produce `≈90k` for snapped values, plain
    `30k` / `1.5M` for clean ones. Kept as a separate symbol so future
    divergence (e.g. a tables-only variant that prefers comma-grouped digits)
    doesn't churn the call sites.
    """
    return shorten_step(step_idx)


def format_step_detail(step_idx: int) -> dict:
    """Snap + emit (display, raw, snapped, is_snapped, tooltip) for a 0-based `step_idx`.

    `step_idx` is a 0-based Levanter `info.step` value. Mirrors
    `formatStepDetail` in `site/src/lib/runNames.ts`. The tooltip explains the
    Levanter end-of-segment naming convention (info.step = state.step − 1) so
    consumers can surface the raw value when a viewer hovers a `≈90k`-style cell.
    """
    snapped = snap_step(step_idx)
    is_snapped = snapped != step_idx
    display = shorten_step(step_idx)
    if is_snapped:
        tooltip = (
            f"raw step {step_idx:,} on disk; snapped to {snapped:,}. "
            f"Levanter saves the final ckpt of each training segment using "
            f"info.step (0-indexed, = state.step − 1), so step-{step_idx} is the "
            f"artifact name for what is logically the {snapped:,}th step "
            f"completed. Periodic ckpts within the same run save at clean N."
        )
    else:
        tooltip = f"step {step_idx:,}"
    return {
        "display": display,
        "is_snapped": is_snapped,
        "raw": step_idx,
        "snapped": snapped,
        "tooltip": tooltip,
    }


_STEP_RE = re.compile(r"^[≈]?(\d+(?:\.\d+)?)([kM])?\*?$")


def expand_step(short: str) -> int | None:
    """Inverse of `shorten_step`. `30k` → 30000, `1.5M` → 1500000.

    Returns a 0-based `step_idx` (Levanter `info.step` convention; see
    `docs/step-conventions.md`).

    Accepts a leading `≈` (the snap marker emitted by `shorten_step`) and
    discards it — the round-trip is already lossy when snapping fires
    (`89999 → ≈90k → 90000`), so we honor the snapped value.

    Returns `None` for unparseable inputs.
    """
    m = _STEP_RE.match(short.strip())
    if m is None:
        return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "M":
        return int(round(val * 1_000_000))
    if suffix == "k":
        return int(round(val * 1000))
    return int(round(val))


# ── Source specs ────────────────────────────────────────────────────────────
#
# Spec grammar (matches the viz CLI):
#   gt:<split>                                   e.g. `gt:val`, `gt:train`
#   pred:<run>:<setmode>:<step>                  e.g. `pred:train-mg-kl-bin5-fs-tpu:val_200-maskgit:30000`
#
# Short forms:
#   GT (val), GT (train)                         from `gt:<split>`
#   <run-short> @ <step-short>                   from `pred:` with `maskgit` setmode
#   <run-short> @ <step-short>-K1                from `pred:` with `oneshot` setmode


def _split_short(split: str) -> str:
    if split.startswith("val"):
        return "val"
    if split.startswith("train"):
        return "train"
    return split


def shorten_spec(spec: str) -> str:
    """Map a source spec to its short label (no split tag returned)."""
    parts = spec.split(":")
    if parts and parts[0] == "gt" and len(parts) == 2:
        return f"GT ({_split_short(parts[1])})"
    if parts and parts[0] == "pred" and len(parts) == 4:
        _, run, setmode, step = parts
        # `step` here is a stringified 0-based `step_idx` from the spec grammar
        # (matches `step-{step_idx}` in GCS). See `docs/step-conventions.md`.
        try:
            step_idx = int(step)
        except ValueError:
            step_idx = 0
        run_short = shorten_run(run)
        step_short = shorten_step(step_idx)
        suffix = "-K1" if "oneshot" in setmode else ""
        return f"{run_short} @ {step_short}{suffix}"
    return spec


_SHORT_GT_RE = re.compile(r"^GT \((val|train)\)$")
_SHORT_PRED_RE = re.compile(r"^(?P<run>.+?) @ (?P<step>≈?\d+(?:\.\d+)?[kM]?)(?P<k1>-K1)?$")


def expand_spec(short: str) -> str | None:
    """Inverse of `shorten_spec`. Returns `None` when the run-short isn't registered.

    Defaults setmode to `val_200-maskgit` for pred specs (or `val_200-oneshot`
    when the short ends with `-K1`).
    """
    m = _SHORT_GT_RE.match(short)
    if m is not None:
        return f"gt:{m.group(1)}"
    m = _SHORT_PRED_RE.match(short)
    if m is not None:
        run_short = m.group("run")
        step_short = m.group("step")
        is_oneshot = m.group("k1") is not None
        canonical = expand_run(run_short)
        if canonical is None:
            return None
        step_idx = expand_step(step_short)
        if step_idx is None:
            return None
        setmode = "val_200-oneshot" if is_oneshot else "val_200-maskgit"
        return f"pred:{canonical}:{setmode}:{step_idx}"
    return None
