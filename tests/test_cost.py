"""Tests for `tomat.cost` — Modal-from-manifest + TPU wandb fallback (spec 46).

Phase B added wallclock-from-manifest derivation: Modal runs read
`history.ts_max - ts_min`, and TPU runs use the same fallback when the
attempts sidecar undercounts vs job-level cycle counters.
"""
from tomat.cost import (
    CostSegment,
    MODAL_GPU_RATES_USD_PER_HR,
    MODAL_CPU_MEM_ADDER,
    ModalGpu,
    TpuVariant,
    build_cost_record,
    compute_modal_segment_from_manifest,
    compute_tpu_segment_from_manifest,
    parse_modal_gpu,
)


# ── Modal: run-name parsing ─────────────────────────────────────────────────


def test_parse_modal_gpu_h200x8():
    g = parse_modal_gpu("train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42")
    assert g == ModalGpu(gpu="h200", num_gpus=8)


def test_parse_modal_gpu_h100x8_short_name():
    g = parse_modal_gpu("train-modal-h100x8-foo")
    assert g == ModalGpu(gpu="h100", num_gpus=8)


def test_parse_modal_gpu_all_families():
    # Exercise every family in the rate table so a future family addition
    # without a rate-table entry trips a test.
    families = ["h100", "h200", "b200", "a100", "l40s"]
    parsed = [parse_modal_gpu(f"train-foo-{fam}x4-bar") for fam in families]
    assert parsed == [ModalGpu(gpu=fam, num_gpus=4) for fam in families]


def test_parse_modal_gpu_non_modal_run():
    assert parse_modal_gpu("train-mg-tz-11") is None
    assert parse_modal_gpu("train-full-v3-200M-v6e16-shuf1k") is None


def test_modal_gpu_rate():
    # h200×8 with the 1.005 CPU/mem adder.
    g = ModalGpu(gpu="h200", num_gpus=8)
    expected = 8 * MODAL_GPU_RATES_USD_PER_HR["h200"] * MODAL_CPU_MEM_ADDER
    assert g.rate_usd_per_hr == expected
    assert g.label() == "H200×8"


def test_modal_gpu_unknown_family():
    g = ModalGpu(gpu="mi300", num_gpus=8)
    assert g.rate_usd_per_hr is None


# ── Modal: wallclock derivation from manifest ───────────────────────────────


def _manifest(ts_min=None, ts_max=None, runtime=None, ts=None):
    """Tiny manifest fixture matching the shape `runs sync` writes."""
    return {
        "history": {"ts_min": ts_min, "ts_max": ts_max},
        "summary": {"_runtime": runtime, "_timestamp": ts},
        "run": {"state": "running"},
    }


def test_modal_segment_from_ts_span():
    # ts_max - ts_min = 7200s (2h). 8 × h200 × 1.005 = $36.66/hr.
    seg = compute_modal_segment_from_manifest(
        "train-modal-h200x8-foo",
        _manifest(ts_min=1000.0, ts_max=8200.0),
    )
    expected_rate = 8 * MODAL_GPU_RATES_USD_PER_HR["h200"] * MODAL_CPU_MEM_ADDER
    expected_msrp = 2.0 * expected_rate
    assert seg == CostSegment(
        kind="modal",
        label="H200×8",
        wallclock_sec=7200.0,
        rate_per_hr_usd=expected_rate,
        msrp_usd=expected_msrp,
        source="wandb history ts span",
        started_at_ms=1_000_000,
        finished_at_ms=8_200_000,
    )


def test_modal_segment_runtime_fallback_when_ts_missing():
    # No ts span; fall back to _runtime + _timestamp.
    seg = compute_modal_segment_from_manifest(
        "train-modal-h100x8-foo",
        _manifest(runtime=3600.0, ts=5000.0),
    )
    assert seg is not None
    assert seg.source == "wandb summary._runtime"
    assert seg.wallclock_sec == 3600.0
    assert seg.started_at_ms == 1_400_000  # (5000 - 3600) * 1000
    assert seg.finished_at_ms == 5_000_000


def test_modal_segment_no_wallclock_returns_none():
    # No ts span AND no _runtime → no segment.
    assert compute_modal_segment_from_manifest(
        "train-modal-h200x8-foo",
        _manifest(),
    ) is None


def test_modal_segment_non_modal_label():
    # Not a Modal run name → None (TPU pricing handles it elsewhere).
    assert compute_modal_segment_from_manifest(
        "train-mg-tz-11",
        _manifest(ts_min=0.0, ts_max=3600.0),
    ) is None


def test_modal_segment_no_manifest():
    assert compute_modal_segment_from_manifest(
        "train-modal-h200x8-foo", None,
    ) is None


def test_modal_segment_sub_second_wallclock():
    # Crashed-immediately run with a degenerate span: must not emit
    # a nonsensical fractional segment.
    assert compute_modal_segment_from_manifest(
        "train-modal-h200x8-foo",
        _manifest(ts_min=1000.0, ts_max=1000.5),
    ) is None


# ── TPU: wandb-history fallback ─────────────────────────────────────────────


def test_tpu_segment_from_manifest_v6e16_preempt():
    # v6e-16 preempt rate: 16 × $1.08/hr = $17.28/hr. ts span = 1h.
    variant = TpuVariant(variant="v6e", num_chips=16, allocation_class="preemptible")
    seg = compute_tpu_segment_from_manifest(
        variant, _manifest(ts_min=0.0, ts_max=3600.0),
    )
    assert seg == CostSegment(
        kind="tpu",
        label="v6e-16 preemptible",
        wallclock_sec=3600.0,
        rate_per_hr_usd=17.28,
        msrp_usd=17.28,
        source="wandb history ts span (covers all iris re-fires)",
        started_at_ms=0,
        finished_at_ms=3_600_000,
    )


def test_tpu_segment_from_manifest_v5p16_ondemand():
    # v5p-16 on-demand: 16 × $4.20/hr = $67.20/hr. ts span = 2h.
    variant = TpuVariant(variant="v5p", num_chips=16, allocation_class="on-demand")
    seg = compute_tpu_segment_from_manifest(
        variant, _manifest(ts_min=0.0, ts_max=7200.0),
    )
    assert seg is not None
    assert seg.label == "v5p-16 on-demand"
    assert seg.wallclock_sec == 7200.0
    assert seg.msrp_usd == 2.0 * 67.20


def test_tpu_segment_from_manifest_no_history():
    variant = TpuVariant(variant="v6e", num_chips=16, allocation_class="preemptible")
    assert compute_tpu_segment_from_manifest(variant, _manifest()) is None
    assert compute_tpu_segment_from_manifest(variant, None) is None


def test_tpu_segment_unknown_variant():
    # Unknown variant → no rate → no segment.
    variant = TpuVariant(variant="v7e", num_chips=32, allocation_class="preemptible")
    assert compute_tpu_segment_from_manifest(
        variant, _manifest(ts_min=0.0, ts_max=3600.0),
    ) is None


# ── build_cost_record: schema ───────────────────────────────────────────────


def test_build_cost_record_modal_segment_sorted():
    big = CostSegment(
        kind="modal", label="H200×8", wallclock_sec=3600.0,
        rate_per_hr_usd=36.66, msrp_usd=36.66, source="wandb history ts span",
    )
    small = CostSegment(
        kind="modal", label="H200×8", wallclock_sec=60.0,
        rate_per_hr_usd=36.66, msrp_usd=0.61, source="wandb summary._runtime",
    )
    rec = build_cost_record(
        [small, big],
        is_complete=False,
        computed_at_iso="2026-06-05T00:00:00Z",
    )
    # Larger segment first (sorted by msrp desc).
    assert [b["msrp_usd"] for b in rec["breakdown"]] == [36.66, 0.61]
    assert rec["msrp_usd"] == 37.27  # rounded sum
    assert rec["is_complete"] is False
    assert rec["pricing_table_version"] == "2026-06-02"
    assert rec["schema_version"] == 1
    assert rec["notes"] == []
