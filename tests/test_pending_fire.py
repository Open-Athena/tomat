"""Tests for `tomat modal pending-fire {add,list,gc}` (R2 sidecar mgmt).

The CLI lives in the `tomat` shebang script at the repo root, not a python
package — we load it via `importlib.util.spec_from_file_location` so the
tests can exercise its Click commands directly.

R2 I/O is mocked: `_load_pending_fires` / `_save_pending_fires` /
`_r2_list_synced_runs` are patched to operate against an in-memory dict so
the tests don't shell out to `aws s3 cp` (and don't need R2 creds).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOMAT_SCRIPT = PROJECT_ROOT / "tomat"


def _load_tomat_module():
    """Load the `tomat` shebang script as an importable module.

    Has no `.py` extension so `spec_from_file_location` returns None on
    its own — we hand it an explicit `SourceFileLoader` to bypass the
    extension check.
    """
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("tomat_cli", str(TOMAT_SCRIPT))
    spec = importlib.util.spec_from_loader("tomat_cli", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture
def tomat_mod():
    return _load_tomat_module()


@pytest.fixture
def fake_r2(tomat_mod, monkeypatch):
    """In-memory R2 for pending-fires + synced-runs list."""
    state: dict = {"pending_fires": None, "synced_runs": []}

    def fake_load(bucket: str = tomat_mod.R2_BUCKET) -> dict:
        if state["pending_fires"] is None:
            return {
                "schema_version": tomat_mod._PENDING_FIRES_SCHEMA_VERSION,
                "synced_at": "1970-01-01T00:00:00+00:00",
                "fires": [],
            }
        return json.loads(json.dumps(state["pending_fires"]))

    def fake_save(doc: dict, bucket: str = tomat_mod.R2_BUCKET, dry_run: bool = False):
        if dry_run:
            return
        state["pending_fires"] = json.loads(json.dumps(doc))

    def fake_list_runs(bucket: str, prefix: str) -> list[str]:
        return list(state["synced_runs"])

    monkeypatch.setattr(tomat_mod, "_load_pending_fires", fake_load)
    monkeypatch.setattr(tomat_mod, "_save_pending_fires", fake_save)
    monkeypatch.setattr(tomat_mod, "_r2_list_synced_runs", fake_list_runs)
    return state


def _add_args(
    *,
    fc_id: str,
    run_id: str,
    results_label: str,
    loss: str = "ce",
    spawned_at_ms: int = 1700000000000,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "modal", "pending-fire", "add",
        "--app-name", "tomat-train-smoke",
        "--batch-size", "128",
        "--fc-id", fc_id,
        "--function-name", "train_bakeoff_h200x8",
        "--hardware", "h200x8",
        "--loss-type", loss,
        "--model-preset", "200M",
        "--maskgit-prior", "absorbing",
        "--results-label", results_label,
        "--run-id", run_id,
        "--seed", "42",
        "--intended-steps", "1001",
        "--wandb-project", "tomat-lmq-P19",
        "--spawned-at-ms", str(spawned_at_ms),
    ]
    if extra:
        args += extra
    return args


def test_add_writes_canonical_entry(tomat_mod, fake_r2):
    runner = CliRunner()
    r = runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-AAA", run_id="train-mg-modal-h200x8-tz-fs-ce-bs128-seed42",
        results_label="train-mg-modal-h200x8-tz-fs-ce",
        extra=["--tags", "maskgit", "--tags", "mg-loss-ce"],
    ))
    assert r.exit_code == 0, r.output
    fires = fake_r2["pending_fires"]["fires"]
    assert fires == [{
        "run_id": "train-mg-modal-h200x8-tz-fs-ce-bs128-seed42",
        "fc_id": "fc-AAA",
        "app_name": "tomat-train-smoke",
        "function_name": "train_bakeoff_h200x8",
        "spawned_at_ms": 1700000000000,
        "hardware": "h200x8",
        "model_preset": "200M",
        "batch_size": 128,
        "seed": 42,
        "intended_steps": 1001,
        "results_label": "train-mg-modal-h200x8-tz-fs-ce",
        "wandb_project": "tomat-lmq-P19",
        "loss_type": "ce",
        "kl_sigma": None,
        "maskgit_prior": "absorbing",
        "tags": ["maskgit", "mg-loss-ce"],
    }]


def test_add_overwrites_same_run_id(tomat_mod, fake_r2):
    runner = CliRunner()
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-OLD", run_id="train-x-bs128-seed42", results_label="train-x",
    ))
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-NEW", run_id="train-x-bs128-seed42", results_label="train-x",
    ))
    fires = fake_r2["pending_fires"]["fires"]
    assert [(f["run_id"], f["fc_id"]) for f in fires] == [
        ("train-x-bs128-seed42", "fc-NEW"),
    ]


def test_add_keeps_other_entries(tomat_mod, fake_r2):
    runner = CliRunner()
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-1", run_id="train-a-bs128-seed42", results_label="train-a",
    ))
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-2", run_id="train-b-bs128-seed42", results_label="train-b",
    ))
    fires = fake_r2["pending_fires"]["fires"]
    assert sorted(f["run_id"] for f in fires) == [
        "train-a-bs128-seed42",
        "train-b-bs128-seed42",
    ]


def test_add_kl_sigma_passes_through(tomat_mod, fake_r2):
    runner = CliRunner()
    r = runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-K", run_id="train-kl-bs128-seed42", results_label="train-kl",
        loss="kl_gauss", extra=["--kl-sigma", "0.5"],
    ))
    assert r.exit_code == 0, r.output
    fires = fake_r2["pending_fires"]["fires"]
    assert fires[0]["loss_type"] == "kl_gauss"
    assert fires[0]["kl_sigma"] == 0.5


def test_list_outputs_entries(tomat_mod, fake_r2):
    runner = CliRunner()
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-L", run_id="train-list-bs128-seed42",
        results_label="train-list",
    ))
    r = runner.invoke(tomat_mod.cli, ["modal", "pending-fire", "list", "--json"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.output)
    assert doc["schema_version"] == 1
    assert [f["run_id"] for f in doc["fires"]] == ["train-list-bs128-seed42"]


def test_gc_drops_manifested_runs(tomat_mod, fake_r2, monkeypatch):
    runner = CliRunner()
    # Freeze "now" close to the fires' spawned_at_ms so the age-out branch
    # doesn't fire — we want to exercise the manifested-set branch only.
    monkeypatch.setattr(tomat_mod.time, "time", lambda: 1700000010.0)
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-1", run_id="train-a-bs128-seed42", results_label="train-a",
    ))
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-2", run_id="train-b-bs128-seed42", results_label="train-b",
    ))
    fake_r2["synced_runs"] = ["train-a-bs128-seed42"]
    r = runner.invoke(tomat_mod.cli, ["modal", "pending-fire", "gc"])
    assert r.exit_code == 0, r.output
    fires = fake_r2["pending_fires"]["fires"]
    assert [f["run_id"] for f in fires] == ["train-b-bs128-seed42"]


def test_gc_drops_aged_out_entries(tomat_mod, fake_r2, monkeypatch):
    runner = CliRunner()
    # Spawn epoch-ms ~70 hours ago; with default --max-age-hours=48 this drops.
    spawned = 1700000000_000
    monkeypatch.setattr(tomat_mod.time, "time",
                        lambda: (spawned + 70 * 3600 * 1000) / 1000)
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-stale", run_id="train-stale-bs128-seed42",
        results_label="train-stale", spawned_at_ms=spawned,
    ))
    r = runner.invoke(tomat_mod.cli, ["modal", "pending-fire", "gc"])
    assert r.exit_code == 0, r.output
    fires = fake_r2["pending_fires"]["fires"]
    assert fires == []


def test_gc_keeps_recent_unmanifested(tomat_mod, fake_r2, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(tomat_mod.time, "time", lambda: 1700000010.0)
    runner.invoke(tomat_mod.cli, _add_args(
        fc_id="fc-keep", run_id="train-keep-bs128-seed42",
        results_label="train-keep",
    ))
    r = runner.invoke(tomat_mod.cli, ["modal", "pending-fire", "gc"])
    assert r.exit_code == 0, r.output
    fires = fake_r2["pending_fires"]["fires"]
    assert [f["run_id"] for f in fires] == ["train-keep-bs128-seed42"]
