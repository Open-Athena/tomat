"""Roundtrip tests for the rho_gga Zarr loader."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from tomat.data.zarr_io import load_rho_gga, load_rho_gga_as_chgcar_like


def _build_fake_rho_gga(out: Path, shape: tuple[int, int, int] = (16, 16, 20)) -> np.ndarray:
    """Mirror the on-disk layout of one rho_gga ``<task>.zarr`` directory."""
    if out.exists():
        shutil.rmtree(out)

    structure_dict = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0.0,
        "lattice": {
            "matrix": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            "pbc": [True, True, True],
            "a": 4.0, "b": 4.0, "c": 4.0,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "volume": 64.0,
        },
        "properties": {},
        "sites": [
            {
                "species": [{"element": "H", "occu": 1}],
                "abc": [0.0, 0.0, 0.0],
                "properties": {},
                "label": "H",
                "xyz": [0.0, 0.0, 0.0],
            }
        ],
    }

    g = zarr.open_group(str(out), mode="w")
    g.attrs["structure"] = json.dumps(structure_dict)
    g.attrs["metadata"] = json.dumps({"task_id": out.stem.removesuffix(".zarr")})

    rng = np.random.default_rng(0)
    density = rng.uniform(0.01, 1.0, size=shape).astype(np.float32)
    g.create_array("charge_density_total", shape=density.shape, dtype=density.dtype)
    g["charge_density_total"][:] = density
    return density


def test_load_rho_gga_roundtrip(tmp_path):
    path = tmp_path / "mp-fake-1.zarr"
    original = _build_fake_rho_gga(path)

    sample = load_rho_gga(path)

    assert sample.task_id == "mp-fake-1"
    assert sample.data["total"].shape == original.shape
    assert np.array_equal(sample.data["total"], original)
    assert [site.specie.symbol for site in sample.structure] == ["H"]
    assert sample.structure.volume == pytest.approx(64.0)


def test_load_rho_gga_as_chgcar_like_duck_typing(tmp_path):
    """The shim must expose ``data['total']`` + ``structure`` so existing
    tokenizers (direct, cutoff, fourier, delta) consume it unchanged."""
    path = tmp_path / "mp-fake-2.zarr"
    _build_fake_rho_gga(path)

    shim = load_rho_gga_as_chgcar_like(path)
    assert "total" in shim.data
    assert shim.data["total"].ndim == 3
    assert len(shim.structure) == 1  # one atom in the fake


def test_load_rho_gga_nonuniform_shape(tmp_path):
    """rho_gga grid shapes are typically non-cubic — loader must handle that."""
    path = tmp_path / "mp-fake-3.zarr"
    _build_fake_rho_gga(path, shape=(20, 24, 32))

    sample = load_rho_gga(path)
    assert sample.data["total"].shape == (20, 24, 32)
