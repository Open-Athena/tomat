"""Roundtrip tests that don't require pymatgen IO — we stub a Chgcar-like object."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tomat.pads import GaussianPADS, SlaterPADS
from tomat.tokenizers import CutoffTokenizer, DeltaDensityTokenizer, DirectTokenizer, FourierTokenizer


def make_density(shape: tuple[int, int, int] = (16, 18, 20), seed: int = 0) -> np.ndarray:
    """Synthetic smooth-ish density: a few Gaussian lumps on a grid."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(
        np.linspace(0, 1, shape[0], endpoint=False),
        np.linspace(0, 1, shape[1], endpoint=False),
        np.linspace(0, 1, shape[2], endpoint=False),
        indexing="ij",
    )
    density = np.zeros(shape, dtype=np.float64)
    for _ in range(4):
        cx, cy, cz = rng.uniform(0.2, 0.8, size=3)
        sigma = rng.uniform(0.05, 0.15)
        density += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma**2))
    return density


def fake_chgcar(density: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(data={"total": density})


def nmae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / np.abs(a).sum())


def test_direct_is_lossless():
    density = make_density()
    recon = DirectTokenizer().roundtrip(fake_chgcar(density))
    assert recon.shape == density.shape
    # float64 → float32 → float64 loses ~ulp; should be effectively lossless.
    assert nmae(density, recon) < 1e-6


def test_cutoff_top_fraction_one_is_lossless():
    density = make_density()
    recon = CutoffTokenizer(top_fraction=1.0).roundtrip(fake_chgcar(density))
    assert nmae(density, recon) < 1e-6


def test_cutoff_top_fraction_keeps_exactly_k_voxels():
    density = make_density(shape=(10, 10, 10))
    encoded = CutoffTokenizer(top_fraction=0.1).encode(fake_chgcar(density))
    assert encoded.flat_indices.size == 100


def test_cutoff_nmae_decreases_with_more_voxels():
    density = make_density()
    fractions = [0.01, 0.05, 0.25, 1.0]
    nmaes = [nmae(density, CutoffTokenizer(top_fraction=f).roundtrip(fake_chgcar(density))) for f in fractions]
    assert nmaes == sorted(nmaes, reverse=True)


def test_fourier_full_basis_is_near_lossless():
    density = make_density()
    recon = FourierTokenizer(coefficient_fraction=1.0).roundtrip(fake_chgcar(density))
    assert nmae(density, recon) < 1e-6


def test_fourier_nmae_decreases_with_more_coefficients():
    density = make_density()
    fractions = [0.01, 0.05, 0.25, 1.0]
    nmaes = [nmae(density, FourierTokenizer(coefficient_fraction=f).roundtrip(fake_chgcar(density))) for f in fractions]
    assert nmaes == sorted(nmaes, reverse=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"top_k": 10, "top_fraction": 0.1},
        {"top_k": 10, "threshold": 0.1},
        {"top_fraction": 0.1, "threshold": 0.1},
        {"top_fraction": 0.0},
        {"top_fraction": 1.5},
    ],
)
def test_cutoff_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        CutoffTokenizer(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_coefficients": 10, "coefficient_fraction": 0.1},
        {"coefficient_fraction": 0.0},
        {"coefficient_fraction": 1.5},
        {"n_coefficients": 0},
    ],
)
def test_fourier_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        FourierTokenizer(**kwargs)


def fake_chgcar_with_structure(density: np.ndarray) -> SimpleNamespace:
    """Structure with two fake atoms in a small cubic cell for Δρ tests."""
    lattice = Mock()
    lattice.matrix = np.eye(3) * 4.0  # 4 Å cube
    structure = Mock()
    structure.lattice = lattice
    structure.volume = 64.0
    site_a = Mock()
    site_a.specie = Mock(Z=6)
    site_a.frac_coords = np.array([0.25, 0.25, 0.25])
    site_b = Mock()
    site_b.specie = Mock(Z=8)
    site_b.frac_coords = np.array([0.75, 0.75, 0.75])
    structure.__iter__ = lambda self: iter([site_a, site_b])
    return SimpleNamespace(data={"total": density}, structure=structure)


def test_delta_fourier_full_basis_is_lossless():
    density = make_density()
    chg = fake_chgcar_with_structure(density)
    # At 100% coefs, Fourier is lossless, and PADS cancels exactly on decode.
    wrapped = DeltaDensityTokenizer(FourierTokenizer(coefficient_fraction=1.0), pads=GaussianPADS())
    recon = wrapped.roundtrip(chg)
    # FourierEncoded stores coefs as complex64, so 1e-5 is the float32 roundoff floor.
    assert nmae(density, recon) < 1e-4


def test_delta_preserves_sum_within_noise():
    """At full Fourier basis the roundtrip should preserve total mass; PADS is a
    deterministic add/subtract so it can't introduce net mass."""
    density = make_density()
    chg = fake_chgcar_with_structure(density)
    wrapped = DeltaDensityTokenizer(FourierTokenizer(coefficient_fraction=1.0), pads=SlaterPADS())
    recon = wrapped.roundtrip(chg)
    # Float32 roundoff in FourierEncoded; SlaterPADS's sharp core amplifies it.
    assert abs(density.sum() - recon.sum()) < 1e-4 * abs(density.sum())


def test_pads_sums_match_total_electrons_under_integration():
    """∫ ρ_PADS dV should equal Σ Z over atoms (for both PADS variants)."""
    density = make_density()
    chg = fake_chgcar_with_structure(density)
    voxel_volume = 64.0 / density.size  # cell volume / N_voxels
    expected_electrons = 6 + 8  # C + O

    for pads in (GaussianPADS(sigma_angstrom=0.3), SlaterPADS()):
        grid = pads.compute(chg)
        total = grid.sum() * voxel_volume
        # Periodic-image truncation + coarse grid means this isn't exact,
        # but should be within a few percent for reasonable σ/α.
        assert abs(total - expected_electrons) / expected_electrons < 0.1
