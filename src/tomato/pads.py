"""Pro-atomic density sum (PADS): ρ_PADS(r) = Σ atoms ρ_atom(r − R_atom).

PADS is the superposition of isolated-atom electron densities placed at
the nuclear positions of a structure. It's a cheap, physically-motivated
approximation of the total electron density; the *deformation density*
Δρ = ρ − ρ_PADS captures only the redistribution that happens upon
bonding (the chemically interesting part).

This module currently implements only a **crude Gaussian approximation**:

* Real atomic densities are roughly exponential with cusps at the
  nucleus and a long tail; the 1s core has significant high-\|G\|
  spectral content.
* A Gaussian has none of those features. It's smooth everywhere and
  has strictly Gaussian-decaying \|G\| content.

So Gaussian PADS *won't* remove the high-frequency core contributions
that drive the oxide-Fourier gap — the stated motivation for pursuing
Δρ in the first place. It will remove the *low-frequency* "atoms are
here, with roughly this much charge" component, which is also useful
but less targeted.

Treat this as a pipeline smoke test. Replace with Clementi-Raimondi
Slater densities or pyscf-computed neutral-atom RHF/PBE densities for
quantitative claims.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar

BOHR_ANGSTROM = 0.5291772109


def _minimum_image_r(grid_frac: np.ndarray, frac_pos: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Minimum-image Cartesian distance (Å) from each grid point to a fractional position."""
    disp_frac = grid_frac - frac_pos
    disp_frac -= np.round(disp_frac)
    disp_cart = disp_frac @ lattice
    return np.sqrt((disp_cart * disp_cart).sum(axis=-1))


def _grid_frac(grid_shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(0, 1, n, endpoint=False) for n in grid_shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


@dataclass
class GaussianPADS:
    """One isotropic Gaussian per atom. Smooth — won't remove core-cusp high-|G| content.

    Retained as a baseline for comparison; prefer :class:`SlaterPADS` for the
    actual Δρ hypothesis test.
    """

    sigma_angstrom: float = 0.4

    def compute(self, chgcar: "Chgcar") -> np.ndarray:
        grid_shape = chgcar.data["total"].shape
        lattice = np.asarray(chgcar.structure.lattice.matrix)
        grid_frac = _grid_frac(grid_shape)

        pads = np.zeros(grid_shape, dtype=np.float64)
        norm = 1.0 / (self.sigma_angstrom * np.sqrt(2 * np.pi)) ** 3
        two_sigma_sq = 2 * self.sigma_angstrom**2

        for site in chgcar.structure:
            z = site.specie.Z
            r = _minimum_image_r(grid_frac, np.asarray(site.frac_coords), lattice)
            pads += z * norm * np.exp(-(r**2) / two_sigma_sq)

        return pads


@dataclass
class SlaterPADS:
    """Two-component atomic density: Slater-1s core (2 electrons, cusp) + Gaussian valence.

    For each atom with nuclear charge Z:

    * **core**: up to 2 electrons in a Slater-1s exponential
      ``2 α³ / (8π) · exp(−α r)`` with ``α = 2 Z_eff / a0`` and
      ``Z_eff = Z − 0.30`` (Slater's rule for 1s). Hydrogen uses
      ``Z_eff = 1`` and only 1 electron.
    * **valence**: remaining ``Z − 2`` electrons in an isotropic
      Gaussian with width ``valence_sigma_angstrom`` (default 0.5 Å).
      Not physical, but gives a smooth outer shell that roughly
      matches typical covalent radii.

    The two-component form is deliberate: the core has the exponential
    high-|G| content needed to cancel ρ's nuclear cusps (the motivation
    for Δρ); the valence mass goes somewhere smooth where Fourier
    handles it easily.

    Still **crude** compared to proper multi-shell Slater or
    pyscf-computed densities — no angular structure, Z_eff from simple
    Slater rules rather than Clementi-Raimondi fits, single Gaussian
    for all valence electrons regardless of element. But should suffice
    to test the "Δρ helps on oxides" hypothesis directionally.
    """

    valence_sigma_angstrom: float = 0.5
    bohr_angstrom: float = BOHR_ANGSTROM

    def core_alpha(self, z: int) -> float:
        """1s Slater's-rule α in 1/Å."""
        z_eff = 1.0 if z == 1 else z - 0.30
        return 2.0 * z_eff / self.bohr_angstrom

    def compute(self, chgcar: "Chgcar") -> np.ndarray:
        grid_shape = chgcar.data["total"].shape
        lattice = np.asarray(chgcar.structure.lattice.matrix)
        grid_frac = _grid_frac(grid_shape)

        val_norm = 1.0 / (self.valence_sigma_angstrom * np.sqrt(2 * np.pi)) ** 3
        two_sigma_sq = 2 * self.valence_sigma_angstrom**2

        pads = np.zeros(grid_shape, dtype=np.float64)
        for site in chgcar.structure:
            z = site.specie.Z
            r = _minimum_image_r(grid_frac, np.asarray(site.frac_coords), lattice)

            n_core_electrons = min(z, 2)
            a = self.core_alpha(z)
            pads += n_core_electrons * a**3 / (8 * np.pi) * np.exp(-a * r)

            n_valence_electrons = max(0, z - 2)
            if n_valence_electrons:
                pads += n_valence_electrons * val_norm * np.exp(-(r**2) / two_sigma_sq)

        return pads
