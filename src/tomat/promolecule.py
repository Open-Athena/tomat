"""Promolecule density: ρ_pro(r) = Σ atoms ρ_atom(r − R_atom).

The *promolecule* model (also: Independent Atom Model, IAM) replaces the
molecule's self-consistent density with a sum of isolated-atom densities
placed at the nuclear positions. It is a cheap, physically-motivated
approximation of the total electron density; the *deformation density*
Δρ = ρ − ρ_pro captures the redistribution upon bonding — the chemically
interesting part.

This module provides three analytic promolecule-density implementations
used by :class:`~tomat.tokenizers.delta.DeltaDensityTokenizer` to turn
scheme 4 (Δρ tokenization) into a concrete computation:

1. :class:`GaussianPromolecule` — one Gaussian per atom. Smooth; no core cusps.
2. :class:`SlaterPromolecule` — Slater-1s core (2 e⁻) + Gaussian valence.
   A two-shell toy model.
3. :class:`MultiShellSlaterPromolecule` — full multi-shell Slater-type
   orbitals per occupied shell, with Z_eff from Slater's rules. All-electron,
   no fitted parameters; works for any element.

For quantitative claims, the next step up is **Clementi-Raimondi 1963**
fitted ζ values (trivial table lookup for Z ≤ 36, more accurate than
Slater's rules) or **pyscf RHF/PBE** computed atomic densities.

**Not to be confused with OA's PADS.** OA's PADS (*Pre-tabulated Atomic
Density Superposition*) is a different thing: a VASP-derived tabulated
per-element radial density used by RHOAR-Net to generate *input* density
guesses without a VASP license at inference. PADS does not subtract from
ρ — it replaces VASP SAD as the low-resolution input to the super-
resolution model. The classes here build an analytic promolecule density
used as a **subtraction** against the target CHGCAR for Δρ tokenization.
See ``docs/discussion-notes.md`` for the longer note.

Note on VASP-PAW mismatch: VASP CHGCARs contain frozen-core + valence
densities. An all-electron promolecule (any of our three) roughly matches
this *in principle* since the PAW core is itself from an atomic
calculation, but any residual mismatch appears as near-nucleus noise in
Δρ. A POTCAR-derived pseudo-valence density would close the gap.
"""

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core.periodic_table import Element
    from pymatgen.io.vasp.outputs import Chgcar

BOHR_ANGSTROM = 0.5291772109
L_SYM_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3}
# Slater's effective principal quantum numbers (Slater 1930)
SLATER_N_STAR = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2, 7: 4.3}


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
class GaussianPromolecule:
    """One isotropic Gaussian per atom. Smooth — won't remove core-cusp high-|G| content.

    Retained as a baseline for comparison; prefer
    :class:`MultiShellSlaterPromolecule` for the actual Δρ hypothesis test.
    """

    sigma_angstrom: float = 0.4

    def compute(self, chgcar: "Chgcar") -> np.ndarray:
        grid_shape = chgcar.data["total"].shape
        lattice = np.asarray(chgcar.structure.lattice.matrix)
        grid_frac = _grid_frac(grid_shape)

        out = np.zeros(grid_shape, dtype=np.float64)
        norm = 1.0 / (self.sigma_angstrom * np.sqrt(2 * np.pi)) ** 3
        two_sigma_sq = 2 * self.sigma_angstrom**2

        for site in chgcar.structure:
            z = site.specie.Z
            r = _minimum_image_r(grid_frac, np.asarray(site.frac_coords), lattice)
            out += z * norm * np.exp(-(r**2) / two_sigma_sq)

        return out


@dataclass
class SlaterPromolecule:
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
    for all valence electrons regardless of element.
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

        out = np.zeros(grid_shape, dtype=np.float64)
        for site in chgcar.structure:
            z = site.specie.Z
            r = _minimum_image_r(grid_frac, np.asarray(site.frac_coords), lattice)

            n_core_electrons = min(z, 2)
            a = self.core_alpha(z)
            out += n_core_electrons * a**3 / (8 * np.pi) * np.exp(-a * r)

            n_valence_electrons = max(0, z - 2)
            if n_valence_electrons:
                out += n_valence_electrons * val_norm * np.exp(-(r**2) / two_sigma_sq)

        return out


def slater_zeff(z: int, target_n: int, target_l: int, config: list[tuple[int, str, int]]) -> float:
    """Effective nuclear charge Z_eff for a target electron in the (target_n, target_l) shell.

    Applies Slater's rules (Slater 1930) given an element's occupied-shell
    configuration ``config = [(n, 's'/'p'/..., n_electrons), ...]``.

    Groupings per Slater:

    * (1s) is its own group.
    * (ns, np) share a group for each n ≥ 2.
    * (nd), (nf) each stand alone.

    Screening contribution from electrons in each group:

    * Same group: 0.35 each (0.30 each if the target is 1s).
    * For ns/np targets, (n-1) shell contributes 0.85 each; deeper 1.00 each.
      Electrons in nd/nf of *the same n* contribute 1.00 each (per Slater).
    * For nd/nf targets, all electrons in any lower group contribute 1.00 each.
    * Higher shells contribute 0 (no screening).
    """
    target_is_sp = target_l in (0, 1)
    screen = 0.0
    for n, l_sym, ne in config:
        if n > target_n:
            continue
        l = L_SYM_TO_INT[l_sym]

        same_group = n == target_n and (
            (target_is_sp and l in (0, 1)) or l == target_l
        )
        if same_group:
            # Exclude the target electron itself from the screening count.
            effective = ne - 1 if (n == target_n and l == target_l) else ne
            screen += (0.30 if target_n == 1 else 0.35) * effective
            continue

        if target_is_sp:
            if n == target_n - 1:
                screen += 0.85 * ne
            elif n < target_n - 1:
                screen += 1.00 * ne
            elif n == target_n and l > 1:
                # nd/nf of same n screens ns/np fully.
                screen += 1.00 * ne
        else:  # nd / nf target
            if n < target_n or (n == target_n and l < target_l):
                screen += 1.00 * ne

    return z - screen


def _slater_shell_density(r: np.ndarray, n: int, alpha: float, n_electrons: int) -> np.ndarray:
    """Slater-type spherically-averaged density from ``n_electrons`` in the nl shell.

    ρ_{nl}(r) = n_electrons × (2α)^{2n+1} / ((2n)! × 4π) × r^{2n−2} × exp(−2α r)

    Uses integer n (not Slater's non-integer n*) for the r-exponent to keep
    the normalization clean (factorial instead of Γ). The exp decay constant
    uses α = Z_eff / (n* a₀), so heavier-shell orbitals extend further as
    intended.
    """
    prefactor = n_electrons * (2 * alpha) ** (2 * n + 1) / (factorial(2 * n) * 4 * np.pi)
    return prefactor * r ** (2 * n - 2) * np.exp(-2 * alpha * r)


@dataclass
class MultiShellSlaterPromolecule:
    """Sum of Slater-type radial densities over each atom's occupied shells.

    Pymatgen gives ``Element.full_electronic_structure`` as a list of
    ``(n, l, n_electrons)``. For each occupied shell we compute
    Z_eff via Slater's rules, set α = Z_eff / (n* a₀) with Slater's
    effective principal quantum number n*, and evaluate the Slater-type
    density ``(2α)^{2n+1} r^{2n-2} e^{-2αr} / ((2n)! · 4π)``.

    Accuracy: reproduces Clementi-Raimondi ζ values to within ~1% for
    first-row atoms (verified: O 1s α ≈ 7.70 vs CR's 7.66; O 2s α ≈
    2.275 vs CR's 2.246). Worse for d-block metals where Slater's rules
    break down; a Clementi-Raimondi table lookup is the obvious upgrade.

    Works for any element (full periodic table) since Slater's rules
    don't require a table.

    **valence_only**: when True (default), drops all but the outermost
    principal-shell electrons — approximates what VASP's pseudopotential
    CHGCARs contain (frozen-core, only valence is self-consistent). For
    row-2 atoms (e.g. oxygen) this keeps ``2s/2p``, drops ``1s``. Not a
    perfect match to any specific POTCAR — VASP sometimes includes
    semi-core electrons (e.g. Li_sv, Ga_d) — but captures the right
    electron count for most standard PPs and is vastly better than an
    all-electron form for subtracting from CHGCAR totals.

    When False, gives the true all-electron atomic density (matches
    tabulated HF references); suitable for Δρ against genuinely
    all-electron reference data.
    """

    valence_only: bool = True
    bohr_angstrom: float = BOHR_ANGSTROM

    def compute(self, chgcar: "Chgcar") -> np.ndarray:
        grid_shape = chgcar.data["total"].shape
        lattice = np.asarray(chgcar.structure.lattice.matrix)
        grid_frac = _grid_frac(grid_shape)

        out = np.zeros(grid_shape, dtype=np.float64)
        for site in chgcar.structure:
            element: Element = site.specie
            z = element.Z
            config = element.full_electronic_structure
            if self.valence_only:
                max_n = max(n for n, _, ne in config if ne > 0)
                config = [s for s in config if s[0] == max_n]
            r = _minimum_image_r(grid_frac, np.asarray(site.frac_coords), lattice)

            for n, l_sym, ne in config:
                if ne == 0:
                    continue
                l = L_SYM_TO_INT[l_sym]
                z_eff = slater_zeff(z, n, l, element.full_electronic_structure)
                n_star = SLATER_N_STAR.get(n, float(n))
                alpha = z_eff / (n_star * self.bohr_angstrom)  # 1/Å
                out += _slater_shell_density(r, n, alpha, ne)

        return out
