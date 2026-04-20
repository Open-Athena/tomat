"""Scheme 4: deformation density tokenization.

Wraps a base tokenizer and applies it to Δρ = ρ − ρ_promolecule instead
of ρ directly, adding ρ_promolecule back on decode. The base tokenizer's
reconstruction error on Δρ becomes the total reconstruction error on ρ
(the promolecule cancels exactly).

The motivation (per the design doc): Δρ has smaller dynamic range, no
nuclear cusps, centered on zero, and concentrates chemically interesting
signal (bonds, charge transfer). Expected win depends on how well the
promolecule captures the core-electron contribution the base tokenizer
would otherwise fail to represent; see ``tomat.promolecule``.

(*Not* to be confused with OA's PADS — *Pre-tabulated Atomic Density
Superposition* — which is a VASP-derived tabulated density used by
RHOAR-Net to generate low-resolution inputs, not a subtraction.)
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from tomat.promolecule import GaussianPromolecule, MultiShellSlaterPromolecule, SlaterPromolecule
from tomat.token_count import delta_overhead
from tomat.tokenizers.base import DensityTokenizer

PromoleculeImpl = GaussianPromolecule | SlaterPromolecule | MultiShellSlaterPromolecule

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class DeltaEncoded:
    base_encoded: Any
    promolecule: np.ndarray
    n_atoms: int


class DeltaDensityTokenizer(DensityTokenizer):
    """Wrap a base tokenizer to operate on Δρ = ρ − ρ_promolecule.

    NMAE is measured against ρ (not Δρ), so what the base tokenizer
    loses in relative Δρ terms gets scaled by ``sum|Δρ| / sum|ρ|`` in
    ρ units — typically a factor of 0.05–0.1 for a well-formed
    promolecule density.
    """

    def __init__(self, base: DensityTokenizer, promolecule: PromoleculeImpl | None = None):
        self.base = base
        self.promolecule = promolecule or MultiShellSlaterPromolecule()
        self.name = f"delta-{base.name}"

    def encode(self, chgcar: "Chgcar") -> DeltaEncoded:
        rho = np.asarray(chgcar.data["total"], dtype=np.float64)
        # chgcar.data["total"] stores ρ × V_cell (VASP CHGCAR convention); our
        # promolecule returns e/Å³. Multiply by V_cell so units match.
        pro = self.promolecule.compute(chgcar) * float(chgcar.structure.volume)
        delta = rho - pro
        fake = SimpleNamespace(data={"total": delta}, structure=chgcar.structure)
        return DeltaEncoded(
            base_encoded=self.base.encode(fake),
            promolecule=pro,
            n_atoms=len(chgcar.structure),
        )

    def decode(self, encoded: DeltaEncoded) -> np.ndarray:
        delta_recon = self.base.decode(encoded.base_encoded)
        return encoded.promolecule + delta_recon

    def token_count(self, encoded: DeltaEncoded) -> int:
        return self.base.token_count(encoded.base_encoded) + delta_overhead(encoded.n_atoms)
