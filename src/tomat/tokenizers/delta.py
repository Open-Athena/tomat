"""Scheme 4: deformation density tokenization.

Wraps a base tokenizer and applies it to Δρ = ρ − ρ_PADS instead of ρ
directly, adding ρ_PADS back on decode. The base tokenizer's
reconstruction error on Δρ becomes the total reconstruction error on ρ
(PADS cancels exactly).

The motivation (per the design doc): Δρ has smaller dynamic range, no
nuclear cusps, centered on zero, and concentrates chemically
interesting signal (bonds, charge transfer). Expected win depends on
how well the PADS captures the core-electron contribution the base
tokenizer would otherwise fail to represent; see ``tomat.pads``.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from tomat.pads import GaussianPADS, MultiShellSlaterPADS, SlaterPADS
from tomat.tokenizers.base import DensityTokenizer

PADSImpl = GaussianPADS | SlaterPADS | MultiShellSlaterPADS

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class DeltaEncoded:
    base_encoded: Any
    pads: np.ndarray


class DeltaDensityTokenizer(DensityTokenizer):
    """Wrap a base tokenizer to operate on Δρ = ρ − ρ_PADS.

    NMAE is measured against ρ (not Δρ), so what the base tokenizer
    loses in relative Δρ terms gets scaled by ``sum|Δρ| / sum|ρ|`` in
    ρ units — typically a factor of 0.05–0.1 for well-formed PADS.
    """

    def __init__(self, base: DensityTokenizer, pads: PADSImpl | None = None):
        self.base = base
        self.pads = pads or MultiShellSlaterPADS()
        self.name = f"delta-{base.name}"

    def encode(self, chgcar: "Chgcar") -> DeltaEncoded:
        rho = np.asarray(chgcar.data["total"], dtype=np.float64)
        # chgcar.data["total"] stores ρ × V_cell (VASP CHGCAR convention); our
        # PADS returns e/Å³. Multiply by V_cell so units match for subtraction.
        pads = self.pads.compute(chgcar) * float(chgcar.structure.volume)
        delta = rho - pads
        fake = SimpleNamespace(data={"total": delta}, structure=chgcar.structure)
        return DeltaEncoded(base_encoded=self.base.encode(fake), pads=pads)

    def decode(self, encoded: DeltaEncoded) -> np.ndarray:
        delta_recon = self.base.decode(encoded.base_encoded)
        return encoded.pads + delta_recon
