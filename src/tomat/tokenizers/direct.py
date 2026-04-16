"""Scheme 1: direct CHGCAR serialization.

This first cut is a lossless round trip — float32 copy of the density grid.
The point is to exercise the encode/decode pipeline and establish a zero-loss
baseline so that reconstruction error reported for other schemes is clearly
attributable to the compression step, not the plumbing.

A future revision will add a `FloatCodec` (e.g. tomol-style
signed-exponent + mantissa-digit quantization) layered on top; the resulting
NMAE floor is what "scheme 1 at production fidelity" really costs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomat.tokenizers.base import DensityTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class DirectEncoded:
    density: np.ndarray


class DirectTokenizer(DensityTokenizer):
    name = "direct"

    def encode(self, chgcar: "Chgcar") -> DirectEncoded:
        return DirectEncoded(density=np.asarray(chgcar.data["total"], dtype=np.float32))

    def decode(self, encoded: DirectEncoded) -> np.ndarray:
        return encoded.density.astype(np.float64)
