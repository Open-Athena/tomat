"""Scheme 1 at production fidelity: direct float-per-voxel, routed through the FP16 codec.

Each voxel is encoded as 3 bytes (SE, M0, M1) via :class:`~tomat.float_codec.FP16Codec`
— a log-uniform 24-bit quantiser fit to per-channel ``(log_min, log_max)``.
The reconstruction error of the round-trip is the scheme-1 *token*
floor (as opposed to :class:`~tomat.tokenizers.direct.DirectTokenizer`'s
float32 memory-copy baseline, which is effectively lossless).

At 128³ voxels, one CHGCAR is ≈ 2.1 M values; at 3 tokens per value that's
~6.3 M tokens — well beyond any context length we're modelling today.
The point here isn't to fit it into a model; it's to fix the "what does
production-fidelity scheme 1 actually cost in reconstruction error"
blank on the sweep.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomat.float_codec import FP16Codec
from tomat.tokenizers.base import DensityTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class DirectCodedEncoded:
    components: np.ndarray  # (N_voxels, 3) int32, columns [SE, M0, M1]
    shape: tuple[int, int, int]


@dataclass
class DirectCodedTokenizer(DensityTokenizer):
    """Route each voxel through :class:`FP16Codec` (signed, 24-bit)."""

    codec: FP16Codec
    name: str = "direct-coded"

    def encode(self, chgcar: "Chgcar") -> DirectCodedEncoded:
        density = np.asarray(chgcar.data["total"], dtype=np.float64)
        components = self.codec.encode_signed(density.ravel())
        return DirectCodedEncoded(components=components, shape=density.shape)

    def decode(self, encoded: DirectCodedEncoded) -> np.ndarray:
        flat = self.codec.decode_signed(encoded.components)
        return flat.reshape(encoded.shape)
