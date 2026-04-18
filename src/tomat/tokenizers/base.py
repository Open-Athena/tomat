"""Density tokenizer ABC.

A tokenizer is a lossy (or lossless) compression of a 3D electron-density
field into a discrete representation that an LLM can be trained to predict.
For the first fidelity sweep we only need an `encode` / `decode` round trip —
the integer-token vocabulary mapping is a later concern.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


class DensityTokenizer(ABC):
    """Encode a CHGCAR into a scheme-specific representation and decode it back.

    Subclasses pick what to store; the common contract is:

        roundtrip(chgcar) ≈ chgcar.data["total"]

    up to the scheme's reconstruction error.
    """

    name: ClassVar[str]

    @abstractmethod
    def encode(self, chgcar: "Chgcar") -> Any:
        """Return a scheme-specific encoded representation."""

    @abstractmethod
    def decode(self, encoded: Any) -> np.ndarray:
        """Reconstruct a density grid matching the original ``chgcar.data['total']`` shape."""

    def token_count(self, encoded: Any) -> int:
        """Tokens needed to transmit ``encoded`` under FP16 codec fidelity.

        Default is 0 (scheme hasn't reported one). Subclasses override.
        See :mod:`tomat.token_count` for the per-component constants.
        """
        return 0

    def roundtrip(self, chgcar: "Chgcar") -> np.ndarray:
        return self.decode(self.encode(chgcar))
