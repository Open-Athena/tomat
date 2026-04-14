"""Scheme 3: high-density voxels with cutoff (Yael).

Keep the CHGCAR header (lattice, atoms, grid shape) but replace the voxel
grid with an ordered list of ``(flat_index, density)`` pairs — the top-K
voxels by absolute density, with the rest implicitly zero on decode.

Two selection modes:

* ``top_k``: keep exactly K voxels (sequence length is deterministic per
  structure regardless of density distribution).
* ``threshold``: keep voxels with density above a fixed threshold in e/Å³
  (sequence length varies with sparsity).

For the fidelity sweep we care about the reconstruction NMAE as a function
of the retained fraction. Sparsity statistics in the design doc suggest
intermetallics retain essentially all voxels above 0.05 e/Å³ whereas
halides retain well under half — so both modes are worth exercising.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomato.tokenizers.base import DensityTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class CutoffEncoded:
    grid_shape: tuple[int, int, int]
    flat_indices: np.ndarray
    values: np.ndarray
    mass_captured: float
    effective_threshold: float


class CutoffTokenizer(DensityTokenizer):
    """Keep the top-K voxels (or all voxels above ``threshold``)."""

    name = "cutoff"

    def __init__(
        self,
        *,
        top_k: int | None = None,
        top_fraction: float | None = None,
        threshold: float | None = None,
    ):
        if sum(x is not None for x in (top_k, top_fraction, threshold)) != 1:
            raise ValueError("Pass exactly one of top_k, top_fraction, threshold")
        if top_fraction is not None and not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        self.top_k = top_k
        self.top_fraction = top_fraction
        self.threshold = threshold

    def encode(self, chgcar: "Chgcar") -> CutoffEncoded:
        density = np.asarray(chgcar.data["total"], dtype=np.float32)
        flat = density.ravel()
        if self.top_k is not None:
            k = min(self.top_k, flat.size)
            idx = np.argpartition(flat, -k)[-k:]
        elif self.top_fraction is not None:
            k = max(1, min(flat.size, int(round(flat.size * self.top_fraction))))
            idx = np.argpartition(flat, -k)[-k:]
        else:
            idx = np.flatnonzero(flat >= self.threshold)
        values = flat[idx]
        total_mass = float(np.abs(flat).sum())
        return CutoffEncoded(
            grid_shape=density.shape,
            flat_indices=idx.astype(np.int64),
            values=values,
            mass_captured=float(np.abs(values).sum() / total_mass) if total_mass else 0.0,
            effective_threshold=float(values.min()) if values.size else float("nan"),
        )

    def decode(self, encoded: CutoffEncoded) -> np.ndarray:
        out = np.zeros(int(np.prod(encoded.grid_shape)), dtype=np.float64)
        out[encoded.flat_indices] = encoded.values
        return out.reshape(encoded.grid_shape)
