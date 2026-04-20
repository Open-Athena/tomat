"""Average-pool downsampling composed with an existing tokenizer.

Partitions the CHGCAR grid into ``factor × factor × factor`` voxel blocks,
averages each block to a single scalar, and tokenizes the resulting coarser
grid with any wrapped :class:`DensityTokenizer`. On decode, the coarse grid
is expanded back by repeating each coarse voxel ``factor³`` times.

A 128³ → 32³ avg-pool reduces voxel count ``8³ = 512×``; coupled with the
one-token FP16 codec that's ~2 k tokens per structure — trainable at 30M-scale
within 4–16k context windows.

Reconstruction NMAE floor is set by the averaging (high-frequency spatial
detail is destroyed and cannot be recovered) plus the wrapped tokenizer's
own floor. Useful for a "hello-world" training run where capacity and
context budget both need to be small; **not** a path to matching electrAI.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from tomat.tokenizers.base import DensityTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


def avg_pool_3d(grid: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool a ``(N, N, N)`` array by ``factor`` along each axis.

    Requires ``N % factor == 0``. Preserves integrated density (the mean
    of each block, summed, equals the original sum divided by ``factor³``
    — so the *sum* is preserved after multiplying back by ``factor³`` at
    decode, which is what :class:`DownsampledTokenizer.decode` does).
    """
    nx, ny, nz = grid.shape
    if nx % factor or ny % factor or nz % factor:
        raise ValueError(f"grid shape {grid.shape} not divisible by factor={factor}")
    return grid.reshape(
        nx // factor, factor, ny // factor, factor, nz // factor, factor,
    ).mean(axis=(1, 3, 5))


def upsample_repeat_3d(grid: np.ndarray, factor: int) -> np.ndarray:
    """Inverse of :func:`avg_pool_3d` assuming within-block values were uniform.

    Each coarse voxel is repeated ``factor³`` times. Reconstruction is exact
    only for grids that were uniform within each pooled block; real data
    loses within-block structure."""
    return np.repeat(np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1), factor, axis=2)


@dataclass
class DownsampledEncoded:
    base_encoded: Any
    factor: int
    original_shape: tuple[int, int, int]


@dataclass
class DownsampledTokenizer(DensityTokenizer):
    """Wrap a base tokenizer to operate on an ``factor×``-downsampled grid.

    At encode time the CHGCAR is replaced with a coarsened one (same
    lattice, same structure) and handed to the base tokenizer. At decode,
    the coarse grid is upsampled by value-repeat back to the original
    shape.
    """

    base: DensityTokenizer
    factor: int = 4
    name: str = ""

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ValueError(f"factor must be ≥ 1, got {self.factor}")
        if not self.name:
            self.name = f"down{self.factor}-{self.base.name}"

    def encode(self, chgcar: "Chgcar") -> DownsampledEncoded:
        rho = np.asarray(chgcar.data["total"], dtype=np.float64)
        coarse = avg_pool_3d(rho, self.factor) if self.factor > 1 else rho
        # Forward the structure iff the wrapped CHGCAR carries one — Δρ
        # needs it, direct/cutoff/fourier don't, and the test stubs for
        # the latter omit it.
        attrs: dict[str, object] = {"data": {"total": coarse}}
        structure = getattr(chgcar, "structure", None)
        if structure is not None:
            attrs["structure"] = structure
        fake = SimpleNamespace(**attrs)
        return DownsampledEncoded(
            base_encoded=self.base.encode(fake),
            factor=self.factor,
            original_shape=rho.shape,
        )

    def decode(self, encoded: DownsampledEncoded) -> np.ndarray:
        coarse_recon = self.base.decode(encoded.base_encoded)
        if encoded.factor == 1:
            return coarse_recon
        return upsample_repeat_3d(coarse_recon, encoded.factor)

    def token_count(self, encoded: DownsampledEncoded) -> int:
        return self.base.token_count(encoded.base_encoded)
