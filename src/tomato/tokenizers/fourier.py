"""Scheme 5: Fourier coefficients (reciprocal space).

Decompose ρ into plane waves via FFT, keep the ``n_coefficients`` lowest-|G|
coefficients, zero-fill the rest and invert. Since ρ is real, the FFT is
conjugate-symmetric; we use ``np.fft.rfftn`` to avoid storing redundant
coefficients.

This scheme is directly analogous to how VASP stores charge density
internally and has a natural ordering by spatial frequency.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomato.tokenizers.base import DensityTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class FourierEncoded:
    grid_shape: tuple[int, int, int]
    flat_indices: np.ndarray
    coefficients: np.ndarray


class FourierTokenizer(DensityTokenizer):
    """Real-to-complex FFT with |G|-truncation."""

    name = "fourier"

    def __init__(
        self,
        *,
        n_coefficients: int | None = None,
        coefficient_fraction: float | None = None,
    ):
        if (n_coefficients is None) == (coefficient_fraction is None):
            raise ValueError("Pass exactly one of n_coefficients, coefficient_fraction")
        if n_coefficients is not None and n_coefficients <= 0:
            raise ValueError("n_coefficients must be positive")
        if coefficient_fraction is not None and not 0 < coefficient_fraction <= 1:
            raise ValueError("coefficient_fraction must be in (0, 1]")
        self.n_coefficients = n_coefficients
        self.coefficient_fraction = coefficient_fraction

    def encode(self, chgcar: "Chgcar") -> FourierEncoded:
        density = np.asarray(chgcar.data["total"], dtype=np.float64)
        coefs = np.fft.rfftn(density)
        g_squared = self._g_squared_grid(density.shape, coefs.shape)
        flat_g2 = g_squared.ravel()
        target = (
            self.n_coefficients
            if self.n_coefficients is not None
            else max(1, int(round(flat_g2.size * self.coefficient_fraction)))
        )
        k = min(target, flat_g2.size)
        idx = np.argpartition(flat_g2, k - 1)[:k]
        return FourierEncoded(
            grid_shape=density.shape,
            flat_indices=idx.astype(np.int64),
            coefficients=coefs.ravel()[idx].astype(np.complex64),
        )

    def decode(self, encoded: FourierEncoded) -> np.ndarray:
        rfft_shape = (*encoded.grid_shape[:-1], encoded.grid_shape[-1] // 2 + 1)
        flat = np.zeros(int(np.prod(rfft_shape)), dtype=np.complex128)
        flat[encoded.flat_indices] = encoded.coefficients
        axes = tuple(range(len(encoded.grid_shape)))
        return np.fft.irfftn(flat.reshape(rfft_shape), s=encoded.grid_shape, axes=axes)

    @staticmethod
    def _g_squared_grid(grid_shape: tuple[int, ...], rfft_shape: tuple[int, ...]) -> np.ndarray:
        """Integer |G|² for each point of the rfftn output (fractional reciprocal coords)."""
        nx, ny, nz = grid_shape
        fx = np.fft.fftfreq(nx) * nx
        fy = np.fft.fftfreq(ny) * ny
        fz = np.arange(rfft_shape[-1])
        gx, gy, gz = np.meshgrid(fx, fy, fz, indexing="ij")
        return gx**2 + gy**2 + gz**2
