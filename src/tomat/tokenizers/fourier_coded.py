"""Scheme 5 at production fidelity: Fourier with FP16 codec on each kept coefficient.

Each kept complex coefficient decomposes into ``(real, imag)``; each real
value routes through :class:`~tomat.float_codec.FP16Codec` (signed, 3
tokens). Total per kept coef = 6 tokens + 3 index tokens.

Real and imaginary parts have different scales — imag ~0 at the DC bin,
both parts grow with |G| for high-entropy content — so a single
``(log_min, log_max)`` pair fitted from a real-space density scan isn't
a perfect match. A more careful design would fit separate configs for
``real`` / ``imag`` over a training-set coefficient sample. For now we
reuse the density-space config; codec contribution is expected to be
~1e-6 on the density side regardless, dominated by truncation at any
realistic retention.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomat.float_codec import FP16Codec
from tomat.token_count import fourier_tokens
from tomat.tokenizers.base import DensityTokenizer
from tomat.tokenizers.fourier import FourierTokenizer

if TYPE_CHECKING:
    from pymatgen.io.vasp.outputs import Chgcar


@dataclass
class FourierCodedEncoded:
    grid_shape: tuple[int, int, int]
    flat_indices: np.ndarray
    real_components: np.ndarray  # (k, 3) int32
    imag_components: np.ndarray  # (k, 3) int32


@dataclass
class FourierCodedTokenizer(DensityTokenizer):
    """Wrap :class:`FourierTokenizer` so that each kept complex coefficient
    is encoded via the signed FP16 codec on its real and imaginary parts."""

    base: FourierTokenizer
    codec: FP16Codec
    name: str = "fourier-coded"

    def encode(self, chgcar: "Chgcar") -> FourierCodedEncoded:
        raw = self.base.encode(chgcar)
        coefs = np.asarray(raw.coefficients, dtype=np.complex128)
        return FourierCodedEncoded(
            grid_shape=raw.grid_shape,
            flat_indices=raw.flat_indices,
            real_components=self.codec.encode_signed(coefs.real),
            imag_components=self.codec.encode_signed(coefs.imag),
        )

    def decode(self, encoded: FourierCodedEncoded) -> np.ndarray:
        real_recon = self.codec.decode_signed(encoded.real_components)
        imag_recon = self.codec.decode_signed(encoded.imag_components)
        coefs = real_recon + 1j * imag_recon
        rfft_shape = (*encoded.grid_shape[:-1], encoded.grid_shape[-1] // 2 + 1)
        flat = np.zeros(int(np.prod(rfft_shape)), dtype=np.complex128)
        flat[encoded.flat_indices] = coefs
        axes = tuple(range(len(encoded.grid_shape)))
        return np.fft.irfftn(flat.reshape(rfft_shape), s=encoded.grid_shape, axes=axes)

    def token_count(self, encoded: FourierCodedEncoded) -> int:
        return fourier_tokens(
            int(encoded.flat_indices.size),
            tokens_per_real=self.codec.tokens_per_value_signed,
        )
