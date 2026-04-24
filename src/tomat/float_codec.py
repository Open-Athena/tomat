"""FP16-like log-uniform float codec, parameterised by per-token bit widths.

Each value is encoded by taking ``log10(|v|)``, clipping to a channel-specific
``[log_min, log_max]``, normalising to ``[0, 1]``, and quantising to a
fixed-precision integer bin index. The bin index is then split across a
configurable number of tokens, optionally carrying a sign bit in the first
token.

**Signed variants** (the common case — density can be negative near
FFT-artifact regions, Fourier real/imag are signed):

========================  ==========  ==========  ===============
name                       layout       vocabs      precision bits
========================  ==========  ==========  ===============
``tomol_3byte``            (8, 8, 8)   512+256+256   24 (1 sign + 23 mag)
``two_token_9_12``         (8, 12)     512+4096      21 (1 sign + 20 mag)
``fp16_1token``            (15,)       65536         16 (1 sign + 15 mag)
========================  ==========  ==========  ===============

Built via the matching class method, e.g.::

    codec = FP16Codec.tomol_3byte(log_min=-4.13, log_max=4.97)

The first token always packs the sign bit — so its vocabulary is
``2 * (1 << bits[0])``. Remaining tokens encode straight magnitude bits,
vocab ``1 << bits[i]``.

Zero (or any ``|v| < 1e-15``) round-trips exactly via a ``bin_index == 0``
shortcut. Values outside ``[10^log_min, 10^log_max]`` are clipped to the
endpoints.

This module is deliberately *offset-free*: it returns component indices in
``[0, vocab_i)`` rather than IDs within a flat LLM vocabulary. Mapping to
flat token IDs is a downstream concern.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_MAGNITUDE_FLOOR = 1e-15  # anything below this encodes as bin_index 0 → decodes as 0.0


@dataclass
class FP16Codec:
    """Log-uniform quantizer.

    ``token_mag_bits`` lists magnitude-bit widths per token, MSB-first. The
    sign bit is always packed into the first signed token (so the first
    token's vocab is ``2 * 2^token_mag_bits[0]``; the rest are just
    ``2^token_mag_bits[i]``). Total signed precision = 1 + sum(mag bits).

    ``token_mag_bits_unsigned`` is the analogous list for the always-positive
    form (e.g. |Fourier coefficient|). No sign bit; vocabs are just
    ``2^bits[i]`` per token.
    """

    log_min: float
    log_max: float
    token_mag_bits: tuple[int, ...] = (8, 8, 8)
    token_mag_bits_unsigned: tuple[int, ...] = (8, 8, 8, 8)

    def __post_init__(self) -> None:
        if not self.token_mag_bits:
            raise ValueError("token_mag_bits must be non-empty")
        if not self.token_mag_bits_unsigned:
            raise ValueError("token_mag_bits_unsigned must be non-empty")

    # ---- named builders ---------------------------------------------------

    @classmethod
    def tomol_3byte(cls, log_min: float, log_max: float) -> "FP16Codec":
        """Tomol's 3-token layout: SE(9) + M0(8) + M1(8). 24-bit precision."""
        return cls(log_min=log_min, log_max=log_max, token_mag_bits=(8, 8, 8))

    @classmethod
    def two_token_9_12(cls, log_min: float, log_max: float) -> "FP16Codec":
        """2-token hierarchy: SE(9 bits = 1 sign + 8 mag, 512 vocab) + M(12 bits, 4096 vocab).

        Preserves coarse-to-fine SE-then-M structure, 33% shorter than the
        3-byte layout. Vocab is small (~4.6k) so fits 30M-param models.
        """
        return cls(log_min=log_min, log_max=log_max, token_mag_bits=(8, 12))

    @classmethod
    def fp16_1token(cls, log_min: float, log_max: float) -> "FP16Codec":
        """Single token per value: 1 sign + 15 magnitude bits → 65 536 vocab.

        Flat layout (no hierarchy). 3× shorter than tomol's 3-byte; vocab is
        in the modern-LLM normal zone (Llama 3 ≈ 128k) but a ~15 % embedding
        tax on a 1B-parameter model. For 30M-scale, prefer
        :meth:`tomol_3byte` or :meth:`two_token_9_12`.
        """
        return cls(log_min=log_min, log_max=log_max, token_mag_bits=(15,))

    # ---- derived ---------------------------------------------------------

    @property
    def tokens_per_value_signed(self) -> int:
        return len(self.token_mag_bits)

    @property
    def tokens_per_value_unsigned(self) -> int:
        return len(self.token_mag_bits_unsigned)

    @property
    def total_mag_bits_signed(self) -> int:
        return sum(self.token_mag_bits)

    @property
    def total_mag_bits_unsigned(self) -> int:
        return sum(self.token_mag_bits_unsigned)

    @property
    def signed_vocabs(self) -> tuple[int, ...]:
        """Vocabulary size per token (signed form). First token includes sign bit."""
        return tuple(
            (2 if i == 0 else 1) << b for i, b in enumerate(self.token_mag_bits)
        )

    @property
    def unsigned_vocabs(self) -> tuple[int, ...]:
        return tuple(1 << b for b in self.token_mag_bits_unsigned)

    # ---- signed ----------------------------------------------------------

    def encode_signed(self, values: np.ndarray) -> np.ndarray:
        """Encode ``values`` into ``(N, tokens_per_value_signed)`` component indices."""
        values = np.asarray(values, dtype=np.float64)
        signs_positive = values >= 0
        bin_indices = self._log_bin_index(values, self.total_mag_bits_signed)

        # Split bin_indices into chunks, MSB-first. First chunk carries sign.
        out = np.empty((bin_indices.size, self.tokens_per_value_signed), dtype=np.int32)
        shift = self.total_mag_bits_signed
        first_half = 1 << self.token_mag_bits[0]
        for i, b in enumerate(self.token_mag_bits):
            shift -= b
            chunk = ((bin_indices >> shift) & ((1 << b) - 1)).astype(np.int32)
            if i == 0:
                out[:, 0] = np.where(signs_positive, chunk, first_half + chunk)
            else:
                out[:, i] = chunk
        return out

    def decode_signed(self, components: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`encode_signed`."""
        components = np.asarray(components, dtype=np.int64)
        first_half = 1 << self.token_mag_bits[0]
        sign_positive = components[:, 0] < first_half
        first_chunk = np.where(sign_positive, components[:, 0], components[:, 0] - first_half)

        bin_index = first_chunk.copy()
        for i in range(1, self.tokens_per_value_signed):
            bin_index = (bin_index << self.token_mag_bits[i]) | components[:, i]

        magnitude = self._decode_bin_index(bin_index, self.total_mag_bits_signed)
        return np.where(sign_positive, magnitude, -magnitude)

    # ---- unsigned --------------------------------------------------------

    def encode_unsigned(self, values: np.ndarray) -> np.ndarray:
        """Encode non-negative ``values`` into ``(N, tokens_per_value_unsigned)`` components."""
        values = np.asarray(values, dtype=np.float64)
        bin_indices = self._log_bin_index(values, self.total_mag_bits_unsigned)

        out = np.empty((bin_indices.size, self.tokens_per_value_unsigned), dtype=np.int32)
        shift = self.total_mag_bits_unsigned
        for i, b in enumerate(self.token_mag_bits_unsigned):
            shift -= b
            chunk = ((bin_indices >> shift) & ((1 << b) - 1)).astype(np.int32)
            out[:, i] = chunk
        return out

    def decode_unsigned(self, components: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`encode_unsigned`."""
        components = np.asarray(components, dtype=np.int64)
        bin_index = components[:, 0].copy()
        for i in range(1, self.tokens_per_value_unsigned):
            bin_index = (bin_index << self.token_mag_bits_unsigned[i]) | components[:, i]
        return self._decode_bin_index(bin_index, self.total_mag_bits_unsigned)

    # ---- internals -------------------------------------------------------

    def _log_bin_index(self, values: np.ndarray, mag_bits: int) -> np.ndarray:
        magnitudes = np.abs(values)
        safe_mag = np.maximum(magnitudes, _MAGNITUDE_FLOOR)
        log_vals = np.log10(safe_mag)
        log_vals = np.clip(log_vals, self.log_min, self.log_max)

        normalized = (log_vals - self.log_min) / (self.log_max - self.log_min)
        total_bins = 1 << mag_bits
        bin_indices = (normalized * (total_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, total_bins - 1)
        # Anything below the floor forces bin_index 0 → decodes as exact 0.
        return np.where(magnitudes < _MAGNITUDE_FLOOR, 0, bin_indices)

    def _decode_bin_index(self, bin_index: np.ndarray, mag_bits: int) -> np.ndarray:
        total_bins = 1 << mag_bits
        normalized = bin_index / (total_bins - 1)
        log_val = normalized * (self.log_max - self.log_min) + self.log_min
        magnitude = 10 ** log_val
        return np.where(bin_index == 0, 0.0, magnitude)

    # ---- persistence -----------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, *, channel: str | None = None, **kwargs) -> "FP16Codec":
        """Load ``(log_min, log_max)``. If the JSON has a ``channels`` key,
        ``channel`` picks one; otherwise the top-level entry is used.

        Extra kwargs (e.g. ``token_mag_bits``) pass through to the constructor.
        """
        import json
        with open(path) as f:
            cfg = json.load(f)
        if "channels" in cfg:
            if channel is None:
                raise ValueError(
                    f"{path} has channels {list(cfg['channels'])!r}; pass channel=..."
                )
            cfg = cfg["channels"][channel]
        return cls(log_min=float(cfg["log_min"]), log_max=float(cfg["log_max"]), **kwargs)

    def to_json(self, path: str | Path) -> None:
        import json
        with open(path, "w") as f:
            json.dump({"log_min": self.log_min, "log_max": self.log_max}, f, indent=2)
            f.write("\n")


# ---- Lloyd-Max Quantizer (empirical, 1-token) ----------------------------


@dataclass
class LMQCodec:
    """Empirical Lloyd-Max quantizer — 1-token density codec for unsigned values.

    Fit via `scripts/fit_lmq_codec_modal.py` on an empirical PDF (all
    train-full voxel densities). See `specs/18-lmq-codec.md`.

    Compatible with PatchTokenizer's encode_signed/decode_signed/signed_vocabs
    interface so it can drop in at the density-codec slot. Though the name
    "signed" is a misnomer here (density is always ≥ 0), we keep the method
    names for compatibility.
    """

    boundaries: np.ndarray   # shape (n_bins - 1,), sorted ascending
    recon_points: np.ndarray  # shape (n_bins,)
    clip_max: float          # values > clip_max get the top bin
    codec_name: str = "lmq"

    def __post_init__(self) -> None:
        if self.boundaries.ndim != 1:
            raise ValueError("boundaries must be 1-D")
        if self.recon_points.ndim != 1:
            raise ValueError("recon_points must be 1-D")
        if len(self.recon_points) != len(self.boundaries) + 1:
            raise ValueError(
                f"len(recon_points)={len(self.recon_points)} != "
                f"len(boundaries) + 1 = {len(self.boundaries) + 1}"
            )

    @property
    def n_bins(self) -> int:
        return len(self.recon_points)

    # ---- sizing compatible with FP16Codec --------------------------------

    @property
    def tokens_per_value_signed(self) -> int:
        return 1

    @property
    def tokens_per_value_unsigned(self) -> int:
        return 1

    @property
    def signed_vocabs(self) -> tuple[int, ...]:
        return (self.n_bins,)

    @property
    def unsigned_vocabs(self) -> tuple[int, ...]:
        return (self.n_bins,)

    # Kept for shape-compat with FP16Codec callers that peek at mag_bits.
    @property
    def token_mag_bits(self) -> tuple[int, ...]:
        import math

        return (int(math.ceil(math.log2(self.n_bins))),)

    # Not meaningful for a learned quantizer, but callers (meta.json dump)
    # expect these fields. Store the min/max of the reconstruction range.
    @property
    def log_min(self) -> float:
        import math

        return math.log10(max(float(self.recon_points.min()), 1e-30))

    @property
    def log_max(self) -> float:
        import math

        return math.log10(max(float(self.clip_max), 1e-30))

    # ---- encode / decode -------------------------------------------------

    def encode_signed(self, values: np.ndarray) -> np.ndarray:
        """Map floats → bin indices via boundary search.

        Returns shape ``(N, 1)`` to mirror the FP16Codec's component layout.
        """
        values = np.asarray(values, dtype=np.float64)
        clipped = np.clip(values, 0.0, self.clip_max)
        bin_idx = np.searchsorted(self.boundaries, clipped, side="right")
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        return bin_idx.astype(np.int32).reshape(-1, 1)

    def decode_signed(self, components: np.ndarray) -> np.ndarray:
        """Map bin indices (shape (N, 1) or (N,)) → floats (shape (N,))."""
        components = np.asarray(components, dtype=np.int64)
        if components.ndim == 2:
            if components.shape[1] != 1:
                raise ValueError(f"expected (N, 1) components, got {components.shape}")
            indices = components[:, 0]
        else:
            indices = components
        indices = np.clip(indices, 0, self.n_bins - 1)
        return self.recon_points[indices]

    # Same thing for the "unsigned" branch — density is always unsigned, we
    # just alias to signed versions.
    def encode_unsigned(self, values: np.ndarray) -> np.ndarray:
        return self.encode_signed(values)

    def decode_unsigned(self, components: np.ndarray) -> np.ndarray:
        return self.decode_signed(components)

    # ---- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "LMQCodec":
        """Load from a saved .npz (produced by `fit_lmq_codec_modal.py`).

        Supports local paths and gs:// URLs (uses `fsspec` if available).
        """
        try:
            import fsspec  # type: ignore
            with fsspec.open(path, "rb") as f:
                data = np.load(f, allow_pickle=True)
                bounds = data["boundaries"]
                recon = data["recon_points"]
                clip_max = float(data["clip_max"])
        except Exception:
            data = np.load(path, allow_pickle=True)
            bounds = data["boundaries"]
            recon = data["recon_points"]
            clip_max = float(data["clip_max"])
        codec_name = "lmq"
        if path.endswith(".npz"):
            codec_name = Path(path).stem
        return cls(
            boundaries=bounds.astype(np.float32),
            recon_points=recon.astype(np.float32),
            clip_max=clip_max,
            codec_name=codec_name,
        )
