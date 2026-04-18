"""FP16-like log-uniform float codec, ported from tomol
(``Open-Athena/tomol:serialize_molecules.py``, lines 132-330).

Each value is encoded by taking ``log10(|v|)``, clipping to a
channel-specific ``[log_min, log_max]``, normalising to ``[0, 1]``, and
quantising to a 24-bit (signed) or 32-bit (unsigned) integer. The bin index
is then sliced into 8-bit components.

Signed form — 3 components per value:
* **SE** ∈ ``[0, 512)``: combined sign + exponent byte.
  ``[0, 256)`` ⇒ positive; ``[256, 512)`` ⇒ negative. Each covers 256
  exponent codes.
* **M0** ∈ ``[0, 256)``: high mantissa byte.
* **M1** ∈ ``[0, 256)``: low mantissa byte.

Unsigned form (always-positive, e.g. |Fourier coefficient|) — 4 components,
32-bit bin index, no sign bit: ``[Exp, M0, M1, M2]``, each in ``[0, 256)``.

Resolution: 24 bits over (log_max - log_min) decades. For density data
over ~8 decades that's ~6 decimal digits of relative precision per
encoded value. Zero (or any ``|v| < 1e-15``) round-trips exactly via a
``bin_index == 0`` shortcut.

The class is deliberately *offset-free* — unlike tomol's encoder, it
returns component indices in ``[0, 256/512)`` rather than token IDs
within a specific vocabulary. Mapping those components to flat token
IDs is a downstream concern.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MAGNITUDE_FLOOR = 1e-15  # anything below this encodes as bin_index 0 → decodes as 0.0


@dataclass
class FP16Codec:
    """Log-uniform quantizer parameterised by ``(log_min, log_max)`` in log10.

    Use :meth:`encode_signed` / :meth:`decode_signed` for values that can
    be negative; :meth:`encode_unsigned` / :meth:`decode_unsigned` for
    strictly non-negative values with full 32-bit precision.
    """

    log_min: float
    log_max: float

    # ---- signed (3 components per value, 24-bit precision) ---------------

    def encode_signed(self, values: np.ndarray) -> np.ndarray:
        """Encode ``values`` to component indices of shape ``(N, 3)``.

        Columns: ``[SE, M0, M1]`` with ``SE ∈ [0, 512)``, ``M0, M1 ∈ [0, 256)``.
        """
        values = np.asarray(values, dtype=np.float64)
        signs_positive = values >= 0
        magnitudes = np.abs(values)

        safe_mag = np.maximum(magnitudes, _MAGNITUDE_FLOOR)
        log_vals = np.log10(safe_mag)
        log_vals = np.clip(log_vals, self.log_min, self.log_max)

        normalized = (log_vals - self.log_min) / (self.log_max - self.log_min)

        total_bins = 256 * 65536  # 2^24 = 16 777 216 bins
        bin_indices = (normalized * (total_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, total_bins - 1)

        # Values below the floor should round-trip to exactly 0, which the
        # decoder achieves via bin_index == 0. Force that here.
        bin_indices = np.where(magnitudes < _MAGNITUDE_FLOOR, 0, bin_indices)

        exp = bin_indices // 65536
        mant = bin_indices % 65536
        m0 = mant // 256
        m1 = mant % 256

        se = np.where(signs_positive, exp, 256 + exp).astype(np.int32)
        return np.column_stack([se, m0.astype(np.int32), m1.astype(np.int32)])

    def decode_signed(self, components: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`encode_signed`. ``components`` is ``(N, 3)``."""
        components = np.asarray(components, dtype=np.int64)
        se = components[:, 0]
        m0 = components[:, 1]
        m1 = components[:, 2]

        sign_positive = se < 256
        exp = np.where(sign_positive, se, se - 256)
        mant = m0 * 256 + m1
        bin_index = exp * 65536 + mant

        total_bins = 256 * 65536
        normalized = bin_index / (total_bins - 1)
        log_val = normalized * (self.log_max - self.log_min) + self.log_min
        magnitude = 10 ** log_val
        magnitude = np.where(bin_index == 0, 0.0, magnitude)
        return np.where(sign_positive, magnitude, -magnitude)

    # ---- unsigned (4 components per value, 32-bit precision) -------------

    def encode_unsigned(self, values: np.ndarray) -> np.ndarray:
        """Encode non-negative ``values`` to ``(N, 4)`` components ``[E, M0, M1, M2]``."""
        values = np.asarray(values, dtype=np.float64)
        magnitudes = np.abs(values)

        safe_mag = np.maximum(magnitudes, _MAGNITUDE_FLOOR)
        log_vals = np.log10(safe_mag)
        log_vals = np.clip(log_vals, self.log_min, self.log_max)

        normalized = (log_vals - self.log_min) / (self.log_max - self.log_min)

        total_bins = 256 ** 4  # 2^32
        bin_indices = (normalized * (total_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, total_bins - 1)
        bin_indices = np.where(magnitudes < _MAGNITUDE_FLOOR, 0, bin_indices)

        exp = bin_indices // (256 ** 3)
        rest = bin_indices % (256 ** 3)
        m0 = rest // (256 ** 2)
        m1 = (rest // 256) % 256
        m2 = rest % 256
        return np.column_stack([
            exp.astype(np.int32),
            m0.astype(np.int32),
            m1.astype(np.int32),
            m2.astype(np.int32),
        ])

    def decode_unsigned(self, components: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`encode_unsigned`. ``components`` is ``(N, 4)``."""
        components = np.asarray(components, dtype=np.int64)
        exp, m0, m1, m2 = components[:, 0], components[:, 1], components[:, 2], components[:, 3]
        bin_index = exp * (256 ** 3) + m0 * (256 ** 2) + m1 * 256 + m2

        total_bins = 256 ** 4
        normalized = bin_index / (total_bins - 1)
        log_val = normalized * (self.log_max - self.log_min) + self.log_min
        magnitude = 10 ** log_val
        return np.where(bin_index == 0, 0.0, magnitude)

    # ---- persistence -----------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, *, channel: str | None = None) -> "FP16Codec":
        """Load ``(log_min, log_max)``. If the JSON has a ``channels`` key,
        ``channel`` picks one; otherwise the top-level entry is used."""
        import json
        with open(path) as f:
            cfg = json.load(f)
        if "channels" in cfg:
            if channel is None:
                raise ValueError(
                    f"{path} has channels {list(cfg['channels'])!r}; pass channel=..."
                )
            cfg = cfg["channels"][channel]
        return cls(log_min=float(cfg["log_min"]), log_max=float(cfg["log_max"]))

    def to_json(self, path: str | Path) -> None:
        import json
        with open(path, "w") as f:
            json.dump({"log_min": self.log_min, "log_max": self.log_max}, f, indent=2)
            f.write("\n")
