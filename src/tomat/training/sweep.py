"""Enumerate + parameterise the tomat hello-world sweep grid.

Shared between the preprocessing script (``scripts/sweep_preprocess.py``)
and the training entrypoint (``experiments/tomat_patch_30m.py``). Keeps
one authoritative source of truth for which ``(codec, patch_size)``
combinations are valid under a given context budget, and what the
training-time knobs (patches-per-material ``M``, shuffle-buffer ``N``)
are allowed to be.

This module is *config only* — no Marin or Levanter imports, so it can
be read by the preprocessing CLI without pulling in JAX/TPU deps.
"""

from __future__ import annotations

from dataclasses import dataclass


# Codec → tokens per density value. Used for token-budget estimates.
CODEC_TOKENS_PER_VALUE: dict[str, int] = {
    "tomol_3byte":    3,
    "two_token_9_12": 2,
    "fp16_1token":    1,
}

# Codec → vocabulary size for the density channel (includes signed codec
# component dimensions summed). Used for embedding-param estimates.
CODEC_DENSITY_VOCAB: dict[str, int] = {
    "tomol_3byte":     512 + 256 + 256,  # 1024
    "two_token_9_12":  512 + 4096,       # 4608
    "fp16_1token":     65_536,
}

# Fixed (non-density) vocab — specials + atoms + integers + positions.
# See tomat.tokenizers.patch for the exact layout.
NONDENSITY_VOCAB = 16 + 118 + 1024 + 1024  # = 2182

# Conservative preamble budget (tokens used by specials + atoms + positions
# + shape/offset + delimiters), allowing ~80-atom structures. Real
# preambles for MP val are typically 150–250 tokens.
PREAMBLE_BUDGET = 300


@dataclass(frozen=True)
class SweepConfig:
    """One point in the preprocessing grid."""

    codec: str
    patch_size: int

    @property
    def density_tokens(self) -> int:
        return self.patch_size ** 3 * CODEC_TOKENS_PER_VALUE[self.codec]

    @property
    def estimated_context(self) -> int:
        return self.density_tokens + PREAMBLE_BUDGET

    @property
    def total_vocab_size(self) -> int:
        return NONDENSITY_VOCAB + CODEC_DENSITY_VOCAB[self.codec]

    @property
    def label(self) -> str:
        return f"{self.codec}-P{self.patch_size}"

    def fits(self, context_budget: int) -> bool:
        return self.estimated_context <= context_budget


# Default sweep grid — 3 codecs × 3 patch sizes.
CODECS_TO_SWEEP: tuple[str, ...] = ("tomol_3byte", "two_token_9_12", "fp16_1token")
PATCH_SIZES_TO_SWEEP: tuple[int, ...] = (12, 14, 16)

# Training-time knobs (orthogonal to preprocessing).
PATCHES_PER_MATERIAL_TO_SWEEP: tuple[int, ...] = (16, 32, 64, 128)
SHUFFLE_BUFFER_SIZES_TO_SWEEP: tuple[int, ...] = (1024, 4096, 16_384)


def all_configs() -> list[SweepConfig]:
    """Full ``(codec, patch_size)`` grid regardless of fit."""
    return [SweepConfig(c, p) for c in CODECS_TO_SWEEP for p in PATCH_SIZES_TO_SWEEP]


def valid_configs(context_budget: int = 8192) -> list[SweepConfig]:
    """Configs whose estimated context fits in ``context_budget``."""
    return [c for c in all_configs() if c.fits(context_budget)]
