"""Host-side MaskGIT masking collator (reference / utility).

Note: the primary training path applies masking inside JAX (in
`Qwen3MaskGITLMHeadModel.compute_next_token_loss` via `_jax_apply_maskgit`)
so that everything remains JIT-traceable. This module provides a NumPy-based
reference implementation of the same masking logic, useful for:
  - Offline pre-processing experiments (host-side masking before a JIT call).
  - Unit tests (`tests/test_maskgit_collator.py`).
  - Debugging / visualizing the mask-ratio distribution.

Mask-ratio sampling schedules:
  cosine   (default): r ~ cos(u·π/2), u ~ U(0,1) — favours high ratios,
                      standard MaskGIT prior.
  uniform:            r ~ U(0, 1) — uniform over all mask ratios.
  high:               r ~ U(0.5, 1) — biased towards structure-only training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import haliax as hax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample


MASK_PRIOR_CHOICES = ("cosine", "uniform", "high")


@dataclass
class MaskGITCollator:
    """Online masking collator for a single tokenized sequence batch.

    Parameters
    ----------
    density_offset:   inclusive start of the density vocab range.
    n_density_bins:   number of density bins (density vocab range = [offset, offset+n_bins)).
    mask_id:          the [MASK] token id (= old vocab_size before the bump).
    seq_len:          padded sequence length (pad_to).
    prior:            mask-ratio sampling distribution — "cosine" | "uniform" | "high".
    rng_seed:         seed for the NumPy RNG; bumped per-call for diversity.
    """

    density_offset: int
    n_density_bins: int
    mask_id: int
    seq_len: int
    prior: str = "cosine"
    rng_seed: int = 42

    def __post_init__(self):
        if self.prior not in MASK_PRIOR_CHOICES:
            raise ValueError(
                f"prior must be one of {MASK_PRIOR_CHOICES}, got {self.prior!r}"
            )
        self._rng = np.random.default_rng(self.rng_seed)
        self._step = 0

    def _sample_mask_ratio(self) -> float:
        """Sample one mask ratio r ∈ [0, 1] from the configured prior."""
        u = self._rng.random()
        if self.prior == "cosine":
            return math.cos(u * math.pi / 2.0)
        elif self.prior == "uniform":
            return u
        else:  # "high": r ~ U(0.5, 1.0)
            return 0.5 + 0.5 * u

    def _density_positions(self, row: np.ndarray) -> np.ndarray:
        """Return indices of density tokens in a single sequence row.

        Scans for tokens in [density_offset, density_offset + n_density_bins).
        Density positions are the same for every example in a batch (the
        preamble length only varies by n_atoms for v3), so we use the first row.
        """
        lo = self.density_offset
        hi = lo + self.n_density_bins
        return np.where((row >= lo) & (row < hi))[0]

    def __call__(self, example: LmExample) -> LmExample:
        """Apply online masking to one LmExample.

        The example's `tokens` NamedArray has shape (Batch, Pos) or (Pos,).
        We operate per-sequence (for simplicity and correctness); the result
        retains the same batch structure.
        """
        # --- Extract raw arrays ---
        tokens_ha: NamedArray = example.tokens
        tokens_arr = np.asarray(tokens_ha.array, dtype=np.int32)  # (B, Pos) or (Pos,)
        batched = tokens_arr.ndim == 2
        if not batched:
            tokens_arr = tokens_arr[None, :]  # (1, Pos)

        B, L = tokens_arr.shape
        masked = tokens_arr.copy()
        loss_weight = np.zeros((B, L), dtype=np.float32)

        # Store originals for the loss function (stashed as module global).
        originals = tokens_arr.copy()

        for b in range(B):
            dens_pos = self._density_positions(tokens_arr[b])
            P3 = len(dens_pos)
            if P3 == 0:
                continue
            r = self._sample_mask_ratio()
            mask_count = math.ceil(r * P3)
            # Sample without replacement which positions to mask.
            chosen = self._rng.choice(P3, size=mask_count, replace=False)
            chosen_positions = dens_pos[chosen]
            masked[b, chosen_positions] = self.mask_id
            loss_weight[b, chosen_positions] = 1.0

        self._step += 1

        if not batched:
            masked = masked[0]
            loss_weight = loss_weight[0]
            originals = originals[0]

        # Rebuild NamedArrays with the same axes as the input.
        axes = tokens_ha.axes
        masked_ha = hax.named(jnp.asarray(masked), axes)
        lw_ha = hax.named(jnp.asarray(loss_weight), axes)
        orig_ha = hax.named(jnp.asarray(originals), axes)

        # Bidirectional attention mask (is_causal=False, no explicit mask →
        # materialize() returns None → downstream attention treats as full bidir).
        bidir_mask = AttentionMask(is_causal=False)

        return LmExample(
            tokens=masked_ha,
            loss_weight=lw_ha,
            attn_mask=bidir_mask,
        )
