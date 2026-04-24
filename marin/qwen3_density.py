"""Qwen3 subclass with a density-aware L_1 loss term at density-codec positions.

At each target position whose token falls inside the density-codec vocab range,
replace (or augment) the standard CE loss with:

    L_1 = | E[ρ] − ρ_true |

where

    E[ρ] = Σ_v  softmax(logits)_v  ·  decode_all_vec[v]
    decode_all_vec[v] = codec.decode(v - DENSITY_OFFSET)  if v in density range
                      = PENALTY                             otherwise

This makes the loss aware of:
    (a) bin ordinality (bin 256 vs 257 is closer than bin 256 vs 10), and
    (b) non-density-token emissions at density positions (penalized in the
        same L_1 unit via the PENALTY term).

Paired with the LMQ codec (spec 18), this is Formulation Y. With an old
2-tok codec, it's a per-token approximation of Formulation X.

Env vars:
    TOMAT_DENSITY_L1_WEIGHT   float (default 0.0 = pure CE, no change).
                              1.0 = add L_1 loss at density positions alongside CE.
    TOMAT_DENSITY_L1_MODE     "add" (default) or "replace". "replace" zeroes
                              CE at density positions.
    TOMAT_DENSITY_PENALTY     float (default 10.0 × density max). Used for
                              non-density tokens at density target positions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

from levanter.models.lm_model import LmConfig, LmExample
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel


@dataclass(frozen=True)
class DensityLossArgs:
    """Everything the density-L_1 loss needs at forward time, as a frozen struct.

    `decode_all` has shape (Vocab,) and encodes:
      - for density tokens: the codec's decoded float value
      - for all other tokens: PENALTY
    """

    decode_all: NamedArray           # (Vocab,) — precomputed
    density_lo: int                  # inclusive start of density vocab range
    density_hi: int                  # exclusive end of density vocab range
    weight: float                    # λ multiplier on the L_1 term
    mode: str                        # "add" or "replace"


def build_density_loss_args(
    *,
    Vocab: hax.Axis,
    density_offset: int,
    n_density_bins: int,
    codec_recon: np.ndarray,         # shape (n_density_bins,), the codec's decode_all for density range
    penalty: float,
    weight: float,
    mode: str = "add",
) -> DensityLossArgs:
    """Build the decode_all NamedArray + other args from codec + config."""
    if mode not in ("add", "replace"):
        raise ValueError(f"mode must be 'add' or 'replace', got {mode!r}")
    decode_all_np = np.full(Vocab.size, float(penalty), dtype=np.float32)
    decode_all_np[density_offset : density_offset + n_density_bins] = codec_recon.astype(np.float32)
    decode_all = hax.named(decode_all_np, Vocab)
    return DensityLossArgs(
        decode_all=decode_all,
        density_lo=density_offset,
        density_hi=density_offset + n_density_bins,
        weight=weight,
        mode=mode,
    )


def density_aware_loss(
    *,
    Pos: hax.Axis,
    Vocab: hax.Axis,
    logits: NamedArray,              # (Batch, Pos, Vocab)
    input_tokens: NamedArray,        # (Batch, Pos)
    loss_weight: Optional[NamedArray],
    args: DensityLossArgs,
) -> NamedArray:
    """CE + λ·L_1 per position, optionally replacing CE at density positions.

    Returns scalar (mean over all non-masked positions).

    Heavy math uses raw jnp on the underlying arrays to sidestep some haliax
    named-indexing subtleties.
    """
    # Raw array handles
    logits_arr = logits.astype(jnp.float32).array  # (B, Pos, Vocab)
    tokens_arr = input_tokens.array                # (B, Pos) int
    decode_all_arr = args.decode_all.astype(jnp.float32).array  # (Vocab,)

    # Shift tokens -1 along Pos axis: target at position t = tokens[t+1]
    targets_arr = jnp.roll(tokens_arr, -1, axis=-1)

    # Mask out last position
    B, P = tokens_arr.shape
    not_last = jnp.arange(P) < (P - 1)
    not_last = not_last.astype(jnp.float32)  # (P,)
    not_last = jnp.broadcast_to(not_last[None, :], tokens_arr.shape)  # (B, Pos)

    if loss_weight is None:
        lw = not_last
    else:
        lw = loss_weight.astype(jnp.float32).array * not_last

    # CE per position
    log_probs = jax.nn.log_softmax(logits_arr, axis=-1)  # (B, Pos, V)
    ce_per_pos = -jnp.take_along_axis(log_probs, targets_arr[..., None], axis=-1).squeeze(-1)  # (B, Pos)

    # Density-L_1
    probs = jax.nn.softmax(logits_arr, axis=-1)  # (B, Pos, V)
    e_rho = jnp.einsum("bpv,v->bp", probs, decode_all_arr)  # (B, Pos)
    rho_true = decode_all_arr[targets_arr]  # (B, Pos)
    l1_per_pos = jnp.abs(e_rho - rho_true)  # (B, Pos)

    is_density_target = (targets_arr >= args.density_lo) & (targets_arr < args.density_hi)
    l1_per_pos = jnp.where(is_density_target, l1_per_pos, 0.0)

    if args.mode == "replace":
        ce_per_pos = jnp.where(is_density_target, 0.0, ce_per_pos)

    combined = ce_per_pos + args.weight * l1_per_pos  # (B, Pos)
    weighted = combined * lw
    total = jnp.sum(weighted)
    denom = jnp.maximum(jnp.sum(lw), 1.0)
    loss_scalar = total / denom

    # Wrap back to NamedArray (no-axes scalar)
    return hax.named(loss_scalar, ())


class Qwen3DensityConfig(Qwen3Config):
    """Qwen3Config whose model_type is Qwen3DensityLMHeadModel, so
    `config.build(Vocab, key=...)` returns the density-aware subclass.
    """

    @property  # type: ignore[override]
    def model_type(self):
        return Qwen3DensityLMHeadModel


class Qwen3DensityLMHeadModel(Qwen3LMHeadModel):
    """Qwen3 + density-aware loss. Same weights as Qwen3LMHeadModel; only the
    loss function is different.

    Instantiate via `model_cfg.build(Vocab, key=...)` the same as a base Qwen3,
    then wrap with .set_density_loss_args(...) OR attach DensityLossArgs as a
    class-level attribute via monkey-patch.

    Simpler path (what this module does): attach density loss args as a
    module-level global set by `configure_density_loss(...)`, and the subclass's
    `compute_next_token_loss` uses them.
    """

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction=None,
        reduction_axis=None,
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
        logit_soft_cap=None,
    ) -> NamedArray:
        args = _DENSITY_LOSS_ARGS
        if args is None or args.weight == 0.0:
            # Fall through to default Qwen3 CE loss.
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        # Custom path: compute logits explicitly → density-aware loss.
        activations = self.activations(example.tokens, example.attn_mask, key=key)
        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations
        head = self.get_lm_head()
        logits = hax.dot(activations, head, axis=self.Embed)

        Pos = self.Pos
        Vocab = self.Vocab
        loss = density_aware_loss(
            Pos=Pos,
            Vocab=Vocab,
            logits=logits,
            input_tokens=example.tokens,
            loss_weight=example.loss_weight,
            args=args,
        )
        return loss + aux_loss


# Module-level state (set once at training-script startup)
_DENSITY_LOSS_ARGS: Optional[DensityLossArgs] = None


def configure_density_loss(args: DensityLossArgs | None) -> None:
    global _DENSITY_LOSS_ARGS
    _DENSITY_LOSS_ARGS = args
