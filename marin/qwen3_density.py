"""Qwen3 subclass with a density-aware loss term at density-codec positions.

Two density-loss formulations are supported (selected via `loss_type` arg):

  l1   (legacy):   |E[ρ] − ρ_true|        — degenerate: diffuse distributions
                                              with correct mean get zero loss.
  emd  (W₁):       E_v[|ρ − ρ_true|]
                 = Σ_v softmax(logits)_v · |decode_all[v] − ρ_true|
                                            — Wasserstein-1 distance from
                                              predicted distribution to delta-
                                              at-target. Strictly stronger;
                                              matched to NMAE (L_1 metric).

`mode` ∈ {add, replace}:
  add      — loss = CE + λ·density_loss at all positions
  replace  — loss = CE at non-density tokens + λ·density_loss at density tokens
             (CE zeroed at density positions; recommended with emd since EMD
             enforces peakiness on its own).

`density_only` (bool):
  False (default) — CE active at non-density-target positions (atoms, positions,
                    structure delimiters). Standard NTP loss there.
  True            — Zero CE at non-density-target positions too; only the EMD
                    term contributes. Loss is normalized by density-position
                    count so the per-density-token gradient magnitude is
                    interpretable. Use this when you don't care about NTP on
                    preamble tokens (atoms are unordered, patch offset is
                    chosen, etc.) — no reason to spend gradient there.

Env vars:
    TOMAT_DENSITY_L1_WEIGHT     float (default 0.0 = pure CE).
    TOMAT_DENSITY_L1_MODE       "add" (default) or "replace".
    TOMAT_DENSITY_LOSS_TYPE     "l1" (default, back-compat) or "emd".
    TOMAT_DENSITY_ONLY_LOSS     "1" → zero CE on non-density tokens too.
    TOMAT_DENSITY_PENALTY       float (default 10.0 × density max).
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
    weight: float                    # λ multiplier on the density-loss term
    mode: str                        # "add" or "replace"
    loss_type: str = "l1"            # "l1" (legacy |E[ρ]−ρ_true|) or "emd" (W₁)
    density_only: bool = False       # zero CE on non-density tokens, normalize by density count


def build_density_loss_args(
    *,
    Vocab: hax.Axis,
    density_offset: int,
    n_density_bins: int,
    codec_recon: np.ndarray,         # shape (n_density_bins,), the codec's decode_all for density range
    penalty: float,
    weight: float,
    mode: str = "add",
    loss_type: str = "l1",
    density_only: bool = False,
) -> DensityLossArgs:
    """Build the decode_all NamedArray + other args from codec + config."""
    if mode not in ("add", "replace"):
        raise ValueError(f"mode must be 'add' or 'replace', got {mode!r}")
    if loss_type not in ("l1", "emd"):
        raise ValueError(f"loss_type must be 'l1' or 'emd', got {loss_type!r}")
    decode_all_np = np.full(Vocab.size, float(penalty), dtype=np.float32)
    decode_all_np[density_offset : density_offset + n_density_bins] = codec_recon.astype(np.float32)
    decode_all = hax.named(decode_all_np, Vocab)
    return DensityLossArgs(
        decode_all=decode_all,
        density_lo=density_offset,
        density_hi=density_offset + n_density_bins,
        weight=weight,
        mode=mode,
        loss_type=loss_type,
        density_only=density_only,
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

    is_density_target = (targets_arr >= args.density_lo) & (targets_arr < args.density_hi)

    # CE per position — skip entirely when density_only=True (would be zeroed).
    if args.density_only:
        ce_per_pos = jnp.zeros_like(tokens_arr, dtype=jnp.float32)
    else:
        log_probs = jax.nn.log_softmax(logits_arr, axis=-1)  # (B, Pos, V)
        ce_per_pos = -jnp.take_along_axis(log_probs, targets_arr[..., None], axis=-1).squeeze(-1)
        if args.mode == "replace":
            ce_per_pos = jnp.where(is_density_target, 0.0, ce_per_pos)

    # Density loss term — two formulations (always L_1 norm; NMAE is L_1):
    #  l1  (legacy): |E[ρ] − ρ_true|       — degenerate; spread tolerated
    #  emd (W₁):    E_v[|ρ_v − ρ_true|]    — penalizes distribution spread
    probs = jax.nn.softmax(logits_arr, axis=-1)  # (B, Pos, V)
    rho_true = decode_all_arr[targets_arr]  # (B, Pos)
    if args.loss_type == "emd":
        # E_v[|dec_v - rho_true|] under predicted distribution.
        # XLA fuses the abs with the einsum; no (B,P,V) materialization.
        diff = decode_all_arr[None, None, :] - rho_true[..., None]  # (B, Pos, V)
        density_per_pos = jnp.einsum("bpv,bpv->bp", probs, jnp.abs(diff))
    else:  # legacy "l1": |E[ρ] - ρ_true|
        e_rho = jnp.einsum("bpv,v->bp", probs, decode_all_arr)  # (B, Pos)
        density_per_pos = jnp.abs(e_rho - rho_true)

    density_per_pos = jnp.where(is_density_target, density_per_pos, 0.0)

    combined = ce_per_pos + args.weight * density_per_pos  # (B, Pos)
    weighted = combined * lw
    total = jnp.sum(weighted)
    if args.density_only:
        # Normalize by density-position count so per-density-token gradient
        # magnitude is invariant to seq packing / preamble length.
        denom_mask = lw * is_density_target.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(denom_mask), 1.0)
    else:
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
