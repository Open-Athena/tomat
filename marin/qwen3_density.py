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

Env vars (AR density-aware loss):
    TOMAT_DENSITY_L1_WEIGHT     float (default 0.0 = pure CE).
    TOMAT_DENSITY_L1_MODE       "add" (default) or "replace".
    TOMAT_DENSITY_LOSS_TYPE     "l1" (default, back-compat) or "emd".
    TOMAT_DENSITY_ONLY_LOSS     "1" → zero CE on non-density tokens too.
    TOMAT_DENSITY_PENALTY       float (default 10.0 × density max).

Env vars (MaskGIT mode):
    TOMAT_MG_MODE               "1" → enable MaskGIT masked-token training.
    TOMAT_MG_MASK_PRIOR         "cosine" (default) | "uniform" | "high".
    TOMAT_MG_LOSS_TYPE          "ce" (default) | "ce_emd".

Env vars (scheduled-sampling / FR-aware AR mode — spec 31):
    TOMAT_SS_MODE               "1" → enable token-level scheduled sampling on
                                density tokens during AR training (Bengio 2015,
                                always-sample variant restricted to density).
    TOMAT_SS_EPS_MAX            float, default 0.25. Per-batch ε is sampled
                                Uniform(0, ε_max) inside JIT; ε is the
                                probability of replacing each density-target
                                input with the model's own one-step prediction.
    TOMAT_SS_SAMPLER            "median" (default, matches FR eval decoder) |
                                "argmax" | "sample" (true categorical from
                                softmax over the density vocab range).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, cast

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis, NamedArray

from levanter.layers.attention import AttentionMask
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
    pad_id: int | None = None        # spec 36: when set, zero loss at positions
                                     # whose target equals pad_id (the segment-
                                     # boundary sentinel in packed datasets).
                                     # Also masks the trailing PAD region.


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
    pad_id: int | None = None,
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
        pad_id=pad_id,
    )


def density_aware_loss(
    *,
    Pos: hax.Axis,
    Vocab: hax.Axis,
    logits: NamedArray,              # (Batch, Pos, Vocab)
    input_tokens: NamedArray,        # (Batch, Pos)
    loss_weight: Optional[NamedArray],
    args: DensityLossArgs,
    reduction=None,                  # None → return per-position; else apply
) -> NamedArray:
    """CE + λ·L_1 per position, optionally replacing CE at density positions.

    Returns rank-2 (Batch, Pos) per-position loss when reduction is None
    (the path Levanter's eval shape-inference takes). When reduction is a
    callable (e.g. hax.mean from the training-time default), returns a scalar
    using a custom denom (number of density positions or all non-masked
    positions, depending on `density_only`).

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

    # Sequence-packing (spec 36): in packed datasets, a single PAD sentinel
    # separates adjacent sub-sequences (and pads the row tail). The roll(-1)
    # target at the *last* position of each sub-sequence is therefore PAD —
    # we don't want to grade the model on "predict PAD after density". Mask
    # these positions out of the loss weight. Also catches the trailing-PAD
    # region (rolls forward to PAD or to the wrap-around token at pos 0).
    if args.pad_id is not None:
        non_pad_target = (targets_arr != args.pad_id).astype(jnp.float32)
        lw = lw * non_pad_target

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
    weighted = combined * lw                                # (B, Pos)

    if reduction is None:
        # Eval path: per-position loss, rank-2. Levanter's accumulator
        # will sum + divide by its own count across the eval batches.
        # Reconstruct NamedArray with the original input's axes.
        return hax.named(weighted, input_tokens.axes)

    # Training path: scalar with custom normalization.
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

    NOTE on `init` override (2026-05-25): Levanter's `Qwen3LMHeadModel.init`
    hard-codes `return Qwen3LMHeadModel(...)` instead of `return cls(...)`, so
    `Qwen3DensityLMHeadModel.init(...)` silently returns a base Qwen3LMHeadModel
    with the subclass discarded — `compute_next_token_loss` dispatches to the
    base, never to this override. Every "density EMD"-mode run prior to this
    fix was actually trained with standard CE. Re-wrap via parent's tuple to
    restore the subclass.
    """

    @classmethod
    def init(cls, Vocab, config, *, key):
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction=hax.mean,        # match Levanter base default; eval passes None
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
            reduction=reduction,
        )
        return loss + aux_loss


# Module-level state (set once at training-script startup)
_DENSITY_LOSS_ARGS: Optional[DensityLossArgs] = None


def configure_density_loss(args: DensityLossArgs | None) -> None:
    global _DENSITY_LOSS_ARGS
    _DENSITY_LOSS_ARGS = args


# ---------------------------------------------------------------------------
# MaskGIT masked-token loss
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskGITLossArgs:
    """Configuration for the MaskGIT masked-token loss and online masking.

    `decode_all` has shape (Vocab,) identical to `DensityLossArgs.decode_all`:
      - density tokens: codec decoded float value
      - all other tokens: PENALTY

    `mask_id`:  the [MASK] token id (= old vocab_size before the +1 bump).
    `prior`:    mask-ratio sampling schedule — "cosine" | "uniform" | "high"
                | "absorbing" (r=1 always; pure structure→density, no
                in-patch GT conditioning ever; see spec 40).
    `loss_type`: "ce" (pure CE) or "ce_emd" (CE + Wasserstein-1 density term).
    `weight`:   λ on the EMD term (only used when loss_type == "ce_emd").

    Masking is applied inside `Qwen3MaskGITLMHeadModel.compute_next_token_loss`
    using JAX random ops (so everything is JIT-traceable). No host-side collator
    is needed.
    """

    decode_all: NamedArray       # (Vocab,) — precomputed
    density_lo: int              # inclusive start of density vocab range
    density_hi: int              # exclusive end of density vocab range
    mask_id: int                 # [MASK] token id = old vocab_size
    prior: str = "cosine"        # mask-ratio schedule: cosine|uniform|high|absorbing
    weight: float = 1.0          # λ multiplier on the EMD density term (ce_emd)
    loss_type: str = "ce"        # "ce" | "ce_emd" | "emd" | "kl_gauss" | "crps"
    # KL-Gaussian: stddev (in ρ-units) of the soft target distribution.
    # Target Q(v) ∝ exp(-(ρ_v − ρ_true)² / 2σ²), normalized over the density
    # range. σ→0 recovers vanilla CE; σ large gives EMD-like smoothness.
    # See `posts/13-why-pure-emd-collapses.md` §5.1.
    kl_sigma: float = 0.5


_VALID_LOSS_TYPES = ("ce", "ce_emd", "emd", "kl_gauss", "crps")


def build_maskgit_loss_args(
    *,
    Vocab: hax.Axis,
    density_offset: int,
    n_density_bins: int,
    codec_recon: np.ndarray,     # shape (n_density_bins,)
    penalty: float,
    mask_id: int,
    prior: str = "cosine",
    weight: float = 1.0,
    loss_type: str = "ce",
    kl_sigma: float = 0.5,
) -> MaskGITLossArgs:
    """Build `MaskGITLossArgs` from codec + config."""
    if loss_type not in _VALID_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {_VALID_LOSS_TYPES}, got {loss_type!r}")
    if prior not in ("cosine", "uniform", "high", "absorbing"):
        raise ValueError(f"prior must be cosine/uniform/high/absorbing, got {prior!r}")
    if loss_type == "kl_gauss" and not (kl_sigma > 0):
        raise ValueError(f"kl_sigma must be > 0 for kl_gauss, got {kl_sigma!r}")
    decode_all_np = np.full(Vocab.size, float(penalty), dtype=np.float32)
    decode_all_np[density_offset : density_offset + n_density_bins] = codec_recon.astype(np.float32)
    decode_all = hax.named(decode_all_np, Vocab)
    return MaskGITLossArgs(
        decode_all=decode_all,
        density_lo=density_offset,
        density_hi=density_offset + n_density_bins,
        mask_id=mask_id,
        prior=prior,
        weight=weight,
        loss_type=loss_type,
        kl_sigma=kl_sigma,
    )


def maskgit_aware_loss(
    *,
    Pos: hax.Axis,
    Vocab: hax.Axis,
    logits: NamedArray,              # (Batch, Pos, Vocab)
    input_tokens: NamedArray,        # (Batch, Pos) — the MASKED input
    loss_weight: Optional[NamedArray],  # (Batch, Pos) — 1 at masked positions, 0 elsewhere
    original_tokens: NamedArray,     # (Batch, Pos) — pre-mask ground-truth tokens
    args: MaskGITLossArgs,
    reduction=None,                  # None → return per-position; else apply
) -> NamedArray:
    """MaskGIT masked-token loss: CE (+ optional EMD) at masked positions only.

    Key differences from `density_aware_loss`:
    - No token shift: target at position t = original_tokens[t], NOT input[t+1].
    - Loss restricted to masked positions (loss_weight=1 where [MASK] was
      applied; collator zeros the rest).
    - Normalized by sum(loss_weight) — masked-position count — to keep
      gradient magnitude stable across varying mask ratios.
    - EMD density term applied only at masked density positions.

    Returns scalar (mean over masked positions).
    """
    logits_arr = logits.astype(jnp.float32).array          # (B, Pos, V)
    targets_arr = original_tokens.array                     # (B, Pos) int
    decode_all_arr = args.decode_all.astype(jnp.float32).array  # (V,)

    # loss_weight: 1 at masked positions, 0 elsewhere
    if loss_weight is None:
        lw = jnp.ones(targets_arr.shape, dtype=jnp.float32)
    else:
        lw = loss_weight.astype(jnp.float32).array          # (B, Pos)

    # Loss term selection. Mirrors AR's `density_aware_loss`:
    #   ce      : pure CE at masked positions (default; MaskGIT paper).
    #   ce_emd  : CE + λ·EMD at masked density positions.
    #   emd     : pure EMD at masked density positions (no CE) — matches the
    #             AR `density_only=True, loss_type="emd"` recipe that got us
    #             0.91% NMAE TF on cont33k. Skip CE entirely.
    if args.loss_type == "emd":
        is_density_target = (targets_arr >= args.density_lo) & (targets_arr < args.density_hi)
        probs = jax.nn.softmax(logits_arr, axis=-1)         # (B, Pos, V)
        rho_true = decode_all_arr[targets_arr]              # (B, Pos)
        diff = decode_all_arr[None, None, :] - rho_true[..., None]  # (B, Pos, V)
        emd_per_pos = jnp.einsum("bpv,bpv->bp", probs, jnp.abs(diff))
        combined = jnp.where(is_density_target, emd_per_pos, 0.0)
    elif args.loss_type in ("kl_gauss", "crps"):
        # Soft-target losses at density positions; CE at non-density positions
        # so the non-density preamble keeps the same gradient signal as the
        # plain-CE baseline (clean LF-only ablation).
        is_density_target = (targets_arr >= args.density_lo) & (targets_arr < args.density_hi)
        log_probs = jax.nn.log_softmax(logits_arr, axis=-1)     # (B, Pos, V)
        ce_per_pos = -jnp.take_along_axis(
            log_probs, targets_arr[..., None], axis=-1,
        ).squeeze(-1)                                           # (B, Pos)

        rho_true = decode_all_arr[targets_arr]                  # (B, Pos)
        V = decode_all_arr.shape[0]
        dens_idx = jnp.arange(V)
        density_bin_mask = (dens_idx >= args.density_lo) & (dens_idx < args.density_hi)  # (V,)

        if args.loss_type == "kl_gauss":
            # Q(v) ∝ exp(-(ρ_v - ρ_true)² / 2σ²) over density range only.
            # Loss = -Σ_v Q(v) log P(v) (= KL(Q‖P) up to entropy(Q) const).
            sigma = jnp.asarray(args.kl_sigma, dtype=jnp.float32)
            log_q_unnorm = -((decode_all_arr[None, None, :] - rho_true[..., None]) ** 2) / (2.0 * sigma ** 2)
            log_q_masked = jnp.where(density_bin_mask[None, None, :], log_q_unnorm, -jnp.inf)
            log_q = jax.nn.log_softmax(log_q_masked, axis=-1)   # (B, Pos, V), normalized over density range
            q = jnp.exp(log_q)
            kl_ce_per_pos = -jnp.einsum("bpv,bpv->bp", q, log_probs)
            combined = jnp.where(is_density_target, kl_ce_per_pos, ce_per_pos)
        else:  # crps
            # CRPS = ∫(F_P(x) − 1[x≥ρ_true])² dx, discretized over sorted
            # density bins. LMQCodec.recon_points is sorted ascending (see
            # src/tomat/float_codec.py boundaries). Integral from ρ_0 to ρ_{n-1};
            # tails contribute 0 when ρ_true ∈ [ρ_0, ρ_{n-1}] (always true since
            # the codec covers the full density range).
            probs = jax.nn.softmax(logits_arr, axis=-1)         # (B, Pos, V)
            p_dens = probs[..., args.density_lo : args.density_hi]   # (B, Pos, n_dens)
            sorted_rho = decode_all_arr[args.density_lo : args.density_hi]  # (n_dens,)
            gaps = jnp.diff(sorted_rho)                              # (n_dens - 1,)
            f_p = jnp.cumsum(p_dens, axis=-1)[..., :-1]              # (B, Pos, n_dens - 1)
            f_t = (sorted_rho[None, None, :-1] >= rho_true[..., None]).astype(jnp.float32)
            crps_per_pos = jnp.einsum("bpv,v->bp", (f_p - f_t) ** 2, gaps)
            combined = jnp.where(is_density_target, crps_per_pos, ce_per_pos)
    else:
        # CE at masked positions: no shift — target at t is original token at t.
        log_probs = jax.nn.log_softmax(logits_arr, axis=-1)     # (B, Pos, V)
        ce_per_pos = -jnp.take_along_axis(
            log_probs, targets_arr[..., None], axis=-1,
        ).squeeze(-1)                                           # (B, Pos)
        combined = ce_per_pos

        if args.loss_type == "ce_emd":
            # EMD density term at masked density positions only.
            is_density_target = (targets_arr >= args.density_lo) & (targets_arr < args.density_hi)
            probs = jax.nn.softmax(logits_arr, axis=-1)         # (B, Pos, V)
            rho_true = decode_all_arr[targets_arr]              # (B, Pos)
            diff = decode_all_arr[None, None, :] - rho_true[..., None]  # (B, Pos, V)
            emd_per_pos = jnp.einsum("bpv,bpv->bp", probs, jnp.abs(diff))
            emd_per_pos = jnp.where(is_density_target, emd_per_pos, 0.0)
            combined = combined + args.weight * emd_per_pos

    weighted = combined * lw
    if reduction is None:
        # Eval path: return per-position rank-2 (Batch, Pos) NamedArray.
        return hax.named(weighted, input_tokens.axes)
    # Training path: scalar normalized by masked-position count.
    total = jnp.sum(weighted)
    denom = jnp.maximum(jnp.sum(lw), 1.0)
    loss_scalar = total / denom
    return hax.named(loss_scalar, ())


class Qwen3MaskGITConfig(Qwen3Config):
    """Qwen3Config whose `model_type` is `Qwen3MaskGITLMHeadModel`."""

    @property  # type: ignore[override]
    def model_type(self):
        return Qwen3MaskGITLMHeadModel


class Qwen3MaskGITLMHeadModel(Qwen3LMHeadModel):
    """Qwen3 + MaskGIT masked-token loss.

    Same weights as Qwen3LMHeadModel; only the loss function differs.

    Masking is applied *inside* `compute_next_token_loss` using JAX random ops
    so everything remains JIT-traceable without a host-side collator. On entry
    `example.tokens` are the original unmasked tokens; the model applies the
    cosine/uniform/high mask schedule using `key`, then:
      - Sets masked positions → MASK_ID in the token buffer.
      - Builds `attn_mask=AttentionMask(is_causal=False)` (full bidirectional).
      - Sets `loss_weight=1` at masked positions, 0 elsewhere.
      - Forwards the masked sequence and computes `maskgit_aware_loss`.

    The mask schedule prior and MASK_ID are read from `_MASKGIT_LOSS_ARGS`
    (set at training-script startup via `configure_maskgit_loss`).

    See `Qwen3DensityLMHeadModel.init` for why we need to override `init`
    (Levanter's parent `Qwen3LMHeadModel.init` hard-codes the return class).
    """

    @classmethod
    def init(cls, Vocab, config, *, key):
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction=hax.mean,        # match Levanter base default; eval passes None
        reduction_axis=None,
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
        logit_soft_cap=None,
    ) -> NamedArray:
        args = _MASKGIT_LOSS_ARGS
        # Trace-time guard: caught the mg-2 silent-fallback regression where
        # args was None despite TOMAT_MG_MODE=1 → all 3 prior variants trained
        # as plain bidirectional AR with bit-identical loss curves. Make it
        # loud now — either confirm MG path or fail fast.
        if args is None:
            import os as _os
            print("[maskgit] compute_next_token_loss: _MASKGIT_LOSS_ARGS is None")
            if _os.environ.get("TOMAT_MG_MODE") == "1":
                raise RuntimeError(
                    "Qwen3MaskGITLMHeadModel.compute_next_token_loss called "
                    "but _MASKGIT_LOSS_ARGS is None despite TOMAT_MG_MODE=1. "
                    "configure_maskgit_loss() must be called before training "
                    "starts (and survive the JIT trace). Silent fallback "
                    "would train as plain bidirectional AR — that's the "
                    "mg-2 sweep regression we're hardening against."
                )
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        # Trace-time confirmation print — fires once per JIT trace.
        print(f"[maskgit] compute_next_token_loss TRACE: prior={args.prior!r} "
              f"loss_type={args.loss_type!r} mask_id={args.mask_id} "
              f"key_is_none={key is None}")

        # Original tokens are the ground-truth targets (no shift).
        original_tokens = example.tokens  # (Batch, Pos) NamedArray

        # Apply MaskGIT masking inside JIT using JAX random.
        # `key` may be None during eval — in that case skip masking and
        # fall through to the standard forward (no loss for eval).
        if key is not None:
            masked_tokens, loss_weight = _jax_apply_maskgit(
                original_tokens=original_tokens,
                args=args,
                key=key,
            )
        else:
            masked_tokens = original_tokens
            loss_weight = example.loss_weight

        # Bidirectional attention (no causal mask). Propagate any segment_ids
        # from the incoming attn_mask so packed sub-sequences keep their
        # block-diagonal isolation (spec 36).
        bidir_mask = AttentionMask(is_causal=False)
        incoming = example.attn_mask
        if isinstance(incoming, AttentionMask) and incoming.segment_ids is not None:
            q_seg, kv_seg = incoming.segment_ids
            bidir_mask = bidir_mask.with_segment_ids(q_seg, kv_seg)
        masked_example = LmExample(
            tokens=masked_tokens,
            loss_weight=loss_weight,
            attn_mask=bidir_mask,
        )

        activations = self.activations(masked_example.tokens, masked_example.attn_mask, key=None)
        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations
        head = self.get_lm_head()
        logits = hax.dot(activations, head, axis=self.Embed)

        loss = maskgit_aware_loss(
            Pos=self.Pos,
            Vocab=self.Vocab,
            logits=logits,
            input_tokens=masked_example.tokens,
            loss_weight=masked_example.loss_weight,
            original_tokens=original_tokens,
            args=args,
            reduction=reduction,
        )
        return loss + aux_loss


def _jax_apply_maskgit(
    original_tokens: NamedArray,   # (Batch, Pos)
    args: "MaskGITLossArgs",
    key,
) -> tuple[NamedArray, NamedArray]:
    """Apply MaskGIT masking inside a JIT-traced context.

    For each example in the batch:
      1. Identify density positions (tokens in [density_lo, density_hi)).
      2. Sample mask ratio r from the configured prior (cosine/uniform/high).
      3. Randomly pick ceil(r * P³) density positions and replace with MASK_ID.
      4. Return (masked_tokens, loss_weight) NamedArrays with the same axes.

    All ops are JAX-native so this runs under JIT without Python-side RNG.
    """
    tokens_arr = original_tokens.array.astype(jnp.int32)   # raw (B, Pos)
    B, L = tokens_arr.shape
    axes = original_tokens.axes

    lo = args.density_lo
    hi = args.density_hi
    mask_id = args.mask_id
    prior = args.prior

    is_density = (tokens_arr >= lo) & (tokens_arr < hi)  # (B, Pos) bool

    def mask_one_example(tokens_b, is_dens_b, key_b):
        # Density positions for this example.
        P3 = jnp.sum(is_dens_b)
        key_r, key_sel = jax.random.split(key_b)
        # Sample mask ratio from prior.
        u = jax.random.uniform(key_r)
        if prior == "cosine":
            r = jnp.cos(u * jnp.pi / 2.0)
        elif prior == "high":
            r = 0.5 + 0.5 * u
        elif prior == "absorbing":
            # r=1 always. Every density position is replaced by MASK; the
            # model never sees in-patch GT density at training time. Pure
            # structure→field objective. See spec 40 ("tomat zero").
            r = jnp.ones_like(u)
        else:  # "uniform"
            r = u
        mask_count = jnp.ceil(r * P3.astype(jnp.float32)).astype(jnp.int32)
        mask_count = jnp.minimum(mask_count, P3)

        # Assign random scores to all positions; highest scores at density
        # positions among the top-mask_count are masked.
        scores = jax.random.uniform(key_sel, shape=(L,))
        # Penalty: non-density positions get score -1 so they're never picked.
        scores = jnp.where(is_dens_b, scores, -1.0)
        # Top-k indices: argsort descending, take first mask_count.
        order = jnp.argsort(-scores)  # (L,) descending by score

        # Mark positions whose rank < mask_count as masked.
        rank = jnp.argsort(order)     # rank[i] = rank of position i
        to_mask = rank < mask_count   # (L,) bool

        masked = jnp.where(to_mask, mask_id, tokens_b)
        lw = to_mask.astype(jnp.float32)
        return masked, lw

    keys = jax.random.split(key, B)
    masked_arr, lw_arr = jax.vmap(mask_one_example)(tokens_arr, is_density, keys)

    return (
        hax.named(masked_arr, axes),
        hax.named(lw_arr, axes),
    )


# Module-level state for MaskGIT (set once at training-script startup).
_MASKGIT_LOSS_ARGS: Optional[MaskGITLossArgs] = None


def configure_maskgit_loss(args: "MaskGITLossArgs | None") -> None:
    global _MASKGIT_LOSS_ARGS
    _MASKGIT_LOSS_ARGS = args


# ---------------------------------------------------------------------------
# Scheduled-sampling (FR-aware AR training) — spec 31
# ---------------------------------------------------------------------------

# ε distribution mini-DSL — supports per-batch sampling from a mixture of
# constants and uniforms. Examples:
#   "1"             → ε = 1 always               (pure FR-style)
#   "U(0,1)"        → ε ~ Uniform(0, 1)          (mix without ε=1 guarantee)
#   ".3(1)+.7U(0,1)" → 30% ε=1 + 70% ε ~ U(0,1)  (mix WITH ε=1 mass)
# Grammar:
#   DIST      := COMPONENT ('+' COMPONENT)*
#   COMPONENT := WEIGHT? BASE
#   WEIGHT    := NUMBER '*'?           # e.g. '.3' or '.3*' or '0.3*'
#   BASE      := NUMBER                # const, e.g. '1'
#              | '(' NUMBER ')'        # parenthesized const, e.g. '(1)'
#              | 'U(' NUMBER ',' NUMBER ')'  # uniform interval
# Weights default to 1 and are normalized to sum to 1. Whitespace ignored.

@dataclass(frozen=True)
class _EpsComponent:
    kind: str       # 'const' or 'uniform'
    a: float        # constant value (kind='const'), or interval lo (kind='uniform')
    b: float        # interval hi (kind='uniform'), or 0 (kind='const')


@dataclass(frozen=True)
class _EpsDist:
    weights: tuple[float, ...]
    components: tuple[_EpsComponent, ...]


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep`, ignoring separators inside parens."""
    depth = 0
    parts: list[str] = []
    last = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(s[last:i])
            last = i + 1
    parts.append(s[last:])
    return parts


def _parse_eps_component(s: str) -> tuple[float, _EpsComponent]:
    import re as _re
    s = s.strip()
    # Try to peel off a leading weight followed by '*' or '('.
    # Heuristic: if string contains 'U(' or starts with '(', the bit before is
    # the (optional) weight; otherwise bare numbers are constants without weight.
    m_uniform = _re.match(r'^([\d.]*)\*?\s*U\(\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*\)$', s)
    m_paren_const = _re.match(r'^([\d.]+)\*?\s*\(\s*([\-\d.]+)\s*\)$', s)
    m_just_paren = _re.match(r'^\(\s*([\-\d.]+)\s*\)$', s)
    m_bare = _re.match(r'^([\-\d.]+)$', s)
    if m_uniform:
        w_str, a_str, b_str = m_uniform.groups()
        weight = float(w_str) if w_str else 1.0
        return weight, _EpsComponent('uniform', float(a_str), float(b_str))
    if m_paren_const:
        w_str, v_str = m_paren_const.groups()
        return float(w_str), _EpsComponent('const', float(v_str), 0.0)
    if m_just_paren:
        return 1.0, _EpsComponent('const', float(m_just_paren.group(1)), 0.0)
    if m_bare:
        return 1.0, _EpsComponent('const', float(m_bare.group(1)), 0.0)
    raise ValueError(f"Cannot parse ε component {s!r}")


def parse_eps_dist(s: str) -> _EpsDist:
    """Parse a `TOMAT_SS_EPS_DIST` mini-language string. See docstring above."""
    parts = _split_top_level(s.replace(' ', ''), '+')
    if not parts or any(not p for p in parts):
        raise ValueError(f"Empty ε-dist: {s!r}")
    weights = []
    components = []
    for part in parts:
        w, comp = _parse_eps_component(part)
        weights.append(w)
        components.append(comp)
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"ε-dist weights sum to {total}: {s!r}")
    # Validate every component yields ε ∈ [0, 1].
    for c in components:
        if c.kind == 'const' and not (0.0 <= c.a <= 1.0):
            raise ValueError(f"const ε out of [0,1]: {c.a} in {s!r}")
        if c.kind == 'uniform' and not (0.0 <= c.a <= c.b <= 1.0):
            raise ValueError(f"uniform [a,b] not in [0,1]: ({c.a},{c.b}) in {s!r}")
    return _EpsDist(
        weights=tuple(w / total for w in weights),
        components=tuple(components),
    )


@dataclass(frozen=True)
class SSArgs:
    """Configuration for token-level scheduled sampling on density positions.

    The base loss is the existing density-aware AR loss; SS only mutates the
    *input* tokens before the real forward pass. The base `DensityLossArgs` is
    reused for the loss term itself — see `_SS_DENSITY_ARGS_REQUIRED` for why
    SS requires the density-loss path to be configured too.

    `density_lo` / `density_hi`:  inclusive/exclusive density vocab range
                                  (same convention as `DensityLossArgs`).
    `eps_dist`:   Parsed `_EpsDist` describing the per-batch ε distribution.
                  Replaces the legacy `eps_min`/`eps_max` knobs (which are
                  translated to `U(eps_min, eps_max)` if `eps_dist` is None
                  for back-compat).
    `sampler`:    "median" (default, matches FR eval) | "argmax" | "sample".
                  All operate on the density-bin slice of the predicted-logits
                  distribution (not the full vocab), so they always return a
                  valid density token.
    """

    density_lo: int                  # inclusive start of density vocab range
    density_hi: int                  # exclusive end of density vocab range
    eps_dist: _EpsDist               # parsed ε mixture distribution
    sampler: str = "median"          # "median" | "argmax" | "sample"


def build_ss_args(
    *,
    density_offset: int,
    n_density_bins: int,
    eps_dist: str | None = None,
    eps_min: float = 0.0,
    eps_max: float = 0.25,
    sampler: str = "median",
) -> SSArgs:
    """Build SSArgs from codec config.

    Args:
      eps_dist: mini-DSL string for the per-batch ε distribution (see
        `parse_eps_dist` docstring). If None, falls back to the legacy
        `U(eps_min, eps_max)` parameterization for back-compat.
    """
    if sampler not in ("median", "argmax", "sample"):
        raise ValueError(f"sampler must be 'median'/'argmax'/'sample', got {sampler!r}")
    if eps_dist is None:
        # Legacy fallback.
        if not (0.0 <= eps_min <= 1.0):
            raise ValueError(f"eps_min must be in [0, 1], got {eps_min!r}")
        if not (0.0 <= eps_max <= 1.0):
            raise ValueError(f"eps_max must be in [0, 1], got {eps_max!r}")
        if eps_min > eps_max:
            raise ValueError(f"eps_min ({eps_min!r}) must be <= eps_max ({eps_max!r})")
        if eps_min == eps_max:
            parsed = parse_eps_dist(str(eps_min))
        else:
            parsed = parse_eps_dist(f"U({eps_min},{eps_max})")
    else:
        parsed = parse_eps_dist(eps_dist)
    return SSArgs(
        density_lo=density_offset,
        density_hi=density_offset + n_density_bins,
        eps_dist=parsed,
        sampler=sampler,
    )


def _jax_apply_ss(
    original_tokens: NamedArray,   # (Batch, Pos) — GT input tokens
    pre_logits: NamedArray,        # (Batch, Pos, Vocab) — pre-forward logits, stop_grad'd
    args: SSArgs,
    key,
) -> NamedArray:
    """Mix GT density-target inputs with the model's own predictions.

    AR convention: target at position t is `tokens[t+1]`. The model's
    one-step-ahead prediction at output position t is therefore the prediction
    for the token that should appear at position t+1. For each density-target
    position (i.e. positions t where `tokens[t+1]` is a density token), we:

      1. Restrict `pre_logits[t, density_lo:density_hi]` to the density vocab.
      2. Decode a single bin index via `sampler` (median / argmax / sample).
      3. With Bernoulli(ε) per position, replace `tokens[t+1]` with the decoded
         bin's vocab id. Non-density positions are never touched (the AR loss
         doesn't grade them as densities, and the input semantics on preamble
         tokens are categorical not continuous).

    ε is sampled per-batch as `Uniform(0, eps_max)` (single scalar shared
    across the whole batch) inside JIT. Mirrors how MaskGIT samples its mask
    ratio per-example, but per-batch here keeps the schedule simple — Bengio
    2015's "always-sample" variant restricted to density tokens.

    Returns:
      mixed_tokens: (Batch, Pos) NamedArray, same axes as `original_tokens`.
      Non-density input positions and positions whose *next* token is not a
      density token are guaranteed identical to the input.
    """
    tokens_arr = original_tokens.array.astype(jnp.int32)   # (B, L)
    logits_arr = pre_logits.array.astype(jnp.float32)      # (B, L, V)
    axes = original_tokens.axes
    B, L = tokens_arr.shape

    lo = args.density_lo
    hi = args.density_hi
    n_bins = hi - lo

    # Sample per-batch ε from the configured mixture distribution. See
    # `parse_eps_dist` for the mini-DSL grammar. Inside JIT we always sample
    # ONE ε per batch: pick one component index via weighted choice, then
    # sample from that component's distribution.
    key_eps, key_mask, key_sample = jax.random.split(key, 3)
    key_comp, key_val = jax.random.split(key_eps)
    dist = args.eps_dist
    weights = jnp.asarray(dist.weights, dtype=jnp.float32)
    # Per-component samples (compiled-in; component count is small and static).
    per_comp = []
    for c in dist.components:
        if c.kind == 'const':
            per_comp.append(jnp.asarray(c.a, dtype=jnp.float32))
        else:  # uniform
            per_comp.append(
                jax.random.uniform(key_val, shape=(), minval=c.a, maxval=c.b)
                .astype(jnp.float32)
            )
    samples_stacked = jnp.stack(per_comp)
    comp_idx = jax.random.choice(key_comp, len(dist.weights), p=weights)
    eps = samples_stacked[comp_idx]

    # density_logits[t] is the predicted distribution for token at position t+1.
    # Restrict to the density vocab range so the decoded bin is always a valid
    # density token (avoids the fp32 underflow bug fixed in `decode_density`).
    density_logits = logits_arr[..., lo:hi]                # (B, L, n_bins)

    # Decode per-position predicted bin via sampler.
    if args.sampler == "argmax":
        pred_bin = jnp.argmax(density_logits, axis=-1).astype(jnp.int32)  # (B, L)
    elif args.sampler == "median":
        probs = jax.nn.softmax(density_logits, axis=-1)    # (B, L, n_bins)
        cumP = jnp.cumsum(probs, axis=-1)
        pred_bin = jnp.clip(
            jnp.sum(cumP < 0.5, axis=-1).astype(jnp.int32), 0, n_bins - 1,
        )                                                  # (B, L)
    else:  # "sample"
        # Categorical sample, vmap across (B, L) via flatten.
        flat = density_logits.reshape(B * L, n_bins)
        keys = jax.random.split(key_sample, B * L)
        sampled = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(keys, flat)
        pred_bin = sampled.reshape(B, L).astype(jnp.int32)

    pred_tokens = pred_bin + lo                            # (B, L) density vocab id

    # The prediction at output position t is the candidate replacement for the
    # INPUT token at position t+1. Build a (B, L) candidate-input buffer by
    # shifting pred_tokens one to the right; position 0 has no upstream
    # predictor and is filled with the original token (never replaced).
    # candidate[b, t+1] = pred_tokens[b, t].
    cand_tokens = jnp.concatenate(
        [tokens_arr[:, :1], pred_tokens[:, :-1]], axis=-1,
    )                                                      # (B, L)

    # Replace only at positions whose original token is a density token (i.e.
    # positions that are density-target inputs, t+1 in AR terms). Equivalent
    # to: the prediction at t-1 generated this density bin.
    is_density_input = (tokens_arr >= lo) & (tokens_arr < hi)  # (B, L) bool

    # Per-position Bernoulli(ε) mask.
    u = jax.random.uniform(key_mask, shape=tokens_arr.shape, dtype=jnp.float32)
    bernoulli = u < eps                                    # (B, L) bool

    do_replace = is_density_input & bernoulli              # (B, L) bool
    mixed_arr = jnp.where(do_replace, cand_tokens, tokens_arr).astype(tokens_arr.dtype)

    return hax.named(mixed_arr, axes)


class Qwen3SSConfig(Qwen3Config):
    """Qwen3Config whose `model_type` is `Qwen3SSLMHeadModel`."""

    @property  # type: ignore[override]
    def model_type(self):
        return Qwen3SSLMHeadModel


class Qwen3SSLMHeadModel(Qwen3DensityLMHeadModel):
    """Qwen3 + density-aware loss + scheduled-sampling on density inputs.

    Inherits from `Qwen3DensityLMHeadModel` so the loss term (CE / EMD /
    density-only) is identical; SS only mutates the *input* tokens before the
    real forward pass via `_jax_apply_ss`.

    Two-step forward per train step:
      1. Pre-forward `example.tokens` → logits, wrapped in `stop_gradient`.
         No gradients from this pass; it only supplies candidate predictions
         for the mixing step.
      2. Mix predicted-vs-GT density inputs per a per-batch ε ~ U(0, ε_max)
         + per-position Bernoulli(ε) mask, restricted to density-target
         positions.
      3. Real forward on the mixed input → existing `density_aware_loss` vs
         the **original** ground-truth targets.

    Eval path (`key is None`): SS is skipped — eval is teacher-forced by
    construction, identical to `Qwen3DensityLMHeadModel`.

    See `Qwen3DensityLMHeadModel.init` for the `init` re-wrap necessitated by
    Levanter's parent `Qwen3LMHeadModel.init` hard-coding the return class
    (specs/done/30-levanter-init-cls-postmortem.md).
    """

    @classmethod
    def init(cls, Vocab, config, *, key):
        base = Qwen3LMHeadModel.init(Vocab, config, key=key)
        return cls(base.transformer, base.embeddings, base.lm_head)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction=hax.mean,        # match Levanter base default; eval passes None
        reduction_axis=None,
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
        logit_soft_cap=None,
    ) -> NamedArray:
        ss_args = _SS_ARGS

        # Trace-time guard: mirror the mg-2 silent-fallback hardening. If
        # TOMAT_SS_MODE=1 is set but configure_ss() was never called (or was
        # called with None), raise loudly — silent fallback to plain AR would
        # waste a multi-hour TPU run training the same model we already have.
        if ss_args is None:
            print("[ss] compute_next_token_loss: _SS_ARGS is None")
            if os.environ.get("TOMAT_SS_MODE") == "1":
                raise RuntimeError(
                    "Qwen3SSLMHeadModel.compute_next_token_loss called but "
                    "_SS_ARGS is None despite TOMAT_SS_MODE=1. "
                    "configure_ss() must be called before training starts "
                    "(and survive the JIT trace). Silent fallback would "
                    "train as plain AR — the mg-2-style regression we're "
                    "hardening against."
                )
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        # Eval path (key is None): SS is skipped — eval is teacher-forced by
        # construction. Fall through to the parent density-aware loss with the
        # original tokens.
        if key is None:
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        # Trace-time confirmation print — fires once per JIT trace.
        print(f"[ss] compute_next_token_loss TRACE: sampler={ss_args.sampler!r} "
              f"eps_max={ss_args.eps_max} density_range="
              f"[{ss_args.density_lo}, {ss_args.density_hi})")

        # Split RNG: one half drives the pre-forward (in case the model has
        # dropout-style stochastic ops; for plain Qwen3 this is a no-op but
        # keeps the contract right), the other half drives mixing.
        key_pre, key_mix = jax.random.split(key, 2)

        # --- Pre-forward (stop_gradient) -----------------------------------
        # Get the model's own one-step-ahead predictions on the GT inputs.
        # stop_gradient on the logits ensures the pre-forward contributes
        # NO gradients — its sole purpose is to sample replacement tokens.
        pre_act = self.activations(example.tokens, example.attn_mask, key=key_pre)
        if isinstance(pre_act, tuple):
            pre_act, _ = pre_act
        pre_head = self.get_lm_head()
        pre_logits = hax.dot(pre_act, pre_head, axis=self.Embed)
        pre_logits_sg = jax.lax.stop_gradient(pre_logits)

        # --- Mix GT inputs with predicted density tokens -------------------
        mixed_tokens = _jax_apply_ss(
            original_tokens=example.tokens,
            pre_logits=pre_logits_sg,
            args=ss_args,
            key=key_mix,
        )

        mixed_example = LmExample(
            tokens=mixed_tokens,
            loss_weight=example.loss_weight,
            attn_mask=example.attn_mask,
        )

        # --- Real forward on the mixed input → density-aware loss ----------
        # Reuse the parent density-aware loss path; it shifts targets via
        # roll(-1) on the *original* tokens (passed as `example.tokens`)
        # which would normally equal the input. Here we want the loss graded
        # against the GROUND-TRUTH targets, not the model's own predictions,
        # so we compute logits on the mixed input but pass the ORIGINAL
        # tokens as the `input_tokens` arg (which `density_aware_loss` rolls
        # for the target sequence).
        args = _DENSITY_LOSS_ARGS
        if args is None or args.weight == 0.0:
            # Latent bug per task #155: the original early-return passed
            # `mixed_example` to the parent CE loss, which uses
            # `mixed_example.tokens` for BOTH the forward AND the
            # `roll(-1)`-derived targets — so the SS objective became "predict
            # your own next sample" instead of "predict GT next token". The
            # correct fix splits input (mixed) from targets (original) but the
            # parent's CE loss path doesn't support that without a refactor.
            # Until that lands, fail fast rather than silently misbehave.
            raise NotImplementedError(
                "SS with TOMAT_DENSITY_L1_WEIGHT=0 (or no density-loss args) "
                "is unsupported — the parent CE-loss path uses the mixed "
                "input as its own target, defeating the SS objective. Either "
                "set TOMAT_DENSITY_L1_WEIGHT>0 (use density-aware loss with "
                "correct input/target split, the path below) or implement an "
                "inline CE that takes mixed input + original targets."
            )

        activations = self.activations(mixed_example.tokens, mixed_example.attn_mask, key=key)
        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations
        head = self.get_lm_head()
        logits = hax.dot(activations, head, axis=self.Embed)

        loss = density_aware_loss(
            Pos=self.Pos,
            Vocab=self.Vocab,
            logits=logits,
            input_tokens=example.tokens,   # ORIGINAL (GT) tokens — provides targets
            loss_weight=example.loss_weight,
            args=args,
            reduction=reduction,
        )
        return loss + aux_loss


# Module-level state for SS (set once at training-script startup).
_SS_ARGS: Optional[SSArgs] = None


def configure_ss(args: "SSArgs | None") -> None:
    global _SS_ARGS
    _SS_ARGS = args


# ---------------------------------------------------------------------------
# Spec 34 F1 — sinusoidal continuous-xyz atom embedding
# ---------------------------------------------------------------------------
#
# F1 layout (see `tomat.tokenizers.patch_v3` module docstring):
#   - Each atom contributes ONE preamble token (the atom-type id).
#   - Per-atom continuous (x, y, z) coords are carried as a side array
#     `atom_xyz: float32[Pos, 3]` (NaN at non-atom positions).
#
# Model side: we add a learned linear projection of a NeRF-style sinusoidal
# encoding of (x, y, z) to the token embedding at atom-token positions. The
# rest of the Qwen3 stack is unchanged.
#
# Atom-token detection uses the [ATOM_OFFSET, ATOM_END) id range from
# `tomat.tokenizers.patch.SPECIAL_TOKENS`. The model receives that range as
# a config field (no import of tomat from inside the model module, so the
# levanter/JIT path doesn't depend on tomat).
#
# JIT-traceability: `atom_xyz` enters `activations` as an explicit kwarg
# alongside `input_ids`/`attn_mask`. Callers that don't have it (e.g. eval
# paths that haven't been F1-plumbed yet) can omit it — `activations` falls
# back to the standard embedding lookup.
#
# Loader plumbing: threading `atom_xyz` from a parquet column through the
# Levanter `LmExample` (whose fields are frozen at the library layer) to
# `compute_next_token_loss` requires subclassing `LmExample` (or replacing
# the `PrebuiltLmDataset` adapter). That work is deferred — this module
# provides only the model-side machinery; see spec 34 for the broader
# integration plan.


@dataclass(frozen=True)
class F1Args:
    """Spec-34 F1 configuration.

    `num_freqs` (K): number of log-spaced frequencies in the sinusoidal
    encoding `γ`. Output dim of γ is 6K (cos+sin per axis).
    `atom_token_lo` / `atom_token_hi`: half-open [lo, hi) id range that
    identifies atom-type tokens. Defaults match `tomat.tokenizers.patch`
    (ATOM_OFFSET=20, ATOM_END=138).
    `coord_frame`: "fractional" (default — matches v3 tokenizer's
    translated-frac convention) or "cartesian" (caller pre-multiplies
    fractional coords by the lattice).
    """

    num_freqs: int = 10
    atom_token_lo: int = 20   # ATOM_OFFSET in tomat.tokenizers.patch
    atom_token_hi: int = 138  # ATOM_END
    coord_frame: str = "fractional"


def f1_sinusoidal_embed(xyz: jax.Array, num_freqs: int) -> jax.Array:
    """NeRF-style sinusoidal positional encoding.

    Input: `xyz` of shape (..., 3) — coords in [0, 1) (fractional) or any
    Cartesian range (works as long as it's consistent within a run).
    Output: array of shape (..., 6 * num_freqs).

    For each axis a ∈ {x, y, z} and frequency k ∈ {0, …, K-1}:
        γ_{a, k}(p) = [sin(2^k · π · p_a), cos(2^k · π · p_a)]
    Concatenated across (a, k, {sin, cos}).
    """
    # `freqs` shape (K,)
    freqs = jnp.pi * (2.0 ** jnp.arange(num_freqs, dtype=xyz.dtype))
    # `scaled` shape (..., 3, K)
    scaled = xyz[..., None] * freqs
    sin = jnp.sin(scaled)
    cos = jnp.cos(scaled)
    # Concatenate sin/cos along the freq axis → (..., 3, 2K),
    # then flatten the last two dims into 6K.
    combined = jnp.concatenate([sin, cos], axis=-1)
    out_shape = xyz.shape[:-1] + (3 * 2 * num_freqs,)
    return combined.reshape(out_shape)


@dataclass(frozen=True)
class Qwen3F1Config(Qwen3Config):
    """Qwen3Config whose `model_type` is `Qwen3F1LMHeadModel`.

    Spec 34 F1: 1 token/atom + sinusoidal continuous-xyz at the embed layer.

    Adds F1-specific knobs as config fields so `config.build(Vocab, key)`
    (which Levanter calls via `LmConfig.build → model_type.init(Vocab,
    config, key=...)`) can thread them into `Qwen3F1LMHeadModel.init`.
    The `build` override below is what wires these through.
    """

    f1_num_freqs: int = 10
    f1_atom_token_lo: int = 20   # ATOM_OFFSET in tomat.tokenizers.patch
    f1_atom_token_hi: int = 138  # ATOM_END
    f1_coord_frame: str = "fractional"

    @property  # type: ignore[override]
    def model_type(self):
        return Qwen3F1LMHeadModel

    def build(self, Vocab, *, key):
        # Translate config-time F1 settings → runtime `F1Args` for `init`.
        f1_args = F1Args(
            num_freqs=self.f1_num_freqs,
            atom_token_lo=self.f1_atom_token_lo,
            atom_token_hi=self.f1_atom_token_hi,
            coord_frame=self.f1_coord_frame,
        )
        return Qwen3F1LMHeadModel.init(Vocab, self, key=key, f1_args=f1_args)


class Qwen3F1LMHeadModel(Qwen3DensityLMHeadModel):
    """Spec-34 F1 model: standard Qwen3 + learned linear projection of a
    sinusoidal encoding of per-atom (x, y, z), added to the atom-token
    embedding at the embed layer.

    Extra field (vs `Qwen3DensityLMHeadModel`):
        `f1_atom_proj`: `hnn.Linear(In=PosFeat=6K, Out=Embed, use_bias=False)`

    Forward path (override of `activations`):
        1. Look up token embeddings the usual way (`self.embeddings.embed`).
        2. If `atom_xyz` is None: return as-is (F1-OFF fallback — model
           weights are still well-defined, just unused).
        3. Compute γ(atom_xyz) on the *whole* `Pos` axis (NaN at non-atom
           positions → finite output via `jnp.where(is_atom, xyz, 0.0)`
           before γ is evaluated; γ(0)=cos(0)=1 leaks a constant but is
           masked out by `is_atom` below).
        4. Project to `Embed` and add at atom-token positions only.

    Inherits `compute_next_token_loss` from `Qwen3DensityLMHeadModel`. The
    density-loss path already calls `self.activations(...)` explicitly; F1
    plumbing for *that* path requires passing `atom_xyz` through the
    `LmExample`-adjacent kwargs chain (deferred — see module-docstring note).

    See `Qwen3DensityLMHeadModel.init` for the `init` re-wrap necessitated by
    Levanter's parent `Qwen3LMHeadModel.init` hard-coding the return class.
    """

    f1_atom_proj: hnn.Linear
    f1_args: F1Args = eqx.field(static=True)

    @classmethod
    def init(cls, Vocab, config, *, key, f1_args: F1Args | None = None):
        # Allow callers to pass F1Args; default to a fresh F1Args() for
        # backward-compat with config-only call sites.
        f1_args = f1_args if f1_args is not None else F1Args()
        k_base, k_proj = jrandom.split(key, 2)
        base = Qwen3LMHeadModel.init(Vocab, config, key=k_base)
        PosFeat = Axis("pos_feat", 6 * f1_args.num_freqs)
        # Standard small-variance init; Linear.init in haliax handles fan-in
        # scaling. `use_bias=False` matches Llama/Qwen embed-projection style.
        f1_atom_proj = hnn.Linear.init(
            In=PosFeat,
            Out=config.Embed,
            key=k_proj,
            use_bias=False,
            out_first=True,
        )
        return cls(
            base.transformer,
            base.embeddings,
            base.lm_head,
            f1_atom_proj,
            f1_args,
        )

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
        atom_xyz: NamedArray | None = None,
    ) -> NamedArray:
        """F1 forward: standard embed + sinusoidal-xyz addition at atom positions.

        `atom_xyz` is expected to have axes (Pos, XYZ) where XYZ has size 3.
        NaN at non-atom positions is OK — we mask before the linear projection.
        If `atom_xyz` is None (e.g. eval paths not yet F1-plumbed), this
        reduces to standard Qwen3 forward.
        """
        x = self.embeddings.embed(input_ids)
        if atom_xyz is not None:
            x = x + self._f1_pos_addend(input_ids, atom_xyz)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def _f1_pos_addend(
        self,
        input_ids: NamedArray,
        atom_xyz: NamedArray,
    ) -> NamedArray:
        """Compute the additive sinusoidal-xyz embedding contribution.

        Returns a `NamedArray` shaped like `self.embeddings.embed(input_ids)`
        (i.e. {..., Pos, Embed}), zero at non-atom positions, equal to
        `f1_atom_proj(γ(xyz))` at atom positions.
        """
        f1 = self.f1_args
        # Atom-mask: True where token id ∈ [lo, hi).
        is_atom = hax.logical_and(
            input_ids >= f1.atom_token_lo,
            input_ids < f1.atom_token_hi,
        )  # axes: same as input_ids → {..., Pos}

        # Sanitize NaN at non-atom positions before applying γ (γ(NaN)=NaN
        # would taint the masked-zero contribution via fp arithmetic).
        # Bypass haliax for the elementwise jnp ops on the underlying array,
        # then rewrap.
        xyz_arr = atom_xyz.array  # shape (..., Pos, 3)
        xyz_safe = jnp.where(jnp.isnan(xyz_arr), 0.0, xyz_arr)
        gamma_arr = f1_sinusoidal_embed(xyz_safe, f1.num_freqs)
        # Wrap γ as a NamedArray with axes (..., Pos, PosFeat).
        pos_feat_axis = self.f1_atom_proj.In  # AxisSpec — could be tuple, but our init uses a single Axis
        # `self.f1_atom_proj.In` may be a single Axis or a tuple; project
        # safely by constructing a NamedArray with the right axes.
        new_axes = atom_xyz.axes[:-1] + (pos_feat_axis,)
        gamma = hax.named(gamma_arr, new_axes)
        # Project to Embed.
        proj = self.f1_atom_proj(gamma)  # axes (..., Pos, Embed)
        # Mask to atom positions only.
        mask = is_atom.astype(proj.dtype)
        return proj * mask

    def compute_next_token_loss(
        self,
        example,
        *,
        key=None,
        reduction=hax.mean,
        reduction_axis=None,
        logsumexp_weight=None,
        loss_dtype=jnp.float32,
        logit_soft_cap=None,
    ) -> NamedArray:
        """F1-aware loss dispatch.

        If `example` is an `F1LmExample` carrying a non-None `atom_xyz`,
        compute activations with the F1 path (sinusoidal-xyz addend at
        atom positions) and then re-use the standard CE/density loss tail.
        Otherwise fall through to the parent `Qwen3DensityLMHeadModel`
        path (which itself falls through to plain Qwen3 CE when no density
        loss args are configured).

        This override only kicks in when there's actual F1 data to thread;
        it preserves byte-for-byte parity with the parent loss when
        `atom_xyz` is None (the TOMAT_F1_MODE=0 back-compat guarantee at
        the loss layer).
        """
        # Late-import to avoid a forward-reference loop with the class
        # definition above (F1LmExample lives in this module).
        atom_xyz = getattr(example, "atom_xyz", None)
        if atom_xyz is None:
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

        # Forward with atom_xyz, then re-use the parent's logit→loss tail.
        activations = self.activations(
            example.tokens, example.attn_mask, key=key, atom_xyz=atom_xyz,
        )
        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations
        head = self.get_lm_head()
        logits = hax.dot(activations, head, axis=self.Embed)

        # Decide loss flavor based on whether a density-loss config is set.
        args = _DENSITY_LOSS_ARGS
        if args is not None and args.weight != 0.0:
            loss = density_aware_loss(
                Pos=self.Pos,
                Vocab=self.Vocab,
                logits=logits,
                input_tokens=example.tokens,
                loss_weight=example.loss_weight,
                args=args,
                reduction=reduction,
            )
            return loss + aux_loss

        # Plain CE tail (mirrors `LmHeadModel.compute_next_token_loss`
        # internals — we already paid for activations so we can't call
        # super() to recompute them; we also bypass `head` recomputation).
        from levanter.models.loss import maybe_fused_next_token_loss
        loss = maybe_fused_next_token_loss(
            self.Pos,
            self.Embed,
            self.Vocab,
            activations,
            head,
            example.tokens,
            loss_weight=example.loss_weight,
            reduction=reduction,
            reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            logit_soft_cap=logit_soft_cap,
        )
        return loss + aux_loss


class F1LmExample(LmExample):
    """`LmExample` subclass that carries a per-position `atom_xyz` sidecar.

    Spec 34 F1: `atom_xyz: NamedArray` with axes `(..., Pos, XYZ)` (XYZ
    size 3), float32, NaN at non-atom positions, finite at atom positions
    (translated fractional coords in the patch frame).

    The field is `Optional` so an `F1LmExample` with `atom_xyz=None`
    behaves like a plain `LmExample` (back-compat path).

    Levanter's `LmExample` is an `eqx.Module`, so subclassing is the
    sanctioned extension mechanism. Equinox `field(...)` is required for
    optional NamedArray fields because `eqx.Module` is frozen.

    NOTE: `LmExample.causal` (used by Levanter's collator) returns an
    instance of `LmExample`, not `cls(...)`, so we can't just call
    `F1LmExample.causal(...)`. The F1 loader plumbing constructs
    `F1LmExample` instances directly from `LmExample.causal`'s output.
    """

    atom_xyz: NamedArray | None = eqx.field(default=None)
