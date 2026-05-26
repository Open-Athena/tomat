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
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

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
    `prior`:    mask-ratio sampling schedule — "cosine" | "uniform" | "high".
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
    prior: str = "cosine"        # mask-ratio schedule: "cosine" | "uniform" | "high"
    weight: float = 1.0          # λ multiplier on the EMD density term
    loss_type: str = "ce"        # "ce" or "ce_emd"


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
) -> MaskGITLossArgs:
    """Build `MaskGITLossArgs` from codec + config."""
    if loss_type not in ("ce", "ce_emd", "emd"):
        raise ValueError(f"loss_type must be 'ce'/'ce_emd'/'emd', got {loss_type!r}")
    if prior not in ("cosine", "uniform", "high"):
        raise ValueError(f"prior must be 'cosine'/'uniform'/'high', got {prior!r}")
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

        # Bidirectional attention (no causal mask).
        bidir_mask = AttentionMask(is_causal=False)
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

@dataclass(frozen=True)
class SSArgs:
    """Configuration for token-level scheduled sampling on density positions.

    The base loss is the existing density-aware AR loss; SS only mutates the
    *input* tokens before the real forward pass. The base `DensityLossArgs` is
    reused for the loss term itself — see `_SS_DENSITY_ARGS_REQUIRED` for why
    SS requires the density-loss path to be configured too.

    `density_lo` / `density_hi`:  inclusive/exclusive density vocab range
                                  (same convention as `DensityLossArgs`).
    `eps_max`:    Upper bound on the per-batch ε ~ Uniform(0, eps_max). ε is
                  the Bernoulli probability that each density-target input is
                  replaced with the model's own prediction at that step.
    `sampler`:    "median" (default, matches FR eval) | "argmax" | "sample".
                  All operate on the density-bin slice of the predicted-logits
                  distribution (not the full vocab), so they always return a
                  valid density token.
    """

    density_lo: int                  # inclusive start of density vocab range
    density_hi: int                  # exclusive end of density vocab range
    eps_max: float = 0.25            # ε ~ Uniform(0, eps_max) per batch
    sampler: str = "median"          # "median" | "argmax" | "sample"


def build_ss_args(
    *,
    density_offset: int,
    n_density_bins: int,
    eps_max: float = 0.25,
    sampler: str = "median",
) -> SSArgs:
    """Build SSArgs from codec config."""
    if sampler not in ("median", "argmax", "sample"):
        raise ValueError(f"sampler must be 'median'/'argmax'/'sample', got {sampler!r}")
    if not (0.0 <= eps_max <= 1.0):
        raise ValueError(f"eps_max must be in [0, 1], got {eps_max!r}")
    return SSArgs(
        density_lo=density_offset,
        density_hi=density_offset + n_density_bins,
        eps_max=float(eps_max),
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

    # Sample per-batch ε ~ Uniform(0, eps_max).
    key_eps, key_mask, key_sample = jax.random.split(key, 3)
    eps = jax.random.uniform(key_eps, shape=(), minval=0.0, maxval=args.eps_max)

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
            # SS without density loss → CE on plain vocab, mixed input.
            # Re-forward via the mixed example and let the parent compute CE.
            return Qwen3LMHeadModel.compute_next_token_loss(
                self,
                mixed_example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
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
