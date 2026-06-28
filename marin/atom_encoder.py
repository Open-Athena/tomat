# Copyright Open Athena
# SPDX-License-Identifier: Apache-2.0
"""Atom-position encoding strategies for tomat density LMs.

Factors the "how do atom positions enter the model" axis out of the
training-objective class hierarchy (Density / MaskGIT / SS). Each
training-objective class holds an `atom_encoder: AtomEncoder` field;
their `activations()` calls `self.atom_encoder.apply(...)` between the
token-embedding lookup and the transformer. The training objective is
agnostic to which encoding strategy is in use.

Spec 34 mapping:
    F0 → F0AtomEncoder (no-op; atom info is already discretely encoded
         in the input_ids stream as `[POS_START] + 3 xyz-bucket tokens
         + [POS_END]` per atom).
    F1 → F1AtomEncoder (1 token/atom + continuous xyz sidecar; sinusoidal
         encoding linearly projected to `Embed` and added at atom-token
         positions).
    F2 (3D RoPE), F3 (k-NN encoder) — future extensions to this interface.

The encoder is an `eqx.Module` so it can carry learned parameters
(`F1AtomEncoder.atom_proj`) and be JIT-traced alongside the parent model.
"""

from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray


# F1 atom-token id range — defaults match `tomat.tokenizers.patch.SPECIAL_TOKENS`
# (ATOM_OFFSET=20, ATOM_END=138). Kept here so the model code doesn't import
# tomat (levanter/JIT path stays decoupled from the local package).
_DEFAULT_ATOM_TOKEN_LO = 20
_DEFAULT_ATOM_TOKEN_HI = 138


def f1_sinusoidal_embed(xyz: jax.Array, num_freqs: int) -> jax.Array:
    """NeRF-style sinusoidal positional encoding for atom xyz coords.

    Input: `xyz` of shape (..., 3) — fractional or Cartesian, consistent
    within a run.
    Output: array of shape (..., 6 * num_freqs).

    For each axis a ∈ {x, y, z} and frequency k ∈ {0, …, K-1}:
        γ_{a, k}(p) = [sin(2^k · π · p_a), cos(2^k · π · p_a)]
    Concatenated across (a, k, {sin, cos}).
    """
    freqs = jnp.pi * (2.0 ** jnp.arange(num_freqs, dtype=xyz.dtype))
    scaled = xyz[..., None] * freqs                       # (..., 3, K)
    sin = jnp.sin(scaled)
    cos = jnp.cos(scaled)
    combined = jnp.concatenate([sin, cos], axis=-1)       # (..., 3, 2K)
    out_shape = xyz.shape[:-1] + (3 * 2 * num_freqs,)
    return combined.reshape(out_shape)


class AtomEncoder(eqx.Module):
    """Interface for atom-position encoding strategies.

    Implementations: `F0AtomEncoder` (no-op; the F0 baseline) and
    `F1AtomEncoder` (sinusoidal continuous xyz added at atom positions).
    Future: F2 (3D RoPE), F3 (k-NN encoder).

    Subclasses ARE eqx.Module subclasses themselves so they can hold
    parameters (F1's atom_proj weights) and survive Levanter's JIT path.
    """

    @abstractmethod
    def apply(
        self,
        embeddings: NamedArray,
        input_ids: NamedArray,
        atom_xyz: NamedArray | None,
    ) -> NamedArray:
        """Return possibly-modified token embeddings.

        Called by the parent model's `activations()` between
        `self.embeddings.embed(input_ids)` and the transformer stack.

        Args:
            embeddings: `(..., Pos, Embed)` — output of `self.embeddings.embed`.
            input_ids:  `(..., Pos)` — needed to detect atom-token positions.
            atom_xyz:   `(..., Pos, XYZ=3)` or None. NaN at non-atom positions.
                        None means "no atom sidecar available"; encoders
                        should fall back to a no-op when this happens (so
                        F1-trained models survive eval paths that haven't
                        been F1-plumbed yet, and F0-trained models trivially
                        ignore the field).
        """
        ...


class F0AtomEncoder(AtomEncoder):
    """No-op encoder: atom positions are already discretely encoded in input_ids.

    Has no parameters. Identity on the embeddings.
    """

    def apply(
        self,
        embeddings: NamedArray,
        input_ids: NamedArray,
        atom_xyz: NamedArray | None,
    ) -> NamedArray:
        return embeddings


@dataclass(frozen=True)
class F1Args:
    """Spec-34 F1 configuration."""
    num_freqs: int = 10
    atom_token_lo: int = _DEFAULT_ATOM_TOKEN_LO
    atom_token_hi: int = _DEFAULT_ATOM_TOKEN_HI
    coord_frame: str = "fractional"


class F1AtomEncoder(AtomEncoder):
    """Sinusoidal continuous xyz encoding added at atom-token positions.

    Parameters:
        atom_proj: `hnn.Linear(In=PosFeat=6K, Out=Embed, use_bias=False)`.
    """

    atom_proj: hnn.Linear
    args: F1Args = eqx.field(static=True)

    @classmethod
    def init(cls, Embed: Axis, *, key, args: F1Args | None = None) -> "F1AtomEncoder":
        args = args if args is not None else F1Args()
        PosFeat = Axis("pos_feat", 6 * args.num_freqs)
        atom_proj = hnn.Linear.init(
            In=PosFeat,
            Out=Embed,
            key=key,
            use_bias=False,
            out_first=True,
        )
        return cls(atom_proj, args)

    def apply(
        self,
        embeddings: NamedArray,
        input_ids: NamedArray,
        atom_xyz: NamedArray | None,
    ) -> NamedArray:
        if atom_xyz is None:
            # F1-trained model called via an eval path that lacks the
            # sidecar — fall back to plain embeddings. Model weights are
            # still well-defined (just unused). See spec 34 §"backward
            # compat at the loss layer".
            return embeddings
        return embeddings + self._addend(input_ids, atom_xyz)

    def _addend(
        self,
        input_ids: NamedArray,
        atom_xyz: NamedArray,
    ) -> NamedArray:
        """Compute the additive sinusoidal-xyz embedding contribution."""
        is_atom = hax.logical_and(
            input_ids >= self.args.atom_token_lo,
            input_ids <  self.args.atom_token_hi,
        )

        # γ(NaN) would taint the masked-zero contribution via fp; sanitize.
        xyz_arr = atom_xyz.array
        xyz_safe = jnp.where(jnp.isnan(xyz_arr), 0.0, xyz_arr)
        gamma_arr = f1_sinusoidal_embed(xyz_safe, self.args.num_freqs)

        # Wrap γ as a NamedArray with (..., Pos, PosFeat) axes, then project.
        new_axes = atom_xyz.axes[:-1] + (self.atom_proj.In,)
        gamma = hax.named(gamma_arr, new_axes)
        proj = self.atom_proj(gamma)                       # (..., Pos, Embed)
        mask = is_atom.astype(proj.dtype)
        return proj * mask


def build_atom_encoder(
    atom_encoding: str,
    Embed: Axis,
    *,
    key: jax.Array,
    f1_args: F1Args | None = None,
) -> AtomEncoder:
    """Construct the right `AtomEncoder` for a string config value.

    Used by model `init()` to dispatch on `config.atom_encoding`.
    `key` is only consumed when the encoder has learned parameters
    (F1+); F0 ignores it.
    """
    if atom_encoding == "f0":
        return F0AtomEncoder()
    if atom_encoding == "f1":
        return F1AtomEncoder.init(Embed=Embed, key=key, args=f1_args)
    raise ValueError(
        f"unknown atom_encoding={atom_encoding!r}; supported: 'f0', 'f1'"
    )
