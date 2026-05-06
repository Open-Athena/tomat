"""v3 patch tokenizer: per-patch translated atoms, drop SHAPE/OFFSET/HI.

Differences vs v2 (:mod:`tomat.tokenizers.patch`):

1. **Atoms are translated to the patch frame** before being tokenized.
   v2 emits global fractional coords (same for every patch from the
   same material); v3 subtracts ``(ix/nx, iy/ny, iz/nz)`` and re-mods
   1, so each patch carries its own atom positions in its own frame.
   This means the model doesn't have to combine OFFSET+POS to get the
   relative geometry — it's directly observable.

2. **No SHAPE / OFFSET / HI blocks** when the patch matches the
   default shape. With translated atoms, OFFSET+HI become irrelevant
   (the patch is at the origin of its own frame), and SHAPE is
   constant across the run. They get dropped from the preamble,
   freeing tokens for a larger density block.

3. **Fallback patch shape** for materials where the default ``P×P×P``
   density block won't fit in context: emit a SHAPE block (only) and
   use the smaller shape. The model sees one of two preamble
   variants — default (no SHAPE) or fallback (SHAPE present).

4. **Default P=19** (was 14 in v2), giving a 6859-token density block
   per patch. Combined with M=64 (configured at sampling time, not in
   the tokenizer itself), this raises voxel coverage per epoch from
   ~5% (v2) to ~10%.

Token layout
------------
Default::

    [BOS]
    [GRID_START]    nx ny nz [GRID_END]
    [LATTICE_START] qa qb qc qα qβ qγ [LATTICE_END]
    [ATOMS_START]   Z₁ … Zₙ [ATOMS_END]
    [POS_START]     ⟨translated frac coords for each atom⟩ [POS_END]
    [DENS_START]    P³ density tokens [DENS_END]
    [EOS]

Fallback (when ``patch_shape != (P,P,P)``)::

    [BOS] [GRID_START]…[LATTICE_START]…[ATOMS_START]…[POS_START]…
    [SHAPE_START]   Px Py Pz [SHAPE_END]    ← extra block
    [DENS_START]    Px*Py*Pz density tokens [DENS_END]
    [EOS]

Detokenize auto-detects fallback by checking for ``[SHAPE_START]``
between ``[POS_END]`` and ``[DENS_START]``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tomat.float_codec import FP16Codec
from tomat.tokenizers.patch import (
    LATTICE_ANGLE_RES_DEG,
    LATTICE_LENGTH_RES_A,
    PatchSample,
    PatchTokenizer,
    PatchVocab,
    SPECIAL_TOKENS,
    INT_OFFSET,
    INT_END,
    ATOM_OFFSET,
    ATOM_END,
)

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@dataclass
class PatchTokenizerV3(PatchTokenizer):
    """v3 patch tokenizer. See module docstring for layout details.

    Inherits ``extract_patch``, ``make_sample``, ``random_offsets``,
    ``vocab`` and codec config from v2. Overrides ``tokenize`` and
    ``detokenize``.
    """

    patch_size: int = 19
    # Default density codec is overridden to LMQ-style 1-token/voxel at the
    # call site; v3 doesn't bake in a specific density codec.
    density_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.two_token_9_12(log_min=-4.13, log_max=4.97),
    )
    position_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.tomol_3byte(log_min=-4.0, log_max=0.0),
    )

    @property
    def default_patch_shape(self) -> tuple[int, int, int]:
        return (self.patch_size, self.patch_size, self.patch_size)

    # ---- per-patch atom translation --------------------------------------

    def translate_frac_coords(
        self,
        frac_coords: np.ndarray,
        offset: tuple[int, int, int],
        grid_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Translate fractional coords into the patch's own frame.

        ``frac_coords`` is (N, 3) in [0, 1) (the global crystal frame).
        Patch's lower-corner offset in voxel indices is ``offset``;
        grid dims are ``grid_shape``. Returns (N, 3) in [0, 1) where
        the patch's lower corner is at (0, 0, 0) in fractional space.
        """
        nx, ny, nz = grid_shape
        ix, iy, iz = offset
        delta = np.array([ix / nx, iy / ny, iz / nz], dtype=np.float64)
        return (frac_coords - delta) % 1.0

    # ---- tokenization ----------------------------------------------------

    def tokenize(self, sample: PatchSample) -> list[int]:
        """Emit a v3 token sequence for ``sample``."""
        vocab = self.vocab
        S = SPECIAL_TOKENS

        is_fallback = sample.patch_shape != self.default_patch_shape

        tokens: list[int] = [S["[BOS]"]]

        # Grid shape (still emitted — the model needs it to interpret
        # lattice + atom positions).
        tokens.append(S["[GRID_START]"])
        tokens.extend(vocab.int_token(int(n)) for n in sample.grid_shape)
        tokens.append(S["[GRID_END]"])

        # Lattice
        tokens.append(S["[LATTICE_START]"])
        tokens.extend(vocab.lattice_tokens(sample.lattice))
        tokens.append(S["[LATTICE_END]"])

        # Atomic inventory
        tokens.append(S["[ATOMS_START]"])
        tokens.extend(vocab.atom_token(int(z)) for z in sample.atomic_numbers)
        tokens.append(S["[ATOMS_END]"])

        # Translated positions (patch frame)
        translated = self.translate_frac_coords(
            sample.frac_coords, sample.offset, sample.grid_shape,
        )
        tokens.append(S["[POS_START]"])
        for xyz in translated:
            for c in xyz:
                tokens.extend(vocab.position_tokens(float(c)))
        tokens.append(S["[POS_END]"])

        # SHAPE only when patch_shape differs from the default (fallback).
        if is_fallback:
            tokens.append(S["[SHAPE_START]"])
            tokens.extend(vocab.int_token(int(p)) for p in sample.patch_shape)
            tokens.append(S["[SHAPE_END]"])

        # Density
        tokens.append(S["[DENS_START]"])
        flat = sample.patch_density.ravel().astype(np.float64)
        comps = vocab.density_codec.encode_signed(flat)
        cum = 0
        offsets = []
        for width in vocab.density_codec.signed_vocabs:
            offsets.append(vocab.density_offset + cum)
            cum += width
        for row in comps:
            for o, c in zip(offsets, row, strict=True):
                tokens.append(o + int(c))
        tokens.append(S["[DENS_END]"])

        tokens.append(S["[EOS]"])
        return tokens

    # ---- detokenization --------------------------------------------------

    def detokenize(self, tokens: list[int] | np.ndarray) -> PatchSample:
        """Invert :meth:`tokenize`. ``offset`` is recovered as ``(0,0,0)``
        — the patch is in its own frame.

        Auto-detects fallback shape via presence of ``[SHAPE_START]``
        between ``[POS_END]`` and ``[DENS_START]``.
        """
        toks = list(tokens)
        vocab = self.vocab
        S = SPECIAL_TOKENS

        if not toks or toks[0] != S["[BOS]"] or toks[-1] != S["[EOS]"]:
            raise ValueError("expected sequence to be BOS-prefixed and EOS-suffixed")

        def find_block(open_tok: int, close_tok: int, start: int) -> tuple[int, int]:
            try:
                i = toks.index(open_tok, start)
            except ValueError as e:
                raise ValueError(f"missing open token {open_tok} after index {start}") from e
            try:
                j = toks.index(close_tok, i + 1)
            except ValueError as e:
                raise ValueError(f"missing close token {close_tok} after index {i}") from e
            return i + 1, j

        # GRID
        gi, gj = find_block(S["[GRID_START]"], S["[GRID_END]"], 1)
        grid_shape = tuple(self._decode_int(t) for t in toks[gi:gj])
        if len(grid_shape) != 3:
            raise ValueError(f"expected 3 grid dims, got {len(grid_shape)}")

        # LATTICE
        li, lj = find_block(S["[LATTICE_START]"], S["[LATTICE_END]"], gj + 1)
        lat_ints = [self._decode_int(t) for t in toks[li:lj]]
        if len(lat_ints) != 6:
            raise ValueError(f"expected 6 lattice params, got {len(lat_ints)}")
        lattice = (
            lat_ints[0] * LATTICE_LENGTH_RES_A,
            lat_ints[1] * LATTICE_LENGTH_RES_A,
            lat_ints[2] * LATTICE_LENGTH_RES_A,
            lat_ints[3] * LATTICE_ANGLE_RES_DEG,
            lat_ints[4] * LATTICE_ANGLE_RES_DEG,
            lat_ints[5] * LATTICE_ANGLE_RES_DEG,
        )

        # ATOMS
        ai, aj = find_block(S["[ATOMS_START]"], S["[ATOMS_END]"], lj + 1)
        atomic_numbers = np.array([self._decode_atom(t) for t in toks[ai:aj]], dtype=np.int32)

        # POSITIONS (in patch frame — caller may translate back to global if needed)
        pi, pj = find_block(S["[POS_START]"], S["[POS_END]"], aj + 1)
        pos_tokens = toks[pi:pj]
        coord_stride = vocab.position_codec.tokens_per_value_signed
        expected_pos = len(atomic_numbers) * 3 * coord_stride
        if len(pos_tokens) != expected_pos:
            raise ValueError(
                f"position block length {len(pos_tokens)} != expected {expected_pos}"
            )
        coords_flat = self._decode_codec(pos_tokens, vocab.position_codec, vocab.position_offset)
        frac_coords = np.array(coords_flat, dtype=np.float64).reshape(-1, 3)

        # SHAPE — present only when patch_shape != default.
        cursor = pj + 1
        if toks[cursor] == S["[SHAPE_START]"]:
            si, sj = find_block(S["[SHAPE_START]"], S["[SHAPE_END]"], cursor)
            patch_shape = tuple(self._decode_int(t) for t in toks[si:sj])
            if len(patch_shape) != 3:
                raise ValueError(f"expected 3 patch dims, got {len(patch_shape)}")
            cursor = sj + 1
        else:
            patch_shape = self.default_patch_shape

        # DENSITY
        di, dj = find_block(S["[DENS_START]"], S["[DENS_END]"], cursor)
        dens_tokens = toks[di:dj]
        dens_stride = vocab.density_codec.tokens_per_value_signed
        expected_dens = int(np.prod(patch_shape)) * dens_stride
        if len(dens_tokens) != expected_dens:
            raise ValueError(
                f"density block length {len(dens_tokens)} != expected {expected_dens}"
            )
        density_flat = self._decode_codec(dens_tokens, vocab.density_codec, vocab.density_offset)
        patch_density = np.array(density_flat).reshape(patch_shape).astype(np.float32)

        return PatchSample(
            task_id="",
            offset=(0, 0, 0),  # patch is in its own frame; no offset to recover
            patch_shape=patch_shape,
            grid_shape=grid_shape,
            lattice=lattice,
            atomic_numbers=atomic_numbers,
            frac_coords=frac_coords,
            patch_density=patch_density,
        )
