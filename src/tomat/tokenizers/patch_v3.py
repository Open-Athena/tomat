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

Atom-position encoding mode (spec 34, ``atom_encoding``)
-------------------------------------------------------
``f0`` (default, status quo): each atom contributes ~4 preamble tokens
— one atom-type token in the [ATOMS_START]…[ATOMS_END] block plus
``tokens_per_value_signed × 3`` discrete xyz-bucket tokens inside the
[POS_START]…[POS_END] block (3 tokens/axis for ``tomol_3byte``).

``f1`` (new, spec 34): each atom contributes **one** preamble token —
the atom-type token. The [POS_START]…[POS_END] block is dropped
entirely. The atom's continuous fractional ``(x, y, z)`` is carried
as a parallel side array ``atom_xyz``, aligned with the token stream
(NaN at non-atom positions). The model is expected to consume the
sidecar at the embedding layer (see ``Qwen3F1LMHeadModel``).

Token layout (f0)
-----------------
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

Token layout (f1)
-----------------
Default::

    [BOS]
    [GRID_START]    nx ny nz [GRID_END]
    [LATTICE_START] qa qb qc qα qβ qγ [LATTICE_END]
    [ATOMS_START]   Z₁ … Zₙ [ATOMS_END]
    [DENS_START]    P³ density tokens [DENS_END]
    [EOS]

Side array ``atom_xyz``: ``float32[L, 3]`` (same L as the row's
``input_ids`` length, NaN at non-atom positions, fractional coords in
the **patch frame**, matching ``f0``'s translated convention).
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
    # Spec 34: atom-position encoding scheme.
    #   "f0" — discrete xyz buckets inside [POS_START]…[POS_END] (status quo).
    #   "f1" — 1 token/atom + sidecar ``atom_xyz`` continuous coords (no POS block).
    atom_encoding: str = "f0"

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
        """Emit a v3 token sequence for ``sample``.

        For ``atom_encoding="f1"``, the per-atom xyz sidecar is dropped;
        callers wanting both tokens and xyz should use
        :meth:`tokenize_with_xyz` instead.
        """
        tokens, _xyz = self.tokenize_with_xyz(sample)
        return tokens

    def tokenize_with_xyz(
        self, sample: PatchSample,
    ) -> tuple[list[int], np.ndarray]:
        """Emit a v3 token sequence + aligned ``atom_xyz`` sidecar.

        Returns ``(tokens, atom_xyz)`` where ``atom_xyz`` is
        ``float32[len(tokens), 3]`` with NaN at non-atom positions and
        the patch-frame translated fractional coords at atom positions.

        For ``atom_encoding="f0"``, ``atom_xyz`` is still populated at
        the atom-token positions (callers can ignore it). The
        ``f0`` token stream is byte-identical to the original
        :meth:`tokenize`.

        For ``atom_encoding="f1"``, the [POS_START]…[POS_END] block is
        omitted; the per-atom continuous xyz is carried only via
        ``atom_xyz``.
        """
        vocab = self.vocab
        S = SPECIAL_TOKENS

        if self.atom_encoding not in {"f0", "f1"}:
            raise ValueError(
                f"atom_encoding={self.atom_encoding!r} not in {{'f0', 'f1'}}"
            )

        is_fallback = sample.patch_shape != self.default_patch_shape

        # Translated positions (patch frame) — computed for both f0 and f1.
        # f0 emits them as discrete tokens; f1 stores them on the sidecar.
        translated = self.translate_frac_coords(
            sample.frac_coords, sample.offset, sample.grid_shape,
        )
        n_atoms = int(translated.shape[0])

        # Track xyz per token-index. NaN sentinel marks non-atom positions.
        xyz_by_pos: list[tuple[float, float, float] | None] = []

        def _emit(tok: int, xyz: tuple[float, float, float] | None = None) -> None:
            tokens.append(tok)
            xyz_by_pos.append(xyz)

        tokens: list[int] = []

        _emit(S["[BOS]"])

        # Grid shape (still emitted — the model needs it to interpret
        # lattice + atom positions).
        _emit(S["[GRID_START]"])
        for n in sample.grid_shape:
            _emit(vocab.int_token(int(n)))
        _emit(S["[GRID_END]"])

        # Lattice
        _emit(S["[LATTICE_START]"])
        for t in vocab.lattice_tokens(sample.lattice):
            _emit(t)
        _emit(S["[LATTICE_END]"])

        # Atomic inventory — F1 stamps xyz onto each atom-type token.
        _emit(S["[ATOMS_START]"])
        for i, z in enumerate(sample.atomic_numbers):
            xyz = tuple(float(c) for c in translated[i])
            _emit(vocab.atom_token(int(z)), xyz)
        _emit(S["[ATOMS_END]"])

        # F0: emit the discrete-bucket [POS_START]…[POS_END] block.
        # F1: skip; per-atom xyz lives on the sidecar at the atom-token positions.
        if self.atom_encoding == "f0":
            _emit(S["[POS_START]"])
            for xyz in translated:
                for c in xyz:
                    for t in vocab.position_tokens(float(c)):
                        _emit(t)
            _emit(S["[POS_END]"])

        # SHAPE only when patch_shape differs from the default (fallback).
        if is_fallback:
            _emit(S["[SHAPE_START]"])
            for p in sample.patch_shape:
                _emit(vocab.int_token(int(p)))
            _emit(S["[SHAPE_END]"])

        # Density
        _emit(S["[DENS_START]"])
        flat = sample.patch_density.ravel().astype(np.float64)
        comps = vocab.density_codec.encode_signed(flat)
        cum = 0
        offsets = []
        for width in vocab.density_codec.signed_vocabs:
            offsets.append(vocab.density_offset + cum)
            cum += width
        for row in comps:
            for o, c in zip(offsets, row, strict=True):
                _emit(o + int(c))
        _emit(S["[DENS_END]"])

        _emit(S["[EOS]"])

        # Build aligned NaN-default xyz array. Use the special value
        # ``np.nan`` as the sentinel so downstream collation can mask.
        atom_xyz = np.full((len(tokens), 3), np.nan, dtype=np.float32)
        for i, xyz in enumerate(xyz_by_pos):
            if xyz is not None:
                atom_xyz[i] = xyz
        assert (
            int(np.isfinite(atom_xyz).all(axis=1).sum()) == n_atoms
        ), "atom_xyz row count mismatch with atomic_numbers"
        return tokens, atom_xyz

    # ---- detokenization --------------------------------------------------

    def detokenize(
        self,
        tokens: list[int] | np.ndarray,
        atom_xyz: np.ndarray | None = None,
    ) -> PatchSample:
        """Invert :meth:`tokenize`. ``offset`` is recovered as ``(0,0,0)``
        — the patch is in its own frame.

        Auto-detects fallback shape via presence of ``[SHAPE_START]``
        between ``[POS_END]`` (f0) or ``[ATOMS_END]`` (f1) and ``[DENS_START]``.

        For ``atom_encoding="f1"``, the caller must pass the row's
        ``atom_xyz`` sidecar (NaN-padded float32 array shape ``(L, 3)``)
        so per-atom positions can be recovered.
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

        # POSITIONS: f0 reads them from [POS_START]…[POS_END]; f1 reads them
        # from the ``atom_xyz`` sidecar at the atom-token positions.
        if self.atom_encoding == "f1":
            if atom_xyz is None:
                raise ValueError(
                    "atom_encoding='f1' detokenize requires atom_xyz sidecar"
                )
            atom_xyz = np.asarray(atom_xyz)
            if atom_xyz.shape != (len(toks), 3):
                raise ValueError(
                    f"atom_xyz shape {atom_xyz.shape} != expected ({len(toks)}, 3)"
                )
            # Pull xyz at the atom-token positions (indices ai..aj).
            frac_coords = np.asarray(atom_xyz[ai:aj], dtype=np.float64)
            if frac_coords.shape != (len(atomic_numbers), 3) or not np.isfinite(
                frac_coords
            ).all():
                raise ValueError(
                    f"atom_xyz at atom-token slice [{ai}:{aj}] not finite or "
                    f"shape mismatch: got {frac_coords.shape}"
                )
            cursor = aj + 1
        else:
            pi, pj = find_block(S["[POS_START]"], S["[POS_END]"], aj + 1)
            pos_tokens = toks[pi:pj]
            coord_stride = vocab.position_codec.tokens_per_value_signed
            expected_pos = len(atomic_numbers) * 3 * coord_stride
            if len(pos_tokens) != expected_pos:
                raise ValueError(
                    f"position block length {len(pos_tokens)} != expected {expected_pos}"
                )
            coords_flat = self._decode_codec(
                pos_tokens, vocab.position_codec, vocab.position_offset,
            )
            frac_coords = np.array(coords_flat, dtype=np.float64).reshape(-1, 3)
            cursor = pj + 1

        # SHAPE — present only when patch_shape != default.
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
