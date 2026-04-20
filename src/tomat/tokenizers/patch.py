"""Patch tokenizer: one training example = one ``P × P × P`` sub-cube of a
material's full-grid density, prefixed with the material's atomic inventory
and the patch's offset/shape within the parent grid.

Motivating the design (per the Betsy/Ryan sync, April 2026):

* Training on full-grid data without downsampling. Native rho_gga grid
  shapes vary from 40³ to ~256³; one structure's grid is too big for 8k
  context directly, so we cut patches.
* Atomic preamble + patch-offset let the model predict local density
  from crystal-wide context. Independent patches (no autoregressive
  inter-patch dependency) for the first run.
* PBC wrap at patch extraction means any anchor voxel is valid — natural
  data augmentation via different anchor offsets on the same structure.

Token layout
------------
::

    [BOS]
    [GRID_START]  ⟨nx⟩ ⟨ny⟩ ⟨nz⟩                          # grid shape (ints)
    [ATOMS_START] ⟨Z₁⟩ … ⟨Zₙ⟩ [ATOMS_END]
    [POS_START]   ⟨x₁ SE M0 M1⟩ ⟨y₁ …⟩ ⟨z₁ …⟩ …          # frac coords, 3-byte codec
    [POS_END]
    [SHAPE_START] ⟨P⟩ ⟨P⟩ ⟨P⟩                             # patch dims (ints)
    [OFFSET_START] ⟨ix⟩ ⟨iy⟩ ⟨iz⟩                         # patch anchor (ints)
    [DENS_START]  ⟨d₀ SE M⟩ ⟨d₁ SE M⟩ … ⟨d_{P³−1} SE M⟩   # density, 2-token codec
    [EOS]

Vocabulary (default codecs: positions = tomol 3-byte; density = 2-token 9+12)::

    specials          :    0 …    15
    atom Z=1..118     :   16 …   133
    int range 0..1023 :  134 … 1157      # grid dims, offsets, sizes share this
    position codec    : 1158 … 2181      # 512 SE + 256 M0 + 256 M1
    density codec     : 2182 … 6789      # 512 SE + 4096 M
    TOTAL             : 6790 tokens

At hidden=512 tied embeddings this is ~3.5 M params — small fraction of a
30 M transformer body. At patch ``P=14``, density payload = 14³ × 2 = 5488
tokens; preamble for a 20-atom structure ≈ 220 tokens; total ≈ 5.7k.
Fits in an 8k context with ~2 k headroom for 100-atom structures or
a bigger patch.

Not a :class:`DensityTokenizer` — this emits a flat token sequence for LLM
training, not a scheme with an encode/decode roundtrip against a full grid.
A patch's ``decode`` reconstructs just the patch region; whole-structure
reconstruction requires tiling multiple patches (out of scope for v1).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tomat.float_codec import FP16Codec

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


# ---- vocab layout ---------------------------------------------------------

SPECIAL_TOKENS = {
    "[PAD]":            0,
    "[BOS]":            1,
    "[EOS]":            2,
    "[ATOMS_START]":    3,
    "[ATOMS_END]":      4,
    "[POS_START]":      5,
    "[POS_END]":        6,
    "[GRID_START]":     7,
    "[GRID_END]":       8,
    "[SHAPE_START]":    9,
    "[SHAPE_END]":     10,
    "[OFFSET_START]":  11,
    "[OFFSET_END]":    12,
    "[DENS_START]":    13,
    "[DENS_END]":      14,
    "[NL]":            15,
}
N_SPECIALS = 16

ATOM_OFFSET = N_SPECIALS  # 16
MAX_ATOMIC_NUMBER = 118   # Z ≤ 118 (Oganesson) covers all real chemistry
ATOM_END = ATOM_OFFSET + MAX_ATOMIC_NUMBER  # 134

INT_OFFSET = ATOM_END     # 134
INT_VOCAB_SIZE = 1024     # grid dims / offsets / sizes fit in [0, 1024)
INT_END = INT_OFFSET + INT_VOCAB_SIZE  # 1158


@dataclass(frozen=True)
class PatchVocab:
    """Offsets and sizes for each vocab region; derived from the codec choice."""

    position_codec: FP16Codec
    density_codec: FP16Codec
    # Fractional-coord range for atomic positions (always [0, 1) in crystal
    # coords — ``log_min`` / ``log_max`` are codec params, not chemistry).
    position_log_min: float = -4.0  # 10⁻⁴ resolution on fractional coords
    position_log_max: float = 0.0

    @property
    def position_offset(self) -> int:
        return INT_END  # 1158

    @property
    def position_vocab_size(self) -> int:
        return sum(self.position_codec.signed_vocabs)

    @property
    def density_offset(self) -> int:
        return self.position_offset + self.position_vocab_size

    @property
    def density_vocab_size(self) -> int:
        return sum(self.density_codec.signed_vocabs)

    @property
    def total_vocab_size(self) -> int:
        return self.density_offset + self.density_vocab_size

    # ---- per-component token-ID encoders / decoders ----------------------

    def atom_token(self, z: int) -> int:
        if not 1 <= z <= MAX_ATOMIC_NUMBER:
            raise ValueError(f"Z={z} out of range [1, {MAX_ATOMIC_NUMBER}]")
        return ATOM_OFFSET + (z - 1)

    def int_token(self, n: int) -> int:
        if not 0 <= n < INT_VOCAB_SIZE:
            raise ValueError(f"int {n} out of range [0, {INT_VOCAB_SIZE})")
        return INT_OFFSET + n

    def position_tokens(self, coord: float) -> list[int]:
        """Encode one fractional coordinate ∈ [0, 1) to position-codec tokens."""
        comps = self.position_codec.encode_signed(np.asarray([coord], dtype=np.float64))[0]
        # Layout components into a flat token-id sequence.
        out: list[int] = []
        cum = 0
        for width, comp in zip(self.position_codec.signed_vocabs, comps, strict=True):
            out.append(self.position_offset + cum + int(comp))
            cum += width
        return out

    def density_tokens(self, value: float) -> list[int]:
        """Encode one density value to density-codec tokens."""
        comps = self.density_codec.encode_signed(np.asarray([value], dtype=np.float64))[0]
        out: list[int] = []
        cum = 0
        for width, comp in zip(self.density_codec.signed_vocabs, comps, strict=True):
            out.append(self.density_offset + cum + int(comp))
            cum += width
        return out


# ---- patch sampling + tokenizer ------------------------------------------


@dataclass(frozen=True)
class PatchSample:
    task_id: str
    offset: tuple[int, int, int]
    patch_shape: tuple[int, int, int]
    grid_shape: tuple[int, int, int]
    atomic_numbers: np.ndarray    # (N,) int
    frac_coords: np.ndarray       # (N, 3) float in [0, 1)
    patch_density: np.ndarray     # (Px, Py, Pz) float


@dataclass
class PatchTokenizer:
    patch_size: int = 14
    density_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.two_token_9_12(log_min=-4.13, log_max=4.97)
    )
    position_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.tomol_3byte(log_min=-4.0, log_max=0.0)
    )

    @property
    def vocab(self) -> PatchVocab:
        return PatchVocab(position_codec=self.position_codec, density_codec=self.density_codec)

    # ---- patch extraction -------------------------------------------------

    def extract_patch(
        self,
        density: np.ndarray,
        offset: tuple[int, int, int],
    ) -> np.ndarray:
        """Extract a ``(P, P, P)`` patch from ``density`` with PBC wrap."""
        P = self.patch_size
        ix, iy, iz = offset
        nx, ny, nz = density.shape
        # np.take with mode='wrap' handles negative / out-of-range indices.
        xs = np.arange(ix, ix + P) % nx
        ys = np.arange(iy, iy + P) % ny
        zs = np.arange(iz, iz + P) % nz
        return density[np.ix_(xs, ys, zs)]

    def make_sample(
        self,
        task_id: str,
        density: np.ndarray,
        structure: "Structure",
        offset: tuple[int, int, int],
    ) -> PatchSample:
        P = self.patch_size
        patch = self.extract_patch(density, offset)
        return PatchSample(
            task_id=task_id,
            offset=offset,
            patch_shape=(P, P, P),
            grid_shape=density.shape,  # type: ignore[arg-type]
            atomic_numbers=np.array([site.specie.Z for site in structure], dtype=np.int32),
            frac_coords=np.array([site.frac_coords for site in structure], dtype=np.float64) % 1.0,
            patch_density=patch,
        )

    def random_offsets(
        self,
        grid_shape: tuple[int, int, int],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return ``(n, 3)`` int offsets, uniformly over ``[0, grid_shape[i])``."""
        return np.stack([rng.integers(0, s, size=n) for s in grid_shape], axis=1)

    # ---- tokenization -----------------------------------------------------

    def tokenize(self, sample: PatchSample) -> list[int]:
        """Emit a flat token sequence for one ``PatchSample``."""
        vocab = self.vocab
        S = SPECIAL_TOKENS

        tokens: list[int] = [S["[BOS]"]]

        # Full grid shape
        tokens.append(S["[GRID_START]"])
        tokens.extend(vocab.int_token(int(n)) for n in sample.grid_shape)
        tokens.append(S["[GRID_END]"])

        # Atomic inventory
        tokens.append(S["[ATOMS_START]"])
        tokens.extend(vocab.atom_token(int(z)) for z in sample.atomic_numbers)
        tokens.append(S["[ATOMS_END]"])

        # Fractional positions (x, y, z per atom), all via the position codec
        tokens.append(S["[POS_START]"])
        for xyz in sample.frac_coords:
            for c in xyz:
                tokens.extend(vocab.position_tokens(float(c)))
        tokens.append(S["[POS_END]"])

        # Patch shape
        tokens.append(S["[SHAPE_START]"])
        tokens.extend(vocab.int_token(int(p)) for p in sample.patch_shape)
        tokens.append(S["[SHAPE_END]"])

        # Patch offset
        tokens.append(S["[OFFSET_START]"])
        tokens.extend(vocab.int_token(int(o)) for o in sample.offset)
        tokens.append(S["[OFFSET_END]"])

        # Density values (row-major flatten; decoder knows patch_shape)
        tokens.append(S["[DENS_START]"])
        flat = sample.patch_density.ravel().astype(np.float64)
        # Batch the codec call — much faster than per-voxel.
        comps = vocab.density_codec.encode_signed(flat)
        # Append each row's codec tokens with the right offset.
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
