"""Ball-shaped patch tokenizer (ablation vs cube `patch.py`).

Emits sequences for the **ball** patch variant described in
`specs/10-ball-patches.md`: a patch is the set of voxels within integer
squared-radius R² of a chosen center voxel, traversed in canonical
radial order (sort by r² then lex tiebreak).

Design notes:

* This tokenizer has its **own vocabulary** (24 specials instead of 18 —
  adds RADIUS / CENTER / BOUNDS blocks), so ball-tokenized parquets are
  not cross-compatible with cube models, and vice versa. That's
  intentional for clean ablation: train one model per tokenizer, compare
  final losses at matched voxel-count-per-patch.
* All per-element encoders (atom Z, int, position codec, density codec)
  are numerically identical to `patch.PatchVocab` — just shifted by the
  6 extra specials at the head of the vocab.
* Balls near the grid edge are **clipped** to the in-grid region. The
  BOUNDS block records the actual extent so the decoder can reconstruct
  which voxels were emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from tomat.float_codec import FP16Codec

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


# ---- vocab layout ---------------------------------------------------------

SPECIAL_TOKENS = {
    "[PAD]":              0,
    "[BOS]":              1,
    "[EOS]":              2,
    "[ATOMS_START]":      3,
    "[ATOMS_END]":        4,
    "[POS_START]":        5,
    "[POS_END]":          6,
    "[GRID_START]":       7,
    "[GRID_END]":         8,
    "[RADIUS_START]":     9,
    "[RADIUS_END]":      10,
    "[CENTER_START]":    11,
    "[CENTER_END]":      12,
    "[BOUNDS_START]":    13,
    "[BOUNDS_END]":      14,
    "[DENS_START]":      15,
    "[DENS_END]":        16,
    "[NL]":              17,
}
N_SPECIALS = 18

ATOM_OFFSET = N_SPECIALS
MAX_ATOMIC_NUMBER = 118
ATOM_END = ATOM_OFFSET + MAX_ATOMIC_NUMBER  # 136

INT_OFFSET = ATOM_END
INT_VOCAB_SIZE = 1024
INT_END = INT_OFFSET + INT_VOCAB_SIZE  # 1160


@dataclass(frozen=True)
class BallVocab:
    """Mirrors `patch.PatchVocab`; only `total_vocab_size` differs (same specials
    layout since we kept SPECIAL_TOKENS the same size as cube's — we just
    relabeled some slots for ball semantics)."""

    position_codec: FP16Codec
    density_codec: FP16Codec
    position_log_min: float = -4.0
    position_log_max: float = 0.0

    @property
    def position_offset(self) -> int:
        return INT_END

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

    def atom_token(self, z: int) -> int:
        if not 1 <= z <= MAX_ATOMIC_NUMBER:
            raise ValueError(f"Z={z} out of range [1, {MAX_ATOMIC_NUMBER}]")
        return ATOM_OFFSET + (z - 1)

    def int_token(self, n: int) -> int:
        if not 0 <= n < INT_VOCAB_SIZE:
            raise ValueError(f"int {n} out of range [0, {INT_VOCAB_SIZE})")
        return INT_OFFSET + n

    def position_tokens(self, coord: float) -> list[int]:
        comps = self.position_codec.encode_signed(np.asarray([coord], dtype=np.float64))[0]
        out: list[int] = []
        cum = 0
        for width, comp in zip(self.position_codec.signed_vocabs, comps, strict=True):
            out.append(self.position_offset + cum + int(comp))
            cum += width
        return out


# ---- ball geometry (shared cache) -----------------------------------------


@lru_cache(maxsize=32)
def ball_offsets(r2_max: int) -> np.ndarray:
    """Integer voxel-offsets (dx, dy, dz) inside a ball of squared-radius r2_max.

    Returned array is sorted in canonical order: primary by r² ascending,
    then lexicographic by (dy, dx, dz). Cached since the same r2_max is
    reused across every patch of every material.
    """
    R = int(np.ceil(np.sqrt(r2_max))) + 1
    coords = []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= r2_max:
                    coords.append((r2, dy, dx, dz))
    coords.sort()
    return np.array([(dx, dy, dz) for (_, dy, dx, dz) in coords], dtype=np.int32)


# ---- sample + tokenizer ---------------------------------------------------


@dataclass(frozen=True)
class BallSample:
    task_id: str
    center: tuple[int, int, int]            # voxel coord of ball center
    r2_max: int                             # squared-radius threshold
    grid_shape: tuple[int, int, int]
    bounds: tuple[int, int, int, int, int, int]  # x_lo, x_hi, y_lo, y_hi, z_lo, z_hi (inclusive; clipped)
    atomic_numbers: np.ndarray              # (N,) int
    frac_coords: np.ndarray                 # (N, 3) float in [0, 1)
    ball_density: np.ndarray                # (K,) float — density values in canonical ball order, clipped


@dataclass
class BallTokenizer:
    """Ball patch tokenizer. Voxel-count-matched to cube P=14 at r2_max=75,
    or to cube P=15 at r2_max=86 (see `specs/10-ball-patches.md`)."""

    r2_max: int = 75  # default matches cube P=14 (2,777 voxels vs 2,744)
    density_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.two_token_9_12(log_min=-4.13, log_max=4.97)
    )
    position_codec: FP16Codec = field(
        default_factory=lambda: FP16Codec.tomol_3byte(log_min=-4.0, log_max=0.0)
    )

    @property
    def vocab(self) -> BallVocab:
        return BallVocab(position_codec=self.position_codec, density_codec=self.density_codec)

    # ---- ball extraction -------------------------------------------------

    def extract_ball(
        self,
        density: np.ndarray,
        center: tuple[int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
        """Extract density values at all voxels within r²≤r2_max of `center`.

        Edge-clipped (no PBC wrap — balls that spill outside the grid just
        lose those voxels). Returns `(values, bounds)` where `values` is
        the (K,) density array in canonical ball order (sorted by r², then
        dy, dx, dz) and `bounds = (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)`
        is the inclusive bounding box of actually-emitted voxels.
        """
        offsets = ball_offsets(self.r2_max)
        cx, cy, cz = center
        nx, ny, nz = density.shape

        abs_coords = offsets + np.array([cx, cy, cz], dtype=np.int32)
        in_grid = (
            (abs_coords[:, 0] >= 0) & (abs_coords[:, 0] < nx)
            & (abs_coords[:, 1] >= 0) & (abs_coords[:, 1] < ny)
            & (abs_coords[:, 2] >= 0) & (abs_coords[:, 2] < nz)
        )
        kept = abs_coords[in_grid]
        values = density[kept[:, 0], kept[:, 1], kept[:, 2]].astype(np.float64)

        if len(kept) == 0:
            # Shouldn't happen if center is in-grid, but guard defensively.
            raise ValueError(f"empty ball at center={center} grid_shape=({nx},{ny},{nz})")

        bounds = (
            int(kept[:, 0].min()), int(kept[:, 0].max()),
            int(kept[:, 1].min()), int(kept[:, 1].max()),
            int(kept[:, 2].min()), int(kept[:, 2].max()),
        )
        return values, bounds

    def make_sample(
        self,
        task_id: str,
        density: np.ndarray,
        structure: "Structure",
        center: tuple[int, int, int],
    ) -> BallSample:
        values, bounds = self.extract_ball(density, center)
        return BallSample(
            task_id=task_id,
            center=center,
            r2_max=self.r2_max,
            grid_shape=density.shape,  # type: ignore[arg-type]
            bounds=bounds,
            atomic_numbers=np.array([site.specie.Z for site in structure], dtype=np.int32),
            frac_coords=np.array([site.frac_coords for site in structure], dtype=np.float64) % 1.0,
            ball_density=values,
        )

    def random_centers(
        self,
        grid_shape: tuple[int, int, int],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """(n, 3) uniform-random ball centers over the full grid interior.

        Centers can be any voxel (balls near edges get clipped). Matches
        cube tokenizer's `random_offsets` semantics for direct ablation.
        """
        return np.stack([rng.integers(0, s, size=n) for s in grid_shape], axis=1)

    # ---- tokenization -----------------------------------------------------

    def tokenize(self, sample: BallSample) -> list[int]:
        vocab = self.vocab
        S = SPECIAL_TOKENS

        tokens: list[int] = [S["[BOS]"]]

        tokens.append(S["[GRID_START]"])
        tokens.extend(vocab.int_token(int(n)) for n in sample.grid_shape)
        tokens.append(S["[GRID_END]"])

        tokens.append(S["[ATOMS_START]"])
        tokens.extend(vocab.atom_token(int(z)) for z in sample.atomic_numbers)
        tokens.append(S["[ATOMS_END]"])

        tokens.append(S["[POS_START]"])
        for xyz in sample.frac_coords:
            for c in xyz:
                tokens.extend(vocab.position_tokens(float(c)))
        tokens.append(S["[POS_END]"])

        tokens.append(S["[RADIUS_START]"])
        tokens.append(vocab.int_token(int(sample.r2_max)))
        tokens.append(S["[RADIUS_END]"])

        tokens.append(S["[CENTER_START]"])
        tokens.extend(vocab.int_token(int(c)) for c in sample.center)
        tokens.append(S["[CENTER_END]"])

        tokens.append(S["[BOUNDS_START]"])
        tokens.extend(vocab.int_token(int(b)) for b in sample.bounds)
        tokens.append(S["[BOUNDS_END]"])

        tokens.append(S["[DENS_START]"])
        flat = sample.ball_density.astype(np.float64)
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

    def _decode_int(self, t: int) -> int:
        if not INT_OFFSET <= t < INT_END:
            raise ValueError(f"token {t} not in int range [{INT_OFFSET}, {INT_END})")
        return int(t - INT_OFFSET)

    def detokenize(self, tokens: list[int] | np.ndarray) -> BallSample:
        toks = list(tokens)
        S = SPECIAL_TOKENS

        if not toks or toks[0] != S["[BOS]"] or toks[-1] != S["[EOS]"]:
            raise ValueError("expected BOS-prefixed and EOS-suffixed sequence")

        def find_block(open_tok: int, close_tok: int, start: int) -> tuple[int, int]:
            try:
                i = toks.index(open_tok, start)
                j = toks.index(close_tok, i + 1)
            except ValueError as e:
                raise ValueError(f"missing {open_tok}/{close_tok} block after idx {start}") from e
            return i + 1, j

        gi, gj = find_block(S["[GRID_START]"], S["[GRID_END]"], 1)
        grid_shape = tuple(self._decode_int(t) for t in toks[gi:gj])
        if len(grid_shape) != 3:
            raise ValueError(f"expected 3 grid dims, got {len(grid_shape)}")

        ai, aj = find_block(S["[ATOMS_START]"], S["[ATOMS_END]"], gj + 1)
        atomic_numbers = np.array(
            [t - ATOM_OFFSET + 1 for t in toks[ai:aj]], dtype=np.int32
        )

        pi, pj = find_block(S["[POS_START]"], S["[POS_END]"], aj + 1)
        # position tokens decode not implemented here; skip for now (only needed
        # for full roundtrip tests — tokenize is what tokenize_patches.py needs).
        n_pos_tokens = sum(self.position_codec.signed_vocabs) * 3  # per atom
        n_atoms = len(atomic_numbers)
        frac_coords = np.zeros((n_atoms, 3), dtype=np.float64)  # placeholder

        ri, rj = find_block(S["[RADIUS_START]"], S["[RADIUS_END]"], pj + 1)
        if rj - ri != 1:
            raise ValueError(f"expected single r² int in RADIUS block, got {rj-ri}")
        r2_max = self._decode_int(toks[ri])

        ci, cj = find_block(S["[CENTER_START]"], S["[CENTER_END]"], rj + 1)
        center = tuple(self._decode_int(t) for t in toks[ci:cj])
        if len(center) != 3:
            raise ValueError(f"expected 3 center coords, got {len(center)}")

        bi, bj = find_block(S["[BOUNDS_START]"], S["[BOUNDS_END]"], cj + 1)
        bounds = tuple(self._decode_int(t) for t in toks[bi:bj])
        if len(bounds) != 6:
            raise ValueError(f"expected 6 bounds ints, got {len(bounds)}")

        # Density decode — slice tokens until DENS_END.
        di, dj = find_block(S["[DENS_START]"], S["[DENS_END]"], bj + 1)
        codec = self.density_codec
        widths = codec.signed_vocabs
        n_per_voxel = len(widths)
        dens_toks = toks[di:dj]
        if len(dens_toks) % n_per_voxel != 0:
            raise ValueError(f"density tokens {len(dens_toks)} not divisible by {n_per_voxel}")
        K = len(dens_toks) // n_per_voxel

        vocab = self.vocab
        base = vocab.density_offset
        comps = np.zeros((K, n_per_voxel), dtype=np.int64)
        for k in range(K):
            cum = 0
            for j, w in enumerate(widths):
                comps[k, j] = dens_toks[k * n_per_voxel + j] - (base + cum)
                cum += w
        values = codec.decode_signed(comps)

        return BallSample(
            task_id="",
            center=center,
            r2_max=r2_max,
            grid_shape=grid_shape,
            bounds=bounds,  # type: ignore[arg-type]
            atomic_numbers=atomic_numbers,
            frac_coords=frac_coords,
            ball_density=values.astype(np.float64),
        )
