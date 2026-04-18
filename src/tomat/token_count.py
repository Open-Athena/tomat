"""Per-scheme token-count accounting.

Token counts assume FP16-codec fidelity end-to-end — every real value
costs 3 tokens (signed 24-bit codec, from :class:`tomat.float_codec.FP16Codec`),
each complex value costs 6 tokens (real + imag), and each integer index on
a ≤ 128³ grid fits in 3 bytes at a 256-entry vocab (21 bits for 2 M voxels).
Atoms in a Δρ structure cost 10 tokens each (atomic number + 3 position
components × 3 tokens).

The counts are a modeling-side quantity; they don't affect reconstruction
NMAE at all — they just answer "at what scheme / retention does this fit
in a 4k / 16k / 64k context?".
"""

TOKENS_PER_REAL = 3  # signed 24-bit codec: SE, M0, M1
TOKENS_PER_COMPLEX = 2 * TOKENS_PER_REAL
TOKENS_PER_INDEX = 3  # 3 bytes of 256-vocab → 24 bits → up to 16 M grid points
TOKENS_PER_ATOM = 1 + 3 * TOKENS_PER_REAL  # Z + (x, y, z) × signed codec


def direct_tokens(n_voxels: int) -> int:
    return n_voxels * TOKENS_PER_REAL


def cutoff_tokens(k: int) -> int:
    return k * (TOKENS_PER_REAL + TOKENS_PER_INDEX)


def fourier_tokens(k: int) -> int:
    return k * (TOKENS_PER_COMPLEX + TOKENS_PER_INDEX)


def delta_overhead(n_atoms: int) -> int:
    return n_atoms * TOKENS_PER_ATOM
