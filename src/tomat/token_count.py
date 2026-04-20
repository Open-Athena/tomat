"""Per-scheme token-count accounting.

The value-token count is codec-dependent — 3 for tomol's 3-byte layout, 2 for
the 9+12 2-token variant, 1 for a fp16-single-token codec. The index token
count is fixed: 3 bytes at 256-vocab = 24 bits, enough for any 128³-or-smaller
grid (2²¹ voxels fit in 3 bytes). Atoms (Δρ overhead) are ``1 + 3 × T`` tokens
each, where ``T`` is the codec's per-value count (Z + 3 coord channels).
"""

TOKENS_PER_INDEX = 3  # 3 bytes of 256-vocab → 24 bits → up to 16 M grid points


def direct_tokens(n_voxels: int, *, tokens_per_real: int = 3) -> int:
    return n_voxels * tokens_per_real


def cutoff_tokens(k: int, *, tokens_per_real: int = 3) -> int:
    return k * (tokens_per_real + TOKENS_PER_INDEX)


def fourier_tokens(k: int, *, tokens_per_real: int = 3) -> int:
    return k * (2 * tokens_per_real + TOKENS_PER_INDEX)


def delta_overhead(n_atoms: int, *, tokens_per_real: int = 3) -> int:
    return n_atoms * (1 + 3 * tokens_per_real)
