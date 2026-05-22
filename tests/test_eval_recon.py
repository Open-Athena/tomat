"""No-GPU tests for the mat-NMAE eval's patch tiling + grid reconstruction.

These exercise the *pure-numpy* parts of `eval_mat_nmae.py` — the parts that
assemble `rho_pred` (the predicted-density grid that NMAE is computed against).
A bug here mis-places predicted patches → NMAE garbage while NEMD (a
placement-invariant mean) stays fine. No model / GPU needed.
"""
import numpy as np
import pytest

from marin.eval_mat_nmae import tile_full_coverage_offsets

P = 19
GRIDS = [
    (19, 19, 19),      # exactly P
    (38, 57, 76),      # multiples of P
    (96, 96, 96),      # n % P != 0 on every axis
    (56, 56, 96),      # the worst-violator grid from cont33k step-50000
    (40, 80, 120),     # mixed
    (160, 160, 192),
]


@pytest.mark.parametrize("grid", GRIDS)
def test_tile_full_coverage(grid):
    """Every voxel is covered exactly once — no gaps, no overlap."""
    offsets, local_slices = tile_full_coverage_offsets(grid, P)
    cover = np.zeros(grid, dtype=np.int32)
    for (ox, oy, oz), (lx0, lx1, ly0, ly1, lz0, lz1) in zip(offsets, local_slices):
        cover[ox + lx0:ox + lx1, oy + ly0:oy + ly1, oz + lz0:oz + lz1] += 1
    assert cover.min() == 1 and cover.max() == 1, (
        f"grid {grid}: coverage counts {sorted(np.unique(cover).tolist())}"
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_extract_reconstruct_identity(grid):
    """Extract every patch from a density grid (exactly as
    `build_all_patch_input_ids_v3`'s DENS block does) then reconstruct it (as
    the eval loop fills `rho_pred`). The round-trip must be the identity — if it
    isn't, predicted patches land at wrong voxels and NMAE is meaningless."""
    rng = np.random.default_rng(0)
    density = rng.random(grid, dtype=np.float64).astype(np.float32)
    offsets, local_slices = tile_full_coverage_offsets(grid, P)
    off = np.asarray(offsets, dtype=np.int32)
    gnp = np.asarray(grid, dtype=np.int32)

    # --- extract: mirrors build_all_patch_input_ids_v3 lines 536-542 ---
    iota = np.arange(P, dtype=np.int32)
    di, dj, dk = np.meshgrid(iota, iota, iota, indexing="ij")
    idx_x = (off[:, 0, None, None, None] + di[None]) % gnp[0]
    idx_y = (off[:, 1, None, None, None] + dj[None]) % gnp[1]
    idx_z = (off[:, 2, None, None, None] + dk[None]) % gnp[2]
    patches = density[idx_x, idx_y, idx_z]            # (n, P, P, P)
    flat = patches.reshape(len(offsets), P ** 3)      # the "DENS tokens" payload

    # --- reconstruct: mirrors the eval loop lines 891-900 ---
    recon = np.zeros(grid, dtype=np.float32)
    blocks = flat.reshape(-1, P, P, P)
    for i, ((ox, oy, oz), (lx0, lx1, ly0, ly1, lz0, lz1)) in enumerate(
        zip(offsets, local_slices)
    ):
        recon[ox + lx0:ox + lx1, oy + ly0:oy + ly1, oz + lz0:oz + lz1] = \
            blocks[i, lx0:lx1, ly0:ly1, lz0:lz1]

    assert np.array_equal(recon, density), (
        f"grid {grid}: extract→reconstruct is not identity — "
        f"{int((recon != density).sum())} / {density.size} voxels differ"
    )
