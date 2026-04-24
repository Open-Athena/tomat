#!/usr/bin/env python
"""Fit an empirical Lloyd-Max quantizer to all train-full voxel densities.

Streaming pass via Modal: each worker mounts the `tomat-rho-gga-train`
volume, reads its stripe of Zarrs, accumulates a fine-grained histogram
of density values. Histograms merge into a global histogram. Then Lloyd-
Max iteration on the histogram produces (boundaries, reconstruction
points) for the target vocab size. Saved to GCS.

Run:
    modal run scripts/fit_lmq_codec_modal.py::run \\
        --n-bins 16384 \\
        --codec-name lmq-v1

Output: gs://marin-eu-west4/tomat/codecs/<codec-name>.npz
"""

from __future__ import annotations

import os
from typing import Any

import modal

VOLUME_NAME = os.environ.get("TOMAT_VOLUME", "tomat-rho-gga-train")
MOUNT = "/vol"
BUCKET = "gs://marin-eu-west4/tomat"

# Fine histogram setup: linear bins over [LIN_LO, LIN_HI].
# Densities are non-negative; MP rho_gga typically in [0, ~150] e/bohr³ but
# log_max=4.97 (in the old codec) implies max ≈ 144. Extend to 200 for safety.
LIN_LO = 0.0
LIN_HI = 200.0
N_FINE = 1_000_000  # 1M fine bins over [0, 200] → 2e-4 resolution
FINE_DX = (LIN_HI - LIN_LO) / N_FINE

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("zarr>=3", "numpy", "gcsfs")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")

app = modal.App("tomat-fit-lmq", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def setup_gcp_creds():
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


@app.function(volumes={MOUNT: volume}, cpu=2, memory=4096, timeout=3600)
def list_mats() -> list[str]:
    """List all task_ids from the train split by reading split_limit_22M.json."""
    import json
    import pathlib

    split_path = pathlib.Path(MOUNT) / "split_limit_22M.json"
    with open(split_path) as f:
        split = json.load(f)
    entries = split["train"]
    if entries and isinstance(entries[0], int):
        filelist_path = pathlib.Path(MOUNT) / "mp_filelist.txt"
        with open(filelist_path) as f:
            filelist = [l.strip() for l in f if l.strip()]
        return [filelist[i] for i in entries]
    return list(entries)


@app.function(
    volumes={MOUNT: volume},
    cpu=1,
    memory=4096,
    timeout=1800,
    retries=5,
)
def histogram_stripe(mat_ids: list[str]) -> tuple[Any, dict]:
    """Accumulate a (N_FINE,) density histogram + stats across a chunk of mats.

    Returns (histogram_array, stats_dict).
    """
    import numpy as np
    import zarr
    import pathlib

    hist = np.zeros(N_FINE, dtype=np.int64)
    n_ok = 0
    n_missing = 0
    n_voxels = 0
    sum_rho = 0.0
    max_rho = 0.0
    min_rho = 1e30
    for mat_id in mat_ids:
        zarr_path = pathlib.Path(MOUNT) / "label" / f"{mat_id}.zarr"
        if not zarr_path.exists():
            n_missing += 1
            continue
        try:
            # Zarr v3 group layout: density lives at ``charge_density_total`` per
            # src/tomat/data/zarr_io.py::load_rho_gga.
            group = zarr.open_group(str(zarr_path), mode="r")
            density = np.asarray(group["charge_density_total"][:]).astype(np.float32).ravel()
        except Exception as e:
            if n_missing < 5:
                print(f"[stripe] read FAIL {mat_id}: {type(e).__name__}: {e}")
            n_missing += 1
            continue
        n_voxels += density.size
        # Clip before binning — preserves everything below LIN_HI exactly; any
        # outliers go to the top bin (tracked separately via stats).
        n_over = int((density > LIN_HI).sum())
        n_neg = int((density < 0).sum())
        dens_clip = np.clip(density, LIN_LO, LIN_HI - 1e-9)
        bin_idx = np.minimum(
            (dens_clip / FINE_DX).astype(np.int64),
            N_FINE - 1,
        )
        hist += np.bincount(bin_idx, minlength=N_FINE)[:N_FINE]
        sum_rho += float(density.sum())
        mx = float(density.max())
        mn = float(density.min())
        if mx > max_rho:
            max_rho = mx
        if mn < min_rho:
            min_rho = mn
        n_ok += 1
    return hist, {
        "n_ok": n_ok,
        "n_missing": n_missing,
        "n_voxels": n_voxels,
        "sum_rho": sum_rho,
        "max_rho": max_rho,
        "min_rho": min_rho,
    }


@app.function(cpu=4, memory=8192, secrets=[gcp_secret], timeout=1800)
def fit_and_save(hist_bytes: bytes, total_stats: dict, n_bins: int, codec_name: str):
    """Lloyd-Max fit on merged histogram; save codec to GCS."""
    import io
    import json
    import numpy as np
    import gcsfs  # type: ignore

    setup_gcp_creds()

    hist = np.load(io.BytesIO(hist_bytes))
    fine_centers = (np.arange(N_FINE) + 0.5) * FINE_DX  # float64
    total = hist.sum()
    print(f"[fit] histogram total = {total:,} voxels, n_fine_bins = {N_FINE}")

    # Mask zero bins for efficiency.
    nz = hist > 0
    centers_nz = fine_centers[nz]
    counts_nz = hist[nz].astype(np.float64)
    print(f"[fit] {len(centers_nz):,} non-empty fine bins used in fitting")

    # Initial boundaries: equal-mass quantiles.
    cdf = counts_nz.cumsum() / counts_nz.sum()
    qtiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # (n_bins-1,) interior boundaries
    init_bounds = np.interp(qtiles, cdf, centers_nz)
    bounds = init_bounds.copy()
    print(f"[fit] initial boundary range: [{bounds.min():.6f}, {bounds.max():.6f}]")

    # Lloyd-Max iteration (L2-optimal reconstruction = mean of bin).
    for iter_idx in range(30):
        # Assign every fine-bin center to an output bin
        bin_idx = np.searchsorted(bounds, centers_nz)  # shape (n_fine_nz,), range [0, n_bins]
        # Reconstruction points: mean of fine centers weighted by counts per output bin
        sum_num = np.bincount(bin_idx, weights=centers_nz * counts_nz, minlength=n_bins)
        sum_den = np.bincount(bin_idx, weights=counts_nz, minlength=n_bins)
        recon = np.where(sum_den > 0, sum_num / np.maximum(sum_den, 1e-30), np.nan)
        # For empty output bins: fill by interpolation from neighbors
        empty_mask = np.isnan(recon)
        if empty_mask.any():
            # Fill empties with linear interp between adjacent non-empty points
            idxs = np.arange(n_bins)
            non_empty = ~empty_mask
            recon[empty_mask] = np.interp(idxs[empty_mask], idxs[non_empty], recon[non_empty])
        # New boundaries: midpoints between consecutive recon points
        new_bounds = 0.5 * (recon[:-1] + recon[1:])
        diff = np.abs(new_bounds - bounds).max() if iter_idx > 0 else float("inf")
        bounds = new_bounds
        print(f"[fit] iter {iter_idx}: max |Δbound| = {diff:.6e}")
        if diff < 1e-8:
            print(f"[fit] converged at iter {iter_idx}")
            break

    # Final recon computation with stable bounds
    bin_idx = np.searchsorted(bounds, centers_nz)
    sum_num = np.bincount(bin_idx, weights=centers_nz * counts_nz, minlength=n_bins)
    sum_den = np.bincount(bin_idx, weights=counts_nz, minlength=n_bins)
    recon = np.where(sum_den > 0, sum_num / np.maximum(sum_den, 1e-30), np.nan)
    empty = np.isnan(recon)
    if empty.any():
        idxs = np.arange(n_bins)
        ne = ~empty
        recon[empty] = np.interp(idxs[empty], idxs[ne], recon[ne])
    # Per-bin counts for diagnostics
    per_bin_count = sum_den.astype(np.int64)

    # Compute diagnostic: expected MAE if we quantize data → closest recon point
    fine_bin_idx = np.searchsorted(bounds, fine_centers)
    quant_err = np.abs(fine_centers - recon[fine_bin_idx])
    mae = (quant_err * hist.astype(np.float64)).sum() / max(hist.sum(), 1)
    mean_rho = total_stats["sum_rho"] / max(total_stats["n_voxels"], 1)
    nmae_approx = mae / max(mean_rho, 1e-30)
    print(f"[fit] quantization MAE = {mae:.6e}, mean ρ = {mean_rho:.6e}, NMAE lower-bound = {nmae_approx:.4%}")

    # Save.
    clip_max = LIN_HI
    bundle = {
        "boundaries": bounds.astype(np.float32),
        "recon_points": recon.astype(np.float32),
        "clip_max": np.float32(clip_max),
        "n_bins": np.int32(n_bins),
        "lin_lo": np.float32(LIN_LO),
        "lin_hi": np.float32(LIN_HI),
        "per_bin_count": per_bin_count,
        "total_stats": np.array([json.dumps(total_stats)], dtype=object),
    }
    buf = io.BytesIO()
    np.savez(buf, **bundle)
    buf.seek(0)
    out_path = f"{BUCKET}/codecs/{codec_name}.npz"
    fs = gcsfs.GCSFileSystem()
    with fs.open(out_path, "wb") as f:
        f.write(buf.getvalue())
    print(f"[fit] wrote {out_path} ({len(buf.getvalue())/1024:.1f} KB)")

    return {
        "path": out_path,
        "n_bins": n_bins,
        "mae": float(mae),
        "nmae_approx": float(nmae_approx),
        "recon_range": [float(recon.min()), float(recon.max())],
    }


@app.local_entrypoint()
def run(n_bins: int = 16384, codec_name: str = "lmq-v1", chunk_size: int = 200, max_mats: int = 0):
    """Fit LMQ codec on train-full densities."""
    import sys
    import io
    import numpy as np

    err = lambda *a: print(*a, file=sys.stderr)

    mat_ids = list_mats.remote()
    if max_mats > 0:
        mat_ids = mat_ids[:max_mats]
    err(f"[fit-lmq] {len(mat_ids):,} mats; chunk_size={chunk_size}")

    # Chunk mats for parallel dispatch.
    chunks = [mat_ids[i:i + chunk_size] for i in range(0, len(mat_ids), chunk_size)]
    err(f"[fit-lmq] dispatching {len(chunks)} chunks")

    global_hist = np.zeros(N_FINE, dtype=np.int64)
    total_stats = {
        "n_ok": 0, "n_missing": 0, "n_voxels": 0,
        "sum_rho": 0.0, "max_rho": 0.0, "min_rho": 1e30,
    }
    n_done = 0
    for hist, stats in histogram_stripe.map(chunks, order_outputs=False):
        global_hist += hist
        total_stats["n_ok"] += stats["n_ok"]
        total_stats["n_missing"] += stats["n_missing"]
        total_stats["n_voxels"] += stats["n_voxels"]
        total_stats["sum_rho"] += stats["sum_rho"]
        total_stats["max_rho"] = max(total_stats["max_rho"], stats["max_rho"])
        total_stats["min_rho"] = min(total_stats["min_rho"], stats["min_rho"])
        n_done += 1
        if n_done % 20 == 0:
            err(f"[fit-lmq] {n_done}/{len(chunks)} chunks merged; "
                f"{total_stats['n_voxels']/1e9:.2f} G voxels so far")

    err(f"[fit-lmq] DONE accumulating: {total_stats['n_voxels']/1e9:.2f} G voxels "
        f"from {total_stats['n_ok']:,} mats ({total_stats['n_missing']:,} missing), "
        f"ρ range [{total_stats['min_rho']:.4e}, {total_stats['max_rho']:.4e}]")

    # Serialize histogram + stats, send to fit worker.
    buf = io.BytesIO()
    np.save(buf, global_hist)
    result = fit_and_save.remote(buf.getvalue(), total_stats, n_bins, codec_name)
    err(f"[fit-lmq] saved codec: {result}")
