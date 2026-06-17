"""Voxel-zarr writer + converter — shared by eval_mat_nmae.py and the
`tomat zarr convert` CLI.

Writes per-mat charge-density volumes as zarr v3 groups with NGFF-style
`multiscales` attribute and 2×-downsampled pyramid levels. Schema is
designed for elvis's diff-mode renderer (see
`/Users/ryan/c/oa/elvis/specs/zarr-v3-multires-support.md`) — same
shape used for GT and predictions so the diff loader treats them
interchangeably.

Layout:

    <root>.zarr/
      zarr.json          (group meta + `multiscales` attr + structure attrs)
      0/                 (full res — shape (Nx, Ny, Nz))
        zarr.json
        c/0/0/0          (single chunk per level for grids ≤200³)
      1/                 (2× downsampled)
        zarr.json
        c/0/0/0
      2/                 (4× downsampled — only if grid > 64 on shortest axis)

Downsampling: simple 2×2×2 average per level. Cropped to be divisible
by 2 (drops at most 1 voxel per axis at each level).
"""

from __future__ import annotations
import json
from typing import Any

import numpy as np


# Stop adding pyramid levels when the shortest axis falls below this many
# voxels — below that, "load coarse first, refine on zoom" loses
# resolution faster than network savings buy back.
MIN_PYRAMID_AXIS = 8
# Cap pyramid depth so even a 200³ grid doesn't go past 5 levels.
MAX_PYRAMID_LEVELS = 5


def _downsample_2x(arr: np.ndarray) -> np.ndarray:
    """2× downsample by averaging 2×2×2 cubes. Drops the last voxel along
    each axis if the size is odd (matches NGFF convention)."""
    nx, ny, nz = arr.shape
    nx2 = (nx // 2) * 2
    ny2 = (ny // 2) * 2
    nz2 = (nz // 2) * 2
    cropped = arr[:nx2, :ny2, :nz2]
    return cropped.reshape(nx2 // 2, 2, ny2 // 2, 2, nz2 // 2, 2).mean(axis=(1, 3, 5))


def build_pyramid(
    rho: np.ndarray,
    min_axis: int = MIN_PYRAMID_AXIS,
    max_levels: int = MAX_PYRAMID_LEVELS,
) -> list[np.ndarray]:
    """Return a list of arrays [level0, level1, …] where level0 is `rho` and
    each subsequent level is 2× downsampled. Stops at `max_levels` or when
    the shortest axis would drop below `min_axis`.
    """
    levels: list[np.ndarray] = [rho]
    current = rho
    for _ in range(max_levels - 1):
        next_shape = tuple((d // 2) for d in current.shape)
        if min(next_shape) < min_axis:
            break
        current = _downsample_2x(current)
        levels.append(current)
    return levels


def build_multiscales_attr(
    levels: list[np.ndarray],
    name: str = "charge_density_total",
) -> dict[str, Any]:
    """NGFF / OME-Zarr 0.5 `multiscales` attribute. Each level entry
    declares a `scale` coordinateTransform = 2^level along all axes.
    """
    datasets = []
    for i in range(len(levels)):
        s = float(2 ** i)
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [s, s, s]},
            ],
        })
    return {
        "version": "0.5",
        "name": name,
        "axes": [
            {"name": "z", "type": "space", "unit": "voxel"},
            {"name": "y", "type": "space", "unit": "voxel"},
            {"name": "x", "type": "space", "unit": "voxel"},
        ],
        "datasets": datasets,
    }


def build_elvis_attr(
    rho: np.ndarray,
    material_id: str,
    role: str,
    structure_json: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Build ELVis's custom metadata block — see
    `~/c/oa/elvis/pkgs/core/src/storage/zarr-volume.ts:5-22`.

    `role` is `'input'` (= reference / GT / "before") or `'label'`
    (= prediction / "after"). For tomat: GT=input, pred=label so that
    the diff `v1 − v0 = pred − GT` reads as "where did the model over-
    or under-predict" (positive = overpredicted; matches the
    DFT-vs-SAD convention from App.tsx:1270-1273).

    `structure_json` is the pymatgen Structure JSON (either a JSON
    string from a zarr attr, or an already-parsed dict). If None or
    unparseable, lattice/atoms are emitted empty — ELVis renders
    fine, just without the atom overlay.
    """
    if role not in ("input", "label"):
        raise ValueError(f"role must be 'input' or 'label', got {role!r}")
    # ---- lattice + atoms from pymatgen Structure ----
    lattice: list[list[float]] = []
    atoms: list[dict[str, Any]] = []
    if structure_json is not None:
        try:
            s = (json.loads(structure_json)
                 if isinstance(structure_json, str) else structure_json)
            lat = s.get("lattice", {})
            mat = lat.get("matrix") or []
            lattice = [[float(x) for x in row] for row in mat]
            for site in s.get("sites", []):
                spec = (site.get("species") or [{}])[0]
                elt = spec.get("element") or site.get("label") or "?"
                abc = site.get("abc") or [0.0, 0.0, 0.0]
                atoms.append({
                    "element": str(elt),
                    "frac": [float(abc[0]), float(abc[1]), float(abc[2])],
                })
        except Exception:
            # Bad/missing structure → emit empty lattice + atoms.
            lattice, atoms = [], []
    # ---- stats + quantiles over the full grid ----
    flat = rho.astype(np.float32).ravel()
    stats = {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
    }
    # 201 equally-spaced quantiles matches App.tsx's client expectation.
    qs = np.linspace(0.0, 1.0, 201)
    quantiles = [float(v) for v in np.quantile(flat, qs)]
    return {
        "material_id": material_id,
        "role": role,
        "lattice": lattice,
        "atoms": atoms,
        "stats": stats,
        "quantiles": quantiles,
    }


def write_voxel_zarr(
    url: str,
    rho: np.ndarray,
    extra_attrs: dict[str, Any] | None = None,
    dtype: str = "<f2",
    elvis_block: dict[str, Any] | None = None,
) -> int:
    """Write a multi-res zarr v3 group at `url` (gs://… or local path) from
    voxel grid `rho` shape `(Nx, Ny, Nz)`.

    `extra_attrs` is merged into the group's attrs alongside `multiscales`
    (used to carry `structure` + `mp_id` + provenance — anything JSON).

    Returns number of pyramid levels written.
    """
    import zarr
    import fsspec
    mapper = fsspec.get_mapper(url)
    zg = zarr.open_group(mapper, mode="w", zarr_format=3)
    levels = build_pyramid(rho)
    arr_dtype = np.dtype(dtype)
    for i, lvl in enumerate(levels):
        # zarr-python 3.x: use create_array(shape=..., dtype=..., chunks=...)
        # then assign data via slice; create_dataset's `data=` keyword is gone.
        arr = zg.create_array(
            str(i),
            shape=lvl.shape,
            chunks=lvl.shape,  # single chunk per level — grids ≤200³ fit fine
            dtype=dtype,
        )
        arr[:] = lvl.astype(arr_dtype)
    attrs: dict[str, Any] = {"multiscales": [build_multiscales_attr(levels)]}
    if elvis_block is not None:
        attrs["elvis"] = elvis_block
    if extra_attrs:
        attrs.update(extra_attrs)
    # zarrita / NGFF readers consume `attrs` from the group's zarr.json.
    # zarr-python writes them when you assign to `zg.attrs`.
    for k, v in attrs.items():
        zg.attrs[k] = v
    return len(levels)


def convert_v2_multires_to_v3(
    src_url: str,
    dst_local_dir: str,
) -> dict[str, Any]:
    """Convert a v2 multi-res zarr group (e.g. elvis's electrai
    `<task>-<role>.zarr/` with `0/`, `1/`, `2/`, `3/` levels + an
    `elvis` attrs block) into a v3 multi-res zarr group preserving
    structure, data, chunks, dtype, and attrs verbatim.

    `src_url` is an HTTPS prefix (trailing slash). Reads via plain
    urllib so 403/404 are uniform "missing" (zarr-python+fsspec raise
    hard 403 from S3-public-read buckets that don't allow ListBucket).
    Writes to a LOCAL directory which the caller is expected to
    `aws s3 sync` up to R2.
    """
    import zarr
    import os
    import urllib.request
    import urllib.error
    import io
    import numcodecs
    import json as _json

    def _http_get(url: str) -> bytes | None:
        try:
            with urllib.request.urlopen(url) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise

    if not src_url.endswith("/"):
        src_url += "/"
    # Group attrs (multiscales + elvis).
    zattrs_bytes = _http_get(f"{src_url}.zattrs")
    if zattrs_bytes is None:
        raise ValueError(f"{src_url}: no .zattrs")
    src_attrs = _json.loads(zattrs_bytes)
    multiscales = src_attrs.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{src_url}: no multiscales attr")
    datasets = multiscales[0]["datasets"]
    os.makedirs(dst_local_dir, exist_ok=True)
    dst = zarr.open_group(dst_local_dir, mode="w", zarr_format=3)
    level_shapes = []
    for ds in datasets:
        path = ds["path"]
        zarray_bytes = _http_get(f"{src_url}{path}/.zarray")
        if zarray_bytes is None:
            raise ValueError(f"{src_url}{path}/.zarray: missing")
        zarray = _json.loads(zarray_bytes)
        shape = tuple(int(d) for d in zarray["shape"])
        chunks = tuple(int(c) for c in zarray["chunks"])
        dtype = np.dtype(zarray["dtype"])
        # Compressor: elvis uses Zstd; numcodecs.get_codec handles all.
        compressor_cfg = zarray.get("compressor")
        codec = numcodecs.get_codec(compressor_cfg) if compressor_cfg else None
        # Re-assemble grid from chunks.
        grid_chunks = tuple(
            (sh + cs - 1) // cs for sh, cs in zip(shape, chunks)
        )
        out = np.empty(shape, dtype=dtype)
        for ci in range(grid_chunks[0]):
            for cj in range(grid_chunks[1]):
                for ck in range(grid_chunks[2]):
                    chunk_url = f"{src_url}{path}/{ci}.{cj}.{ck}"
                    raw = _http_get(chunk_url)
                    if raw is None:
                        # Sparse chunk — fill_value (usually 0).
                        continue
                    decoded = codec.decode(raw) if codec else raw
                    arr = np.frombuffer(decoded, dtype=dtype).reshape(chunks)
                    # Slice into the output, accounting for trailing partial chunks.
                    i0, j0, k0 = ci * chunks[0], cj * chunks[1], ck * chunks[2]
                    i1 = min(i0 + chunks[0], shape[0])
                    j1 = min(j0 + chunks[1], shape[1])
                    k1 = min(k0 + chunks[2], shape[2])
                    out[i0:i1, j0:j1, k0:k1] = arr[: i1 - i0, : j1 - j0, : k1 - k0]
        new_arr = dst.create_array(
            path,
            shape=out.shape,
            chunks=chunks,
            dtype=str(dtype),
        )
        new_arr[:] = out
        level_shapes.append(list(out.shape))
    for k, v in src_attrs.items():
        dst.attrs[k] = v
    return {
        "src": src_url,
        "dst": dst_local_dir,
        "n_levels": len(datasets),
        "level_shapes": level_shapes,
        "attrs_keys": sorted(src_attrs),
    }


def convert_v3_single_array_to_pyramid(
    src_url: str,
    dst_url: str,
    material_id: str | None = None,
    role: str = "input",
    array_name: str = "charge_density_total",
    dtype: str = "<f2",
) -> dict[str, Any]:
    """Convert a v3 single-array zarr group (e.g. `rho_gga_raw/<mp>.zarr`
    with a `charge_density_total` array directly under the root) into a v3
    multi-res pyramid group with NGFF `multiscales` + ELVis `elvis`
    blocks. Schema matches the GT side of ELVis's
    `pkgs/core/src/storage/zarr-volume.ts:5-22`.

    `material_id` defaults to the last path segment of dst_url minus
    `.zarr`. `role` defaults to `'input'` (the GT convention); pass
    `'label'` for prediction zarrs.

    Returns a small dict with `n_levels`, `src_shape`, etc. for inspection.
    """
    import zarr
    import fsspec
    src_mapper = fsspec.get_mapper(src_url)
    src_group = zarr.open_group(src_mapper, mode="r", zarr_format=3)
    arr = src_group[array_name]
    rho = np.asarray(arr[:]).astype(np.float32)
    src_attrs_raw = dict(src_group.attrs)
    # Material id: prefer explicit arg, else strip `.zarr` from dst basename.
    if material_id is None:
        leaf = dst_url.rstrip("/").rsplit("/", 1)[-1]
        material_id = leaf.removesuffix(".zarr")
    # Synthesize the elvis block from the source's `structure` attr (if any).
    elvis_block = build_elvis_attr(
        rho=rho,
        material_id=material_id,
        role=role,
        structure_json=src_attrs_raw.get("structure"),
    )
    # Preserve other source-side attrs (e.g. `metadata`) for provenance —
    # `structure` is no longer needed since elvis_block carries lattice+atoms.
    passthrough = {k: v for k, v in src_attrs_raw.items()
                   if k not in ("multiscales", "elvis", "structure")}
    n = write_voxel_zarr(
        dst_url, rho, extra_attrs=passthrough, dtype=dtype,
        elvis_block=elvis_block,
    )
    return {
        "src": src_url,
        "dst": dst_url,
        "material_id": material_id,
        "role": role,
        "n_levels": n,
        "src_shape": list(rho.shape),
        "n_atoms": len(elvis_block["atoms"]),
        "stats": elvis_block["stats"],
    }
