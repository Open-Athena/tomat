"""Read one rho_gga Zarr directory → ``(density, structure, shape)``.

Format (confirmed against
``/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/``):

    <task>.zarr/
      zarr.json                     # group-level; attributes include
                                    # 'structure' (pymatgen JSON) and
                                    # 'metadata' (task_id etc.)
      charge_density_total/
        zarr.json                   # array-level metadata (shape, dtype,
                                    # codec, chunk_shape)
        c/                          # chunk storage (single-chunk, default)

Zarr v3, float32, zstd-compressed; one sample = a few MB on disk and
typically a few hundred k to a few M voxels.

Mirrors :class:`tomat.data.mp.load_chgcar`'s return shape — a light
CHGCAR-like object with ``.data['total']`` (np.ndarray) and
``.structure`` (pymatgen ``Structure``), so existing tokenizers consume
it unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@dataclass(frozen=True)
class RhoGgaSample:
    """Lightweight CHGCAR-shaped container for a rho_gga Zarr sample."""

    task_id: str
    data: dict[str, np.ndarray]
    structure: "Structure"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data["total"].shape  # type: ignore[return-value]


def load_rho_gga(path: str | Path) -> RhoGgaSample:
    """Load one ``<task>.zarr`` directory into an in-memory sample.

    ``path`` should be the ``*.zarr`` directory (or a ``zarr.store``
    handle). Reads the full ``charge_density_total`` array — a few MB
    per sample after zstd decompression.
    """
    from pymatgen.core.structure import Structure

    group = zarr.open_group(str(path), mode="r")
    density = np.asarray(group["charge_density_total"][:])

    attrs = _group_attrs(group)
    structure = Structure.from_dict(json.loads(attrs["structure"]))
    meta = json.loads(attrs.get("metadata", "{}"))
    task_id = meta.get("task_id") or Path(path).stem.removesuffix(".zarr")

    return RhoGgaSample(
        task_id=task_id,
        data={"total": density},
        structure=structure,
    )


def load_rho_gga_as_chgcar_like(path: str | Path) -> SimpleNamespace:
    """Duck-typed CHGCAR-shim: tokenizers just need ``.data['total']`` and
    ``.structure``. Thin wrapper on top of :func:`load_rho_gga`."""
    s = load_rho_gga(path)
    return SimpleNamespace(data=s.data, structure=s.structure)


def _group_attrs(group: Any) -> dict[str, str]:
    """Zarr v3 exposes attributes via ``.attrs``; v2 via ``.attrs``. Handle both."""
    attrs = getattr(group, "attrs", None)
    if attrs is None:
        raise RuntimeError(f"Zarr group at {group} has no attributes")
    # Coerce to a plain dict; some zarr versions return a lazy Attributes object.
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else attrs)
