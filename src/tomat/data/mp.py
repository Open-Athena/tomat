"""Materials Project CHGCAR loader.

Pulls individual CHGCARs from the electrAI-curated S3 dataset and caches
them locally.

S3 layout (``s3://openathena/electrai/mp/chg_datasets/dataset_4/``):

* ``mp_filelist.txt`` — 2,885 ``mp-<id>`` keys, one per line.
* ``data/<mp-id>.CHGCAR`` — cheap-guess (SAD) inputs used by electrAI.
* ``label/<mp-id>.CHGCAR`` — DFT-converged targets (what we want to tokenize).
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from pymatgen.io.vasp.outputs import Chgcar

S3_PREFIX = "s3://openathena/electrai/mp/chg_datasets/dataset_4"
DEFAULT_CACHE = Path("data/mp-cache")


@dataclass(frozen=True)
class MPEntry:
    mp_id: str
    s3_uri: str
    local_path: Path


def list_mp_ids(filelist_path: Path | None = None) -> list[str]:
    """Return the curated 2,885-entry mp-id list. Caches the filelist locally."""
    path = filelist_path or (DEFAULT_CACHE / "mp_filelist.txt")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _aws_cp(f"{S3_PREFIX}/mp_filelist.txt", path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def fetch_chgcar(
    mp_id: str,
    *,
    split: str = "label",
    cache_dir: Path = DEFAULT_CACHE,
) -> MPEntry:
    """Download ``<cache_dir>/<split>/<mp_id>.CHGCAR`` if missing, return its path."""
    if split not in {"data", "label"}:
        raise ValueError(f"split must be 'data' or 'label', got {split!r}")
    local = cache_dir / split / f"{mp_id}.CHGCAR"
    s3_uri = f"{S3_PREFIX}/{split}/{mp_id}.CHGCAR"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        _aws_cp(s3_uri, local)
    return MPEntry(mp_id=mp_id, s3_uri=s3_uri, local_path=local)


def load_chgcar(mp_id: str, *, split: str = "label", cache_dir: Path = DEFAULT_CACHE) -> Chgcar:
    entry = fetch_chgcar(mp_id, split=split, cache_dir=cache_dir)
    return Chgcar.from_file(str(entry.local_path))


def _aws_cp(s3_uri: str, dest: Path) -> None:
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dest)], check=True)
