#!/usr/bin/env -S uv run --with modal
"""Modal-parallel reconstruction-fidelity sweep for density tokenizers.

Mirrors ``scripts/fidelity_sweep.py`` end-to-end, but fans each MP id out to
a remote Modal worker so the sweep runs in ≈(slowest-sample) wall-clock
instead of (N × slowest-sample). Same tokenizer configs, same metrics, same
CSV schema.

Usage:

    scripts/fidelity_sweep_modal.py                 # 5 samples, default configs
    scripts/fidelity_sweep_modal.py -n 50 -o out.csv

Or invoked as a Modal entrypoint directly:

    modal run scripts/fidelity_sweep_modal.py --n-samples 50 --output-csv out.csv

First-time setup: ``pip install modal && modal setup`` (auths the local CLI).
The CHGCARs themselves are read from the pre-populated ``electrai-data``
Modal Volume — no S3 credentials are required on the Modal side.


================================================================
Modal Volume (CHGCAR cache)
================================================================

We reuse the existing **``electrai-data``** Modal Volume, which already
mirrors ``s3://openathena/electrai/mp/chg_datasets/dataset_4/`` in full
(both ``data/`` — SAD inputs — and ``label/`` — DFT-converged targets).
No download on this sweep; the volume is mounted read-only for sweep runs.

    Volume name : ``electrai-data``
    Mount path  : ``/cache/electrai-data``
    Subpath     : ``mp/chg_datasets/dataset_4``  (within the volume)
    Layout at the subpath:
        ``mp_filelist.txt``                 — curated 2,885 mp-ids
        ``mp_filelist_filtered.txt``        — filtered subset
        ``mp_filelist_benchmark.txt``       — benchmark subset
        ``label/<mp-id>.CHGCAR``            — DFT-converged (what we tokenize)
        ``data/<mp-id>.CHGCAR``             — SAD input (not used here)

Verify with: ``modal volume ls electrai-data /mp/chg_datasets/dataset_4``.


================================================================
Adding more data to the Volume
================================================================

The Volume is a general-purpose persistent filesystem — anything we want
to reuse across Modal invocations (checkpoints, preprocessed tensors,
additional datasets) can live there:

    modal volume put electrai-data <local-path> <remote-path>
    modal volume get electrai-data <remote-path> <local-path>
    modal volume ls  electrai-data <remote-path>
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"

err = partial(print, file=sys.stderr)

# ---- Modal image ---------------------------------------------------------
# Minimal deps to run the sweep: pymatgen for CHGCAR IO, scipy for JSD.
# No boto3 / awscli needed — CHGCARs come from the mounted Volume.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "click",
        "numpy",
        "pymatgen",
        "scipy",
    )
    # Make the local `tomat` package importable inside the container.
    # `add_local_python_source` is the modern (>=0.63) replacement for the
    # old `Mount.from_local_python_packages` pattern.
    .add_local_python_source("tomat", copy=True)
    # Ship the codec config JSONs so `default_configs()` picks up the
    # `*-coded` variants (otherwise it silently skips them).
    .add_local_dir(str(CONFIGS_DIR), remote_path="/root/configs", copy=True)
)

app = modal.App("tomat-fidelity-sweep", image=image)

# ---- CHGCAR cache volume ------------------------------------------------
# Reuse the existing `electrai-data` Modal Volume — already pre-populated
# with the MP CHGCAR dataset (mirrors the S3 layout at
# ``s3://openathena/electrai/mp/chg_datasets/dataset_4/``). Mounted
# read-only: we never download new CHGCARs on this sweep, we just read.
MP_CACHE_VOLUME_NAME = "electrai-data"
MP_CACHE_MOUNT = "/cache/electrai-data"
# Within the volume the CHGCARs live under this subpath — tomat's
# ``load_chgcar`` treats this as its ``cache_dir``.
MP_DATASET_SUBPATH = "mp/chg_datasets/dataset_4"

mp_cache_volume = modal.Volume.from_name(
    MP_CACHE_VOLUME_NAME,
    create_if_missing=False,  # fail loudly if the named volume doesn't exist
)


@app.function(
    volumes={MP_CACHE_MOUNT: mp_cache_volume},
    timeout=60 * 60,  # 1 h per sample — well above the ~1-2 min laptop budget
    cpu=4.0,
    memory=8192,
)
def sweep_one(mp_id: str, split: str = "label") -> list[dict]:
    """Read one CHGCAR from the mounted Volume and run all configured
    tokenizers. Returns metric rows."""
    import os

    import numpy as np

    # Imports deferred so they run inside the container.
    from tomat.data.classify import classify_elements
    from tomat.data.mp import load_chgcar
    from tomat.sweep import compute_metrics, default_configs
    from tomat.tokenizers import CutoffEncoded

    # The codec configs were shipped to /root/configs; `default_configs()`
    # resolves the relative ``configs/fp16-channels.json``, so cwd must be
    # /root. `modal run` sets this; be defensive.
    os.chdir("/root")

    cache_dir = Path(MP_CACHE_MOUNT) / MP_DATASET_SUBPATH
    cache_path = cache_dir / split / f"{mp_id}.CHGCAR"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{mp_id} not in volume {MP_CACHE_VOLUME_NAME!r} at {cache_path}. "
            f"The electrai-data volume is expected to be fully pre-populated; "
            f"verify the mp-id is in dataset_4."
        )

    t0 = time.perf_counter()
    chgcar = load_chgcar(mp_id, split=split, cache_dir=cache_dir)
    density = np.asarray(chgcar.data["total"], dtype=np.float64)
    category = classify_elements(el.symbol for el in chgcar.structure.composition.elements)
    load_s = time.perf_counter() - t0
    print(
        f"[{mp_id}] {category}: grid {density.shape}, sum ρ = {density.sum():.3e}, "
        f"loaded in {load_s:.1f}s",
        flush=True,
    )

    configs = default_configs()
    rows: list[dict] = []
    for cfg in configs:
        t1 = time.perf_counter()
        encoded = cfg.tokenizer.encode(chgcar)
        recon = cfg.tokenizer.decode(encoded)
        elapsed = time.perf_counter() - t1
        metrics = compute_metrics(density, recon)
        tokens = cfg.tokenizer.token_count(encoded)
        row = dict(
            mp_id=mp_id,
            category=category,
            config=cfg.label,
            tokens=tokens,
            seconds=elapsed,
            grid=str(density.shape),
            **metrics,
        )
        if isinstance(encoded, CutoffEncoded):
            row["mass_captured"] = encoded.mass_captured
            row["effective_threshold"] = encoded.effective_threshold
        rows.append(row)

    return rows


@app.local_entrypoint()
def main(
    n_samples: int = 5,
    output_csv: str = "",
    split: str = "label",
    mp_ids: str = "",
):
    """Modal entrypoint (invoked via ``modal run scripts/fidelity_sweep_modal.py``).

    Mirrors the click CLI below; Modal's local entrypoint decorator only
    supports primitive kwargs, so ``mp_ids`` is a comma-separated string.
    """
    # Import inside so `modal run` doesn't trip over the click CLI.
    from tomat.data.mp import list_mp_ids
    from tomat.sweep import CSV_FIELDNAMES

    ids: list[str]
    if mp_ids:
        ids = [s.strip() for s in mp_ids.split(",") if s.strip()]
    else:
        ids = list_mp_ids()[:n_samples]

    err(f"Dispatching {len(ids)} mp ids to Modal (split={split})")

    all_rows: list[dict] = []
    done = 0
    # `.starmap` unpacks each tuple as positional args to `sweep_one`.
    # Using tuples (vs. `.map` + `kwargs=`) keeps this portable across
    # Modal versions where the `kwargs=` param has moved around.
    args = [(mp_id, split) for mp_id in ids]
    for sample_rows in sweep_one.starmap(args, order_outputs=False):
        done += 1
        all_rows.extend(sample_rows)
        mp_id = sample_rows[0]["mp_id"] if sample_rows else "?"
        err(f"  [{done}/{len(ids)}] {mp_id}: {len(sample_rows)} rows")

    if output_csv:
        out = Path(output_csv)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        err(f"Wrote {len(all_rows)} rows to {out}")
    else:
        # Stream to stdout if no file given.
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ---- Standalone click CLI -----------------------------------------------
# `modal run` uses the `@app.local_entrypoint` above. Running the script
# directly (``scripts/fidelity_sweep_modal.py -n 5 -o out.csv``) goes
# through click, which then opens a Modal `app.run()` context and calls
# `sweep_one.map(...)` the same way.
def _cli():
    import click
    from click import argument, command, option

    @command()
    @option("-n", "--n-samples", type=int, default=5, help="Number of MP entries to evaluate")
    @option("-o", "--output-csv", type=click.Path(dir_okay=False, path_type=Path), help="Write per-(sample, config) rows to CSV")
    @option("-s", "--split", type=click.Choice(["data", "label"]), default="label", help="Which CHGCAR to tokenize; 'label' = DFT-converged target")
    @argument("mp_ids", nargs=-1)
    def cli(n_samples: int, output_csv: Path | None, split: str, mp_ids: tuple[str, ...]):
        from tomat.data.mp import list_mp_ids
        from tomat.sweep import CSV_FIELDNAMES

        ids = list(mp_ids) if mp_ids else list_mp_ids()[:n_samples]
        err(f"Dispatching {len(ids)} mp ids to Modal (split={split})")

        all_rows: list[dict] = []
        args = [(mid, split) for mid in ids]
        with app.run():
            done = 0
            for sample_rows in sweep_one.starmap(args, order_outputs=False):
                done += 1
                all_rows.extend(sample_rows)
                mp_id = sample_rows[0]["mp_id"] if sample_rows else "?"
                err(f"  [{done}/{len(ids)}] {mp_id}: {len(sample_rows)} rows")

        if output_csv is not None:
            with output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
            err(f"Wrote {len(all_rows)} rows to {output_csv}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    cli()


if __name__ == "__main__":
    _cli()
