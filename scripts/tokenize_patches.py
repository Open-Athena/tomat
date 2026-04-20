#!/usr/bin/env python
"""Pre-tokenize a rho_gga Zarr subset into parquet shards for LLM training.

For each task ID in the chosen split, open the Zarr, sample ``M`` random
patch offsets (with PBC wrap), tokenize each via :class:`PatchTokenizer`,
and append the resulting ``(task_id, offset_x, offset_y, offset_z, input_ids)``
rows to a parquet shard. Levanter / Marin can then consume the shards
via ``LmDataConfig`` with ``block_shuffle`` to decorrelate patches across
materials at training time.

Layout written to ``--output-dir``::

    <output-dir>/
      shard-00000.parquet      # up to ``--rows-per-shard`` rows each
      shard-00001.parquet
      ...
      meta.json                # run metadata: split, patch_size, codecs, vocab_size, n_rows

Usage::

    scripts/tokenize_patches.py \\
        --rho-gga-dir /path/to/rho_gga \\
        --split-file split_limit_22M.json \\
        --split validation \\
        --patches-per-material 32 \\
        --patch-size 14 \\
        --output-dir data/tokenized/val \\
        --seed 42
"""

import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from click import command, option

from tomat.data.zarr_io import load_rho_gga
from tomat.float_codec import FP16Codec
from tomat.tokenizers.patch import PatchTokenizer, SPECIAL_TOKENS

err = partial(print, file=sys.stderr)


# Density-codec builders keyed by name. These are the variants we sweep over.
# Each takes ``(log_min, log_max)`` fitted from rho_gga densities.
DENSITY_CODECS: dict[str, callable] = {
    "tomol_3byte":    FP16Codec.tomol_3byte,     # 3 tokens/value, 1024 vocab, 24-bit
    "two_token_9_12": FP16Codec.two_token_9_12,  # 2 tokens/value, 4608 vocab, 21-bit
    "fp16_1token":    FP16Codec.fp16_1token,     # 1 token/value,  65 536 vocab, 16-bit
}


def _build_schema() -> pa.Schema:
    return pa.schema([
        pa.field("task_id", pa.string()),
        pa.field("offset_x", pa.int32()),
        pa.field("offset_y", pa.int32()),
        pa.field("offset_z", pa.int32()),
        pa.field("input_ids", pa.list_(pa.int32())),
    ])


@command()
@option('-r', '--rho-gga-dir', type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True, help='Root of rho_gga (expects `label/<task>.zarr` subdirs).')
@option('-s', '--split-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None, help='JSON with train/val/test task-id lists.')
@option('-k', '--split', default='validation',
        help='Key in split file (default: validation).')
@option('-m', '--patches-per-material', type=int, default=32,
        help='Random patch offsets to sample per material.')
@option('-c', '--density-codec', type=click.Choice(sorted(DENSITY_CODECS)),
        default='two_token_9_12',
        help='FP16 codec variant for density values (affects vocab size + tokens/value).')
@option('--density-log-min', type=float, default=-4.127,
        help='Codec log_min (rho_gga p0.01 with padding).')
@option('--density-log-max', type=float, default=4.967,
        help='Codec log_max (rho_gga p99.99 with padding).')
@option('-p', '--patch-size', type=int, default=14,
        help='Patch edge length (voxels). Default 14.')
@option('-o', '--output-dir', type=click.Path(path_type=Path), required=True,
        help='Parquet shard output directory (created).')
@option('-S', '--seed', type=int, default=42, help='RNG seed.')
@option('-R', '--rows-per-shard', type=int, default=2048,
        help='Rows per parquet file (default 2048 ≈ 100 MB at 5k tokens).')
@option('-n', '--n-materials', type=int, default=None,
        help='Debug: cap the number of materials (first N from split).')
def main(
    rho_gga_dir: Path,
    split_file: Path | None,
    split: str,
    patches_per_material: int,
    density_codec: str,
    density_log_min: float,
    density_log_max: float,
    patch_size: int,
    output_dir: Path,
    seed: int,
    rows_per_shard: int,
    n_materials: int | None,
) -> None:
    # Resolve the task-id list for this split.
    if split_file is not None:
        with open(split_file) as f:
            split_data = json.load(f)
        if split not in split_data:
            raise click.UsageError(f"split {split!r} not in {list(split_data)}")
        task_ids = split_data[split]
    else:
        # No split file: use every zarr under label/
        task_ids = sorted(p.stem for p in (rho_gga_dir / 'label').glob('*.zarr'))
    if n_materials:
        task_ids = task_ids[:n_materials]
    err(f"[tokenize] split={split} materials={len(task_ids)} "
        f"patches/material={patches_per_material} patch_size={patch_size} "
        f"density_codec={density_codec}")

    output_dir.mkdir(parents=True, exist_ok=True)
    codec = DENSITY_CODECS[density_codec](log_min=density_log_min, log_max=density_log_max)
    tokenizer = PatchTokenizer(patch_size=patch_size, density_codec=codec)
    rng = np.random.default_rng(seed)
    schema = _build_schema()

    # Streaming shard writer.
    shard_idx = 0
    rows_in_shard = 0
    writer: pq.ParquetWriter | None = None

    def _new_writer(i: int) -> pq.ParquetWriter:
        path = output_dir / f"shard-{i:05d}.parquet"
        err(f"[tokenize] → {path}")
        return pq.ParquetWriter(str(path), schema, compression="zstd")

    total_rows = 0
    missing: list[str] = []
    for mat_idx, task_id in enumerate(task_ids, start=1):
        zarr_path = rho_gga_dir / 'label' / f'{task_id}.zarr'
        if not zarr_path.exists():
            missing.append(task_id)
            continue

        sample = load_rho_gga(zarr_path)
        offsets = tokenizer.random_offsets(
            sample.data['total'].shape, n=patches_per_material, rng=rng,
        )

        batch_task_ids: list[str] = []
        batch_ox: list[int] = []
        batch_oy: list[int] = []
        batch_oz: list[int] = []
        batch_ids: list[list[int]] = []

        for off in offsets:
            patch = tokenizer.make_sample(
                task_id=task_id,
                density=sample.data['total'],
                structure=sample.structure,
                offset=tuple(int(x) for x in off),
            )
            batch_task_ids.append(task_id)
            batch_ox.append(int(off[0]))
            batch_oy.append(int(off[1]))
            batch_oz.append(int(off[2]))
            batch_ids.append(tokenizer.tokenize(patch))

        # Flush the batch (one material) to parquet, starting a new shard
        # whenever rows_per_shard would be exceeded.
        if writer is None:
            writer = _new_writer(shard_idx)
        if rows_in_shard + len(batch_ids) > rows_per_shard:
            writer.close()
            shard_idx += 1
            rows_in_shard = 0
            writer = _new_writer(shard_idx)

        table = pa.table({
            'task_id': batch_task_ids,
            'offset_x': pa.array(batch_ox, type=pa.int32()),
            'offset_y': pa.array(batch_oy, type=pa.int32()),
            'offset_z': pa.array(batch_oz, type=pa.int32()),
            'input_ids': pa.array(batch_ids, type=pa.list_(pa.int32())),
        })
        writer.write_table(table)
        rows_in_shard += len(batch_ids)
        total_rows += len(batch_ids)

        if mat_idx % 100 == 0 or mat_idx == len(task_ids):
            err(f"[tokenize] {mat_idx}/{len(task_ids)} materials, {total_rows:,} rows")

    if writer is not None:
        writer.close()

    # Summarise codec + vocab for downstream consumers.
    vocab = tokenizer.vocab
    meta = {
        "split": split,
        "split_file": str(split_file) if split_file else None,
        "n_materials": len(task_ids),
        "n_materials_missing": len(missing),
        "missing_task_ids": missing[:50],  # truncate
        "patches_per_material": patches_per_material,
        "patch_size": patch_size,
        "density_codec_name": density_codec,
        "seed": seed,
        "total_rows": total_rows,
        "n_shards": shard_idx + 1 if writer is not None or total_rows > 0 else 0,
        "vocab": {
            "total_size": vocab.total_vocab_size,
            "specials": SPECIAL_TOKENS,
            "position_codec": {
                "log_min": vocab.position_codec.log_min,
                "log_max": vocab.position_codec.log_max,
                "token_mag_bits": vocab.position_codec.token_mag_bits,
            },
            "density_codec": {
                "log_min": vocab.density_codec.log_min,
                "log_max": vocab.density_codec.log_max,
                "token_mag_bits": vocab.density_codec.token_mag_bits,
            },
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    err(f"[tokenize] done: {total_rows:,} rows in {meta['n_shards']} shards "
        f"({len(missing)} task ids missing)")


if __name__ == '__main__':
    main()
