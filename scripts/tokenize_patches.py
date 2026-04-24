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
from tomat.float_codec import FP16Codec, LMQCodec
from tomat.tokenizers.ball import BallTokenizer
from tomat.tokenizers.ball import SPECIAL_TOKENS as BALL_SPECIAL_TOKENS
from tomat.tokenizers.patch import INT_VOCAB_SIZE, PatchTokenizer, SPECIAL_TOKENS

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
@option('-c', '--density-codec', type=click.Choice(sorted(DENSITY_CODECS) + ['lmq']),
        default='two_token_9_12',
        help='Codec variant for density values. "lmq" = empirical Lloyd-Max; '
             'pass --lmq-path to point at the saved codec .npz.')
@option('--lmq-path', type=str, default=None,
        help='Path (local or gs://) to a fitted LMQ codec. Required when --density-codec=lmq.')
@option('--density-log-min', type=float, default=-4.127,
        help='Codec log_min (rho_gga p0.01 with padding). Ignored for LMQ.')
@option('--density-log-max', type=float, default=4.967,
        help='Codec log_max (rho_gga p99.99 with padding).')
@option('--shape', type=click.Choice(['cube', 'ball']), default='cube',
        help='Patch shape: cube (P×P×P, default) or ball (voxels with r²≤r2_max of center).')
@option('--r2-max', type=int, default=75,
        help='Ball squared-radius threshold (only for --shape=ball). '
             'Defaults to 75 (2,777 voxels, ≈cube P=14). '
             'Use 86 for ≈cube P=15 (3,407 voxels), 138 for ≈P=19, 153 for ≈P=20.')
@option('-p', '--patch-size', type=int, default=14,
        help='Patch edge length (voxels). Default 14.')
@option('-o', '--output-dir', type=click.Path(path_type=Path), required=True,
        help='Parquet shard output directory (created).')
@option('-S', '--seed', type=int, default=42, help='RNG seed.')
@option('-R', '--rows-per-shard', type=int, default=2048,
        help='Rows per parquet file (default 2048 ≈ 100 MB at 5k tokens).')
@option('-L', '--pad-to', type=int, default=None,
        help='Right-pad every input_ids sequence to this length with [PAD]=0. '
             'Required for Levanter PrebuiltLmDatasetFormat; error if any row '
             "already exceeds --pad-to. Typical value matches the model's "
             'max_seq_len (e.g., 8192 for tomat-30m).')
@option('-w', '--worker-idx', type=int, default=0,
        help='When parallel: this worker processes task_ids[worker_idx::n_workers]. '
             'Default 0 (serial). Output auto-nested under worker-NN/ when '
             'n_workers > 1.')
@option('-W', '--n-workers', type=int, default=1,
        help='Total worker count (default 1 = serial). See --worker-idx.')
@option('-n', '--n-materials', type=int, default=None,
        help='Debug: cap the number of materials (first N from split). Applied '
             'before worker slicing so n_materials is a global cap.')
def main(
    rho_gga_dir: Path,
    split_file: Path | None,
    split: str,
    patches_per_material: int,
    density_codec: str,
    lmq_path: str | None,
    density_log_min: float,
    density_log_max: float,
    shape: str,
    r2_max: int,
    patch_size: int,
    output_dir: Path,
    seed: int,
    rows_per_shard: int,
    pad_to: int | None,
    worker_idx: int,
    n_workers: int,
    n_materials: int | None,
) -> None:
    # Resolve the task-id list for this split. Split files like
    # `split_limit_22M.json` may store either mp-ID strings directly or
    # 0-based int indices into a sibling `mp_filelist.txt` — detect and
    # resolve the latter.
    if split_file is not None:
        with open(split_file) as f:
            split_data = json.load(f)
        if split not in split_data:
            raise click.UsageError(f"split {split!r} not in {list(split_data)}")
        entries = split_data[split]
        if entries and isinstance(entries[0], int):
            filelist_path = rho_gga_dir / "mp_filelist.txt"
            if not filelist_path.exists():
                raise click.UsageError(
                    f"split file stores int indices but {filelist_path} not found — "
                    "can't resolve indices to mp-IDs"
                )
            with open(filelist_path) as f:
                filelist = [line.strip() for line in f if line.strip()]
            task_ids = [filelist[i] for i in entries]
            err(f"[tokenize] resolved {len(entries)} int indices via {filelist_path.name}")
        else:
            task_ids = list(entries)
    else:
        # No split file: use every zarr under label/
        task_ids = sorted(p.stem for p in (rho_gga_dir / 'label').glob('*.zarr'))
    if n_materials:
        task_ids = task_ids[:n_materials]
    all_count = len(task_ids)
    if n_workers > 1:
        if not 0 <= worker_idx < n_workers:
            raise click.UsageError(
                f"worker_idx {worker_idx} out of range [0, {n_workers})"
            )
        task_ids = task_ids[worker_idx::n_workers]
        output_dir = output_dir / f"worker-{worker_idx:02d}"
    shape_desc = f"shape={shape}"
    shape_desc += f" r2_max={r2_max}" if shape == 'ball' else f" patch_size={patch_size}"
    err(f"[tokenize] split={split} materials={len(task_ids)}"
        f"{f' (slice {worker_idx}/{n_workers} of {all_count})' if n_workers > 1 else ''} "
        f"patches/material={patches_per_material} {shape_desc} "
        f"density_codec={density_codec}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if density_codec == 'lmq':
        if not lmq_path:
            raise click.UsageError("--lmq-path is required when --density-codec=lmq")
        codec = LMQCodec.load(lmq_path)
        err(f"[tokenize] loaded LMQ codec: n_bins={codec.n_bins}, clip_max={codec.clip_max:.4f}")
    else:
        codec = DENSITY_CODECS[density_codec](log_min=density_log_min, log_max=density_log_max)
    if shape == 'ball':
        tokenizer = BallTokenizer(r2_max=r2_max, density_codec=codec)
        specials = BALL_SPECIAL_TOKENS
        err(f"[tokenize] ball shape: r²≤{r2_max} ({len(tokenizer.vocab.position_codec.signed_vocabs)} vocab groups)")
    else:
        tokenizer = PatchTokenizer(patch_size=patch_size, density_codec=codec)
        specials = SPECIAL_TOKENS
    # Each worker gets a decorrelated RNG stream via seed XOR worker_idx —
    # identical serial and parallel runs produce different offsets by design;
    # what's reproducible is a single (seed, n_workers) tuple.
    rng = np.random.default_rng(seed ^ worker_idx)
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
    oversized: list[tuple[str, tuple[int, int, int]]] = []
    overflowed: list[str] = []
    for mat_idx, task_id in enumerate(task_ids, start=1):
        zarr_path = rho_gga_dir / 'label' / f'{task_id}.zarr'
        if not zarr_path.exists():
            missing.append(task_id)
            continue

        sample = load_rho_gga(zarr_path)
        density_shape = sample.data['total'].shape

        # The int vocab maxes out at INT_VOCAB_SIZE (=1024 by default); any
        # grid dim ≥ that would blow up `vocab.int_token(n)`. Skip + log so
        # a small oversized-tail doesn't kill the whole run. Spec 03/04
        # described rho_gga as 40³–448³ but at least a handful of val-split
        # materials have dims up to ~1400.
        if any(d >= INT_VOCAB_SIZE for d in density_shape):
            oversized.append((task_id, tuple(int(d) for d in density_shape)))
            continue

        # Cube samples random offsets (low corners); ball samples random centers.
        if shape == 'ball':
            anchors = tokenizer.random_centers(
                density_shape, n=patches_per_material, rng=rng,
            )
        else:
            anchors = tokenizer.random_offsets(
                density_shape, n=patches_per_material, rng=rng,
            )

        batch_task_ids: list[str] = []
        batch_ox: list[int] = []
        batch_oy: list[int] = []
        batch_oz: list[int] = []
        batch_ids: list[list[int]] = []

        for anc in anchors:
            anchor_t = tuple(int(x) for x in anc)
            if shape == 'ball':
                patch = tokenizer.make_sample(
                    task_id=task_id,
                    density=sample.data['total'],
                    structure=sample.structure,
                    center=anchor_t,
                )
            else:
                patch = tokenizer.make_sample(
                    task_id=task_id,
                    density=sample.data['total'],
                    structure=sample.structure,
                    offset=anchor_t,
                )
            batch_task_ids.append(task_id)
            batch_ox.append(anchor_t[0])
            batch_oy.append(anchor_t[1])
            batch_oz.append(anchor_t[2])
            ids = tokenizer.tokenize(patch)
            if pad_to is not None:
                if len(ids) > pad_to:
                    # Happens when preamble (scales with n_atoms) + density fills
                    # > pad_to. Rather than crash the whole worker, skip the
                    # material and log — matches the oversized-grid-dim skip
                    # pattern. Caller sees meta['n_materials_overflow'] to gauge
                    # the effective corpus size of the variant.
                    err(f"[tokenize] SKIP {task_id}: seq_len={len(ids)} > pad_to={pad_to} "
                        f"(anchor={anchor_t}, n_atoms={len(patch.atomic_numbers)})")
                    # Mark the whole material as overflow + break out of this mat's loop.
                    batch_ids = []
                    break
                ids = ids + [specials["[PAD]"]] * (pad_to - len(ids))
            batch_ids.append(ids)

        # If the material overflowed pad_to (any patch), skip the whole material.
        if not batch_ids:
            overflowed.append(task_id)
            continue

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
        "n_materials_oversized": len(oversized),
        "oversized_task_ids": [
            {"task_id": tid, "shape": list(sh)} for tid, sh in oversized[:50]
        ],
        "n_materials_overflow": len(overflowed),
        "overflow_task_ids": overflowed[:50],
        "patches_per_material": patches_per_material,
        "shape": shape,
        "patch_size": patch_size if shape == 'cube' else f"r{r2_max}",
        "r2_max": r2_max if shape == 'ball' else None,
        "density_codec_name": density_codec,
        "seed": seed,
        "pad_to": pad_to,
        "total_rows": total_rows,
        "n_shards": shard_idx + 1 if writer is not None or total_rows > 0 else 0,
        "vocab": {
            "total_size": vocab.total_vocab_size,
            "specials": specials,
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
        f"({len(missing)} missing, {len(oversized)} oversized-skipped)")


if __name__ == '__main__':
    main()
