#!/usr/bin/env python
"""Globally shuffle a tokenized parquet label on Modal.

Levanter's ``BlockShuffleConfig`` only shuffles WITHIN sequence windows; adjacent
windows are physically adjacent slices of the parquet, producing a 1024-step
sawtooth in train/loss (spec 44). The fix is to pre-shuffle the parquet so any
windowed shuffle the trainer applies is a no-op on an already-uniform dataset.

This app reads every ``gs://<bucket>/tomat/tokenized/<label>/worker-*/*.parquet``
shard, deterministically permutes all sequences globally (seeded), and writes a
new label ``<label>-shuffled`` with identical shard structure (same worker
count, same shards per worker, same rows per shard). The permutation is saved
as a sidecar JSON so the (output_index → original_index) mapping is
reproducible.

Strategy (single high-memory container, output-driven, no full concat):

  1. List every input shard and its row count -> manifest.
  2. Build a deterministic permutation ``perm`` of length N (numpy RNG, seeded).
     ``perm[i]`` = global input-row index that ends up at global output-row i.
  3. For each output shard, gather the rows it needs:
       - Compute the input global indices it pulls from (= perm slice).
       - Group them by input shard.
       - For each input shard, read ONLY the requested rows (download once,
         take, append).
     This needs each input shard read once per output shard that pulls from it
     (~2560 reads in worst case). To avoid 2560**2 GCS round-trips we cache
     downloaded input tables in RAM until all of their consumers are done.

  4. Each output shard is small (~1936 rows ~= 25 MB compressed) -> write
     locally, then upload in parallel to eu-west4 + us-east5.

Memory budget: input tables held in RAM total ~55 GiB (compressed) ~= ~80 GiB
in Arrow form (mostly int32 lists). We provision a 256 GiB container; the
``--sample-rows`` smoke path needs <8 GiB.

Idempotency: each output shard is skipped if the destination exists with the
expected byte size > 0 (full row-count check needs a read; size check is the
cheap proxy).

Usage (smoke, 100k rows from the first few input shards)::

    modal run scripts/shuffle_tokenized_modal.py::run \\
        --label train-full-v3 --sample-rows 100000 --dry-run

Usage (full)::

    modal run scripts/shuffle_tokenized_modal.py::run --label train-full-v3

Outputs:
    gs://marin-eu-west4/tomat/tokenized/<label>-shuffled/worker-NNN/shard-XXXXX.parquet
    gs://marin-us-east5/tomat/tokenized/<label>-shuffled/worker-NNN/shard-XXXXX.parquet
    gs://marin-eu-west4/tomat/tokenized/<label>-shuffled/_perm.json  (sidecar)

The smoke output goes under ``<label>-shuffled-smoke/`` instead.
"""

from __future__ import annotations

import os
import sys
from functools import partial

import modal

err = partial(print, file=sys.stderr)

DEFAULT_LABEL = "train-full-v3"
DEFAULT_SEED = 20260602
DEFAULT_BUCKETS = (
    "gs://marin-eu-west4/tomat",  # primary (eu-west4 - Modal training reads here)
    "gs://marin-us-east5/tomat",  # mirror (us-east5 - TPU runs read here)
)
PRIMARY_BUCKET = DEFAULT_BUCKETS[0]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow>=15", "gcsfs", "numpy")
)

gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
app = modal.App("tomat-shuffle-tokenized", image=image)


def setup_gcp_creds():
    """Materialize the tomat-gcp-sa secret env var into a file + ADC pointer."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        return
    path = "/tmp/gcp-sa.json"
    with open(path, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


# ---------------------------------------------------------------------------
# Discovery + manifest
# ---------------------------------------------------------------------------

@app.function(cpu=2, memory=2048, timeout=600, secrets=[gcp_secret])
def list_input_shards(label: str, bucket: str) -> list[dict]:
    """List every input parquet shard + its row count, in deterministic order.

    Order: by (worker_idx, shard_idx) — both parsed from filename. This
    defines the global row indexing space (concat order).
    """
    import re

    import gcsfs  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()
    root = f"{bucket[5:]}/tokenized/{label}"
    paths = sorted(p for p in fs.glob(f"{root}/worker-*/*.parquet") if p.endswith(".parquet"))
    err(f"[discover] {len(paths)} input shards under {bucket}/tokenized/{label}")

    rx = re.compile(r"worker-(\d+)/shard-(\d+)\.parquet$")
    keyed: list[tuple[tuple[int, int], str]] = []
    for p in paths:
        m = rx.search(p)
        if not m:
            raise ValueError(f"unexpected shard path layout: {p}")
        wi = int(m.group(1)); si = int(m.group(2))
        keyed.append(((wi, si), p))
    keyed.sort()

    out: list[dict] = []
    for (wi, si), p in keyed:
        with fs.open(p, "rb") as f:
            pf = pq.ParquetFile(f)
            n = pf.metadata.num_rows
        out.append(dict(worker=wi, shard=si, path=f"gs://{p}", rows=n))
    return out


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

def build_permutation(total_rows: int, seed: int):
    """Return a numpy int64 permutation of ``range(total_rows)`` from ``seed``."""
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rows)
    return perm.astype(np.int64, copy=False)


def assign_output_layout(
    input_shards: list[dict],
    *,
    sample_rows: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Derive output-shard sizes that mirror the input layout 1-to-1.

    Returns (effective_inputs, outputs):
      effective_inputs: same as ``input_shards``, but truncated/clipped if
        ``sample_rows`` is set (smoke mode).
      outputs: list of dicts {worker, shard, rows, global_start, global_end}.

    For full mode the output layout == input layout exactly (same workers,
    same per-shard row counts), which keeps any downstream consumer that
    assumes ``meta.json``-style structure happy.

    For sample mode (smoke) the inputs are taken as the prefix of shards
    needed to cover ``sample_rows``, and the output mirrors that prefix
    (rounded down to ``sample_rows`` exactly, by trimming the last shard).
    """
    if sample_rows is None:
        eff = list(input_shards)
        target = sum(s["rows"] for s in eff)
    else:
        eff = []
        target = 0
        for s in input_shards:
            if target >= sample_rows:
                break
            eff.append(s)
            target += s["rows"]
        target = min(target, sample_rows)

    outputs: list[dict] = []
    cursor = 0
    for s in eff:
        rows = s["rows"]
        if sample_rows is not None:
            remaining = target - cursor
            if remaining <= 0:
                break
            rows = min(rows, remaining)
        outputs.append(dict(
            worker=s["worker"],
            shard=s["shard"],
            rows=rows,
            global_start=cursor,
            global_end=cursor + rows,
        ))
        cursor += rows
    assert cursor == target, (cursor, target)
    # Trim inputs to match the truncated total.
    if sample_rows is not None:
        eff_total = 0
        kept = []
        for s in eff:
            if eff_total >= target:
                break
            kept.append(s)
            eff_total += s["rows"]
        eff = kept
    return eff, outputs


# ---------------------------------------------------------------------------
# Main shuffler — one container holds the full input dataset in RAM and
# emits output shards. Within this container we parallelize input reads
# and output writes via a thread pool.
# ---------------------------------------------------------------------------

@app.function(
    cpu=16,
    memory=327_680,  # 320 GiB — Modal's per-function max is 344_064 MiB ≈ 336
                      # GiB. Smoke at 100k rows confirmed ~3 GB working set;
                      # linear extrapolation to full label is ~150 GiB, so
                      # 320 GiB leaves ~2× headroom and stays under the cap.
    timeout=14_400,  # 4h cap; expected wallclock 1-2h.
    secrets=[gcp_secret],
    # No explicit ephemeral_disk — the all-in-RAM path doesn't use scratch,
    # and the default disk size is enough for the parquet temp buffers.
)
def shuffle_all(
    label: str,
    out_label: str,
    seed: int,
    sample_rows: int | None,
    buckets: list[str],
    dry_run: bool,
    n_io_threads: int,
) -> dict:
    """Read all input parquets, permute, write output shards to all buckets."""
    import io
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import gcsfs  # type: ignore
    import numpy as np
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    setup_gcp_creds()
    fs = gcsfs.GCSFileSystem()

    err(f"[shuffle] label={label} -> {out_label}  seed={seed}  sample_rows={sample_rows}")
    err(f"[shuffle] buckets={buckets}  dry_run={dry_run}")

    # ------------------------------------------------------------------
    # 1. Discover input shards via the lister function's logic, inlined
    #    here so we can call it from inside the container without going
    #    back out to .remote().
    # ------------------------------------------------------------------
    primary = buckets[0]
    import re
    rx = re.compile(r"worker-(\d+)/shard-(\d+)\.parquet$")
    root = f"{primary[5:]}/tokenized/{label}"
    paths = sorted(p for p in fs.glob(f"{root}/worker-*/*.parquet") if p.endswith(".parquet"))
    err(f"[shuffle] {len(paths)} input shards under {primary}/tokenized/{label}")

    keyed = []
    for p in paths:
        m = rx.search(p)
        if not m:
            raise ValueError(f"unexpected shard path: {p}")
        keyed.append(((int(m.group(1)), int(m.group(2))), p))
    keyed.sort()

    def _row_count(gpath: str) -> int:
        with fs.open(gpath[5:], "rb") as f:
            return pq.ParquetFile(f).metadata.num_rows

    err("[shuffle] gathering row counts (parallel)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_io_threads) as ex:
        row_counts = list(ex.map(lambda kp: _row_count(f"gs://{kp[1]}"), keyed))
    err(f"[shuffle] row-count pass: {time.time()-t0:.1f}s")

    input_shards = [
        dict(worker=wi, shard=si, path=f"gs://{p}", rows=n)
        for ((wi, si), p), n in zip(keyed, row_counts)
    ]
    total_rows_input = sum(s["rows"] for s in input_shards)
    err(f"[shuffle] input: {len(input_shards)} shards, {total_rows_input} rows")

    # ------------------------------------------------------------------
    # 2. Output layout — 1-to-1 mirror of input (or truncated for smoke).
    # ------------------------------------------------------------------
    eff_inputs, outputs = assign_output_layout(input_shards, sample_rows=sample_rows)
    total_rows = sum(o["rows"] for o in outputs)
    err(f"[shuffle] effective: {len(eff_inputs)} input shards used "
        f"({sum(s['rows'] for s in eff_inputs)} rows); "
        f"emitting {len(outputs)} output shards covering {total_rows} rows")

    # Effective input row count == total rows we permute.
    eff_input_rows = sum(s["rows"] for s in eff_inputs)

    # Compute the per-shard global-row offset (input side).
    input_offsets = []
    cur = 0
    for s in eff_inputs:
        input_offsets.append(cur)
        cur += s["rows"]
    assert cur == eff_input_rows

    # ------------------------------------------------------------------
    # 3. Permutation. We permute over ``eff_input_rows`` (== total_rows
    #    when sample_rows is None). For the smoke path with sample_rows
    #    < eff_input_rows, we just take the first ``total_rows`` of the
    #    permutation — still a uniform random sample.
    # ------------------------------------------------------------------
    err(f"[shuffle] building permutation (seed={seed}, len={eff_input_rows})")
    perm = build_permutation(eff_input_rows, seed)
    output_to_input_global = perm[:total_rows]
    err(f"[shuffle] perm head: {output_to_input_global[:8].tolist()}")

    # ------------------------------------------------------------------
    # 4. Read all effective input shards into RAM (parallel).
    # ------------------------------------------------------------------
    err(f"[shuffle] downloading {len(eff_inputs)} input shards "
        f"with {n_io_threads} threads...")
    t0 = time.time()

    def _read_shard(s):
        with fs.open(s["path"][5:], "rb") as f:
            tbl = pq.read_table(f)
        if tbl.num_rows != s["rows"]:
            raise RuntimeError(
                f"row-count mismatch on {s['path']}: expected {s['rows']}, got {tbl.num_rows}"
            )
        # Re-cast list<int32> → large_list<int32>. The original schema uses
        # int32 list offsets, which overflow when concat_tables sums offsets
        # across all 4.95M rows × 8192 ints ≈ 40 B ints (> int32-max). large_list
        # uses int64 offsets so `take()` after concat doesn't blow up.
        new_fields = []
        for f_ in tbl.schema:
            if pa.types.is_list(f_.type):
                value_type = f_.type.value_type
                new_fields.append(pa.field(f_.name, pa.large_list(value_type)))
            else:
                new_fields.append(f_)
        new_schema = pa.schema(new_fields)
        return tbl.cast(new_schema)

    tables: list[pa.Table] = [None] * len(eff_inputs)  # type: ignore
    with ThreadPoolExecutor(max_workers=n_io_threads) as ex:
        for i, t in enumerate(ex.map(_read_shard, eff_inputs)):
            tables[i] = t
            if (i + 1) % 100 == 0:
                err(f"[shuffle]   read {i+1}/{len(eff_inputs)} shards "
                    f"({time.time()-t0:.1f}s elapsed)")
    err(f"[shuffle] all input shards read in {time.time()-t0:.1f}s")

    # Sanity: all shards share schema (drop redundant inspection).
    schema = tables[0].schema
    for i, t in enumerate(tables[1:], 1):
        if t.schema != schema:
            err(f"[shuffle] WARN schema drift at shard {i}: {t.schema} vs {schema}")

    # ------------------------------------------------------------------
    # 5. Build a global table view via pa.concat_tables — promote_options
    #    keeps int32 list columns aligned. This materializes a single
    #    table reference; the underlying buffers are shared with the
    #    per-shard tables (zero-copy in Arrow).
    # ------------------------------------------------------------------
    err("[shuffle] concatenating input tables (zero-copy)...")
    t0 = time.time()
    global_table = pa.concat_tables(tables, promote_options="default")
    del tables  # let GC reclaim per-shard refs (concat keeps buffers alive)
    err(f"[shuffle] concat: {time.time()-t0:.1f}s "
        f"(num_rows={global_table.num_rows}, num_chunks={global_table.column(0).num_chunks})")
    assert global_table.num_rows == eff_input_rows, (global_table.num_rows, eff_input_rows)

    # ------------------------------------------------------------------
    # 6. Per-output-shard: gather rows via Table.take, write parquet bytes,
    #    upload to all buckets. Done in parallel via a thread pool — Arrow
    #    releases the GIL for take(), and uploads are I/O-bound.
    # ------------------------------------------------------------------
    err(f"[shuffle] writing {len(outputs)} output shards to {len(buckets)} bucket(s)...")
    t0 = time.time()

    def _out_path(bucket: str, worker: int, shard: int) -> str:
        return f"{bucket}/tokenized/{out_label}/worker-{worker:02d}/shard-{shard:05d}.parquet"

    def _process_one(o: dict) -> dict:
        start, end = o["global_start"], o["global_end"]
        take_idx = output_to_input_global[start:end]
        # pa.Table.take accepts int64 Arrow Array.
        idx_arr = pa.array(take_idx, type=pa.int64())
        out_tbl = global_table.take(idx_arr)
        # Write to in-memory bytes — output shards are ~25 MB each so this is fine.
        buf = pa.BufferOutputStream()
        # Match input parquet conventions: zstd compression (input uses ZSTD per
        # pqm), 64-row groups (input has row_group=64). Keeping row-group size
        # the same makes BlockShuffleConfig's io_block_size math match upstream.
        pq.write_table(
            out_tbl,
            buf,
            compression="zstd",
            row_group_size=64,
        )
        data = buf.getvalue().to_pybytes()
        sz = len(data)
        if dry_run:
            return dict(worker=o["worker"], shard=o["shard"], rows=o["rows"],
                        bytes=sz, uploaded=[])
        uploaded = []
        for bucket in buckets:
            gpath = _out_path(bucket, o["worker"], o["shard"])
            with fs.open(gpath[5:], "wb") as f:
                f.write(data)
            uploaded.append(gpath)
        return dict(worker=o["worker"], shard=o["shard"], rows=o["rows"],
                    bytes=sz, uploaded=uploaded)

    # Use a smaller pool than I/O threads — write() is mostly upload (I/O) but
    # take() peaks transient RAM. n_io_threads is fine.
    results = []
    with ThreadPoolExecutor(max_workers=n_io_threads) as ex:
        for i, res in enumerate(ex.map(_process_one, outputs)):
            results.append(res)
            if (i + 1) % 50 == 0:
                done_bytes = sum(r["bytes"] for r in results)
                err(f"[shuffle]   wrote {i+1}/{len(outputs)} shards, "
                    f"{done_bytes/1e9:.2f} GB out, {time.time()-t0:.1f}s elapsed")
    total_out_bytes = sum(r["bytes"] for r in results)
    err(f"[shuffle] wrote {len(results)} output shards "
        f"({total_out_bytes/1e9:.2f} GB) in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 7. Sidecar permutation JSON, meta.json copy, and a manifest.
    # ------------------------------------------------------------------
    sidecar = dict(
        label=label,
        out_label=out_label,
        seed=seed,
        total_rows=total_rows,
        eff_input_rows=eff_input_rows,
        sample_rows=sample_rows,
        permutation_npy=None,  # filled in below
    )
    # Save the perm as a binary .npy too — int64 of length eff_input_rows is
    # 8 * eff_input_rows bytes (~40 MB for 5M rows).
    if not dry_run:
        perm_npy_buf = io.BytesIO()
        np.save(perm_npy_buf, perm)
        perm_bytes = perm_npy_buf.getvalue()
        for bucket in buckets:
            gpath = f"{bucket}/tokenized/{out_label}/_perm.npy"
            with fs.open(gpath[5:], "wb") as f:
                f.write(perm_bytes)
            sidecar["permutation_npy"] = gpath
            err(f"[shuffle] wrote {gpath} ({len(perm_bytes)/1e6:.1f} MB)")

        sidecar_bytes = json.dumps(sidecar, indent=2).encode()
        for bucket in buckets:
            gpath = f"{bucket}/tokenized/{out_label}/_perm.json"
            with fs.open(gpath[5:], "wb") as f:
                f.write(sidecar_bytes)
            err(f"[shuffle] wrote {gpath}")

        # Copy meta.json from the first input worker — vocab layout, codec
        # paths etc. are unchanged. Pull eff_inputs[0]'s worker meta.
        first_worker = eff_inputs[0]["worker"]
        meta_src = f"{primary}/tokenized/{label}/worker-{first_worker:02d}/meta.json"
        try:
            with fs.open(meta_src[5:], "rb") as f:
                meta_bytes = f.read()
            for bucket in buckets:
                # Drop one meta.json under worker-00/ as a "representative"
                # sentinel so eval/train scripts that read worker-00/meta.json
                # still find it.
                gpath = f"{bucket}/tokenized/{out_label}/worker-00/meta.json"
                with fs.open(gpath[5:], "wb") as f:
                    f.write(meta_bytes)
                err(f"[shuffle] wrote {gpath} ({len(meta_bytes)} bytes)")
        except Exception as e:
            err(f"[shuffle] WARN: meta.json copy failed: {type(e).__name__}: {e}")

    return dict(
        label=label,
        out_label=out_label,
        seed=seed,
        sample_rows=sample_rows,
        n_input_shards=len(input_shards),
        n_eff_input_shards=len(eff_inputs),
        n_output_shards=len(outputs),
        total_rows=total_rows,
        total_out_bytes=total_out_bytes,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run(
    label: str = DEFAULT_LABEL,
    out_label: str = "",
    seed: int = DEFAULT_SEED,
    sample_rows: int = 0,
    eu_only: bool = False,
    dry_run: bool = False,
    n_io_threads: int = 64,
):
    """Globally shuffle a tokenized parquet label on Modal.

    Args:
        label:        input tokenized label under gs://<bucket>/tomat/tokenized/
        out_label:    output label (default: ``<label>-shuffled``, or
                       ``<label>-shuffled-smoke`` if ``sample_rows>0``)
        seed:         deterministic permutation seed (logged + saved as sidecar)
        sample_rows:  if >0, run smoke mode on a prefix of the dataset
        eu_only:      skip the us-east5 mirror (faster smoke)
        dry_run:      build everything but don't upload to GCS
        n_io_threads: per-container thread pool for GCS reads/writes
    """
    if not out_label:
        out_label = f"{label}-shuffled" + ("-smoke" if sample_rows else "")
    buckets = [DEFAULT_BUCKETS[0]] if eu_only else list(DEFAULT_BUCKETS)

    err(f"[run] label={label} -> {out_label}  seed={seed}  "
        f"sample_rows={sample_rows or 'ALL'}  buckets={buckets}  dry_run={dry_run}")

    result = shuffle_all.remote(
        label=label,
        out_label=out_label,
        seed=seed,
        sample_rows=sample_rows or None,
        buckets=buckets,
        dry_run=dry_run,
        n_io_threads=n_io_threads,
    )
    err(f"[run] done: {result}")
