#!/usr/bin/env python
"""CPU-only Levanter cache builder for tomat training data.

Avoids the per-file metadata bottleneck of writing the ~54k zarr files of a
3-shard cache directly to Modal volume (~10+ hours on H200×8 GPU container,
mostly idle due to IO). Builds the cache on container-local SSD (`/tmp/cache`),
then tars + uploads the single archive to the volume for durable storage.
Train function (`train_smoke_modal.py::_train_bakeoff_impl`) accepts
`cache_tarball_path` and extracts on startup.

**Parallel mode**: fires one `build_cache` job per source label in parallel
(each at 64 CPUs — Modal's per-function max), then a single `merge_caches`
job that downloads the per-shard tarballs, renumbers cache file indices
(global_idx + source_idx), stitches the top-level `shard_ledger.json`, and
re-tars into the final output. Per-shard builds + merge complete in ~3h
instead of the ~10h a single 64-CPU build would take.

Usage (from laptop):
    TOMAT_VOLUME=tomat-rho-gga-train \\
    modal run --detach scripts/build_cache_modal.py::run \\
        --labels train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3 \\
        --output-tarball cache_tarballs/v4-epochwin-ts123.tar

See `specs/31-cache-build-modal-cpu.md` for the full writeup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


VOLUME_NAME = os.environ.get("TOMAT_VOLUME", "tomat-rho-gga-train")
MOUNT = "/vol"
LOCAL_CACHE = "/tmp/cache"

# Mirror the training image (`train_smoke_modal.py`) so the cache format is
# bit-identical between the build container (here) and the train container.
# Bump OA_MARIN_SHA in lockstep with the training script.
OA_MARIN_SHA = "97eea237598bfe0d0af1143dce92c0c00526a8f0"
OA_MARIN_GIT = f"git+https://github.com/Open-Athena/marin.git@{OA_MARIN_SHA}"
_marin_git_pkgs = " ".join(
    f"'marin-{p} @ {OA_MARIN_GIT}#subdirectory=lib/{p}'"
    for p in ("haliax", "levanter", "fray")
)
EXTRA_FIND_LINKS = [
    "https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799",
    "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4",
]
_extra_find_links_args = " ".join(f"--find-links {u}" for u in EXTRA_FIND_LINKS)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --pre {_extra_find_links_args} "
        f"{_marin_git_pkgs} dupekit "
        # CPU-only jax; cache build doesn't touch GPU.
        "jax 'pyarrow>=15' fsspec",
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-build-cache", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    volumes={MOUNT: volume},
    cpu=64,  # Modal's per-function ceiling (0.125 to 64 cores). Cache build
             # is CPU-bound (per-row parquet→zarr encode in Python). With
             # 64 CPUs we typically saturate Python's GIL-bound throughput.
    memory=131072,
    ephemeral_disk=600 * 1024,  # MB; per-shard cache ~50-70 GB. Modal min 512 GB.
    timeout=21600,  # 6 hours
)
def build_cache(
    label: str,
    output_tarball: str,
    source_idx: int = 1,
    global_idx_offset: int = 0,
) -> dict:
    """Build a per-shard Levanter cache. Pre-stages the source parquets to
    local SSD, runs `LmDataConfig.build_caches('train')`, renames the cache
    file directories to use `source_idx` and start at `global_idx_offset`,
    then tars + uploads a single archive to `/vol/<output_tarball>`.

    With the rename applied here, the merge step becomes a dumb concat —
    no file collisions across per-shard tarballs because each tarball
    already has globally-correct names. See spec 31, "Push offsets into
    per-shard `build_cache` jobs".

    Defaults (`source_idx=1, global_idx_offset=0`) preserve the original
    single-shard behavior for backwards compat.
    """
    import json
    import re
    import shutil
    import subprocess
    import time
    from datetime import datetime

    err = lambda *a: print(*a, file=sys.stderr, flush=True)

    err(f"[build-cache] label={label}")
    err(f"[build-cache] output_tarball={output_tarball}")
    err(f"[build-cache] source_idx={source_idx} global_idx_offset={global_idx_offset}")

    volume.reload()

    # Pre-stage source parquets to local SSD. Each parquet is read ~2k times
    # during cache build (per-row); pulling them once via bulk `cp -r` puts
    # them on fast local storage and avoids the per-row volume read overhead.
    LOCAL_PARQUETS = "/tmp/parquets"
    Path(LOCAL_PARQUETS).mkdir(parents=True, exist_ok=True)
    src = f"{MOUNT}/tokenized/{label}"
    dst = f"{LOCAL_PARQUETS}/{label}"
    err(f"[build-cache] staging {src} → {dst}")
    t0 = time.time()
    subprocess.run(["cp", "-r", src, dst], check=True)
    n_files = sum(1 for _ in Path(dst).rglob("*.parquet"))
    err(f"[build-cache] staged {n_files} parquets in {time.time() - t0:.1f}s")
    parquet_root = f"{LOCAL_PARQUETS}/{label}"

    wds = sorted(Path(parquet_root).glob("worker-*"))
    if not wds:
        raise FileNotFoundError(f"no worker-*/ under {parquet_root}")
    parquet_glob = f"{parquet_root}/worker-*/*.parquet"
    meta = json.loads((wds[0] / "meta.json").read_text())
    vocab_size = meta["vocab"]["total_size"]
    err(f"[build-cache] vocab_size={vocab_size}, patch_size={meta['patch_size']}, "
        f"codec={meta['density_codec_name']}")

    from levanter.data.text import (
        DatasetComponent, LmDataConfig, PrebuiltLmDatasetFormat,
        UrlDatasetSourceConfig,
    )

    Path(LOCAL_CACHE).mkdir(parents=True, exist_ok=True)
    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    component = DatasetComponent(
        source=source, cache_dir=LOCAL_CACHE,
        format=PrebuiltLmDatasetFormat(input_ids_key="input_ids"),
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=LOCAL_CACHE,
        components={"tomat": component},
        block_cross_document_attention=False,
    )

    err(f"[build-cache] calling LmDataConfig.build_caches('train') → {LOCAL_CACHE}")
    t0 = time.time()
    train_caches = data.build_caches("train")
    err(f"[build-cache] cache build took {time.time() - t0:.1f}s; "
        f"got {len(train_caches)} train cache(s)")

    # Rename cache subdirs to apply caller-supplied global_idx_offset, so
    # the resulting tarball has globally-correct names and the downstream
    # merge is a dumb concat. (source_idx is unused — it's not part of the
    # actual Levanter naming convention for single-source builds, which
    # is `<global_idx>_<worker_num>_shard-<N>_parquet`. Kept as a kwarg
    # for backwards compat but ignored.)
    if global_idx_offset != 0:
        train_dir = Path(LOCAL_CACHE) / "train"
        DIR_RE = re.compile(r"^(\d+)_(.+)$")
        rename_count = 0
        for d in sorted(train_dir.iterdir()):
            if not d.is_dir():
                continue
            m = DIR_RE.match(d.name)
            if not m:
                continue
            old_global = int(m.group(1))
            new_name = f"{old_global + global_idx_offset:05d}_{m.group(2)}"
            new_path = train_dir / new_name
            shutil.move(str(d), str(new_path))
            # Fix the inner ledger's absolute path reference (Levanter
            # writes the build-time path; after rename it's stale).
            inner = new_path / "shard_ledger.json"
            if inner.exists():
                try:
                    j = json.loads(inner.read_text())
                    new_abs = str(new_path)
                    if isinstance(j.get("shard_rows"), dict):
                        rows = list(j["shard_rows"].values())[0] if j["shard_rows"] else 0
                        j["shard_rows"] = {new_abs: rows}
                    if isinstance(j.get("finished_shards"), list):
                        j["finished_shards"] = [new_abs]
                    inner.write_text(json.dumps(j))
                except Exception as e:
                    err(f"[build-cache] WARN: inner ledger update failed for {new_path}: {e}")
            rename_count += 1
        # Rewrite the top-level ledger's keyed entries.
        top_ledger_path = train_dir / "shard_ledger.json"
        if top_ledger_path.exists():
            top = json.loads(top_ledger_path.read_text())
            for key in ("shard_rows", "field_counts_by_shard"):
                if isinstance(top.get(key), dict):
                    new_dict = {}
                    for k, v in top[key].items():
                        m = DIR_RE.match(k)
                        if m:
                            new_k = f"{int(m.group(1)) + global_idx_offset:05d}_{source_idx}_{m.group(2)}"
                            new_dict[new_k] = v
                        else:
                            new_dict[k] = v
                    top[key] = new_dict
            if isinstance(top.get("finished_shards"), list):
                new_list = []
                for f in top["finished_shards"]:
                    base = Path(f).name
                    m = DIR_RE.match(base)
                    if m:
                        new_base = f"{int(m.group(1)) + global_idx_offset:05d}_{source_idx}_{m.group(2)}"
                        new_list.append(str(train_dir / new_base))
                    else:
                        new_list.append(f)
                top["finished_shards"] = new_list
            top_ledger_path.write_text(json.dumps(top))
        err(f"[build-cache] renamed {rename_count} cache dirs to "
            f"source_idx={source_idx}, global_offset={global_idx_offset}")

    cache_files = list(Path(LOCAL_CACHE).rglob("*"))
    err(f"[build-cache] cache has {len(cache_files)} entries on disk")

    tar_path = "/tmp/cache.tar"
    err(f"[build-cache] tarring → {tar_path}")
    t0 = time.time()
    subprocess.run(["tar", "-cf", tar_path, "-C", LOCAL_CACHE, "."], check=True)
    tar_size = Path(tar_path).stat().st_size
    err(f"[build-cache] tar took {time.time() - t0:.1f}s, size={tar_size / 1e9:.2f} GB")

    out_path = f"{MOUNT}/{output_tarball.lstrip('/')}"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    err(f"[build-cache] copying tar → {out_path}")
    t0 = time.time()
    subprocess.run(["cp", tar_path, out_path], check=True)
    err(f"[build-cache] copy took {time.time() - t0:.1f}s")
    volume.commit()
    err(f"[build-cache] committed volume; output at /vol/{output_tarball}")

    return {
        "label": label,
        "output_tarball": output_tarball,
        "tar_size_bytes": tar_size,
        "n_cache_entries": len(cache_files),
        "built_at": datetime.now().isoformat(),
    }


@app.function(
    volumes={MOUNT: volume},
    cpu=16,
    memory=65536,
    ephemeral_disk=900 * 1024,  # 3× per-shard tarball size + merged copy
    timeout=10800,
)
def merge_caches(input_tarballs: list[str], output_tarball: str) -> dict:
    """Combine N per-shard cache tarballs into one. Each input tarball was
    built independently with global_idx 0..N (no offset), so we need to
    extract each into a temp dir, renumber its dir entries to use a
    monotonically-increasing global_idx, then concat into a single
    `/tmp/merged/train/` and stitch the top-level `shard_ledger.json`.

    Levanter's single-source cache file naming is
    `<global_idx>_<worker_num>_shard-<N>_parquet`. Renumbering means
    rewriting just the global_idx prefix.
    """
    import json
    import re
    import shutil
    import subprocess
    import time
    from datetime import datetime

    err = lambda *a: print(*a, file=sys.stderr, flush=True)
    err(f"[merge] inputs={input_tarballs}")
    err(f"[merge] output={output_tarball}")

    volume.reload()

    MERGED = "/tmp/merged"
    MERGED_TRAIN = f"{MERGED}/train"
    Path(MERGED_TRAIN).mkdir(parents=True, exist_ok=True)

    cumulative_global_idx = 0
    cumulative_rows = 0
    cumulative_field_counts: dict[str, int] = {}
    merged_shard_rows: dict[str, int] = {}
    finished_shards: list[str] = []
    template_metadata = None
    template_layout = None

    # `<global_idx>_<rest>` — rest is `<worker>_shard-<N>_parquet`.
    DIR_RE = re.compile(r"^(\d+)_(.+)$")

    for input_tar in input_tarballs:
        src = f"{MOUNT}/{input_tar.lstrip('/')}"
        if not Path(src).exists():
            raise FileNotFoundError(f"input tarball not found: {src}")
        stage = f"/tmp/stage_{Path(input_tar).stem}"
        shutil.rmtree(stage, ignore_errors=True)
        Path(stage).mkdir(parents=True, exist_ok=True)
        err(f"[merge] extracting {src} → {stage}")
        t0 = time.time()
        subprocess.run(["tar", "-xf", src, "-C", stage], check=True)
        err(f"[merge] extract took {time.time() - t0:.1f}s")

        stage_train = Path(stage) / "train"
        if not stage_train.exists():
            raise FileNotFoundError(f"no train/ in {stage}")
        top_ledger_path = stage_train / "shard_ledger.json"
        top_ledger = json.loads(top_ledger_path.read_text()) if top_ledger_path.exists() else {}

        shard_dirs = sorted([p for p in stage_train.iterdir() if p.is_dir()])
        err(f"[merge] source has {len(shard_dirs)} shard dirs; renumbering "
            f"with global_offset={cumulative_global_idx}")

        moved = 0
        for d in shard_dirs:
            m = DIR_RE.match(d.name)
            if not m:
                err(f"[merge] WARN: skipping unexpected dir name {d.name}")
                continue
            old_global = int(m.group(1))
            new_name = f"{cumulative_global_idx + old_global:05d}_{m.group(2)}"
            shutil.move(str(d), str(Path(MERGED_TRAIN) / new_name))
            moved += 1
        err(f"[merge] moved {moved} dirs into {MERGED_TRAIN}/")

        # Renumber + merge top-level ledger entries.
        for old_key, val in top_ledger.get("shard_rows", {}).items():
            m = DIR_RE.match(old_key)
            if m:
                new_key = f"{cumulative_global_idx + int(m.group(1)):05d}_{m.group(2)}"
                merged_shard_rows[new_key] = val
        for finished in top_ledger.get("finished_shards", []):
            base = Path(finished).name
            m = DIR_RE.match(base)
            if m:
                new_base = f"{cumulative_global_idx + int(m.group(1)):05d}_{m.group(2)}"
                finished_shards.append(str(Path(MERGED_TRAIN) / new_base))

        cumulative_rows += int(top_ledger.get("total_num_rows", 0))
        for field, count in top_ledger.get("field_counts", {}).items():
            cumulative_field_counts[field] = cumulative_field_counts.get(field, 0) + int(count)
        if template_metadata is None:
            template_metadata = top_ledger.get("metadata", {})
        if template_layout is None:
            template_layout = top_ledger.get("layout", "consolidated")

        cumulative_global_idx += moved
        # Free the stage dir.
        shutil.rmtree(stage, ignore_errors=True)

    # Stitch + write the merged top-level ledger.
    merged_ledger = {
        "total_num_rows": cumulative_rows,
        "shard_rows": merged_shard_rows,
        "is_finished": True,
        "finished_shards": finished_shards,
        "field_counts": cumulative_field_counts,
        "field_counts_by_shard": {},
        "layout": template_layout,
        "metadata": template_metadata,
    }
    (Path(MERGED_TRAIN) / "shard_ledger.json").write_text(json.dumps(merged_ledger))
    err(f"[merge] wrote merged ledger: {cumulative_rows} total rows, "
        f"{len(merged_shard_rows)} shards")

    # Tar the merged cache.
    tar_path = "/tmp/merged_cache.tar"
    err(f"[merge] tarring merged cache → {tar_path}")
    t0 = time.time()
    subprocess.run(["tar", "-cf", tar_path, "-C", MERGED, "."], check=True)
    tar_size = Path(tar_path).stat().st_size
    err(f"[merge] tar took {time.time() - t0:.1f}s, size={tar_size / 1e9:.2f} GB")

    out_path = f"{MOUNT}/{output_tarball.lstrip('/')}"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    err(f"[merge] copying tar → {out_path}")
    t0 = time.time()
    subprocess.run(["cp", tar_path, out_path], check=True)
    err(f"[merge] copy took {time.time() - t0:.1f}s")
    volume.commit()
    err(f"[merge] committed volume; output at /vol/{output_tarball}")

    return {
        "inputs": input_tarballs,
        "output": output_tarball,
        "total_num_rows": cumulative_rows,
        "n_shards": len(merged_shard_rows),
        "tar_size_bytes": tar_size,
        "merged_at": datetime.now().isoformat(),
    }


@app.function(cpu=1, memory=512, timeout=21600)
def orchestrate(labels: list[str], output_tarball: str) -> dict:
    """Fire 3 parallel `build_cache` jobs, wait for all to complete, then
    spawn `merge_caches` (whose .get() this also blocks on). Returns when
    the final merged tarball lands at /vol/<output_tarball>.

    Runs on Modal so `--detach` keeps it alive across local-process exit;
    the local entrypoint just kicks this off and returns the call_id.
    """
    import json

    err_orch = lambda *a: print(*a, file=sys.stderr, flush=True)
    per_shard_tarballs = [f"cache_tarballs/{lbl}.tar" for lbl in labels]

    # Each builder needs to know (a) its source_idx in the merged cache and
    # (b) the global_idx offset its files should start at. The offset is
    # the sum of cache-entry counts for prior labels — i.e., the number of
    # source parquets in shards 1..N-1. Each PQT shard has 256 workers ×
    # ~10 files each ≈ 2560 entries; we read the actual count from the
    # tokenize-time meta.json to avoid the off-by-1 risk.
    offsets = []
    cumulative = 0
    for lbl in labels:
        offsets.append(cumulative)
        # Sum parquet files across workers for this label.
        worker_dirs = sorted(Path(f"{MOUNT}/tokenized/{lbl}").glob("worker-*"))
        n_files = sum(
            sum(1 for _ in wd.glob("*.parquet")) for wd in worker_dirs
        )
        cumulative += n_files
    err_orch(f"[orchestrate] computed offsets: {dict(zip(labels, offsets))}")

    err_orch(f"[orchestrate] firing {len(labels)} build_cache jobs in parallel")
    build_calls = [
        build_cache.spawn(lbl, tar, source_idx=i + 1, global_idx_offset=offsets[i])
        for i, (lbl, tar) in enumerate(zip(labels, per_shard_tarballs))
    ]
    for c, lbl in zip(build_calls, labels):
        err_orch(f"[orchestrate] {lbl}: call_id={c.object_id}")
    err_orch(f"[orchestrate] waiting for {len(build_calls)} per-shard builds")
    results = [c.get() for c in build_calls]
    for r in results:
        err_orch(f"[orchestrate] build done: {r['label']} "
                 f"({r['tar_size_bytes']/1e9:.1f} GB, {r['n_cache_entries']} entries)")
    err_orch(f"[orchestrate] firing merge_caches")
    merge_call = merge_caches.spawn(per_shard_tarballs, output_tarball)
    err_orch(f"[orchestrate] merge call_id={merge_call.object_id}")
    merge_result = merge_call.get()
    err_orch(f"[orchestrate] merge done: {merge_result}")
    return {
        "per_shard_results": results,
        "merge_result": merge_result,
    }


@app.local_entrypoint()
def merge_only(input_tarballs: str, output_tarball: str):
    """Fire `merge_caches` directly with a comma-separated list of input
    tarball paths. Useful for recovery when the per-shard builds already
    succeeded but the orchestrator's merge step failed (e.g., a regex bug)
    — re-fires merge against the cached per-shard tarballs on the volume.
    """
    err = lambda *a: print(*a, file=sys.stderr, flush=True)
    tars = [s.strip() for s in input_tarballs.split(",") if s.strip()]
    if not tars:
        raise ValueError("--input-tarballs must be non-empty")
    err(f"[merge-only] firing merge_caches with {len(tars)} inputs")
    call = merge_caches.spawn(tars, output_tarball)
    err(f"[merge-only] merge call_id={call.object_id}")
    print(call.object_id)


@app.local_entrypoint()
def run(labels: str, output_tarball: str):
    """labels: comma-separated list of `train-full-*` labels under
    `/vol/tokenized/`. output_tarball: relative path on the volume where
    the final merged tar lands.

    Spawns the `orchestrate` function on Modal (which itself fires builds +
    merge); detached from this local process so it survives ctrl-C.
    """
    err = lambda *a: print(*a, file=sys.stderr, flush=True)
    labels_list = [s.strip() for s in labels.split(",") if s.strip()]
    if not labels_list:
        raise ValueError("--labels must be non-empty")
    err(f"[run] spawning orchestrator for {len(labels_list)} labels")
    call = orchestrate.spawn(labels_list, output_tarball)
    err(f"[run] orchestrator call_id={call.object_id}")
    print(call.object_id)
