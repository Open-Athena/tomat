#!/usr/bin/env python
"""Modal GPU app: run mat-NMAE/NEMD eval on a tomat checkpoint.

One eval path for *all* runs, TPU- or Modal-trained. Levanter/Orbax
checkpoints are device-agnostic, so a TPU-trained checkpoint loads on a GPU
fine — and routing every eval through one GPU path keeps the NMAE numbers
apples-to-apples across runs (no TPU-vs-GPU bf16-rounding drift).

`marin.eval_mat_nmae.main()` does the actual work (env-var driven) and
persists per-mat JSON to `gs://.../eval/results/<RL>/<mat_set>/<step>.json`.
This wraps it in a Modal GPU container and handles checkpoints that live on
the `tomat-rho-gga-train` Modal volume (Modal-trained runs) as well as GCS.

Also exposes `eval_checkpoint_h200x8` (8×H200) as a higher-throughput
backend used by `tomat evals fire --backend modal` — this sidesteps the
iris/marin-iris 14-day client-freshness gate and `marin-*-latest` wheel
rotation that periodically breaks the TPU eval path.

Examples:
    # GCS checkpoint (TPU run) — single A100
    modal run scripts/eval_modal.py --checkpoint \\
      gs://marin-eu-west4/tomat/results/<RL>/checkpoints/<RL>/step-33000

    # Modal-volume checkpoint (Modal run) — path is relative to the volume root
    modal run scripts/eval_modal.py --checkpoint \\
      /vol/results/<RL>/checkpoints/<RID>/step-9999

    # H200×8 backend (used internally by `tomat evals fire --backend modal`)
    modal run scripts/eval_modal.py::main_h200x8 --checkpoint <path> \\
      --run-label train-mg-modal-h200x8-tz-v4-bs128-seed42
"""
from __future__ import annotations

import sys
from functools import partial

import modal

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"
# v3/P19 codec — same one predict_modal.py + the v3 training runs use.
LMQ_PATH = f"{BUCKET}/codecs/lmq-v2-16k.npz"
VOL_NAME = "tomat-rho-gga-train"

# ---- image -------------------------------------------------------------
# Pull marin-* from the Open-Athena fork at a pinned SHA — same scaffold as
# scripts/train_smoke_modal.py. Switched off the upstream
# `marin-{levanter,haliax,fray}-latest` find-links because those wheels are
# rotated daily and an overnight rotation periodically nukes the pin; the
# git+SHA path is immune. See memory `marin-dev-wheel-rotation`.
OA_MARIN_SHA = "97eea237598bfe0d0af1143dce92c0c00526a8f0"
OA_MARIN_GIT = f"git+https://github.com/Open-Athena/marin.git@{OA_MARIN_SHA}"
_marin_git_pkgs = " ".join(
    f"'marin-{p} @ {OA_MARIN_GIT}#subdirectory=lib/{p}'"
    for p in ("haliax", "levanter", "fray")
)

# dupekit + kitoken still come from upstream (no fork divergence there).
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
        "'jax[cuda12]' 'pyarrow>=15' fsspec gcsfs "
        "zarr 'pymatgen<2025' boto3",
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-eval", image=image)
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
train_volume = modal.Volume.from_name(VOL_NAME)


def _results_path(checkpoint: str, mat_set: str, run_label: str | None = None) -> str:
    """Where `eval_mat_nmae.main()` persists this eval's JSON.

    Levanter lays checkpoints out as `<base>/<RL>/checkpoints/<RID>/step-N`,
    so `parts[-4]` is the run-label — matching `eval_mat_nmae`'s own logic.
    `run_label` (when set) overrides parts[-4] — used for Modal-trained runs
    whose ckpt path has RL ≠ leaf and we want results under <leaf>/.
    """
    parts = checkpoint.rstrip("/").split("/")
    rl = run_label or parts[-4]
    ckpt_tail = parts[-1]
    return f"{BUCKET}/eval/results/{rl}/{mat_set or 'default'}/{ckpt_tail}.json"


def _setup_env_and_run(
    checkpoint: str,
    mat_set: str,
    label: str,
    lmq_path: str,
    model: str,
    n_mats: int,
    decoder: str,
    debug_dump_mat: str,
    run_label: str | None,
    mode: str,
    mg_k_steps: int,
    eval_batch: int,
    eval_skip: int,
    output_suffix: str,
    dump_predictions: bool = False,
    voxel_zarrs: bool = False,
    resume_eval: bool = False,
) -> dict:
    """Shared body: set env vars, call `eval_main()`, read back the summary."""
    import json
    import os

    import fsspec

    # Materialize the GCP SA creds from the secret's env var → a file, and
    # point ADC at it — else gcsfs / tensorstore fall back to anonymous (401).
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        with open("/tmp/gcp-sa.json", "w") as f:
            f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-sa.json"

    # train_200 mats live in the `train` zarr split, val_200 in `validation`.
    split = "validation" if mat_set.startswith("val") else "train"
    env: dict[str, str] = dict(
        TOMAT_CHECKPOINT=checkpoint,
        TOMAT_LMQ_PATH=lmq_path,
        TOMAT_LABEL=label,
        TOMAT_MODEL=model,
        TOMAT_EVAL_MAT_SET=mat_set,
        TOMAT_EVAL_N_MATS=str(n_mats),
        TOMAT_EVAL_SKIP=str(eval_skip),
        TOMAT_EVAL_DECODER=decoder,
        TOMAT_EVAL_BATCH=str(eval_batch),
        TOMAT_EVAL_SPLIT=split,
        # Pure-Pallas flash attention — ~5-10× faster than the reference impl
        # at seq-len 8192, no transformer_engine needed. Exact (same math).
        TOMAT_ATTN_BACKEND="JAX_FLASH",
        TOMAT_DEBUG_DUMP_MAT=debug_dump_mat,
        TOMAT_EVAL_MODE=mode,
    )
    if mode == "maskgit":
        env["TOMAT_EVAL_MG_K_STEPS"] = str(mg_k_steps)
    if run_label:
        env["TOMAT_EVAL_RUN_LABEL"] = run_label
    if output_suffix:
        env["TOMAT_EVAL_OUTPUT_SUFFIX"] = output_suffix
    if dump_predictions:
        env["TOMAT_EVAL_DUMP_PREDICTIONS"] = "1"
    if voxel_zarrs:
        env["TOMAT_EVAL_DUMP_VOXEL_ZARRS"] = "1"
    if resume_eval:
        env["TOMAT_EVAL_RESUME"] = "1"
    os.environ.update(env)
    err(f"[eval-modal] mode={mode} mat_set={mat_set} ckpt={checkpoint} "
        f"run_label={run_label or '<auto>'}")
    from marin.eval_mat_nmae import main as eval_main
    eval_main()

    if debug_dump_mat:
        dump = f"gs://marin-eu-west4/tomat/eval/debug/{debug_dump_mat}-dump.npz"
        err(f"[eval-modal] debug per-voxel dump → {dump}")
        return {"debug_dump_mat": debug_dump_mat, "dump_path": dump}

    # eval_main() already persisted the summary to GCS — read it back so the
    # local entrypoint gets the numbers without re-scraping stdout. The mode
    # suffix in `_ms` matches eval_mat_nmae.py's `_mode_suffix` logic.
    mode_suffix = {"free": "-free", "maskgit": "-maskgit"}.get(mode, "")
    ms = (mat_set or "default") + mode_suffix
    parts = checkpoint.rstrip("/").split("/")
    rl = run_label or parts[-4]
    ckpt_tail = parts[-1]
    path = f"{BUCKET}/eval/results/{rl}/{ms}/{ckpt_tail}{output_suffix}.json"
    with fsspec.open(path, "r") as f:
        summary = json.load(f)
    summary["results_path"] = path
    return summary


@app.function(
    gpu="A100-80GB",
    secrets=[gcp_secret],
    volumes={"/vol": train_volume},
    timeout=14400,  # 4h — a 200-mat eval is ~2h; eval_mat_nmae only writes its
                    # GCS JSON after all mats finish, so a timeout loses everything.
)
def eval_checkpoint(
    checkpoint: str,
    mat_set: str,
    label: str = "train-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
    batch: int = 16,
    debug_dump_mat: str = "",
    run_label: str | None = None,
    mode: str = "oneshot",
    mg_k_steps: int = 12,
    eval_skip: int = 0,
    output_suffix: str = "",
    dump_predictions: bool = False,
    voxel_zarrs: bool = False,
    resume_eval: bool = False,
) -> dict:
    """Eval one checkpoint against one mat-set; return the summary dict."""
    return _setup_env_and_run(
        checkpoint=checkpoint, mat_set=mat_set, label=label, lmq_path=lmq_path,
        model=model, n_mats=n_mats, decoder=decoder,
        debug_dump_mat=debug_dump_mat, run_label=run_label,
        mode=mode, mg_k_steps=mg_k_steps, eval_batch=batch,
        eval_skip=eval_skip, output_suffix=output_suffix,
        dump_predictions=dump_predictions,
        voxel_zarrs=voxel_zarrs,
        resume_eval=resume_eval,
    )


@app.function(
    gpu="H200:8",
    secrets=[gcp_secret],
    volumes={"/vol": train_volume},
    timeout=14400,  # 4h cap; a one-shot eval on H200×8 finishes in 5-15 min.
)
def eval_checkpoint_h200x8(
    checkpoint: str,
    mat_set: str,
    label: str = "val-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
    batch: int = 32,
    debug_dump_mat: str = "",
    run_label: str | None = None,
    mode: str = "oneshot",
    mg_k_steps: int = 12,
    eval_skip: int = 0,
    output_suffix: str = "",
    dump_predictions: bool = False,
    voxel_zarrs: bool = False,
    resume_eval: bool = False,
) -> dict:
    """H200×8 backend used by `tomat evals fire --backend modal`. Same body
    as `eval_checkpoint`, just bigger GPU pool — one-shot eval @ n_mats=200
    on 200M should finish in ~5-15 min vs ~2h on a single A100.
    """
    return _setup_env_and_run(
        checkpoint=checkpoint, mat_set=mat_set, label=label, lmq_path=lmq_path,
        model=model, n_mats=n_mats, decoder=decoder,
        debug_dump_mat=debug_dump_mat, run_label=run_label,
        mode=mode, mg_k_steps=mg_k_steps, eval_batch=batch,
        eval_skip=eval_skip, output_suffix=output_suffix,
        dump_predictions=dump_predictions,
        voxel_zarrs=voxel_zarrs,
        resume_eval=resume_eval,
    )


@app.local_entrypoint()
def main(
    checkpoint: str,
    mat_set: str = "both",
    label: str = "train-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
    debug_dump_mat: str = "",
    run_label: str = "",
    mode: str = "oneshot",
):
    """Eval `checkpoint` on `mat_set` (val_200 | train_200 | both).

    `--debug-dump-mat <mp_id>`: eval ONLY that material and dump its per-voxel
    grids (rho_pred / rho_true / emd_grid) to GCS — for pinning the NMAE bug.
    `--run-label <name>`: override the dir under `<BUCKET>/eval/results/`
    (default: ckpt's parts[-4]).
    """
    sets = ["val_200", "train_200"] if mat_set == "both" else [mat_set]
    rl = run_label or None
    # Spawn one container per mat-set so both splits run in parallel.
    calls = {
        ms: eval_checkpoint.spawn(
            checkpoint, ms, label, lmq_path, model, n_mats, decoder,
            debug_dump_mat=debug_dump_mat, run_label=rl, mode=mode,
        )
        for ms in sets
    }
    for ms, call in calls.items():
        s = call.get()
        if "debug_dump_mat" in s:
            print(f"\n=== {ms}: debug dump → {s['dump_path']} ===")
            continue
        print(f"\n=== {ms} ({s['n_mats']} mats) ===")
        for metric in ("nmae", "nemd"):
            print(
                f"  {metric.upper()}  mean {s[f'{metric}_mean']:.4%}  "
                f"median {s[f'{metric}_median']:.4%}  p99 {s[f'{metric}_p99']:.4%}"
            )
        print(f"  → {s['results_path']}")


@app.local_entrypoint()
def main_h200x8(
    checkpoint: str,
    mat_set: str = "both",
    label: str = "val-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
    run_label: str = "",
    mode: str = "oneshot",
    mg_k_steps: int = 12,
):
    """H200×8 entrypoint. Sets-fanout same as `main`, GPU larger."""
    sets = ["val_200", "train_200"] if mat_set == "both" else [mat_set]
    rl = run_label or None
    calls = {
        ms: eval_checkpoint_h200x8.spawn(
            checkpoint, ms, label, lmq_path, model, n_mats, decoder,
            run_label=rl, mode=mode, mg_k_steps=mg_k_steps,
        )
        for ms in sets
    }
    for ms, call in calls.items():
        err(f"[modal-h200x8 spawned] {ms}: call_id={call.object_id}")
    for ms, call in calls.items():
        s = call.get()
        print(f"\n=== {ms} ({s['n_mats']} mats) ===")
        for metric in ("nmae", "nemd"):
            print(
                f"  {metric.upper()}  mean {s[f'{metric}_mean']:.4%}  "
                f"median {s[f'{metric}_median']:.4%}  p99 {s[f'{metric}_p99']:.4%}"
            )
        print(f"  → {s['results_path']}")


@app.local_entrypoint()
def backfill(
    run_label: str,
    steps: str,
    run_id: str = "",
    mat_set: str = "both",
    label: str = "train-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
):
    """Eval many checkpoints of one run in parallel — the NMAE backfill path.

    `steps` is comma-separated (e.g. "33000,40000,50000,60000,70000,79999").
    `run_id` is the checkpoint subdir; defaults to `run_label` (the TPU-run
    convention — Modal runs append `-bs<N>-seed<N>`, pass it explicitly).
    """
    rid = run_id or run_label
    sets = ["val_200", "train_200"] if mat_set == "both" else [mat_set]
    step_list = [s.strip() for s in steps.split(",") if s.strip()]
    base = f"{BUCKET}/results/{run_label}/checkpoints/{rid}"
    calls = {
        (st, ms): eval_checkpoint.spawn(
            f"{base}/step-{st}", ms, label, lmq_path, model, n_mats, decoder
        )
        for st in step_list
        for ms in sets
    }
    print(f"backfill: {len(step_list)} steps × {len(sets)} set(s) = {len(calls)} evals\n")
    for (st, ms), call in calls.items():
        s = call.get()
        print(
            f"  step-{st:<6} {ms:<10} NMAE mean {s['nmae_mean']:.4%} "
            f"median {s['nmae_median']:.4%}  NEMD mean {s['nemd_mean']:.4%}"
        )
