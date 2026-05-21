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

Examples:
    # GCS checkpoint (TPU run)
    modal run scripts/eval_modal.py --checkpoint \\
      gs://marin-eu-west4/tomat/results/<RL>/checkpoints/<RL>/step-33000

    # Modal-volume checkpoint (Modal run) — path is relative to the volume root
    modal run scripts/eval_modal.py --checkpoint \\
      /vol/results/<RL>/checkpoints/<RID>/step-9999
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
# Same find-links recipe as scripts/{train_smoke,predict}_modal.py.
MARIN_FIND_LINKS = [
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-haliax-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-levanter-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-iris-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-zephyr-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-rigging-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-fray-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/marin-finelog-latest",
    "https://github.com/marin-community/marin/releases/expanded_assets/dupekit-0.1.0-40ac799",
    "https://github.com/marin-community/kitoken/releases/expanded_assets/kitoken-0.10.2-a3012f4",
]
_find_links_args = " ".join(f"--find-links {u}" for u in MARIN_FIND_LINKS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --pre {_find_links_args} "
        "marin-levanter marin-haliax marin-fray dupekit "
        "'jax[cuda12]' 'pyarrow>=15' fsspec gcsfs "
        "zarr 'pymatgen<2025' boto3",
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-eval", image=image)
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
train_volume = modal.Volume.from_name(VOL_NAME)


def _results_path(checkpoint: str, mat_set: str) -> str:
    """Where `eval_mat_nmae.main()` persists this eval's JSON.

    Levanter lays checkpoints out as `<base>/<RL>/checkpoints/<RID>/step-N`,
    so `parts[-4]` is the run-label — matching `eval_mat_nmae`'s own logic.
    """
    parts = checkpoint.rstrip("/").split("/")
    run_label, ckpt_tail = parts[-4], parts[-1]
    return f"{BUCKET}/eval/results/{run_label}/{mat_set or 'default'}/{ckpt_tail}.json"


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
) -> dict:
    """Eval one checkpoint against one mat-set; return the summary dict."""
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
    os.environ.update(
        TOMAT_CHECKPOINT=checkpoint,
        TOMAT_LMQ_PATH=lmq_path,
        TOMAT_LABEL=label,
        TOMAT_MODEL=model,
        TOMAT_EVAL_MAT_SET=mat_set,
        TOMAT_EVAL_N_MATS=str(n_mats),
        TOMAT_EVAL_DECODER=decoder,
        TOMAT_EVAL_BATCH=str(batch),
        TOMAT_EVAL_SPLIT=split,
        # Pure-Pallas flash attention — ~5-10× faster than the reference impl
        # at seq-len 8192, no transformer_engine needed. Exact (same math).
        TOMAT_ATTN_BACKEND="JAX_FLASH",
    )
    err(f"[eval-modal] {mat_set}: {checkpoint}")
    from marin.eval_mat_nmae import main as eval_main
    eval_main()

    # eval_main() already persisted the summary to GCS — read it back so the
    # local entrypoint gets the numbers without re-scraping stdout.
    path = _results_path(checkpoint, mat_set)
    with fsspec.open(path, "r") as f:
        summary = json.load(f)
    summary["results_path"] = path
    return summary


@app.local_entrypoint()
def main(
    checkpoint: str,
    mat_set: str = "both",
    label: str = "train-full-v3",
    lmq_path: str = LMQ_PATH,
    model: str = "200M",
    n_mats: int = 200,
    decoder: str = "median",
):
    """Eval `checkpoint` on `mat_set` (val_200 | train_200 | both)."""
    sets = ["val_200", "train_200"] if mat_set == "both" else [mat_set]
    # Spawn one container per mat-set so both splits run in parallel.
    calls = {
        ms: eval_checkpoint.spawn(checkpoint, ms, label, lmq_path, model, n_mats, decoder)
        for ms in sets
    }
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
