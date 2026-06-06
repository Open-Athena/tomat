#!/usr/bin/env python
"""Modal H200×8 app: backfill `eval/loss` for v4's saved checkpoints.

v4 (`train-mg-modal-h200x8-tz-v4-bs128-seed42`) trained *without* a
validation carve-out — the Modal training entry never plumbed
`num_validation_sequences` (see spec 47), so `eval/loss` is never logged
to wandb. v4-cont-3 (forthcoming) sets `val_seqs=256`; this script
retroactively computes the same `eval/loss` for every saved v4 ckpt by:

1. Loading the ckpt as `Qwen3MaskGITLMHeadModel` (+1 vocab for MASK).
2. Configuring the MaskGIT loss args (so the model's
   `compute_next_token_loss` runs the bidir-MG path, not the base AR
   path — same code path v4-cont-3 logs against).
3. Building an `LmDataConfig` with
   `num_validation_sequences={"tomat": <N>}` against the same
   `train-full-v3` parquet on the `tomat-rho-gga-train` volume — this
   carves out the same 256-sequence val split that v4-cont-3 uses
   (fixed `PRNGKey(0)` feistel shuffle, last N after shuffle).
4. Running `TaggedEvaluator.evaluate(model)` once per ckpt and writing
   `EvalResult.micro_avg_loss` to
   `gs://marin-eu-west4/tomat/eval/vl-backfill/<run_label>/step-<N>.json`.

Reuses `scripts/eval_modal.py`'s image scaffold (same Open-Athena marin
SHA, same secrets, same H200×8). One container call per (run, step)
keeps each unit ≤ a few minutes (256-seq bidir forward at bs=64 is
fast).

Examples:
    # smoke (one step):
    PYTHONPATH=src python scripts/backfill_vl_modal.py -s 10000

    # full series (every 1k; default when --steps omitted):
    PYTHONPATH=src python scripts/backfill_vl_modal.py
"""
from __future__ import annotations

import sys
from functools import partial

import modal

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"
LMQ_PATH = f"{BUCKET}/codecs/lmq-v2-16k.npz"
VOL_NAME = "tomat-rho-gga-train"
MOUNT = "/vol"

# Match scripts/eval_modal.py exactly so the image is shared across calls.
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
        "'jax[cuda12]' 'pyarrow>=15' fsspec gcsfs "
        "zarr 'pymatgen<2025' boto3",
    )
    # `add_local_python_source("marin")` is necessary but not
    # sufficient: marin-haliax/levanter transitively install
    # marin-core (`lib/marin/src/marin/`), which lands at
    # `/root/marin/` (Modal places non-installed-package mounts under
    # `/root/`) and resolves to the OA marin contents
    # (`marin.__file__ → /root/marin/__init__.py` with only
    # `__init__.py` + `utils.py` + cluster/core/datakit/... subdirs).
    # Our tomat-local `qwen3_density.py` + `eval_mat_nmae.py` (never
    # upstreamed) are silently dropped, so `from marin.qwen3_density
    # import ...` reports ModuleNotFoundError. Force-copy the
    # tomat-local files into `/root/marin/` so they live alongside the
    # OA modules. `copy=True` so they become part of the image layer.
    .add_local_file(
        "marin/qwen3_density.py",
        "/root/marin/qwen3_density.py",
        copy=True,
    )
    .add_local_file(
        "marin/eval_mat_nmae.py",
        "/root/marin/eval_mat_nmae.py",
        copy=True,
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-vl-backfill", image=image)
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
train_volume = modal.Volume.from_name(VOL_NAME)

# Default v4 layout — every existing v4 ckpt lives at this path.
V4_RUN_LABEL = "train-mg-modal-h200x8-tz-v4"
V4_CKPT_LEAF = "train-mg-modal-h200x8-tz-v4-bs128-seed42"
V4_PARQUET_LABEL = "train-full-v3"

# Steps every checkpoint v4 saved: every-1000 from step-1000 to step-40000.
# (The 10k → 20k continuation also saved step-10000 normally; no OBO step
# in the v4 series despite the typical Levanter num_train_steps→step-(N-1)
# rule, because each segment ran past N exactly.)
def _default_steps() -> list[int]:
    return list(range(1000, 41000, 1000))


@app.function(
    gpu="H200:8",
    secrets=[gcp_secret],
    volumes={MOUNT: train_volume},
    timeout=3600,  # 1h cap — single 256-seq bidir-forward eval is minutes.
)
def backfill_one(
    run_label: str,
    ckpt_leaf: str,
    step: int,
    parquet_label: str = V4_PARQUET_LABEL,
    lmq_path: str = LMQ_PATH,
    model_preset: str = "200M",
    val_seqs: int = 256,
    eval_batch: int = 64,
    seq_len: int = 8192,
    # Optional override of the levanter cache dir on the volume. When None,
    # falls back to a vl-backfill-private cache that the script builds itself
    # (slow first run). For runs whose training cache survives on the volume,
    # pass the training cache dir (e.g.
    # `/vol/results/<run_label>/cache`) to skip the rebuild — Levanter's val
    # carve-out is a derived view of the same cache, so reusing it gives
    # bit-identical val sequences without re-tokenizing.
    cache_dir_override: str | None = None,
) -> dict:
    """Eval ONE ckpt's training-time `eval/loss` against val_seqs held-out
    sequences from train-full-v3 and write a per-step JSON to GCS.
    """
    import json
    import os
    import time

    import equinox as eqx
    import fsspec
    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np

    import haliax as hax
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning

    import levanter
    from levanter.checkpoint import load_checkpoint
    from levanter.data.text import (
        DatasetComponent,
        LmDataConfig,
        PrebuiltLmDatasetFormat,
        UrlDatasetSourceConfig,
    )
    from levanter.eval import TaggedEvaluator, _default_lm_eval_loss_fn
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode

    # Materialize the GCP SA creds from the secret's env var → a file +
    # ADC, so gcsfs/tensorstore can read the codec / write the JSON.
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        with open("/tmp/gcp-sa.json", "w") as f:
            f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-sa.json"

    # MG mode + loss-args installation must happen before tracing the
    # model's compute_next_token_loss — same gating the trainer does.
    os.environ["TOMAT_MG_MODE"] = "1"
    os.environ["TOMAT_MG_MASK_PRIOR"] = "absorbing"
    os.environ["TOMAT_MG_LOSS_TYPE"] = "ce"

    train_volume.reload()

    parquet_dir = f"{MOUNT}/tokenized/{parquet_label}"
    worker_dirs = sorted(__import__("pathlib").Path(parquet_dir).glob("worker-*"))
    if not worker_dirs:
        raise FileNotFoundError(f"no worker-*/ shards under {parquet_dir}")
    parquet_glob = f"{parquet_dir}/worker-*/*.parquet"
    meta_path = f"{worker_dirs[0]}/meta.json"
    meta = json.loads(__import__("pathlib").Path(meta_path).read_text())
    vocab_size_raw = meta["vocab"]["total_size"]
    err(f"[vl-backfill] vocab_raw={vocab_size_raw}, patch={meta['patch_size']}, "
        f"codec={meta['density_codec_name']}, model={model_preset}, step={step}")

    # Codec read → density offset math (mirrors train_smoke_modal.py's
    # _train_bakeoff_impl MG branch, lines 582-637).
    with fsspec.open(lmq_path, "rb") as f:
        codec_data = np.load(f, allow_pickle=True)
        codec_recon = np.asarray(codec_data["recon_points"], dtype=np.float32)

    specials = meta["vocab"].get("specials", {})
    has_lattice = "[LATTICE_START]" in specials or "LATTICE_START" in specials
    n_specials = 20 if has_lattice else 18
    n_atoms = 118
    n_ints = 1024
    pc = meta["vocab"]["position_codec"]
    p_mag = pc["token_mag_bits"]
    pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
    pos_total = sum(pos_signed_vocabs)
    density_offset = n_specials + n_atoms + n_ints + pos_total
    n_density_bins = len(codec_recon)
    penalty = 10.0 * float(codec_recon.max())

    # MaskGIT model + loss config (same as training-side).
    from marin.qwen3_density import (
        Qwen3MaskGITConfig,
        build_maskgit_loss_args,
        configure_maskgit_loss,
    )

    mask_id = vocab_size_raw
    vocab_size = vocab_size_raw + 1  # +1 for MASK

    Vocab_mg = hax.Axis("vocab", vocab_size)
    mg_loss_args = build_maskgit_loss_args(
        Vocab=Vocab_mg,
        density_offset=density_offset,
        n_density_bins=n_density_bins,
        codec_recon=codec_recon,
        penalty=penalty,
        mask_id=mask_id,
        prior="absorbing",
        weight=1.0,
        loss_type="ce",
    )
    configure_maskgit_loss(mg_loss_args)
    err(f"[vl-backfill] MaskGIT loss configured: penalty={penalty:.3f} "
        f"offset={density_offset} bins={n_density_bins} mask_id={mask_id}")

    # Eval data: build the same val split v4-cont-3 will use.
    # `num_validation_sequences={"tomat": val_seqs}` + default
    # `shuffle_before_trainval_split=True` carves the last `val_seqs`
    # sequences (after a PRNGKey(0) feistel shuffle) into the val slice.
    # Persist the levanter cache on the Modal Volume so spawns past the
    # first share the (~hour-long) build cost. Per-parquet-label
    # subdir so different vocabs get separate caches.
    # NOTE: when `cache_dir_override` is provided, point at the *training
    # cache* of the source run (e.g. `.../<rl>/cache`) so we skip a
    # ~$25-an-hour rebuild. Levanter's `num_validation_sequences` carves
    # the val split on read from the same cache.
    cache_dir = cache_dir_override or f"{MOUNT}/vl-backfill-cache/{parquet_label}"
    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt_fmt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source, cache_dir=cache_dir, format=prebuilt_fmt,
    )
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        components={"tomat": component},
        block_cross_document_attention=False,
        shuffle=False,  # eval doesn't need full shuffle
        num_validation_sequences={"tomat": val_seqs},
        shuffle_before_trainval_split=True,
    )

    # Trainer config: only used here to get a device mesh + axis
    # mappings; we never call .train() — we hand-build the evaluator and
    # invoke it directly.
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32,
    )
    trainer_cfg = TrainerConfig(
        id="vl-backfill",
        seed=42,
        num_train_steps=1,
        train_batch_size=eval_batch,
        per_device_eval_parallelism=max(1, eval_batch // 8),
        tracker=(),
        mp=mp_policy,
    )
    levanter.initialize(trainer_cfg)
    compute_mapping = trainer_cfg.compute_axis_mapping
    param_mapping = trainer_cfg.parameter_axis_mapping

    model_cfg = Qwen3MaskGITConfig(
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        gradient_checkpointing=False,  # inference only
        hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16,
        intermediate_dim=4096,
    )
    key = jax.random.PRNGKey(0)

    checkpoint = f"{BUCKET}/results/{run_label}/checkpoints/{ckpt_leaf}/step-{step}"
    err(f"[vl-backfill] checkpoint={checkpoint}")

    with trainer_cfg.use_device_mesh():
        Vocab = round_axis_for_partitioning(
            Axis("vocab", vocab_size), compute_mapping,
        )
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[vl-backfill] loading checkpoint")
            model = load_checkpoint(model, checkpoint, subpath="model")
        model = hax.shard_with_axis_mapping(model, param_mapping)
        model = inference_mode(model, True)
        model = mp_policy.cast_to_compute(model)

        Pos = Axis("position", seq_len)
        tagged_eval_sets = data.tagged_eval_sets(Pos)
        if len(tagged_eval_sets) == 0:
            raise RuntimeError(
                "tagged_eval_sets empty after val carve-out — check "
                "num_validation_sequences / cache build."
            )
        EvalBatch = trainer_cfg.EvalBatch

        def loss_fn(m, batch):
            return _default_lm_eval_loss_fn(
                m, batch, EvalBatch=EvalBatch, mp=mp_policy,
            )

        evaluator = TaggedEvaluator(
            EvalBatch=EvalBatch,
            tagged_eval_sets=tagged_eval_sets,
            loss_fn=loss_fn,
            tokenizer=None,
            device_mesh=trainer_cfg.device_mesh,
            axis_mapping=compute_mapping,
        )
        err(f"[vl-backfill] evaluator built; running evaluate()")
        t0 = time.time()
        # Wrap in BOTH `set_mesh` and `axis_mapping` so the dataloader's
        # `jax.jit`-ed `stack_tree` (it doesn't take an explicit axis_mapping)
        # and haliax's named-jit fns alike see the same GPU mesh +
        # compute-mapping that `accum_for_batch` expects. Without these two
        # layers the inner jit traces with a CPU context and the eval blows
        # up with "incompatible devices … GPU vs CPU".
        with hax.partitioning.set_mesh(trainer_cfg.device_mesh):
            with hax.axis_mapping(compute_mapping):
                result = evaluator.evaluate(model)
        elapsed = time.time() - t0
        err(f"[vl-backfill] eval done: micro_avg_loss={result.micro_avg_loss:.4f} "
            f"(elapsed {elapsed:.1f}s)")

    out = {
        "step": int(step),
        "eval_loss": float(result.micro_avg_loss),
        "n_seqs": int(val_seqs),
        "model_preset": model_preset,
        "run_label": run_label,
        "ckpt_leaf": ckpt_leaf,
        "checkpoint": checkpoint,
        "elapsed_seconds": float(elapsed),
    }
    dst = f"{BUCKET}/eval/vl-backfill/{ckpt_leaf}/step-{step}.json"
    with fsspec.open(dst, "w") as f:
        json.dump(out, f, indent=2)
    err(f"[vl-backfill] wrote {dst}")

    # Commit the volume so the (~hour-long) levanter cache build under
    # `{MOUNT}/vl-backfill-cache/{parquet_label}/` is visible to the next
    # spawn — without this each step pays the full build cost again
    # (~$25/h H200x8). Idempotent if cache already exists.
    try:
        train_volume.commit()
        err(f"[vl-backfill] volume committed (cache persisted for next spawn)")
    except Exception as e:
        err(f"[vl-backfill] WARN: volume.commit() failed: {type(e).__name__}: {e}")
    return out


def main(
    steps: str = "",
    run_label: str = V4_RUN_LABEL,
    ckpt_leaf: str = V4_CKPT_LEAF,
    parquet_label: str = V4_PARQUET_LABEL,
    model_preset: str = "200M",
    val_seqs: int = 256,
    eval_batch: int = 64,
    cache_dir_override: str | None = None,
):
    """Fan out one H200×8 call per step.

    `steps`: comma-separated list, e.g. `10000,20000`. Empty → every
    1k from 1000 to 40000.

    Run as a plain Python script (NOT `modal run`). Deploys the app
    once (so the spawned function calls survive the local process
    exit), then `.spawn()`s one call per step. Same pattern as
    `tmp/spawn_mg_v4_cont_*.py` — deploy + spawn keeps queued inputs
    alive without holding the local process open.
    """
    if steps:
        step_list = [int(s.strip()) for s in steps.split(",") if s.strip()]
    else:
        step_list = _default_steps()
    err(f"[vl-backfill] firing {len(step_list)} step(s): {step_list[:5]}"
        f"{'…' if len(step_list) > 5 else ''}")
    err(f"[vl-backfill] deploying app {app.name!r}...")
    app.deploy()
    err(f"[vl-backfill] deployed; spawning {len(step_list)} call(s)...")
    calls: dict[int, str] = {}
    for st in step_list:
        call = backfill_one.spawn(
            run_label=run_label,
            ckpt_leaf=ckpt_leaf,
            step=st,
            parquet_label=parquet_label,
            model_preset=model_preset,
            val_seqs=val_seqs,
            eval_batch=eval_batch,
            cache_dir_override=cache_dir_override,
        )
        err(f"[vl-backfill spawned] step={st} call_id={call.object_id}")
        calls[st] = call.object_id
    err(f"[vl-backfill] all {len(calls)} call(s) spawned. "
        f"`modal app logs tomat-vl-backfill` to watch.")
    print("\n[modal call IDs]")
    for st, cid in sorted(calls.items()):
        print(f"  step={st:<5}  {cid}")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("-s", "--steps", default="",
                  help="comma-sep step list (default: every 1k from 1000 to 40000)")
    @click.option("-r", "--run-label", default=V4_RUN_LABEL,
                  help="run dir under gs://.../results/")
    @click.option("-l", "--ckpt-leaf", default=V4_CKPT_LEAF,
                  help="ckpt subdir under .../checkpoints/")
    @click.option("-p", "--parquet-label", default=V4_PARQUET_LABEL,
                  help="parquet label under /vol/tokenized/")
    @click.option("-m", "--model-preset", default="200M")
    @click.option("-v", "--val-seqs", default=256, type=int)
    @click.option("-b", "--eval-batch", default=64, type=int)
    @click.option("-c", "--cache-dir", default=None,
                  help="Override levanter cache dir (e.g. "
                       "`/vol/results/<rl>/cache`) — reuse a training cache "
                       "instead of building a fresh one.")
    def _cli(steps, run_label, ckpt_leaf, parquet_label, model_preset,
             val_seqs, eval_batch, cache_dir):
        main(
            steps=steps,
            run_label=run_label,
            ckpt_leaf=ckpt_leaf,
            parquet_label=parquet_label,
            model_preset=model_preset,
            val_seqs=val_seqs,
            eval_batch=eval_batch,
            cache_dir_override=cache_dir,
        )
    _cli()
