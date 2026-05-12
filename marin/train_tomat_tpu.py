#!/usr/bin/env python
"""Train tomat's Qwen3 patch-LM on Marin's shared TPU cluster.

Mirrors `scripts/train_smoke_modal.py` but targets GCS for data +
checkpoints (Marin's standard data path). Run via:

    cd marin
    uv run iris --cluster=marin job run \\
        --tpu v6e-4 \\
        --env-vars WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python train_tomat_tpu.py

Env-var knobs:
    TOMAT_LABEL           data label under gs://.../tokenized/ (default val-full)
    TOMAT_STEPS           num train steps (default 1000)
    TOMAT_BATCH_SIZE      nominal (total) batch size (default 128)
    TOMAT_SEED            seed (default 42)
    TOMAT_RESULTS_LABEL   overrides W&B run id / checkpoint namespace
    TOMAT_MODEL           model size preset: "30M" (default) or "200M"
    TOMAT_VAL_SEQS        num validation sequences held out (default 0 = no val)
    TOMAT_STEPS_PER_EVAL  eval cadence; default steps // 4 when val is on
    TOMAT_LR              peak learning rate (default 3e-4)
    TOMAT_LR_SCHEDULE     cosine (default) | constant | linear | inv_sqrt | …
                          Choose constant/linear for runs you might extend:
                          cosine's decay couples loss trajectory to step budget, so
                          bumping num_train_steps mid-run causes an LR bump.
    TOMAT_WARMUP          warmup fraction (default 0.1)
    TOMAT_COOLDOWN        cooldown fraction (default None; for WSD use with
                          lr_schedule=constant, e.g. cooldown=0.1)
    TOMAT_DECAY           decay fraction (default None = full decay; cosine only)
    TOMAT_MIN_LR_RATIO    min LR / peak LR for cosine floor (default 0.0)

    TOMAT_DENSITY_L1_WEIGHT float λ on the density-L_1 loss term (default 0 = off).
                            When >0, at density-target positions the loss becomes
                            CE + λ·|E[ρ]−ρ_true| ("add" mode) or pure L_1
                            ("replace" mode). Requires `LMQ` or other known codec
                            so we can build the decode vector.
    TOMAT_DENSITY_L1_MODE   "add" (default) or "replace".
    TOMAT_DENSITY_PENALTY   Float value assigned to non-density tokens in the
                            decode vector — their probability mass gets penalized
                            in L_1 units when the model leaks mass outside the
                            density range. Default: 10 × max(decode_vec).

Prereqs:
- `gs://marin-eu-west4/tomat/tokenized/<label>/worker-*/*.parquet` populated
- ADC refreshed for `ryan.williams@openathena.ai` on hai-gcp-models.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import timedelta
from pathlib import Path

# Multihost-capable JAX init. On v6e-16+ Marin spins up 4+ VM replicas; Levanter's
# `WandbConfig.init` calls `jax_utils.multihost_broadcast_sync` before
# `jax.distributed.initialize()` fires internally, which crashes single-process-
# view code paths. We initialize up-front so later multihost calls work. On
# single-host v6e-4 / v6e-8 this is a no-op (one process, auto-discovery).
import jax
try:
    jax.distributed.initialize()
    print(f"[tomat-tpu] jax.distributed.initialize() done "
          f"(process_index={jax.process_index()}/{jax.process_count()})")
except Exception as e:
    # Single-host envs may raise if no coordinator can be discovered — that's
    # fine, Levanter's own init path handles single-process. Log and continue.
    print(f"[tomat-tpu] jax.distributed.initialize() skipped: {type(e).__name__}: {e}")

# Monkey-patch PassthroughTokenizer.encode to handle non-numeric input — Levanter's
# BPB-computation path calls `tokenizer.encode(".")` to estimate bytes-per-token,
# which crashes on the default PassthroughTokenizer (tries `int(".")`). Fallback
# to a benign [0] so BPB math runs (values are meaningless for integer-only
# tokenizer; train/eval loss stays correct).
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
_orig_passthrough_encode = PassthroughTokenizer.encode

def _safe_passthrough_encode(self, text, *, add_special_tokens=False):
    try:
        return _orig_passthrough_encode(self, text, add_special_tokens=add_special_tokens)
    except ValueError:
        return [0]

PassthroughTokenizer.encode = _safe_passthrough_encode

import jax.numpy as jnp
import jmp

from levanter.checkpoint import CheckpointerConfig

# Levanter's checkpointer schedules OCDBT commits asynchronously; metadata.json
# is written in the commit_callback after the async commit drains. If the
# process exits before the final commit drains, the checkpoint dir has the
# weight `d/` blob but no metadata.json, and `_restore_ocdbt` raises
# `FileNotFoundError: Missing paths: ['…/q_proj/weight', …]` because OCDBT
# manifest hasn't been finalized either. Hit on lat-aware step-7999, 1B
# from-scratch step-4400, and cont-from-4711 step-11288.
#
# Fix: track every Checkpointer the trainer creates, and drain them on
# atexit. wait_until_finished() blocks on the GlobalAsyncCheckpointManager
# until all in-flight commits land, which is what writes metadata.json.
import atexit
import weakref

_active_checkpointers: weakref.WeakSet = weakref.WeakSet()
_orig_create_checkpointer = CheckpointerConfig.create


def _create_checkpointer_and_register(self, *args, **kwargs):
    ckpt = _orig_create_checkpointer(self, *args, **kwargs)
    _active_checkpointers.add(ckpt)
    return ckpt


CheckpointerConfig.create = _create_checkpointer_and_register


def _flush_active_checkpointers(label: str = "atexit"):
    """Drain every active checkpointer's async commit thread.

    Called from two places:
      * `main()` right before returning, after `train_lm_main()` finishes,
        as a deterministic save-point. This is the preferred path —
        runs while the interpreter is fully alive and we can log
        success/failure to wandb cleanly.
      * `atexit` as a last-ditch safety net for crashes / non-clean
        exits (`train_lm_main` raises, signal handler fires, etc.).
        Some shutdown paths run partial cleanup before atexit, so this
        is best-effort.
    """
    for ckpt in list(_active_checkpointers):
        try:
            print(f"[tomat-tpu] draining async checkpoint commits ({label}) …", flush=True)
            ckpt.wait_until_finished()
            print(f"[tomat-tpu] drain ({label}) done", flush=True)
        except Exception as e:
            print(f"[tomat-tpu] checkpoint drain ({label}) failed: {e!r}", flush=True)


atexit.register(_flush_active_checkpointers)


# Signal-event telemetry. iris's `preemption_count` is a cumulative scalar
# polled out-of-band; it doesn't tell us *when* the trainer received a
# SIGTERM or whether the gang shutdown barrier completed. We log timestamped
# events to stdout (and best-effort to wandb if a run is live) so we can
# correlate per-event activity with iris's count.
import datetime
import signal as _signal


def _log_lifecycle_event(event: str, **fields):
    """Print a one-line tag the iris log harvester can grep, and best-effort
    log the same event to wandb. Both paths are robust to in-flight teardown
    (wandb may already be finishing when SIGTERM lands).

    `trainer_started` typically fires *before* `wandb.init` completes (it's
    emitted from `main()` right before calling into Levanter), so we defer
    the wandb side to a daemon thread that polls until `wandb.run` is live,
    then logs the spike + bumps a cumulative `lifecycle/resumes` summary
    counter. SIGTERM-path calls run inline (wandb is live by then and the
    process is about to die)."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[tomat-tpu lifecycle] {ts} event={event} {extras}", flush=True)

    def _log_to_wandb():
        import time
        try:
            import wandb
        except ImportError:
            return
        # Wait up to 60 s for wandb.init to complete (it typically takes 5-10 s
        # post-trainer_started, longer on cold caches).
        for _ in range(120):
            if wandb.run is not None:
                break
            time.sleep(0.5)
        else:
            return
        try:
            payload = {f"lifecycle/{event}": 1, **{f"lifecycle/{k}": v for k, v in fields.items()}}
            wandb.run.log(payload)
            if event == "trainer_started":
                cur = wandb.run.summary.get("lifecycle/trainer_starts", 0)
                wandb.run.summary["lifecycle/trainer_starts"] = cur + 1
                wandb.run.summary["lifecycle/resumes"] = cur  # = starts - 1
                wandb.run.summary["lifecycle/last_started_at"] = ts
        except Exception:
            pass

    import threading
    threading.Thread(target=_log_to_wandb, daemon=True).start()


def _handle_sigterm(signum, _frame):
    _log_lifecycle_event("sigterm_received", signum=signum)
    # Re-raise default-handler behavior so the JAX coordination service
    # gets the shutdown signal it expects (don't swallow).
    _signal.signal(signum, _signal.SIG_DFL)
    os.kill(os.getpid(), signum)


_signal.signal(_signal.SIGTERM, _handle_sigterm)
from levanter.data.text import (
    DatasetComponent,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig, main as train_lm_main
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

BUCKET = os.environ.get("TOMAT_BUCKET", "gs://marin-eu-west4/tomat")


MODEL_PRESETS = {
    # (hidden, layers, heads, kv_heads, ffn) — head_dim = hidden // heads
    # 30M: what all earlier runs used (hidden=512, head_dim=128, 6 layers).
    "30M": dict(hidden_dim=512, num_layers=6, num_heads=4, num_kv_heads=4, intermediate_dim=2048),
    # 200M: Chinchilla-zone for ~20 B tokens; hidden=1024, head_dim=64, 12 layers.
    # params ≈ embed(7M tied) + 12 × (4·1024² attn + 3·1024·4096 ffn) ≈ 208M.
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    # 1B: hidden=2048, head_dim=128, 20 layers, ffn=5632 (≈2.75×).
    # params ≈ embed(14M tied) + 20 × (4·2048² + 3·2048·5632) ≈ 1.04 B.
    "1B": dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}


def main():
    label = os.environ.get("TOMAT_LABEL", "val-full")
    steps = int(os.environ.get("TOMAT_STEPS", "1000"))
    batch_size = int(os.environ.get("TOMAT_BATCH_SIZE", "128"))
    seed = int(os.environ.get("TOMAT_SEED", "42"))
    results_label_env = os.environ.get("TOMAT_RESULTS_LABEL")
    model_preset = os.environ.get("TOMAT_MODEL", "30M")
    val_seqs = int(os.environ.get("TOMAT_VAL_SEQS", "0"))
    steps_per_eval_env = os.environ.get("TOMAT_STEPS_PER_EVAL")

    parquet_glob = f"{BUCKET}/tokenized/{label}/worker-*/*.parquet"
    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    import fsspec
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    # Sequence length is the dataset's pad_to (drives both model max_seq_len
    # and trainer train_seq_len). v3-p15 uses 4608 vs v3 baseline's 8192.
    seq_len = int(meta.get("pad_to") or 8192)
    print(f"[tomat-tpu] label={label}, vocab_size={vocab_size}, "
          f"patch={meta['patch_size']}, codec={meta['density_codec_name']}, "
          f"model={model_preset}, val_seqs={val_seqs}, seq_len={seq_len}")

    results_label = results_label_env or f"{label}-tpu-{model_preset}-bs{batch_size}-seed{seed}"
    run_id = results_label

    # cache_dir resolution. Default: per-results-label, so each run rebuilds.
    # Override via TOMAT_CACHE_DIR to share a cache across runs over the same
    # parquet inputs (avoids the Zephyr cache-build crashloop seen on v6e-32
    # under heavy preempt pressure — see marin GH "cache-build-brittleness").
    # Convention for sharing: TOMAT_CACHE_DIR=gs://.../cache/<label>/ keyed on
    # the data label, since cache contents depend only on input parquets +
    # cache_options (currently fixed batch_size=128). Caller is responsible
    # for using the same dir across runs that should share, and a different
    # dir if any cache-determining knob (parquet glob, batch_size) changes.
    cache_dir_env = os.environ.get("TOMAT_CACHE_DIR")
    if cache_dir_env:
        cache_dir = cache_dir_env
        print(f"[tomat-tpu] cache_dir=SHARED {cache_dir}")
    else:
        cache_dir = f"{BUCKET}/results/{results_label}/cache"
        print(f"[tomat-tpu] cache_dir=PER-RUN {cache_dir}")

    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    prebuilt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=prebuilt,
    )
    # Shuffle config. By default Levanter's `LmDataConfig.shuffle=False` —
    # batches are read in cache order, which for tomat means consecutive
    # patches from the same material (M=32 or 64 sequences/mat). That gives
    # batches with only ~BS/M unique mats, hurting gradient quality.
    # `TOMAT_SHUFFLE_WINDOW_BLOCKS > 0` enables `BlockShuffleConfig`:
    #   - `io_block_size` (rows per IO chunk; default = M from meta) keeps
    #     each block as one mat's patches — cache-friendly sequential reads.
    #   - `window_blocks` (default 0 = off) is the within-window mixing
    #     radius; 1024 blocks × M rows ≈ 32–65k rows per shuffle window.
    shuffle_window_blocks = int(os.environ.get("TOMAT_SHUFFLE_WINDOW_BLOCKS", "0"))
    shuffle_io_block_size = int(
        os.environ.get("TOMAT_SHUFFLE_IO_BLOCK_SIZE", "0")
    ) or int(meta.get("patches_per_material", 32))
    if shuffle_window_blocks > 0:
        shuffle_cfg: bool | int | BlockShuffleConfig = BlockShuffleConfig(
            io_block_size=shuffle_io_block_size,
            window_blocks=shuffle_window_blocks,
        )
        print(f"[tomat-tpu] shuffle: BlockShuffle(io_block_size={shuffle_io_block_size}, "
              f"window_blocks={shuffle_window_blocks})")
    else:
        shuffle_cfg = False
        print(f"[tomat-tpu] shuffle: OFF (set TOMAT_SHUFFLE_WINDOW_BLOCKS>0 to enable)")

    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=cache_dir,
        components={"tomat": component},
        block_cross_document_attention=False,
        shuffle=shuffle_cfg,
        # Hold out TOMAT_VAL_SEQS sequences from train for validation. Levanter
        # types this as `dict[str, int]` keyed by component name — one entry per
        # DatasetComponent. We have a single "tomat" component. val_seqs=0 skips.
        num_validation_sequences={"tomat": val_seqs} if val_seqs > 0 else None,
    )

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown TOMAT_MODEL={model_preset!r}; expected one of {list(MODEL_PRESETS)}")

    # Density-loss wiring. Gate on TOMAT_LMQ_PATH presence (the codec is
    # required for both EMD and L_1 density terms). The weight knob is only
    # meaningful in CE+L1 ablations ("add" mode); under density_only=True
    # it's pure LR scaling (vestigial — defaults to 1.0).
    density_l1_weight = float(os.environ.get("TOMAT_DENSITY_L1_WEIGHT", "1.0"))
    density_l1_mode = os.environ.get("TOMAT_DENSITY_L1_MODE", "add")
    density_loss_type = os.environ.get("TOMAT_DENSITY_LOSS_TYPE", "l1")
    density_only_loss = os.environ.get("TOMAT_DENSITY_ONLY_LOSS", "0") == "1"
    density_l1_penalty_env = os.environ.get("TOMAT_DENSITY_PENALTY")
    lmq_path_env = os.environ.get("TOMAT_LMQ_PATH")
    if lmq_path_env:
        from qwen3_density import (
            Qwen3DensityConfig,
            build_density_loss_args,
            configure_density_loss,
        )
        model_config_cls = Qwen3DensityConfig
        print(f"[tomat-tpu] density loss: weight={density_l1_weight}, "
              f"mode={density_l1_mode}, type={density_loss_type}, "
              f"density_only={density_only_loss}, lmq_path={lmq_path_env}")

        # Inline-load the codec .npz since the `tomat` package isn't on the
        # Marin workspace PYTHONPATH (iris bundles only this directory).
        class _LMQCodecInline:
            def __init__(self, boundaries, recon_points, clip_max):
                self.boundaries = boundaries
                self.recon_points = recon_points
                self.clip_max = clip_max
            @property
            def n_bins(self):
                return len(self.recon_points)

        def _load_lmq(path: str) -> _LMQCodecInline:
            import fsspec as _fs
            with _fs.open(path, "rb") as f:
                data = np.load(f, allow_pickle=True)
                return _LMQCodecInline(
                    boundaries=np.asarray(data["boundaries"], dtype=np.float32),
                    recon_points=np.asarray(data["recon_points"], dtype=np.float32),
                    clip_max=float(data["clip_max"]),
                )

        import numpy as np
        lmq_codec = _load_lmq(lmq_path_env)

        # Compute density vocab offsets per PatchVocab layout
        # (specials=18, atoms=118, ints=1024, pos=pos_total, density=lmq_codec.n_bins)
        n_specials = 18
        n_atoms_in_vocab = 118
        n_ints = 1024
        pc = meta["vocab"]["position_codec"]
        p_mag = pc["token_mag_bits"]
        pos_signed_vocabs = tuple((2 if i == 0 else 1) << b for i, b in enumerate(p_mag))
        pos_total = sum(pos_signed_vocabs)
        DENSITY_OFFSET = n_specials + n_atoms_in_vocab + n_ints + pos_total
        print(f"[tomat-tpu] density offset in vocab = {DENSITY_OFFSET}, "
              f"density vocab range = [{DENSITY_OFFSET}, {DENSITY_OFFSET + lmq_codec.n_bins})")

        import haliax as hax
        Vocab = hax.Axis("vocab", vocab_size)
        penalty_val = (
            float(density_l1_penalty_env)
            if density_l1_penalty_env is not None
            else 10.0 * float(lmq_codec.recon_points.max())
        )
        density_loss_args = build_density_loss_args(
            Vocab=Vocab,
            density_offset=DENSITY_OFFSET,
            n_density_bins=lmq_codec.n_bins,
            codec_recon=lmq_codec.recon_points,
            penalty=penalty_val,
            weight=density_l1_weight,
            mode=density_l1_mode,
            loss_type=density_loss_type,
            density_only=density_only_loss,
        )
        configure_density_loss(density_loss_args)
        print(f"[tomat-tpu] density-L_1 configured with PENALTY={penalty_val:.4f}")
    else:
        model_config_cls = Qwen3Config

    grad_ckpt = os.environ.get("TOMAT_GRADIENT_CHECKPOINTING", "1") == "1"
    print(f"[tomat-tpu] gradient_checkpointing={grad_ckpt}")
    model = model_config_cls(
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        gradient_checkpointing=grad_ckpt,
        **MODEL_PRESETS[model_preset],
    )

    # W&B conventions mirror the Modal side so filters/overlays are consistent.
    trackers = (
        WandbConfig(
            id=run_id,
            resume="allow",
            project=f"tomat-{meta['density_codec_name']}-P{meta['patch_size']}",
            group=f"M32-Ntpu-{model_preset}",
            tags=[
                "scale",
                "tpu",
                "marin",
                f"mats{meta['n_materials']}",
                f"bs{batch_size}",
                f"seed{seed}",
                f"model{model_preset}",
                *(["val"] if val_seqs > 0 else []),
            ],
            save_code=False,
        ),
        JsonLoggerConfig(),
    )

    checkpointer = CheckpointerConfig(
        base_path=f"{BUCKET}/results/{results_label}/checkpoints",
        save_interval=timedelta(minutes=10),
        keep=[{"every": 1000}],
    )

    # Eval cadence: if val is on, every steps // 4 by default (so 4 evals in a
    # 2000-step run — useful plot resolution). With no val, default keeps the
    # old behavior (one mid-run eval, effectively a no-op).
    if steps_per_eval_env:
        steps_per_eval = int(steps_per_eval_env)
    elif val_seqs > 0:
        steps_per_eval = max(steps // 4, 1)
    else:
        steps_per_eval = max(steps // 2, 1)

    # bf16 compute + fp32 params/optimizer. TPU v6e has ~31 GB HBM/chip;
    # fp32 activations blow this at 200M/bs=32-per-chip. bf16 compute is also
    # ~2× faster on TPU tensor cores. Standard config for any >30M model.
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )

    profile_enabled = os.environ.get("TOMAT_PROFILE", "1") == "1"
    profile_start = int(os.environ.get("TOMAT_PROFILE_START", "20"))
    profile_num_steps = int(os.environ.get("TOMAT_PROFILE_NUM_STEPS", "25"))
    print(f"[tomat-tpu] profiler: enabled={profile_enabled} start_step={profile_start} num_steps={profile_num_steps}")

    trainer = TrainerConfig(
        id=run_id,
        seed=seed,
        num_train_steps=steps,
        train_batch_size=batch_size,
        steps_per_eval=steps_per_eval,
        tracker=trackers,
        checkpointer=checkpointer,
        mp=mp,
        profiler=ProfilerConfig(
            enabled=profile_enabled,
            start_step=profile_start,
            num_steps=profile_num_steps,
        ),
    )

    lr = float(os.environ.get("TOMAT_LR", "3e-4"))
    lr_schedule = os.environ.get("TOMAT_LR_SCHEDULE", "cosine")
    warmup = float(os.environ.get("TOMAT_WARMUP", "0.1"))
    min_lr_ratio = float(os.environ.get("TOMAT_MIN_LR_RATIO", "0.0"))
    cooldown_env = os.environ.get("TOMAT_COOLDOWN")
    decay_env = os.environ.get("TOMAT_DECAY")

    adam_kwargs: dict = dict(
        learning_rate=lr,
        weight_decay=0.0,
        warmup=warmup,
        min_lr_ratio=min_lr_ratio,
        lr_schedule=lr_schedule,
        beta1=0.9,
        beta2=0.95,
    )
    if cooldown_env is not None:
        adam_kwargs["cooldown"] = float(cooldown_env)
    if decay_env is not None:
        adam_kwargs["decay"] = float(decay_env)
    print(f"[tomat-tpu] optimizer: lr={lr}, schedule={lr_schedule}, warmup={warmup}, "
          f"cooldown={cooldown_env}, decay={decay_env}, min_lr_ratio={min_lr_ratio}")

    optimizer = AdamConfig(**adam_kwargs)

    # Continuation across crashes is handled by Levanter's native checkpoint
    # auto-discovery: same TOMAT_RESULTS_LABEL → same checkpointer.base_path
    # → trainer resumes from the latest step-N ckpt with optimizer state and
    # step counter intact. Use `tomat train --resume LABEL` for that path.
    # Warm-start (model-only load, fresh optimizer) used to live here under
    # TOMAT_INIT_FROM_CHECKPOINT; removed because the only times we reached
    # for it were resume use cases dressed as warm-starts (different job
    # name → different output_dir → auto-discovery couldn't see the prior
    # ckpts), and the fresh-Adam-on-trained-weights collision corrupted the
    # 1B cont-from-4711 run at step ~2000.

    config = TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        train_seq_len=seq_len,
    )

    print("[tomat-tpu] calling levanter.main.train_lm.main …")
    _log_lifecycle_event("trainer_started", label=results_label, steps=steps)
    train_lm_main(config)
    # Drain ckpt commits at a deterministic point (before atexit, before
    # interpreter teardown can race with tensorstore HTTP callbacks). The
    # atexit handler still runs as a safety net for crash paths.
    _flush_active_checkpointers(label="post-train")
    _log_lifecycle_event("trainer_finished")
    print("[tomat-tpu] done")


if __name__ == "__main__":
    main()
