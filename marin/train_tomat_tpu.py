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
import sys
from datetime import timedelta
from pathlib import Path

# Multihost-capable JAX init. Historically we called `jax.distributed.initialize()`
# up-front here because Levanter's `WandbConfig.init` would call
# `multihost_broadcast_sync` before jax was initialized and crash single-process
# code paths. The new levanter / iris pin (rev 7115c21d) handles init itself
# via `iris.runtime.jax_init.initialize_jax` (see `levanter/distributed.py`),
# and our early call now silently fails AND poisons the XLA backend so
# levanter's later attempt raises:
#   RuntimeError: jax.distributed.initialize() must be called before any JAX
#   calls that might initialise the XLA backend.
# Skip our own call entirely; let levanter's init path run unmolested.
import jax  # noqa: F401

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
    counter.

    SIGTERM- and trainer-finished paths are *terminal* events: the
    process is about to exit, so we MUST log inline + flush before
    returning, otherwise the wandb POST never lands (per spec 37 — the
    daemon thread is killed by `wandb.finish()` / interpreter teardown
    before it can flush)."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[tomat-tpu lifecycle] {ts} event={event} {extras}", flush=True)

    def _log_to_wandb_now():
        """Inline log + flush — used for terminal events. Returns when the
        POST has been queued (or fail-fast if wandb isn't up)."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        try:
            payload = {f"lifecycle/{event}": 1, **{f"lifecycle/{k}": v for k, v in fields.items()}}
            wandb.run.log(payload)
            # Ask the wandb backend to flush its outbound queue before we
            # return. Without this the process can exit before the network
            # POST lands — see spec 37 for the trail of bodies (8 runs
            # SUCCEEDED with no `lifecycle/trainer_finished` row).
            try: wandb.run.log({}, commit=True)
            except Exception: pass
        except Exception:
            pass

    def _log_to_wandb_deferred():
        """Daemon-thread path — used for `trainer_started` which fires
        before `wandb.init` finishes. Polls for `wandb.run` up to 60 s."""
        import time
        try:
            import wandb
        except ImportError:
            return
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

    # Terminal events: log inline. `trainer_started` happens before
    # `wandb.init` returns → must use the deferred polling path.
    if event in ("trainer_finished", "sigterm_received"):
        _log_to_wandb_now()
    else:
        import threading
        threading.Thread(target=_log_to_wandb_deferred, daemon=True).start()


def _handle_sigterm(signum, _frame):
    _log_lifecycle_event("sigterm_received", signum=signum)
    # Best-effort wandb finish so the run state flips to "finished" (with
    # the sigterm flagged) rather than lingering "running" → "failed" after
    # heartbeat timeout. Wrapped: we're about to re-raise SIGTERM and don't
    # want to mask the original signal if wandb is misbehaving.
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish(exit_code=143, quiet=True)  # 128 + 15 (SIGTERM)
    except Exception:
        pass
    # Re-raise default-handler behavior so the JAX coordination service
    # gets the shutdown signal it expects (don't swallow).
    _signal.signal(signum, _signal.SIG_DFL)
    os.kill(os.getpid(), signum)


_signal.signal(_signal.SIGTERM, _handle_sigterm)


def _maybe_spawn_pyspy_daemon():
    """If TOMAT_PYSPY=1, dump the trainer's Python stacks to GCS periodically.

    Why: throughput/duration says JAX-side step is fast, but wall-clock is
    much higher (~65 s/step on the v5p run that started this). Want to see
    where the host process is during the gap. Uses `py-spy dump` (text
    snapshot of current stacks across all threads), cheap enough to run
    every TOMAT_PYSPY_INTERVAL seconds without disturbing training.

    Output: `{BUCKET}/results/{results_label}/pyspy/{timestamp}.txt` so
    Rafal / anyone else can grab them without ssh.
    """
    if os.environ.get("TOMAT_PYSPY") != "1":
        return
    interval = int(os.environ.get("TOMAT_PYSPY_INTERVAL", "60"))
    duration = int(os.environ.get("TOMAT_PYSPY_RECORD_SECONDS", "0"))  # 0 = dumps only; >0 = `record` flame graphs of that length
    bucket = os.environ.get("TOMAT_BUCKET", "gs://marin-eu-west4/tomat")
    label = os.environ.get("TOMAT_RESULTS_LABEL") or os.environ.get("TOMAT_LABEL", "unknown")
    out_prefix = f"{bucket}/results/{label}/pyspy"

    import shutil
    import subprocess as _sp
    import threading
    import time as _t
    pyspy = shutil.which("py-spy")
    gsutil = shutil.which("gsutil")
    # Diagnostic prints unconditional — easy to grep in worker logs when
    # debugging "pyspy was supposed to be on but produced nothing in GCS".
    print(f"[pyspy] enable check: which(py-spy)={pyspy} which(gsutil)={gsutil}", flush=True)
    if pyspy is None:
        print("[pyspy] TOMAT_PYSPY=1 set but py-spy not on PATH; skipping. "
              "Add `py-spy` to the deps that the iris worker installs, or "
              "set TOMAT_PYSPY_INSTALL=1 to `uv pip install` it inline.",
              flush=True)
        if os.environ.get("TOMAT_PYSPY_INSTALL") == "1":
            print("[pyspy] TOMAT_PYSPY_INSTALL=1 → attempting `uv pip install py-spy` …", flush=True)
            try:
                _sp.run(["uv", "pip", "install", "py-spy"], check=True, timeout=60)
                pyspy = shutil.which("py-spy")
                print(f"[pyspy] post-install which(py-spy)={pyspy}", flush=True)
            except Exception as e:
                print(f"[pyspy] install failed: {e}", flush=True)
        if pyspy is None:
            return
    pid = os.getpid()
    print(f"[pyspy] enabled: pid={pid} interval={interval}s record={duration}s → {out_prefix}/", flush=True)

    def _loop():
        while True:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            try:
                if duration > 0:
                    # Flame-graph SVG over a fixed window. `record` blocks
                    # for the duration; that's fine — we're in a daemon
                    # thread. Use `--idle` so blocked-on-IO frames show up.
                    local = f"/tmp/pyspy-{label}-{ts}.svg"
                    _sp.run(
                        [pyspy, "record", "-p", str(pid), "-o", local, "-d", str(duration), "--idle"],
                        check=False, capture_output=True, timeout=duration + 30,
                    )
                    remote = f"{out_prefix}/{ts}.svg"
                else:
                    # Text snapshot of all threads. Fast (<1 s).
                    local = f"/tmp/pyspy-{label}-{ts}.txt"
                    with open(local, "wb") as f:
                        _sp.run(
                            [pyspy, "dump", "-p", str(pid)],
                            check=False, stdout=f, stderr=_sp.STDOUT, timeout=30,
                        )
                    remote = f"{out_prefix}/{ts}.txt"
                # Upload to GCS via gsutil so Rafal/others can grab without ssh.
                _sp.run(["gsutil", "-q", "cp", local, remote], check=False, timeout=120)
                print(f"[pyspy] {remote}")
            except Exception as e:
                print(f"[pyspy] iteration failed: {e}")
            _t.sleep(interval)

    threading.Thread(target=_loop, name="pyspy-daemon", daemon=True).start()


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

# Map: GCE region → mirrored cache bucket. Kept in sync with `_CACHE_TARGETS`
# in the `tomat` CLI's `cache mirror` command. When `TOMAT_SHARE_CACHE=1` is
# set, the trainer detects its own GCE zone via the metadata server and reads
# the cache from the local-region bucket — keeps results/ckpts canonical on
# TOMAT_BUCKET while avoiding the cross-region IO that halves MFU on v5p.
_REGION_TO_CACHE_BUCKET = {
    "us-central1":  "gs://marin-us-central1/tomat",
    "us-east1":     "gs://marin-us-east1/tomat",
    "us-east5":     "gs://marin-us-east5/tomat",
    "europe-west4": "gs://marin-eu-west4/tomat",
}


def _detect_gce_region() -> str | None:
    """GCE region (e.g. 'us-east5') from the metadata server; None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            zone_path = r.read().decode("utf-8")  # 'projects/123/zones/us-east5-a'
        zone = zone_path.rsplit("/", 1)[-1]
        return zone.rsplit("-", 1)[0]
    except Exception as e:
        print(f"[tomat-tpu] WARN: GCE zone detect failed: {e}", flush=True)
        return None


def _pick_cache_bucket(default: str) -> str:
    """Region-local cache bucket. Falls back to `default` if region unknown."""
    region = _detect_gce_region()
    if region and region in _REGION_TO_CACHE_BUCKET:
        b = _REGION_TO_CACHE_BUCKET[region]
        print(f"[tomat-tpu] zone-local cache: region={region} → {b}", flush=True)
        return b
    print(f"[tomat-tpu] WARN: no zone-local cache bucket (region={region}); "
          f"falling back to {default}", flush=True)
    return default


def _bucket_region(cache_dir: str) -> str | None:
    """Best-effort 'gs://marin-<short>/...' → GCE region. None if unparseable.

    Bucket-name region suffixes use 'eu-' shorthand (e.g. 'marin-eu-west4')
    while GCE region names spell it 'europe-'; normalize so comparisons
    against `_detect_gce_region()` work.
    """
    import re
    m = re.match(r"gs://marin-([a-z]+-[a-z]+\d+)/", cache_dir)
    if not m:
        return None
    short = m.group(1)
    return "europe-" + short[3:] if short.startswith("eu-") else short


def _assert_cache_local(cache_dir: str) -> None:
    """Fail loud on cross-region cache reads; previously these silently
    halved MFU (the 65 s/step v5p host stall we just spent a week on).

    Compares the bucket's region (parsed from `cache_dir`) against this
    worker's region (GCE metadata). Skipped if either side is undetermined
    or `TOMAT_ALLOW_XREG_CACHE=1` is set.
    """
    if os.environ.get("TOMAT_ALLOW_XREG_CACHE") == "1":
        print(f"[tomat-tpu] TOMAT_ALLOW_XREG_CACHE=1 → skipping x-reg check", flush=True)
        return
    cache_region = _bucket_region(cache_dir)
    worker_region = _detect_gce_region()
    if not cache_region or not worker_region:
        print(f"[tomat-tpu] x-reg check skipped (cache_region={cache_region} "
              f"worker_region={worker_region})", flush=True)
        return
    if cache_region != worker_region:
        raise RuntimeError(
            f"cross-region cache read: cache_dir={cache_dir} (region={cache_region}) "
            f"≠ worker region={worker_region}. Mirror the cache to this region with "
            f"`tomat cache mirror`, or set TOMAT_ALLOW_XREG_CACHE=1 to override."
        )
    print(f"[tomat-tpu] x-reg check: cache+worker both in {worker_region} ✓", flush=True)


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
    _maybe_spawn_pyspy_daemon()
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

    # Sequence packing (scheme 2a, spec 36). Auto-enabled when the dataset's
    # meta.json says it was tokenized with --pack. TOMAT_PACKED env can force-
    # enable for ad-hoc smokes; set =0 to force off even on packed data
    # (debugging only — yields garbage attention).
    _packed_env = os.environ.get("TOMAT_PACKED")
    if _packed_env is None:
        packed = bool(meta.get("packed", False))
    else:
        packed = _packed_env == "1"
    PAD_ID = 0  # specials["[PAD]"] in all tokenizer variants
    if packed:
        # Reuse PAD as the segment-boundary sentinel. Levanter's
        # block_cross_document_attention path computes
        #   segment_ids = cumsum(tokens == eos_id)
        # so each PAD increments the segment counter. Trailing PAD region
        # ends up in its own (zero-loss) segment past the last real
        # sub-sequence. Monkey-patch PassthroughTokenizer.eos_token_id so the
        # rest of Levanter's machinery wires this through automatically.
        from levanter.data.passthrough_tokenizer import (
            PassthroughTokenizer as _PT,
        )
        _orig_eos = _PT.eos_token_id.fget
        _PT.eos_token_id = property(lambda self: PAD_ID)  # type: ignore[assignment]
        print(f"[tomat-tpu] packed mode ON: PAD_ID={PAD_ID} acts as segment boundary "
              f"(meta.packed={meta.get('packed')}, env={_packed_env!r})")
    else:
        print(f"[tomat-tpu] packed mode OFF (meta.packed={meta.get('packed')}, env={_packed_env!r})")

    # MaskGIT mode env vars — read early so vocab_size bump propagates into
    # data config, model config, and everywhere else that uses vocab_size.
    mg_mode = os.environ.get("TOMAT_MG_MODE", "0") == "1"
    mg_mask_prior = os.environ.get("TOMAT_MG_MASK_PRIOR", "cosine")
    mg_loss_type = os.environ.get("TOMAT_MG_LOSS_TYPE", "ce")
    if mg_mask_prior not in ("cosine", "uniform", "high", "absorbing"):
        raise ValueError(
            f"TOMAT_MG_MASK_PRIOR must be cosine/uniform/high/absorbing, "
            f"got {mg_mask_prior!r}"
        )
    if mg_loss_type not in ("ce", "ce_emd", "emd"):
        raise ValueError(
            f"TOMAT_MG_LOSS_TYPE must be ce/ce_emd/emd, got {mg_loss_type!r}"
        )

    # Scheduled-sampling (FR-aware AR training) env vars — spec 31.
    # SS is mutually exclusive with MaskGIT (different model subclass, different
    # forward path); disallow both at once to keep the configuration sane.
    ss_mode = os.environ.get("TOMAT_SS_MODE", "0") == "1"
    # New ε-distribution mini-DSL (TOMAT_SS_EPS_DIST). Examples:
    #   "1"               ε=1 always
    #   "U(0,1)"          ε ~ Uniform(0, 1)
    #   ".3(1)+.7U(0,1)"  30% ε=1 + 70% ε ~ U(0,1)
    # See `parse_eps_dist` in `qwen3_density.py`. Legacy `TOMAT_SS_EPS_MIN/MAX`
    # remain supported; if `TOMAT_SS_EPS_DIST` is empty they're translated to
    # `U(min,max)` internally.
    ss_eps_dist = os.environ.get("TOMAT_SS_EPS_DIST", "").strip() or None
    ss_eps_min = float(os.environ.get("TOMAT_SS_EPS_MIN", "0.0"))
    ss_eps_max = float(os.environ.get("TOMAT_SS_EPS_MAX", "0.25"))
    ss_sampler = os.environ.get("TOMAT_SS_SAMPLER", "median")
    if ss_sampler not in ("median", "argmax", "sample"):
        raise ValueError(
            f"TOMAT_SS_SAMPLER must be median/argmax/sample, got {ss_sampler!r}"
        )
    if ss_eps_dist is None:
        if not (0.0 <= ss_eps_min <= 1.0):
            raise ValueError(
                f"TOMAT_SS_EPS_MIN must be in [0, 1], got {ss_eps_min!r}"
            )
        if not (0.0 <= ss_eps_max <= 1.0):
            raise ValueError(
                f"TOMAT_SS_EPS_MAX must be in [0, 1], got {ss_eps_max!r}"
            )
        if ss_eps_min > ss_eps_max:
            raise ValueError(
                f"TOMAT_SS_EPS_MIN ({ss_eps_min!r}) must be <= TOMAT_SS_EPS_MAX ({ss_eps_max!r})"
            )
    if ss_mode and mg_mode:
        raise ValueError(
            "TOMAT_SS_MODE=1 and TOMAT_MG_MODE=1 are mutually exclusive "
            "(SS is an AR-side training intervention; MaskGIT replaces the AR "
            "objective entirely)."
        )

    # F1 mode (spec 34): 1 token/atom + continuous (x,y,z) sidecar.
    # Requires parquet built with `--atom-encoding f1` (carries an `atom_xyz`
    # column alongside `input_ids`). Mutually exclusive with MG/SS for now —
    # the F1 model is a subclass of Qwen3DensityLMHeadModel without MG/SS
    # forward overrides; combinations would need a richer subclass tree.
    f1_mode = os.environ.get("TOMAT_F1_MODE", "0") == "1"
    f1_num_freqs = int(os.environ.get("TOMAT_F1_NUM_FREQS", "10"))
    if f1_mode and (mg_mode or ss_mode):
        raise ValueError(
            "TOMAT_F1_MODE=1 cannot combine with TOMAT_MG_MODE / TOMAT_SS_MODE "
            "(F1 currently only composes with the base AR / density-only paths)."
        )
    if f1_mode and meta.get("atom_encoding") != "f1":
        raise ValueError(
            "TOMAT_F1_MODE=1 requires parquet built with --atom-encoding f1; "
            f"got meta.atom_encoding={meta.get('atom_encoding')!r}. "
            f"Re-tokenize with `tokenize_patches.py -F f1`."
        )

    # Bump vocab_size by 1 for [MASK] token in MaskGIT mode.
    # MASK_ID = old total_size (new token appended at the end of vocab).
    MASK_ID: int | None = None
    if mg_mode:
        MASK_ID = vocab_size
        vocab_size = vocab_size + 1
        print(f"[tomat-tpu] MaskGIT mode: MASK_ID={MASK_ID}, "
              f"new vocab_size={vocab_size}, prior={mg_mask_prior}, "
              f"loss_type={mg_loss_type}")

    print(f"[tomat-tpu] label={label}, vocab_size={vocab_size}, "
          f"patch={meta['patch_size']}, codec={meta['density_codec_name']}, "
          f"model={model_preset}, val_seqs={val_seqs}, seq_len={seq_len}")

    results_label = results_label_env or f"{label}-tpu-{model_preset}-bs{batch_size}-seed{seed}"
    run_id = results_label

    # cache_dir resolution. Three modes, in priority order:
    #   1. TOMAT_CACHE_DIR — explicit full path, used as-is.
    #   2. TOMAT_SHARE_CACHE=1 — detect this worker's GCE region and read the
    #      mirrored cache from the region-local bucket (cache is RO at train
    #      time; we mirrored it across {us-central1, us-east1, us-east5,
    #      eu-west4} so any zone iris picks has a local copy).
    #   3. Per-run default under TOMAT_BUCKET — rebuilds each run.
    # Results/ckpts always go to TOMAT_BUCKET regardless of which mode is
    # used, so resumes across zone changes still find their checkpoints.
    cache_dir_env = os.environ.get("TOMAT_CACHE_DIR")
    if cache_dir_env:
        cache_dir = cache_dir_env
        print(f"[tomat-tpu] cache_dir=SHARED (explicit) {cache_dir}")
    elif os.environ.get("TOMAT_SHARE_CACHE") == "1":
        cache_bucket = _pick_cache_bucket(default=BUCKET)
        cache_dir = f"{cache_bucket}/cache/{label}/"
        print(f"[tomat-tpu] cache_dir=SHARED (zone-local) {cache_dir}")
    else:
        cache_dir = f"{BUCKET}/results/{results_label}/cache"
        print(f"[tomat-tpu] cache_dir=PER-RUN {cache_dir}")
    _assert_cache_local(cache_dir)

    source = UrlDatasetSourceConfig(train_urls=[parquet_glob])
    if f1_mode:
        # F1: carry both `input_ids` and `atom_xyz` through the cache.
        # `f1_data.F1PrebuiltLmDatasetFormat` registers itself under the
        # `LmDatasetFormatBase` ChoiceRegistry name `f1_prebuilt`.
        from f1_data import F1PrebuiltLmDatasetFormat
        prebuilt = F1PrebuiltLmDatasetFormat(
            input_ids_key="input_ids", atom_xyz_key="atom_xyz",
        )
        print(f"[tomat-tpu] F1 mode ON: num_freqs={f1_num_freqs}")
    else:
        prebuilt = PrebuiltLmDatasetFormat(input_ids_key="input_ids")
    component = DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=prebuilt,
    )
    # Shuffle config. Levanter's `LmDataConfig.shuffle` defaults to False —
    # batches read in cache order, which for tomat means consecutive patches
    # from the same material (M=32 or 64 sequences/mat): only ~BS/M unique
    # mats per batch, hurting gradient quality. So shuffle is ON by default
    # here — `TOMAT_SHUFFLE_WINDOW_BLOCKS` (default 1024) → `BlockShuffleConfig`:
    #   - `io_block_size` (rows per IO chunk; default = M from meta) keeps
    #     each block as one mat's patches — cache-friendly sequential reads.
    #   - `window_blocks` is the within-window mixing radius; 1024 blocks ×
    #     M rows ≈ 32–65k rows per shuffle window. Set to 0 to disable.
    shuffle_window_blocks = int(os.environ.get("TOMAT_SHUFFLE_WINDOW_BLOCKS", "1024"))
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
        print(f"[tomat-tpu] shuffle: OFF (TOMAT_SHUFFLE_WINDOW_BLOCKS=0)")

    # `data_cfg_cls` selects between standard `LmDataConfig` and the F1
    # subclass that routes `F1PrebuiltLmDatasetFormat` components through
    # `F1PrebuiltLmDataset → F1LmExample` (so `atom_xyz` survives to
    # `compute_next_token_loss`).
    if f1_mode:
        from f1_data import F1LmDataConfig
        data_cfg_cls = F1LmDataConfig
    else:
        data_cfg_cls = LmDataConfig
    data = data_cfg_cls(
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

    # Cache-build-only short-circuit. Wired via `tomat cache build <label>` —
    # decouples cache materialization from training so a build failure doesn't
    # leave a half-built cache the trainer then trips over (the missing
    # `shard_ledger.json` + `input_ids/` in us-central1 cost us a day).
    if os.environ.get("TOMAT_BUILD_CACHE_ONLY") == "1":
        print("[tomat-tpu] TOMAT_BUILD_CACHE_ONLY=1 → building caches, will exit before trainer")
        for split in ("train", "validation"):
            print(f"[tomat-tpu] data.build_caches({split!r}) → {cache_dir}{split}/", flush=True)
            data.build_caches(split)
            print(f"[tomat-tpu] build_caches({split!r}) done", flush=True)
        print("[tomat-tpu] all caches built; exiting")
        sys.exit(0)

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

    # Inline-load helper reused by both density-loss and MaskGIT paths.
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

    # Compute density vocab offsets (reused by both paths when lmq_path_env set).
    def _compute_density_offset(meta_dict):
        specials = meta_dict["vocab"].get("specials", {})
        _lat = "[LATTICE_START]" in specials or "LATTICE_START" in specials
        _n_spec = 20 if _lat else 18
        _n_atoms = 118
        _n_ints = 1024
        _pc = meta_dict["vocab"]["position_codec"]
        _p_mag = _pc["token_mag_bits"]
        _pos_sv = tuple((2 if i == 0 else 1) << b for i, b in enumerate(_p_mag))
        return _n_spec + _n_atoms + _n_ints + sum(_pos_sv)

    if f1_mode:
        # F1 (spec 34): 1 token/atom + sinusoidal-xyz addend at the embed layer.
        # Composes with the standard density loss if TOMAT_LMQ_PATH is set
        # (Qwen3F1LMHeadModel inherits the density-aware loss from
        # Qwen3DensityLMHeadModel). With no density loss configured, F1
        # behaves as plain Qwen3 + xyz addend + CE loss.
        from qwen3_density import Qwen3F1Config
        if lmq_path_env:
            from qwen3_density import (
                build_density_loss_args,
                configure_density_loss,
            )
            import numpy as np
            lmq_codec = _load_lmq(lmq_path_env)
            DENSITY_OFFSET = _compute_density_offset(meta)
            print(f"[tomat-tpu] F1+density: density_offset={DENSITY_OFFSET}, "
                  f"density_range=[{DENSITY_OFFSET}, {DENSITY_OFFSET + lmq_codec.n_bins})")
            import haliax as hax
            Vocab_f1 = hax.Axis("vocab", vocab_size)
            penalty_val = (
                float(density_l1_penalty_env)
                if density_l1_penalty_env is not None
                else 10.0 * float(lmq_codec.recon_points.max())
            )
            density_loss_args = build_density_loss_args(
                Vocab=Vocab_f1,
                density_offset=DENSITY_OFFSET,
                n_density_bins=lmq_codec.n_bins,
                codec_recon=lmq_codec.recon_points,
                penalty=penalty_val,
                weight=density_l1_weight,
                mode=density_l1_mode,
                loss_type=density_loss_type,
                density_only=density_only_loss,
                pad_id=PAD_ID if packed else None,
            )
            configure_density_loss(density_loss_args)
            print(f"[tomat-tpu] F1+density-L_1 configured with PENALTY={penalty_val:.4f}")
        else:
            print(f"[tomat-tpu] F1 mode with plain CE loss (no TOMAT_LMQ_PATH)")
        model_config_cls = Qwen3F1Config
        model_extra_kwargs = {"f1_num_freqs": f1_num_freqs}
    elif mg_mode and lmq_path_env:
        # MaskGIT path: configure loss args (masking itself is applied inside
        # Qwen3MaskGITLMHeadModel.compute_next_token_loss via JAX random ops,
        # so no host-side collator is needed — everything stays JIT-traceable).
        from qwen3_density import (
            Qwen3MaskGITConfig,
            build_maskgit_loss_args,
            configure_maskgit_loss,
        )

        import numpy as np
        lmq_codec = _load_lmq(lmq_path_env)
        DENSITY_OFFSET = _compute_density_offset(meta)
        print(f"[tomat-tpu] MaskGIT: density_offset={DENSITY_OFFSET}, "
              f"density_range=[{DENSITY_OFFSET}, {DENSITY_OFFSET + lmq_codec.n_bins})")

        import haliax as hax
        Vocab_mg = hax.Axis("vocab", vocab_size)
        penalty_val = (
            float(density_l1_penalty_env)
            if density_l1_penalty_env is not None
            else 10.0 * float(lmq_codec.recon_points.max())
        )
        mg_loss_args = build_maskgit_loss_args(
            Vocab=Vocab_mg,
            density_offset=DENSITY_OFFSET,
            n_density_bins=lmq_codec.n_bins,
            codec_recon=lmq_codec.recon_points,
            penalty=penalty_val,
            mask_id=MASK_ID,
            prior=mg_mask_prior,
            weight=density_l1_weight,
            loss_type=mg_loss_type,
        )
        configure_maskgit_loss(mg_loss_args)
        print(f"[tomat-tpu] MaskGIT loss configured: penalty={penalty_val:.4f}")
        model_config_cls = Qwen3MaskGITConfig

    elif mg_mode:
        raise ValueError(
            "TOMAT_MG_MODE=1 requires TOMAT_LMQ_PATH to be set "
            "(the codec is needed to identify density positions)."
        )
    elif ss_mode and not lmq_path_env:
        raise ValueError(
            "TOMAT_SS_MODE=1 requires TOMAT_LMQ_PATH to be set "
            "(the codec is needed to identify density positions for SS)."
        )
    elif lmq_path_env:
        from qwen3_density import (
            Qwen3DensityConfig,
            build_density_loss_args,
            configure_density_loss,
        )
        # Scheduled-sampling (spec 31): subclass density model + configure SS.
        if ss_mode:
            from qwen3_density import (
                Qwen3SSConfig,
                build_ss_args,
                configure_ss,
            )
            model_config_cls = Qwen3SSConfig
            print(f"[tomat-tpu] scheduled-sampling: "
                  f"eps_min={ss_eps_min}, eps_max={ss_eps_max}, "
                  f"sampler={ss_sampler}")
        else:
            model_config_cls = Qwen3DensityConfig
        print(f"[tomat-tpu] density loss: weight={density_l1_weight}, "
              f"mode={density_l1_mode}, type={density_loss_type}, "
              f"density_only={density_only_loss}, lmq_path={lmq_path_env}")

        import numpy as np
        lmq_codec = _load_lmq(lmq_path_env)
        DENSITY_OFFSET = _compute_density_offset(meta)
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
            pad_id=PAD_ID if packed else None,
        )
        configure_density_loss(density_loss_args)
        print(f"[tomat-tpu] density-L_1 configured with PENALTY={penalty_val:.4f}")
        if ss_mode:
            ss_args = build_ss_args(
                density_offset=DENSITY_OFFSET,
                n_density_bins=lmq_codec.n_bins,
                eps_dist=ss_eps_dist,
                eps_min=ss_eps_min,
                eps_max=ss_eps_max,
                sampler=ss_sampler,
            )
            configure_ss(ss_args)
            print(f"[tomat-tpu] scheduled-sampling configured: "
                  f"eps_dist={ss_args.eps_dist} sampler={ss_args.sampler!r} "
                  f"density_range=[{ss_args.density_lo}, {ss_args.density_hi})")
    else:
        model_config_cls = Qwen3Config

    grad_ckpt = os.environ.get("TOMAT_GRADIENT_CHECKPOINTING", "1") == "1"
    print(f"[tomat-tpu] gradient_checkpointing={grad_ckpt}")
    # Extra config-time kwargs threaded into the chosen `model_config_cls`.
    # F1 path sets `f1_num_freqs`; other branches default empty so the
    # Qwen3Config / Qwen3DensityConfig / Qwen3MaskGITConfig / Qwen3SSConfig
    # constructors aren't given unexpected keys.
    if "model_extra_kwargs" not in locals():
        model_extra_kwargs = {}
    model = model_config_cls(
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        gradient_checkpointing=grad_ckpt,
        **MODEL_PRESETS[model_preset],
        **model_extra_kwargs,
    )

    # W&B conventions mirror the Modal side so filters/overlays are consistent.
    trackers = (
        WandbConfig(
            # New TPU runs go to OA's corporate team. Resumes of pre-2026-05-15
            # runs in PrinceOA will keep landing there (wandb resume=allow honors
            # the run's original entity if `id` already exists). Override via
            # TOMAT_WANDB_ENTITY env if needed (e.g. local-only smokes).
            entity=os.environ.get("TOMAT_WANDB_ENTITY", "open-athena"),
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
        # Retain every-1000 ckpts long-term + every-100 for recent-resume.
        # Prior `keep=[{"every": 1000}]` lost the v5p smoke (step 588) when
        # a preempt cascade killed the job before reaching step 1000.
        # Save itself runs every 10 min regardless; this is retention-only.
        # On preempt-thrashed pools (e.g. mg-1 on degraded us-east1-d in
        # May-26) `_last_save_time` resets per restart and the 10min temp
        # save can fail to fire — but the fix is to use a healthier pool,
        # not to tighten this config.
        # NOTE: Levanter's CheckpointerConfig.__post_init__ does
        # `interval["until"]` (not `.get`), so the trailing/unbounded entry
        # MUST include `"until": None` explicitly — omitting it raises
        # KeyError at task startup. See marin/lib/levanter/.../checkpoint.py:1076.
        keep=[{"every": 100, "until": 1000}, {"every": 1000, "until": None}],
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
    # Explicit wandb.finish() — otherwise the wandb run state lingers as
    # "running" until heartbeat times out, then flips to "failed/crashed"
    # even on a clean exit (v5p-pyspy-3 showed this).
    try:
        import wandb
        if wandb.run is not None:
            print("[tomat-tpu] wandb.finish() …", flush=True)
            wandb.finish(exit_code=0, quiet=True)
    except Exception as e:
        print(f"[tomat-tpu] wandb.finish() failed: {e}", flush=True)
    print("[tomat-tpu] done")


if __name__ == "__main__":
    main()
