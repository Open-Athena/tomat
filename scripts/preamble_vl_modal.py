#!/usr/bin/env python
"""Modal GPU app: preamble-VL ablation eval.

Question: does the v4-epochwin MaskGIT model actually use the per-patch
preamble (atoms / positions / lattice / grid) when predicting density tokens,
or could it get a comparable VL purely from local density-token context?

Methodology:
  - Pick N deterministic val patches from `train-full-v3` (tail slice).
  - v4-epochwin trains with `maskgit_prior="absorbing"` — at every step
    100% of density tokens are replaced with MASK_ID and the loss is
    bidirectional CE on those MASKED positions vs ground-truth. So we mirror
    exactly that at eval: replace EVERY density token (positions
    [preamble_len+1 .. DENS_END_idx)) with MASK_ID, run one bidir forward
    pass, compute mean CE at those positions against the original tokens.
  - For each patch and each MODE:
        baseline : true preamble (unchanged)
        shuffle  : preamble swapped with a paired patch (random non-self)
        random   : preamble overwritten with random integer tokens in the
                   valid non-special vocab range
  - Report mean ± std per mode across the patch set.

The forward path is identical to MaskGIT-eval's `maskgit_forward` (bidir
attention), and the masking + CE math matches `maskgit_aware_loss` (no token
shift; target at position t = original_tokens[t]; loss restricted to masked
positions). Resulting VL numbers are therefore directly comparable to
in-training `eval/loss` (under absorbing prior).

Output: a single JSON to `gs://marin-eu-west4/tomat/eval/preamble-test/
<run_label>-step-<N>.json` with `{mode: {mean, std, n_patches, ts}}`.

Examples:
    # 8-patch smoke
    modal run scripts/preamble_vl_modal.py --n-patches 8 --gpu-pool h100x8

    # full fire (default 256 patches)
    modal run scripts/preamble_vl_modal.py
"""
from __future__ import annotations

import sys
from functools import partial

import modal

err = partial(print, file=sys.stderr)

BUCKET = "gs://marin-eu-west4/tomat"
LMQ_PATH = f"{BUCKET}/codecs/lmq-v2-16k.npz"
DEFAULT_CKPT = (
    f"{BUCKET}/results/train-mg-modal-h200x8-tz-v4-epochwin/checkpoints/"
    "train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42/step-62000"
)
DEFAULT_LABEL = "train-full-v3"
VOL_NAME = "tomat-rho-gga-train"

# Pin same as scripts/eval_modal.py — same marin SHA → same Levanter / Qwen3
# weights load semantics. Bump in lockstep when the eval-modal pin bumps.
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
        "'jax[cuda12]' 'pyarrow>=15' fsspec gcsfs",
    )
    .add_local_python_source("tomat")
    .add_local_python_source("marin")
)

app = modal.App("tomat-preamble-vl", image=image)
gcp_secret = modal.Secret.from_name("tomat-gcp-sa")
train_volume = modal.Volume.from_name(VOL_NAME)

MODEL_PRESETS = {
    "30M":  dict(hidden_dim=512,  num_layers=6,  num_heads=4,  num_kv_heads=4,  intermediate_dim=2048),
    "200M": dict(hidden_dim=1024, num_layers=12, num_heads=16, num_kv_heads=16, intermediate_dim=4096),
    "1B":   dict(hidden_dim=2048, num_layers=20, num_heads=16, num_kv_heads=16, intermediate_dim=5632),
}

# Special token IDs (subset; lat datasets add LATTICE_{START,END}=18,19).
# Mirrors `marin/eval_mat_nmae.py`'s `TOK` table.
TOK_DENS_START = 15
TOK_PAD = 0


def _materialize_gcp_sa() -> None:
    """Write the secret-injected SA JSON to disk + point ADC at it."""
    import os
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if sa_json:
        with open("/tmp/gcp-sa.json", "w") as f:
            f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp-sa.json"


def _load_patches(label: str, n_patches: int, skip: int = 0):
    """Read the tail `n_patches` sequences (after `skip`) from `train-full-v3`
    parquets, deterministic order.

    Layout: `{BUCKET}/tokenized/{label}/worker-NN/shard-XXX.parquet`, each
    row has an `input_ids` column with int32 lists (pad_to elements).
    Reverse-sorting the (worker, shard, row) tuple gives a stable "tail" slice
    that's both data-deterministic and unlikely to overlap with anything the
    model already saw biased toward (the corpus is shuffled at training time;
    here we just need *some* repeatable carve).

    Returns: int32 ndarray of shape (n_patches, pad_to).
    """
    import fsspec
    import numpy as np
    import pyarrow.parquet as pq

    base = f"{BUCKET}/tokenized/{label}"
    fs = fsspec.filesystem("gs")
    shards = sorted(fs.glob(f"{base.replace('gs://', '')}/worker-*/shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards under {base}")
    # Tail slice — walk shards in reverse order, collecting rows until we have
    # `skip + n_patches`. Each shard is ~hundreds of MB; pulling 2-3 is enough.
    rows: list[np.ndarray] = []
    needed = skip + n_patches
    for shard in reversed(shards):
        with fsspec.open(f"gs://{shard}", "rb") as f:
            tbl = pq.read_table(f, columns=["input_ids"])
        col = tbl["input_ids"]
        # Walk rows from the end of this shard backwards.
        for i in range(len(col) - 1, -1, -1):
            arr = np.asarray(col[i].as_py(), dtype=np.int32)
            rows.append(arr)
            if len(rows) >= needed:
                break
        if len(rows) >= needed:
            break
    if len(rows) < needed:
        raise RuntimeError(f"Only {len(rows)} rows found; needed {needed}")
    # Reverse so the very last shard's last row is index 0 — stable order.
    rows = rows[skip:skip + n_patches]
    return np.stack(rows, axis=0)  # (n_patches, pad_to)


def _find_preamble_lengths(patches):
    """For each patch, locate the index of DENS_START (=15). Preamble length
    is that index (positions 0..idx-1 are the preamble; idx is DENS_START
    itself, which we keep). Returns (n_patches,) int32 of preamble lengths.
    """
    import numpy as np
    n = patches.shape[0]
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        loc = np.where(patches[i] == TOK_DENS_START)[0]
        if len(loc) == 0:
            raise ValueError(f"patch {i}: no DENS_START token (=15) found")
        out[i] = loc[0]
    return out


def _build_perturbed(patches, preamble_lens, mode: str, vocab_size: int, seed: int):
    """Build the (n, pad_to) int32 token grid for `mode`.

    - baseline: identity copy.
    - shuffle : for each patch i, copy preamble from a paired patch j ≠ i.
                Paired = deterministic permutation derived from `seed`.
                If preamble lengths differ between i and j, copy the
                min(len_i, len_j) prefix from j into i; leftover positions
                in i (between len_j and len_i) get a random non-special
                token id (very rare; most patches share lengths exactly).
    - random  : overwrite the preamble (positions [1, preamble_len_i)) with
                uniform random integers in [20, vocab_size). 20 = N_SPECIALS_LAT,
                so we skip the [PAD]/[BOS]/.../[LATTICE_END] specials.
                Position 0 ([BOS]) is preserved so the model isn't
                additionally confused by a missing start token.

    For `shuffle`, the swapped row carries its own [BOS] at position 0 —
    consistent across all modes.

    Density region (positions [preamble_len_i .. end]) is unchanged in ALL
    modes (the masking-with-MASK_ID happens later, inside `_eval_one_mode`,
    and is identical across modes — only preamble varies).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n, pad_to = patches.shape
    out = patches.copy()
    if mode == "baseline":
        return out
    if mode == "shuffle":
        # Random derangement (no fixed points).
        perm = rng.permutation(n)
        fixed = np.where(perm == np.arange(n))[0]
        # Resolve any fixed points by swapping each with the next index mod n.
        for k in fixed:
            j = (k + 1) % n
            perm[k], perm[j] = perm[j], perm[k]
        for i in range(n):
            j = int(perm[i])
            li, lj = int(preamble_lens[i]), int(preamble_lens[j])
            m = min(li, lj)
            out[i, :m] = patches[j, :m]
            if li > lj:
                # Pad the remaining preamble slots with random non-special tokens.
                out[i, lj:li] = rng.integers(20, vocab_size, size=li - lj, dtype=np.int32)
        return out
    if mode == "random":
        for i in range(n):
            li = int(preamble_lens[i])
            # Preserve [BOS] at position 0; random-fill positions [1, li).
            out[i, 1:li] = rng.integers(20, vocab_size, size=li - 1, dtype=np.int32)
        return out
    raise ValueError(f"unknown mode: {mode!r}")


def _eval_one_mode(
    model,
    tokens_np,                # (n, pad_to) int32 — POSSIBLY-PERTURBED preamble + GT density
    original_np,              # (n, pad_to) int32 — UNMODIFIED (used for CE targets)
    preamble_lens,            # (n,) int32
    *,
    mask_id: int,
    batch: int,
    compute_mapping,
    seq_len: int,
):
    """Mask all density positions → MASK_ID and run a single bidirectional
    forward pass; compute mean CE at the masked density positions.

    `tokens_np` carries the (possibly perturbed) preamble; we re-mask its
    density region here so the perturbation contract is just "the preamble
    changes; everything else stays identical across modes".

    `original_np` is the unperturbed token grid — used as CE targets so the
    density GT for the loss is always the *true* per-patch density.

    Returns per-patch mean CE: list[float] of length n.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import haliax as hax
    from haliax import Axis
    from levanter.layers.attention import AttentionMask

    mg_bidir_mask = AttentionMask(is_causal=False)

    @hax.named_jit(axis_resources=compute_mapping)
    def forward_ce(tok_ha, target_ha, weight_ha):
        """Bidirectional forward; CE at weighted (= MASKED density) positions.

        tok_ha:    (Batch, Pos) int32 — masked input (density positions = MASK_ID).
        target_ha: (Batch, Pos) int32 — ground-truth tokens (for CE target).
        weight_ha: (Batch, Pos) float — 1 at masked density positions, else 0.
        Returns: (per_position_ce: (Batch, Pos), per_position_weight).
        """
        act = model.activations(tok_ha, key=None, attn_mask=mg_bidir_mask)
        head = model.get_lm_head()
        logits = hax.dot(act, head, axis=model.Embed)
        logits_arr = logits.array.astype(jnp.float32)  # (B, Pos, V)
        log_probs = jax.nn.log_softmax(logits_arr, axis=-1)
        tgt = target_ha.array
        ce = -jnp.take_along_axis(log_probs, tgt[..., None], axis=-1).squeeze(-1)
        w = weight_ha.array.astype(jnp.float32)
        ce_w = ce * w
        return hax.named(ce_w, tok_ha.axes), hax.named(w, tok_ha.axes)

    n = tokens_np.shape[0]
    per_patch: list[float] = []
    Pos = Axis("position", seq_len)
    Batch = Axis("batch", batch)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        # Carve B rows; pad to `batch` for JIT shape stability.
        inp_chunk = tokens_np[start:end].copy()
        tgt_chunk = original_np[start:end].copy()
        pad_n = batch - (end - start)
        if pad_n:
            pad_rows = np.zeros((pad_n, seq_len), dtype=np.int32)
            inp_chunk = np.concatenate([inp_chunk, pad_rows], axis=0)
            tgt_chunk = np.concatenate([tgt_chunk, pad_rows], axis=0)
        # Build per-row weight mask AND apply MASK_ID at density positions in
        # `inp_chunk` (positions [pl+1 .. DENS_END_idx)).
        w = np.zeros((batch, seq_len), dtype=np.float32)
        for i, idx in enumerate(range(start, end)):
            pl = int(preamble_lens[idx])
            row = original_np[idx]
            de = np.where(row == 16)[0]  # TOK["DENS_END"]
            de_idx = int(de[0]) if len(de) else seq_len
            w[i, pl + 1:de_idx] = 1.0
            inp_chunk[i, pl + 1:de_idx] = mask_id
        tok_ha = hax.named(jnp.asarray(inp_chunk), (Batch, Pos))
        tgt_ha = hax.named(jnp.asarray(tgt_chunk), (Batch, Pos))
        weight_ha = hax.named(jnp.asarray(w), (Batch, Pos))
        ce_arr, w_arr = forward_ce(tok_ha, tgt_ha, weight_ha)
        ce_np = np.asarray(ce_arr.array)
        w_np = np.asarray(w_arr.array)
        for i in range(end - start):
            denom = float(w_np[i].sum())
            num = float(ce_np[i].sum())
            mean_ce = num / max(denom, 1.0)
            per_patch.append(mean_ce)
    return per_patch


@app.function(
    gpu="H100:8",
    secrets=[gcp_secret],
    volumes={"/vol": train_volume},
    timeout=3600,
    memory=64 * 1024,
)
def run_preamble_vl_h100x8(
    checkpoint: str,
    label: str,
    lmq_path: str,
    model_preset: str,
    n_patches: int,
    skip: int,
    batch: int,
    modes: list[str],
    seed: int,
    output_path: str,
) -> dict:
    return _run_preamble_vl(
        checkpoint=checkpoint, label=label, lmq_path=lmq_path,
        model_preset=model_preset, n_patches=n_patches, skip=skip,
        batch=batch, modes=modes, seed=seed, output_path=output_path,
    )


@app.function(
    gpu="H200:8",
    secrets=[gcp_secret],
    volumes={"/vol": train_volume},
    timeout=3600,
    memory=64 * 1024,
)
def run_preamble_vl_h200x8(
    checkpoint: str,
    label: str,
    lmq_path: str,
    model_preset: str,
    n_patches: int,
    skip: int,
    batch: int,
    modes: list[str],
    seed: int,
    output_path: str,
) -> dict:
    return _run_preamble_vl(
        checkpoint=checkpoint, label=label, lmq_path=lmq_path,
        model_preset=model_preset, n_patches=n_patches, skip=skip,
        batch=batch, modes=modes, seed=seed, output_path=output_path,
    )


def _run_preamble_vl(
    *,
    checkpoint: str,
    label: str,
    lmq_path: str,
    model_preset: str,
    n_patches: int,
    skip: int,
    batch: int,
    modes: list[str],
    seed: int,
    output_path: str,
) -> dict:
    """Shared body: load ckpt, load patches, run each mode, persist JSON."""
    import json
    import os
    import time

    _materialize_gcp_sa()

    import fsspec
    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np
    import equinox as eqx
    import haliax as hax
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning

    # PassthroughTokenizer monkey-patch (same as eval_mat_nmae.py). Once the
    # upstream marin/Levanter fix lands and the SHA pin is bumped this is a
    # no-op; keep for safety.
    from levanter.data.passthrough_tokenizer import PassthroughTokenizer
    _orig_pt_encode = PassthroughTokenizer.encode

    def _safe_pt_encode(self, text, *, add_special_tokens=False):
        try:
            return _orig_pt_encode(self, text, add_special_tokens=add_special_tokens)
        except ValueError:
            return [0]
    PassthroughTokenizer.encode = _safe_pt_encode

    import levanter
    from levanter.checkpoint import load_checkpoint
    from levanter.layers.attention import AttentionBackend
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.models.qwen import Qwen3Config
    from levanter.trainer import TrainerConfig
    from levanter.utils.jax_utils import use_cpu_device
    from levanter.utils.tree_utils import inference_mode

    try:
        jax.distributed.initialize()
    except Exception:
        pass

    # ---- Read meta.json for vocab info ----
    meta_url = f"{BUCKET}/tokenized/{label}/worker-00/meta.json"
    with fsspec.open(meta_url, "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab"]["total_size"]
    # v4-epochwin was trained MaskGIT, +1 vocab for MASK_ID.
    vocab_size_with_mask = vocab_size + 1
    seq_len = int(meta.get("pad_to") or 8192)
    err(f"[preamble-vl] label={label} vocab={vocab_size} (+1 for MASK), "
        f"seq_len={seq_len}, preset={model_preset}")

    # ---- Pull patches ----
    t0 = time.time()
    patches = _load_patches(label, n_patches, skip=skip)
    err(f"[preamble-vl] loaded {patches.shape[0]} patches in {time.time()-t0:.1f}s "
        f"({patches.shape[1]} tok each)")
    preamble_lens = _find_preamble_lengths(patches)
    err(f"[preamble-vl] preamble lens: min={preamble_lens.min()} "
        f"p50={int(np.median(preamble_lens))} max={preamble_lens.max()}")

    # ---- Set up model + trainer ----
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32,
    )
    trainer_cfg = TrainerConfig(
        id="preamble-vl",
        seed=seed,
        num_train_steps=1,
        train_batch_size=batch,
        tracker=(),
        mp=mp_policy,
    )
    levanter.initialize(trainer_cfg)
    compute_mapping = trainer_cfg.compute_axis_mapping
    param_mapping = trainer_cfg.parameter_axis_mapping

    # Pure-Pallas flash attention — matches eval_mat_nmae default. Same math
    # as ref attn at bf16 for our shapes.
    os.environ.setdefault("TOMAT_ATTN_BACKEND", "JAX_FLASH")
    attn_backend = os.environ.get("TOMAT_ATTN_BACKEND") or None

    model_cfg = Qwen3Config(
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
        attn_backend=AttentionBackend[attn_backend] if attn_backend else None,
        **MODEL_PRESETS[model_preset],
    )
    key = jax.random.PRNGKey(seed)
    with trainer_cfg.use_device_mesh():
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size_with_mask), compute_mapping)
        with use_cpu_device():
            model = eqx.filter_eval_shape(model_cfg.build, Vocab, key=key)
            err(f"[preamble-vl] loading ckpt {checkpoint}")
            model = load_checkpoint(model, checkpoint, subpath="model")
        model = hax.shard_with_axis_mapping(model, param_mapping)
        model = inference_mode(model, True)
        model = mp_policy.cast_to_compute(model)

        # ---- Run each mode ----
        # MaskGIT MASK_ID = vocab_size (the original total, pre +1 bump). See
        # `Qwen3MaskGITLMHeadModel`: vocab is bumped from V → V+1, and the new
        # last id is reserved for [MASK].
        MASK_ID = vocab_size
        results: dict[str, dict] = {}
        for mode in modes:
            tm = time.time()
            tokens_np = _build_perturbed(
                patches, preamble_lens, mode, vocab_size, seed=seed,
            )
            per_patch = _eval_one_mode(
                model, tokens_np, patches, preamble_lens,
                mask_id=MASK_ID,
                batch=batch, compute_mapping=compute_mapping, seq_len=seq_len,
            )
            arr = np.array(per_patch, dtype=np.float64)
            results[mode] = dict(
                mean=float(arr.mean()),
                std=float(arr.std(ddof=0)),
                std_err=float(arr.std(ddof=0) / np.sqrt(len(arr))),
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                n_patches=int(len(arr)),
                elapsed_s=float(time.time() - tm),
            )
            err(f"[preamble-vl] mode={mode}: mean={results[mode]['mean']:.4f} "
                f"std={results[mode]['std']:.4f} "
                f"n={results[mode]['n_patches']} "
                f"(elapsed {results[mode]['elapsed_s']:.1f}s)")

    # ---- Persist ----
    summary = dict(
        checkpoint=checkpoint,
        label=label,
        n_patches=n_patches,
        skip=skip,
        seed=seed,
        modes=modes,
        results=results,
        ts=int(time.time()),
    )
    with fsspec.open(output_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    err(f"[preamble-vl] wrote {output_path}")
    return summary


def _default_output_path(checkpoint: str) -> str:
    """Pull `<run_label>/step-N` tail out of the checkpoint path."""
    parts = checkpoint.rstrip("/").split("/")
    rl = parts[-4]
    step = parts[-1]
    return f"{BUCKET}/eval/preamble-test/{rl}-{step}.json"


@app.local_entrypoint()
def main(
    checkpoint: str = DEFAULT_CKPT,
    label: str = DEFAULT_LABEL,
    lmq_path: str = LMQ_PATH,
    model_preset: str = "200M",
    n_patches: int = 256,
    skip: int = 0,
    batch: int = 8,
    modes: str = "baseline,shuffle,random",
    seed: int = 42,
    output_path: str = "",
    gpu_pool: str = "h100x8",  # "h100x8" or "h200x8"
):
    """Run preamble-VL ablation on `checkpoint`.

    `modes`: comma-separated subset of {baseline, shuffle, random}.
    `gpu_pool`: h100x8 (cheaper) or h200x8 (faster).
    `output_path`: where to write the JSON summary (default: derived).
    """
    out = output_path or _default_output_path(checkpoint)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    fn = run_preamble_vl_h200x8 if gpu_pool == "h200x8" else run_preamble_vl_h100x8
    print(f"[preamble-vl] firing {gpu_pool}: ckpt={checkpoint} n={n_patches} "
          f"modes={mode_list} → {out}")
    call = fn.spawn(
        checkpoint=checkpoint, label=label, lmq_path=lmq_path,
        model_preset=model_preset, n_patches=n_patches, skip=skip,
        batch=batch, modes=mode_list, seed=seed, output_path=out,
    )
    print(f"[preamble-vl] call_id={call.object_id}")
    summary = call.get()
    print(f"\n=== preamble-VL results (n={summary['n_patches']}) ===")
    for m in mode_list:
        r = summary["results"][m]
        print(f"  {m:<10} mean={r['mean']:.4f}  std={r['std']:.4f}  "
              f"se={r['std_err']:.4f}  p50={r['median']:.4f}  "
              f"(n={r['n_patches']}, {r['elapsed_s']:.1f}s)")
    print(f"\n  → {out}")
