"""Spec-34 F1 loader plumbing: parquet → cache → `F1LmExample`.

Smoke-tests the data layer that lets `Qwen3F1LMHeadModel.compute_next_token_loss`
actually see an `atom_xyz` sidecar at training time. The model + tokenizer
tests live in `test_f1_encoding.py`; this file verifies the parquet→cache→
loader→LmExample chain.

Tests
-----

1. **`F1PrebuiltCacheProcessor` round-trip** — feed mock parquet rows
   (input_ids + atom_xyz) into the processor and confirm the output
   exemplar/schema match expectations.

2. **`F1PrebuiltLmDataset` end-to-end** — build a tiny on-disk parquet,
   materialize an F1 cache via `LmDataConfig.build_caches`, iterate
   the `F1PrebuiltLmDataset`, and confirm each row is an
   `F1LmExample` with `atom_xyz` axis = `(Pos, XYZ)`.

3. **Forward + loss with F1 active** — load a row, run
   `Qwen3F1LMHeadModel.compute_next_token_loss(example)`, confirm the
   loss is finite and DIFFERS from the same forward with `atom_xyz=None`
   (sanity: the F1 path is actually flowing through the loss).

4. **NaN-at-non-atom-positions doesn't taint the loss** — sanity check
   that gradient/loss don't go NaN given the parquet's NaN-padded xyz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402

_MARIN = str(Path(__file__).resolve().parent.parent / "marin")
if _MARIN not in sys.path:
    sys.path.insert(0, _MARIN)

from qwen3_density import (  # noqa: E402
    F1Args,
    F1LmExample,
    Qwen3DensityConfig,
    Qwen3DensityLMHeadModel,
    configure_atom_encoder,
)
from f1_data import (  # noqa: E402
    F1LmDataConfig,
    F1PrebuiltCacheProcessor,
    F1PrebuiltLmDataset,
    F1PrebuiltLmDatasetFormat,
)
from tomat.tokenizers.patch import ATOM_OFFSET  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOM_TOKEN = ATOM_OFFSET + 5  # "C" — any atom-token id will do
SEQ_LEN = 16


def _make_row(atom_positions: list[int], rng: np.random.Generator) -> dict:
    """Build a single F1 row: input_ids + aligned atom_xyz (NaN-padded)."""
    ids = np.zeros(SEQ_LEN, dtype=np.int32)
    xyz = np.full((SEQ_LEN, 3), np.nan, dtype=np.float32)
    for i in atom_positions:
        ids[i] = ATOM_TOKEN
        xyz[i] = rng.uniform(0, 1, size=3).astype(np.float32)
    return {"input_ids": ids, "atom_xyz": xyz}


def _write_tiny_parquet(path: Path, n_rows: int = 6, seed: int = 0) -> dict:
    """Emit `n_rows` F1-formatted rows to a parquet file.

    Each row has 3 atom-token positions scattered through the seq. The
    column types match what `scripts/tokenize_patches.py --atom-encoding f1`
    produces (input_ids: list<int32>, atom_xyz: list<list<float32>>).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rows):
        # Stagger atom positions so rows differ.
        atom_positions = [2 + r % 3, 5 + r % 3, 9 + r % 3]
        rows.append(_make_row(atom_positions, rng))

    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("atom_xyz", pa.list_(pa.list_(pa.float32()))),
    ])
    table = pa.table(
        {
            "input_ids": [r["input_ids"].tolist() for r in rows],
            "atom_xyz": [[row.tolist() for row in r["atom_xyz"]] for r in rows],
        },
        schema=schema,
    )
    pq.write_table(table, str(path))
    return {"rows": rows, "n": n_rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_f1_processor_roundtrip():
    """`F1PrebuiltCacheProcessor` passes through both columns with the
    correct dtypes and shapes."""
    proc = F1PrebuiltCacheProcessor()
    rng = np.random.default_rng(0)
    in_rows = [
        _make_row([2, 5], rng),
        _make_row([3, 6, 11], rng),
    ]
    out = proc(in_rows)
    assert len(out) == 2
    for r_in, r_out in zip(in_rows, out):
        assert r_out["input_ids"].dtype == np.int32
        assert r_out["atom_xyz"].dtype == np.float32
        assert r_out["atom_xyz"].shape == (SEQ_LEN, 3)
        np.testing.assert_array_equal(r_out["input_ids"], r_in["input_ids"])
        # NaN-aware equality (NaN at non-atom positions, finite at atom positions).
        np.testing.assert_array_equal(
            np.isnan(r_out["atom_xyz"]),
            np.isnan(r_in["atom_xyz"]),
        )
        finite = np.isfinite(r_in["atom_xyz"]).all(axis=1)
        np.testing.assert_allclose(
            r_out["atom_xyz"][finite], r_in["atom_xyz"][finite],
        )


def test_f1_processor_exemplar_shape():
    """Cache exemplar must be `{(0,) int32, (0, 3) float32}` so the
    underlying `TreeStore` picks rank-1 + rank-2 storage backends."""
    proc = F1PrebuiltCacheProcessor()
    ex = proc.output_exemplar
    assert set(ex.keys()) == {"input_ids", "atom_xyz"}
    assert ex["input_ids"].shape == (0,)
    assert ex["input_ids"].dtype == np.int32
    assert ex["atom_xyz"].shape == (0, 3)
    assert ex["atom_xyz"].dtype == np.float32


def test_f1_processor_missing_xyz_raises():
    proc = F1PrebuiltCacheProcessor()
    with pytest.raises(ValueError, match="atom_xyz"):
        proc([{"input_ids": np.zeros(4, dtype=np.int32)}])


def test_f1_processor_wrong_xyz_shape_raises():
    proc = F1PrebuiltCacheProcessor()
    bad = {
        "input_ids": np.zeros(4, dtype=np.int32),
        "atom_xyz": np.zeros((4, 2), dtype=np.float32),  # 2 != 3
    }
    with pytest.raises(ValueError, match="shape"):
        proc([bad])


def test_f1_processor_length_mismatch_raises():
    proc = F1PrebuiltCacheProcessor()
    bad = {
        "input_ids": np.zeros(4, dtype=np.int32),
        "atom_xyz": np.zeros((5, 3), dtype=np.float32),  # off-by-one
    }
    with pytest.raises(ValueError, match="length"):
        proc([bad])


def test_f1_dataset_end_to_end(tmp_path: Path):
    """Parquet → cache → F1PrebuiltLmDataset → F1LmExample.

    Verifies the whole loader chain on a 6-row synthetic dataset.
    """
    parquet = tmp_path / "f1_smoke.parquet"
    _write_tiny_parquet(parquet, n_rows=6)

    # Build the cache through Levanter's standard pipeline.
    from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig

    fmt = F1PrebuiltLmDatasetFormat()
    source = UrlDatasetSourceConfig(train_urls=[str(parquet)])
    cache_dir = tmp_path / "cache"
    component = DatasetComponent(
        source=source,
        cache_dir=str(cache_dir),
        format=fmt,
    )
    data = F1LmDataConfig(
        tokenizer="passthrough",
        vocab_size=1024,
        cache_dir=str(cache_dir),
        components={"tomat_f1": component},
        block_cross_document_attention=False,
        shuffle=False,
    )

    # Materialize the train cache.
    caches = data.build_caches("train")
    assert "tomat_f1" in caches
    cache = caches["tomat_f1"]
    n_rows = len(cache.store)
    assert n_rows == 6, f"expected 6 rows in cache, got {n_rows}"

    # Build the F1 dataset directly (skipping the train_set shuffle wrapping
    # for a tighter test of the loader itself).
    Pos = Axis("position", SEQ_LEN)
    ds = F1PrebuiltLmDataset(cache, Pos, eos_id=None, block_cross_document_attention=False)
    sync = ds.as_sync_dataset()
    assert len(sync) == 6

    # Grab one example and verify it's an F1LmExample with the right shape.
    ex = sync[0]
    assert isinstance(ex, F1LmExample), f"expected F1LmExample, got {type(ex).__name__}"
    assert ex.tokens.axes == (Pos,)
    assert ex.atom_xyz is not None, "atom_xyz must be populated, not None"
    XYZ = Axis("xyz", 3)
    assert ex.atom_xyz.axes == (Pos, XYZ)
    # Atom positions should match the parquet (3 atom slots per row, ids = ATOM_TOKEN).
    is_atom = np.asarray(ex.tokens.array) == ATOM_TOKEN
    finite = np.isfinite(np.asarray(ex.atom_xyz.array)).all(axis=1)
    np.testing.assert_array_equal(is_atom, finite)


def _tiny_f1_model(num_freqs: int = 4, max_seq_len: int = SEQ_LEN):
    Vocab = Axis("vocab", 1024)
    config = Qwen3DensityConfig(
        max_seq_len=max_seq_len,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
    )
    configure_atom_encoder("f1", f1_args=F1Args(num_freqs=num_freqs))
    return config, Qwen3DensityLMHeadModel.init(
        Vocab, config, key=jrandom.PRNGKey(0),
    )


def test_f1_forward_and_loss_finite(tmp_path: Path):
    """Materialize an F1 row, run `compute_next_token_loss`, verify the
    loss is finite and differs from the F1-off (atom_xyz=None) baseline.

    This is the headline smoke test: it covers the full forward path
    that the trainer takes when `TOMAT_F1_MODE=1`.
    """
    parquet = tmp_path / "f1_smoke.parquet"
    _write_tiny_parquet(parquet, n_rows=4)

    from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig

    fmt = F1PrebuiltLmDatasetFormat()
    source = UrlDatasetSourceConfig(train_urls=[str(parquet)])
    cache_dir = tmp_path / "cache"
    component = DatasetComponent(
        source=source,
        cache_dir=str(cache_dir),
        format=fmt,
    )
    data = F1LmDataConfig(
        tokenizer="passthrough",
        vocab_size=1024,
        cache_dir=str(cache_dir),
        components={"tomat_f1": component},
        block_cross_document_attention=False,
        shuffle=False,
    )
    caches = data.build_caches("train")
    Pos = Axis("position", SEQ_LEN)
    ds = F1PrebuiltLmDataset(
        caches["tomat_f1"], Pos, eos_id=None, block_cross_document_attention=False,
    )
    ex = ds.as_sync_dataset()[0]

    config, model = _tiny_f1_model(num_freqs=4, max_seq_len=SEQ_LEN)

    # F1 path: pass the example with atom_xyz attached.
    loss_on = model.compute_next_token_loss(ex)
    assert jnp.isfinite(loss_on.array), f"loss must be finite, got {float(loss_on.array)}"

    # F1-off path: same tokens, but no atom_xyz.
    from levanter.models.lm_model import LmExample
    ex_off = LmExample(
        tokens=ex.tokens, loss_weight=ex.loss_weight, attn_mask=ex.attn_mask,
    )
    loss_off = model.compute_next_token_loss(ex_off)
    assert jnp.isfinite(loss_off.array)

    # Sanity: the F1 path actually flows through the loss (otherwise we'd
    # train a model whose `f1_atom_proj` weights had zero gradient).
    diff = float(jnp.abs(loss_on.array - loss_off.array))
    assert diff > 1e-6, (
        f"F1-on vs F1-off loss must differ, got diff={diff} "
        f"(on={float(loss_on.array)}, off={float(loss_off.array)})"
    )


def test_f1_gradient_finite_with_nan_xyz(tmp_path: Path):
    """`atom_xyz` carries NaN at non-atom positions by design. The model's
    `_f1_pos_addend` sanitizes via `jnp.where(isnan, 0, …)` BEFORE the
    sinusoidal encoding, so neither loss nor gradient should be NaN.

    Crucially: a gradient w.r.t. an array containing NaN entries can
    silently turn into a NaN gradient (the `jnp.where` trick doesn't
    save you from this in general). The NaN-safe path is to do
    `jnp.where(isnan, 0, x)` and then NEVER touch the original x. The
    F1 model does exactly that; this test verifies the property.
    """
    parquet = tmp_path / "f1_smoke.parquet"
    _write_tiny_parquet(parquet, n_rows=2)

    from levanter.data.text.datasets import DatasetComponent, UrlDatasetSourceConfig

    fmt = F1PrebuiltLmDatasetFormat()
    source = UrlDatasetSourceConfig(train_urls=[str(parquet)])
    cache_dir = tmp_path / "cache"
    component = DatasetComponent(
        source=source,
        cache_dir=str(cache_dir),
        format=fmt,
    )
    data = F1LmDataConfig(
        tokenizer="passthrough",
        vocab_size=1024,
        cache_dir=str(cache_dir),
        components={"tomat_f1": component},
        block_cross_document_attention=False,
        shuffle=False,
    )
    caches = data.build_caches("train")
    Pos = Axis("position", SEQ_LEN)
    ds = F1PrebuiltLmDataset(
        caches["tomat_f1"], Pos, eos_id=None, block_cross_document_attention=False,
    )
    ex = ds.as_sync_dataset()[0]
    config, model = _tiny_f1_model(num_freqs=4, max_seq_len=SEQ_LEN)

    def loss_fn(m):
        return m.compute_next_token_loss(ex).array

    import equinox as eqx
    grads = eqx.filter_grad(loss_fn)(model)
    flat = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    for g in flat:
        assert jnp.all(jnp.isfinite(g)), (
            f"NaN/inf in grad leaf {g.shape if hasattr(g, 'shape') else g!r}"
        )
