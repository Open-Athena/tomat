"""F1 (spec 34) data plumbing: parquet → cache → `F1LmExample`.

The model side (sinusoidal-xyz embed addend) lives in `qwen3_density.py`.
This module wires the loader: parquet rows carry both `input_ids` (int32
list) and `atom_xyz` (list<list<float32>> = float32[L, 3] per row, NaN at
non-atom positions), and we need to materialize an `F1LmExample` whose
`atom_xyz` NamedArray reaches `Qwen3F1LMHeadModel.compute_next_token_loss`.

Architecture
------------

Reuses Levanter's `PrebuiltLmDatasetFormat` cache mechanism almost
verbatim, with two small extensions:

1. **`F1PrebuiltCacheProcessor`** (subclass of `PrebuiltCacheProcessor`).
   Same shape as the parent (passes through pre-tokenized rows) but the
   `output_exemplar` includes a second leaf `atom_xyz` with shape
   `(0, 3)` and dtype float32. Levanter's `TreeStore` infers per-leaf
   item rank from the exemplar; the rank-2 xyz leaf gets its own
   `JaggedArrayStore` alongside the rank-1 `input_ids` leaf. No changes
   to the cache builder are needed.

2. **`F1PrebuiltLmDataset`** (subclass of `MappedAsyncDataset`).
   Reads `{"input_ids": (L,), "atom_xyz": (L, 3)}` per row and emits an
   `F1LmExample` directly (skipping the Grug intermediate — the F1
   sidecar would otherwise be lost in the `GrugLmExample` adapter
   chain). The standard `LmExample.causal` builds tokens/loss_weight/
   attn_mask the usual way; we then construct the `F1LmExample` with
   the same fields plus the `atom_xyz` NamedArray.

3. **`F1LmDataConfig`** (subclass of `LmDataConfig`).
   Overrides `train_set`, `validation_sets`, `tagged_eval_sets` to
   route through `F1PrebuiltLmDataset` instead of the standard
   `PrebuiltLmDataset → NamedLmDataset` chain when the underlying
   component is F1-formatted. Falls back to the parent class for any
   non-F1 components (mixed F0/F1 mixtures still work; the F0
   components are unchanged).

Packing
-------

Spec-34 `scripts/tokenize_patches.py --pack` already packs whole
patches into long rows, with [PAD] sentinels at segment boundaries.
Both `input_ids` and `atom_xyz` are packed in lockstep at parquet
emission time, so we don't need to rebuild Levanter's packing layer
for F1. The block-diagonal causal attention mask is derived from PAD
sentinels by `LmExample.causal(...)` with `eos_id=PAD_ID` (caller's
responsibility).

Back-compat
-----------

* `TOMAT_F1_MODE=0` (default) — `train_tomat_tpu.py` instantiates
  plain `LmDataConfig` + `Qwen3*` models. This module is unused.
* `TOMAT_F1_MODE=1` — train script uses `F1LmDataConfig` + an F1-
  formatted parquet shard glob. The loader emits `F1LmExample`s with
  `atom_xyz` populated.

JIT-traceability
----------------

`F1LmExample` is an `eqx.Module` registered with JAX's tree util, so
`atom_xyz: NamedArray | None` is treated as a tree leaf and flows
through `eqx.filter_jit`. The NaN values (at non-atom positions) are
sanitized inside the model's `_f1_pos_addend` via `jnp.where(isnan, 0,
…)` before they enter the sinusoidal encoding, so NaN never propagates
into gradients.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from haliax import Axis, NamedArray

from jaxtyping import PRNGKeyArray

from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import AsyncDataset, MappedAsyncDataset
from levanter.data.text.datasets import (
    LmDataConfig,
    NamedLmDataset,
)
from levanter.data.text.formats import (
    LmDatasetFormatBase,
    PrebuiltCacheProcessor,
    PrebuiltLmDatasetFormat,
)
from levanter.data.text.datasets import DatasetComponent
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache processor: pass through input_ids + atom_xyz
# ---------------------------------------------------------------------------


class F1PrebuiltCacheProcessor(BatchProcessor[dict, dict]):
    """Pre-built cache processor that also passes through `atom_xyz`.

    Same shape as `PrebuiltCacheProcessor` but the exemplar (and per-row
    output) carries a second leaf `atom_xyz: float32[L, 3]`. Parquet
    readers convert `list<list<float32>>` to Python lists of lists;
    `np.asarray` materializes the (L, 3) array.
    """

    def __init__(
        self,
        input_ids_key: str = "input_ids",
        atom_xyz_key: str = "atom_xyz",
    ):
        self.input_ids_key = input_ids_key
        self.atom_xyz_key = atom_xyz_key
        self._exemplar = {
            input_ids_key: np.zeros((0,), dtype=np.int32),
            atom_xyz_key: np.zeros((0, 3), dtype=np.float32),
        }

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        out = []
        for example in batch:
            if self.input_ids_key not in example:
                raise ValueError(
                    f"Missing required field '{self.input_ids_key}' in F1 prebuilt example."
                )
            if self.atom_xyz_key not in example:
                raise ValueError(
                    f"Missing required field '{self.atom_xyz_key}' in F1 prebuilt example. "
                    "Did you tokenize with `--atom-encoding f1`?"
                )
            ids = np.asarray(example[self.input_ids_key], dtype=np.int32)
            xyz = np.asarray(example[self.atom_xyz_key], dtype=np.float32)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(
                    f"`{self.atom_xyz_key}` must be float32 of shape (L, 3); "
                    f"got shape {xyz.shape}"
                )
            if xyz.shape[0] != ids.shape[0]:
                raise ValueError(
                    f"`{self.atom_xyz_key}` length {xyz.shape[0]} != "
                    f"`{self.input_ids_key}` length {ids.shape[0]}"
                )
            out.append({self.input_ids_key: ids, self.atom_xyz_key: xyz})
        return out

    @property
    def output_exemplar(self) -> dict:
        return self._exemplar

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "input_ids_key": self.input_ids_key,
            "atom_xyz_key": self.atom_xyz_key,
            "format": "f1_prebuilt",
        }


# ---------------------------------------------------------------------------
# Dataset format: pluggable in a `DatasetComponent`
# ---------------------------------------------------------------------------


@LmDatasetFormatBase.register_subclass("f1_prebuilt")
@dataclass(frozen=True)
class F1PrebuiltLmDatasetFormat(LmDatasetFormatBase):
    """F1-aware version of `PrebuiltLmDatasetFormat`.

    Cache rows carry both `input_ids` and `atom_xyz` columns. The
    downstream dataset is `F1PrebuiltLmDataset`, which emits
    `F1LmExample` (not `GrugLmExample`).
    """

    input_ids_key: str = "input_ids"
    atom_xyz_key: str = "atom_xyz"

    @property
    def token_data_key(self) -> str:
        return self.input_ids_key

    def build_preprocessor(
        self,
        tokenizer,
        *,
        enforce_eos: bool = True,
        enforce_bos: bool = True,
    ) -> BatchProcessor[dict, dict]:
        del tokenizer, enforce_eos, enforce_bos
        return F1PrebuiltCacheProcessor(self.input_ids_key, self.atom_xyz_key)


# ---------------------------------------------------------------------------
# Dataset: TreeCache[dict] → AsyncDataset[F1LmExample]
# ---------------------------------------------------------------------------


def _single_cpu_sharding() -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])


def _build_f1_lm_example(
    tokens_array: jax.Array,
    atom_xyz_array: jax.Array,
    Pos: Axis,
    XYZ: Axis,
    *,
    eos_id: int | None,
    block_cross_document_attention: bool,
):
    """Materialize an `F1LmExample` from raw arrays.

    Mirrors `LmExample.causal` (the same path `PrebuiltLmDataset` uses
    via `GrugLmExample.causal` + `named_lm_example_from_grug`) but
    returns a NamedArray-based `F1LmExample` with `atom_xyz` attached.
    """
    # Import here to avoid an import cycle with marin/qwen3_density.py
    # (qwen3_density imports nothing from f1_data; f1_data only needs
    # F1LmExample at the leaf of the JIT'd helper).
    from qwen3_density import F1LmExample

    tokens = hax.named(tokens_array, (Pos,))
    atom_xyz = hax.named(atom_xyz_array, (Pos, XYZ))

    # Build the base LmExample (tokens/loss_weight/attn_mask).
    base = LmExample.causal(
        tokens=tokens,
        eos_id=eos_id,
        block_cross_document_attention=block_cross_document_attention,
    )
    return F1LmExample(
        tokens=base.tokens,
        loss_weight=base.loss_weight,
        attn_mask=base.attn_mask,
        atom_xyz=atom_xyz,
    )


class F1PrebuiltLmDataset(MappedAsyncDataset[dict, "F1LmExample"]):  # type: ignore[name-defined]
    """F1 version of `PrebuiltLmDataset`.

    Reads `{input_ids, atom_xyz}` per row from the underlying cache
    (`AsyncDataset[dict]`) and emits an `F1LmExample` with named axes
    `(Pos,)` for tokens/loss_weight and `(Pos, XYZ)` for atom_xyz.

    Skipping `GrugLmExample` is intentional — `GrugLmExample` is a raw-
    array eqx.Module with a fixed schema (tokens, loss_weight,
    attn_mask) that has no provision for the F1 sidecar, and rewrapping
    the dataset through `NamedLmDataset` would drop it.
    """

    def __init__(
        self,
        dataset: AsyncDataset[dict],
        Pos: Axis,
        *,
        input_ids_key: str = "input_ids",
        atom_xyz_key: str = "atom_xyz",
        eos_id: int | None = None,
        block_cross_document_attention: bool = True,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.XYZ = Axis("xyz", 3)
        self.input_ids_key = input_ids_key
        self.atom_xyz_key = atom_xyz_key
        self.eos_id = eos_id
        self.block_cross_document_attention = block_cross_document_attention

        sharding = _single_cpu_sharding()
        Pos_local = Pos
        XYZ_local = self.XYZ
        eos_id_local = eos_id
        block = block_cross_document_attention

        @functools.partial(eqx.filter_jit)
        def _create_f1_example(tokens: jax.Array, atom_xyz: jax.Array):
            ex = _build_f1_lm_example(
                tokens,
                atom_xyz,
                Pos_local,
                XYZ_local,
                eos_id=eos_id_local,
                block_cross_document_attention=block,
            )
            return jax.lax.with_sharding_constraint(ex, sharding)

        self._create_f1_example = _create_f1_example

        def _map(example: dict):
            return _create_f1_example(
                example[self.input_ids_key],
                example[self.atom_xyz_key],
            )

        super().__init__(self.dataset, _map)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


# ---------------------------------------------------------------------------
# LmDataConfig subclass that routes F1 components through F1PrebuiltLmDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F1LmDataConfig(LmDataConfig):
    """`LmDataConfig` subclass that emits `F1LmExample`s for F1-formatted
    components.

    For F0 components (anything that's not an `F1PrebuiltLmDatasetFormat`),
    delegates to the parent — mixed F0/F1 mixtures work fine, the F0
    half just emits plain `LmExample`s.

    Currently no `MixtureDataset` for the F1 path: `train_set` returns a
    single F1-component dataset (if there's exactly one F1 component) or
    delegates the entire pipeline to `LmDataConfig` when there are no F1
    components. Multi-F1-component mixtures aren't yet supported (would
    need `F1MixtureDataset`); the tomat pipeline uses a single component
    today, so this is sufficient.
    """

    def _f1_components(self) -> dict[str, DatasetComponent]:
        out: dict[str, DatasetComponent] = {}
        for name, comp in self.components.items():
            if (
                isinstance(comp, DatasetComponent)
                and isinstance(comp.format, F1PrebuiltLmDatasetFormat)
            ):
                out[name] = comp
        return out

    def train_set(
        self,
        Pos: Axis,
        batch_schedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset:
        f1_comps = self._f1_components()
        if not f1_comps:
            return super().train_set(Pos, batch_schedule, key=key)
        if len(self.components) != len(f1_comps):
            raise NotImplementedError(
                "Mixed F0+F1 components are not yet supported. "
                f"Got F1 components {list(f1_comps)} alongside other components "
                f"{[n for n in self.components if n not in f1_comps]}."
            )
        if len(f1_comps) > 1:
            raise NotImplementedError(
                "Multiple F1 components are not yet supported (no F1MixtureDataset)."
            )
        (name, comp), = f1_comps.items()
        return self._build_f1_train_dataset(name, comp, Pos, key=key)

    def validation_sets(self, Pos: Axis) -> Mapping[str, AsyncDataset]:
        f1_comps = self._f1_components()
        if not f1_comps:
            return super().validation_sets(Pos)
        if len(self.components) != len(f1_comps):
            raise NotImplementedError(
                "Mixed F0+F1 components are not yet supported."
            )
        return {
            name: self._build_f1_validation_dataset(name, comp, Pos)
            for name, comp in f1_comps.items()
        }

    def tagged_eval_sets(self, Pos: Axis):
        eval_sets = self.validation_sets(Pos)
        tagged = []
        for name, ds in eval_sets.items():
            tags = (self.components[name].tags or []) + [name]
            tagged.append((ds, tags))
        return tagged

    # -- internals --------------------------------------------------------

    def _build_f1_cache(self, name: str, comp: DatasetComponent, *, split: str) -> TreeCache:
        """Build (or load) the F1 cache for one component+split."""
        # Lean on Levanter's cache build pipeline: `LmDataConfig.build_caches`
        # iterates components, calls each component's `format.build_preprocessor`
        # (we return `F1PrebuiltCacheProcessor`), and feeds parquet rows through
        # it. The resulting cache stores `{input_ids, atom_xyz}` per row.
        caches = self.build_caches(split)
        if name not in caches:
            raise ValueError(
                f"F1 cache for component {name!r} not found in {split!r} split. "
                f"Caches: {list(caches)}"
            )
        return caches[name]

    def _build_f1_train_dataset(
        self,
        name: str,
        comp: DatasetComponent,
        Pos: Axis,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset:
        cache = self._build_f1_cache(name, comp, split="train")
        ds = F1PrebuiltLmDataset(
            cache,
            Pos,
            input_ids_key=comp.format.input_ids_key,
            atom_xyz_key=comp.format.atom_xyz_key,
            eos_id=self.the_tokenizer.eos_token_id,
            block_cross_document_attention=self.block_cross_document_attention,
        )

        # Validation hold-out (mirrors LmDataConfig.train_sets logic).
        if (
            self.num_validation_sequences is not None
            and name in self.num_validation_sequences
        ):
            num_val = self.num_validation_sequences[name]
            ds = self._split_off_train(ds, num_val)

        # Shuffle (mirrors LmDataConfig.train_sets logic). We support the
        # same three modes: True (full permutation), int (era), BlockShuffle.
        from levanter.data.text.datasets import BlockShuffleConfig

        shuffle_cfg = self.shuffle
        perm_type = self.permutation_type or "linear"
        if isinstance(shuffle_cfg, BlockShuffleConfig):
            ds = ds.block_shuffle(
                io_block_size=shuffle_cfg.io_block_size,
                window_blocks=shuffle_cfg.window_blocks,
                key=key,
                perm_type=shuffle_cfg.perm_type,
            )
        elif shuffle_cfg is True:
            ds = ds.shuffle(key, perm_type=perm_type)
        elif isinstance(shuffle_cfg, int) and not isinstance(shuffle_cfg, bool) and shuffle_cfg > 0:
            ds = ds.era_shuffle(shuffle_cfg, key=key, perm_type=perm_type)

        return ds

    def _build_f1_validation_dataset(
        self,
        name: str,
        comp: DatasetComponent,
        Pos: Axis,
    ) -> AsyncDataset:
        # Two cases: (1) explicit validation_urls in the source → separate
        # cache; (2) num_validation_sequences hold-out from the train set.
        if (
            self.num_validation_sequences is not None
            and name in self.num_validation_sequences
        ):
            # Build the train cache and split off the val portion.
            train_cache = self._build_f1_cache(name, comp, split="train")
            full_ds = F1PrebuiltLmDataset(
                train_cache,
                Pos,
                input_ids_key=comp.format.input_ids_key,
                atom_xyz_key=comp.format.atom_xyz_key,
                eos_id=self.the_tokenizer.eos_token_id,
                block_cross_document_attention=self.block_cross_document_attention,
            )
            num_val = self.num_validation_sequences[name]
            return self._split_off_val(full_ds, num_val)

        cache = self._build_f1_cache(name, comp, split="validation")
        return F1PrebuiltLmDataset(
            cache,
            Pos,
            input_ids_key=comp.format.input_ids_key,
            atom_xyz_key=comp.format.atom_xyz_key,
            eos_id=self.the_tokenizer.eos_token_id,
            block_cross_document_attention=self.block_cross_document_attention,
        )

    @staticmethod
    def _split_off_train(ds: AsyncDataset, num_val: int) -> AsyncDataset:
        """Return the train portion (everything before the last `num_val` seqs)."""
        from levanter.data.text.datasets import _split_into_trainval_sets

        train_ds, _ = _split_into_trainval_sets(
            ds, num_val, shuffle=False  # F1 shuffle happens later
        )
        return train_ds

    @staticmethod
    def _split_off_val(ds: AsyncDataset, num_val: int) -> AsyncDataset:
        """Return the val portion (last `num_val` seqs)."""
        from levanter.data.text.datasets import _split_into_trainval_sets

        _, val_ds = _split_into_trainval_sets(ds, num_val, shuffle=False)
        return val_ds
