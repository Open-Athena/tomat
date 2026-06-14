#!/usr/bin/env python3
"""Minimal repro of the Levanter in-training eval crash.

Mimics the exact flow that crashes in production at
`levanter/data/loader.py:377` when `_JaxCpuBackgroundIterator` is the
data-loader's producer thread on multi-host TPU.

The production flow (`_batchify_local_data` in loader.py:353):

  with local_cpu_mesh():        # producer thread, CPU mesh active
      jax.make_array_from_callback(
          shape, NamedSharding(trainer_mesh, P(...)),   # TPU-sharded output
          data_callback,                                 # invoked per device
      )

  # data_callback runs inside the same `local_cpu_mesh()` block:
  def data_callback(indices):
      local_data = [...]                       # bare CPU rows
      return stack_tree(local_data)[leaf]      # module-level @jax.jit

The cache key for `stack_tree`'s compilation captures the active mesh at
trace time (CPU). On multi-host TPU, `make_array_from_callback` then
tries to materialize the result on the TPU mesh — and on certain
host-restart interleavings, the cached CPU-context compilation gets hit
with arguments whose device IDs span the TPU mesh → JAX raises:

  ValueError: Received incompatible devices for jitted computation.
  Got argument ... on platform TPU
  and jit's context mesh with device ids [<CPU id>] on platform CPU

## Run

Single-host smoke:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tmp/mre_stack_tree_mesh_mismatch.py

The single-host CPU smoke is NOT expected to repro the exact error —
JAX is permissive about CPU<->CPU mesh transfers within one process.
But the script structure is the test bed: drop into the Marin worktree,
run with real TPU + multi-host (`jax.distributed.initialize`) and the
production traceback fires identically.

The script prints PASS / FAIL per case so it's CI-checkable. A green
single-host CPU run does NOT prove the bug is fixed; only a green
multi-host TPU run does.
"""
import os
import threading
import contextlib
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402


# ── module-level @jit: same shape as `loader.py:stack_tree` ──────────────
@jax.jit
def stack_tree(items):
    """Mimics `loader.py:564` — module-level `@jax.jit` over a tree-map
    of `jnp.stack`. The actual content doesn't matter; what matters is
    that the cache is module-scoped and captures whatever mesh is active
    at first-trace time."""
    return jnp.stack(items)


# ── mimic Levanter's `local_cpu_mesh` from utils/jax_utils.py:54 ─────────
@contextlib.contextmanager
def local_cpu_mesh():
    """Enter a 1-device CPU mesh. On multi-host TPU this is what
    `_JaxCpuBackgroundIterator._fill_queue_with_batches` enters around
    its producer fn."""
    cpu = jax.local_devices(backend="cpu")[0]
    mesh = Mesh(np.array([cpu]).reshape(1, 1, 1, 1),
                axis_names=("replica", "data", "model", "context"))
    with jax.set_mesh(mesh):
        yield mesh


def make_trainer_mesh() -> Mesh:
    """The trainer's outer mesh (multi-device, all backends present).
    Stand-in for `trainer_cfg.device_mesh` in production."""
    devs = jax.devices()
    if len(devs) < 2:
        raise RuntimeError(
            f"Need >= 2 JAX devices for the trainer mesh; got {len(devs)}. "
            "Re-run with XLA_FLAGS=--xla_force_host_platform_device_count=4."
        )
    return Mesh(devs[:4], axis_names=("data",))


# ── producer flow: exactly what `_batchify_local_data` does ──────────────
def _produce_one_batch(trainer_mesh: Mesh, batch_seq_len: int = 1) -> jax.Array:
    """Build a global TPU-sharded array via `make_array_from_callback`,
    with the data-callback calling `stack_tree` under `local_cpu_mesh`.

    This is the call site that crashes in production: line numbers map
    1:1 to `loader.py:_batchify_local_data` → `make_global_array_for_leaf`
    → `get_data` → `get_local_data_for_leaf` → `get_local_batch` →
    `stack_tree`."""
    # Total batch of `n_devices` rows (one per device of the trainer mesh).
    n_devices = trainer_mesh.size
    sharding = NamedSharding(trainer_mesh, P("data"))
    out_shape = jax.ShapeDtypeStruct(
        shape=(n_devices, batch_seq_len), dtype=jnp.int32,
    )

    # The output's per-device-shape sub-slice is (1, batch_seq_len). The
    # callback gets per-device indices and must return THAT shape.
    def data_callback(indices):
        # Pull `end - begin` rows; they're stacked along axis 0.
        begin, end = indices[0].start or 0, indices[0].stop or n_devices
        # Bare numpy rows — same as Levanter's `local_data.append(...)`.
        local_rows = [
            jnp.array([i], dtype=jnp.int32) for i in range(begin, end)
        ]
        # The @jit'd `stack_tree` call. First-trace happens here, under
        # whatever mesh is active in this thread.
        stacked = stack_tree(local_rows)
        # `stacked` shape is (end - begin, 1).
        return stacked

    return jax.make_array_from_callback(out_shape.shape, sharding, data_callback)


def case_a_single_thread_cpu_mesh(trainer_mesh: Mesh) -> tuple[bool, str]:
    """Producer (= main thread here) inside `local_cpu_mesh`; output
    sharded on the trainer mesh. This is the exact production pattern.

    Expected: on multi-host TPU, raises ValueError.
    On single-host CPU smoke, JAX is permissive and this passes."""
    try:
        with local_cpu_mesh():
            out = _produce_one_batch(trainer_mesh)
        return True, f"out.shape={out.shape}, sharding={out.sharding}"
    except ValueError as e:
        return False, str(e).splitlines()[0]


def case_b_producer_thread_then_consumer(trainer_mesh: Mesh) -> tuple[bool, str]:
    """Producer thread populates the `stack_tree` cache under CPU mesh;
    then a fresh consumer thread invokes `make_array_from_callback` under
    the trainer mesh. Simulates the train+eval cadence (train BG iter
    runs first, eval BG iter constructs later — both call `stack_tree`).
    """
    cache_warmup_result: dict[str, Any] = {}

    def warmup_producer():
        try:
            with local_cpu_mesh():
                # Call `stack_tree` directly (not through
                # make_array_from_callback) so we only seed the cache.
                items = [jnp.array([i], dtype=jnp.int32) for i in range(4)]
                _ = stack_tree(items)
                cache_warmup_result["ok"] = True
        except Exception as e:
            cache_warmup_result["err"] = e

    t = threading.Thread(target=warmup_producer)
    t.start()
    t.join()
    if "err" in cache_warmup_result:
        return False, f"warmup failed: {cache_warmup_result['err']}"

    # Consumer call from main thread, under trainer mesh.
    try:
        with jax.set_mesh(trainer_mesh):
            out = _produce_one_batch(trainer_mesh)
        return True, f"out.shape={out.shape}, sharding={out.sharding}"
    except ValueError as e:
        return False, str(e).splitlines()[0]


def main():
    print(f"jax {jax.__version__} backend={jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print()

    trainer_mesh = make_trainer_mesh()
    print(f"trainer_mesh.devices: {trainer_mesh.devices.flatten()}")
    print()

    cases = [
        ("A: producer-flow under local_cpu_mesh (exact production pattern)",
         case_a_single_thread_cpu_mesh),
        ("B: producer-thread cache warmup, then consumer under trainer mesh",
         case_b_producer_thread_then_consumer),
    ]

    for name, fn in cases:
        print(f"== {name} ==")
        ok, detail = fn(trainer_mesh)
        verdict = "PASS" if ok else "FAIL (expected on multi-host TPU)"
        print(f"  {verdict}: {detail}")
        print()

    print("== verdict ==")
    print("On single-host CPU smoke, both cases typically PASS (JAX is")
    print("permissive about CPU↔CPU mesh transfers within one process).")
    print()
    print("On multi-host TPU (the production failure mode):")
    print(" - Case A raises `Received incompatible devices` with `device")
    print("   ids [<CPU id>] on platform CPU` for the jit-context mesh.")
    print(" - Case B exhibits the intermittent variant — depends on which")
    print("   thread first-traces `stack_tree` under whose mesh.")
    print()
    print("If a candidate fix is in place, both cases should PASS on")
    print("multi-host TPU. The MRE is the litmus test.")


if __name__ == "__main__":
    main()
