# Inference CLI: `tomat predict`

## Goal

Take a model checkpoint + material ID, run inference, write predicted
density to GCS as zarr (CHGCAR-equivalent). Produce small-payload preview
slices for fast visualization. Eventually wire into Elvis.

Two execution backends:
- **Phase 1 (this spec)**: iris/TPU. Reuse the existing `eval_mat_nmae.py`
  inference path. Ship sooner; users with iris access can run.
- **Phase 2 (follow-up)**: Modal/GPU. No TPU rental; sub-minute cold start;
  exposes an HTTP endpoint Elvis can call directly. More plumbing.

## Phase 1: iris-based predict mode

### CLI

    tomat predict <ckpt-label> <mp_id> \
        [--step N | latest]                       # which ckpt to use
        [--out-zarr gs://path/to/output.zarr]     # default: alongside eval results
        [--decoder median|mean|argmax]            # default median
        [--split validation|train]                # for MP-data lookup
        [--n-mats 1] [--tpu v6e-8]

Submits an iris job that runs a new `eval_mat_nmae.py` mode (or a sister
script `predict_density.py` — see "implementation"). Output: a zarr file
containing the predicted density grid, plus a small `summary.json`
sidecar with grid shape, decoder used, source ckpt, etc.

### Implementation options

**Option A: extend `eval_mat_nmae.py`**

Add `TOMAT_PREDICT_ONLY=1` mode. When set:
- Skip true-density loading + NMAE/NEMD computation.
- Get grid shape from the structure's natural tessellation (or from a
  `--grid-shape` override). Currently `eval_mat_nmae.py` reads `density`
  from the val zarr to know the grid; we'd need to derive it from the
  structure metadata instead.
- After stitching `rho_pred`, write to `--out-zarr` instead of computing
  metrics.

Pros: minimal new code; reuses existing model load + tokenize + decode
+ stitch logic.
Cons: pollutes eval script with an unrelated mode; the "get grid shape
from structure" logic is new and needs to match what tokenization
expects.

**Option B: new sister script `predict_density.py`**

Pulls out the model load + forward + stitch helpers into a shared
module (`marin/density_inference.py`), used by both eval_mat_nmae.py
and the new predict script. Cleaner separation.

Pros: cleaner; reusable inference module is good for Phase 2 (Modal).
Cons: ~half a day of refactor before any new behavior.

**Recommendation**: B. The shared module pays for itself when we add
Modal in Phase 2.

### Phase-1 deliverables

1. `marin/density_inference.py` — shared helpers:
   - `load_inference_model(ckpt_path, model_preset, codec_path) -> (model, codec, vocab_offsets)`
   - `tokenize_for_inference(structure, vocab_offsets, codec, P, ctx) -> (B, ctx) int32 ids + offsets/local_slices`
   - `forward_decode_stitch(model, codec, ids, offsets, local_slices, decoder='median') -> (rho_pred_grid, optional_emd_grid)`
2. `marin/eval_mat_nmae.py` refactored to use `marin/density_inference.py`.
3. `marin/predict_density.py` — new, calls the shared module. Inputs:
   - `TOMAT_PREDICT_MP_ID` — single mp_id
   - `TOMAT_PREDICT_OUT_ZARR` — gs:// path
   - `TOMAT_PREDICT_GRID_SHAPE` — optional override
4. `tomat predict` subcommand wiring iris job submission.

## Phase 2: Modal / GPU backend

### Why Modal

- No TPU rental. Pay per second of GPU time (~$1.50/hr H100, ~$2/hr A100).
- Cold start ~30s; per-material inference for 200M model is seconds.
- Exposes an HTTP endpoint Elvis can call directly (no iris/wandb in the
  loop for end users).

### Verification status

**Not yet verified.** Specifically un-verified:
1. Does `jax[cuda]` + `levanter` + `haliax` install cleanly in a Modal
   image with reasonable build time (under 20 min)?
2. Does `eqx.filter_eval_shape(...)` + `load_checkpoint(...)` work for
   our orbax checkpoints on GPU?
3. Does the inference forward pass run on GPU without TPU-specific
   sharding code? (e.g. `hax.shard_with_axis_mapping(model, ...)` — this
   constructs a TPU mesh; on single GPU we'd skip sharding.)

Recommended verification step: a minimal `scripts/verify_modal_jax.py`
that does ckpt load + one forward pass, no real output. Should take
~1-2hr to write and verify.

### Architecture sketch

    Modal app: tomat-density-inference
      └── @app.function(gpu="A100", timeout=600, image=infer_image)
           def predict(ckpt_path: str, mp_id: str, decoder: str = "median") -> dict:
               # 1. download mat metadata (MPDB lookup)
               # 2. tokenize
               # 3. load_inference_model on GPU
               # 4. forward + decode + stitch
               # 5. write rho_pred to zarr (GCS)
               # 6. extract 3 axis-aligned slices, return as base64-PNG or
               #    raw float arrays for client-side rendering
               return {
                   "zarr_url": "gs://.../mp-XXX-cont7k-ext-step-9000.zarr",
                   "preview_slices": {...},  # small payload
                   "grid_shape": [...],
                   "decoder": decoder,
                   "elapsed_s": ...,
               }

      └── @app.function(image=infer_image, allow_concurrent_inputs=4)
           @modal.web_endpoint(method="POST")
           def predict_http(req): -> dict:
               # parse JSON request, call .predict.remote(...), return JSON

### CLI integration (Phase 2)

    tomat predict --backend modal <ckpt-label> <mp_id> ...

When `--backend modal` is set, the CLI calls Modal directly via the
`modal` Python SDK. Same args as Phase 1.

### Elvis integration

Elvis calls the Modal HTTP endpoint with `(ckpt, mp_id)`, gets back
slice data + zarr URL. Displays slices immediately, links "open full
diff" → loads zarr in 3D viewer.

For the diff vs target case: Elvis already supports loading two zarrs
and computing a per-voxel diff. Our endpoint just needs to return:
- predicted zarr URL (this material, this ckpt)
- target zarr URL (already in `gs://.../tomat/rho_gga_raw/...`)

## Phase 2 addendum: Modal as a persistent inference *server* (2026-05-12)

Originally framed as "Modal as a sub-minute one-shot job per material."
Discussion with Betsy reframed this: what we actually want is a
**persistent HTTP endpoint** that Elvis (and humans) can hit ad-hoc,
with a lazy per-(ckpt, mat) cache, instead of:

- (a) firing a fresh Modal job per `(ckpt, mat)` request (cold-start
  overhead each time), or
- (b) pre-emitting Zarrs for every (ckpt, mat) pair from the /1k-step
  eval pipeline (decided against — see below).

Server shape:

    @app.cls(
        image=infer_image,
        gpu="H100", keep_warm=1,   # always-on; cold-start ~30s on first hit
        timeout=300,
    )
    class Predictor:
        @modal.enter()
        def load(self):
            self.ckpt_cache = {}   # (run_label, step) → loaded model
            self.density_cache = R2Cache("openathena", "tomat/predictions/")

        @modal.method()
        def predict(self, run_label: str, step: int, mp_id: str,
                    decoder: str = "median") -> dict:
            cache_key = f"{run_label}/step-{step}/{mp_id}/{decoder}.zarr"
            if zarr_url := self.density_cache.get(cache_key):
                return {"zarr_url": zarr_url, "hit": True, "preview": …}
            model = self.ckpt_cache.setdefault((run_label, step),
                _load_ckpt_from_gcs(run_label, step))
            rho = _forward_decode_stitch(model, _structure(mp_id), decoder)
            zarr_url = self.density_cache.put(cache_key, rho)
            return {"zarr_url": zarr_url, "hit": False, "preview": …,
                    "elapsed_s": …}

        @modal.web_endpoint(method="POST")
        def http(self, req): return self.predict(**req.json())

R2 cache layout (mirrors `specs/23-runs-dashboard.md`'s pattern):

    s3://openathena/tomat/predictions/
        <run_label>/step-<N>/<mp_id>/<decoder>.zarr/   ← zarr v2 dir layout

Cache invalidation: never. (ckpt, mat, decoder) tuples are immutable;
re-deriving costs the inference time anyway, so cache-as-side-effect
is essentially free.

### Why NOT pre-emit Zarrs from /1k-step mat-evals

Tempting alternative: have `eval_mat_nmae.py` also write the predicted
density Zarr for each (ckpt, mat) pair as it computes NMAE. Rejected
(decided 2026-05-12 with ryan):

- **Storage**: 200 mats × ~5 MB each (200³ float16 grid) × ~5 "interesting"
  ckpts per run ≈ 5 GB per run. Bearable but mostly waste — most
  intermediate ckpts get inspected zero times.
- **Indistinguishable from waste at write time**: we don't know which
  ckpts will turn out to be "best" until later. Writing all ckpts'
  predictions to disk is paying the IO cost upfront for a value
  realized on at most a few of them.
- **Live-inference path is needed anyway** for any mat outside the
  200-val/200-train snapshot, and for ad-hoc "what does this ckpt do on
  this material" exploration. Once that path exists, on-demand
  inference + lazy server-side cache strictly dominates pre-emit.
- **Compromise if pain emerges**: emit Zarrs only at
  `lifecycle/trainer_finished` and at post-hoc-identified best-NMAE
  ckpts. Handful of ckpts per run, not all of them.

## Open questions

1. Grid shape for prediction-without-target: do we need the user to
   provide one, or can we derive a reasonable default from the
   structure? For diff-vs-target use case it's already known; for pure
   prediction we might want a default (e.g. `120³`).
2. Decoder choice in CLI: default to `median` (consistent with eval) or
   expose all three (`median`, `mean`, `argmax`)? Recommend default
   `median`, expose all.
3. ZarrCAR format: do we follow Elvis's expected layout, or write a
   simpler `.zarr` and let Elvis adapt? Need to check Elvis's reader.
4. `keep_warm` cost: even with no traffic, `keep_warm=1` on H100 is
   ~$2/h × 24 × 30 ≈ $1.4k/mo on Modal's pricing. Probably want
   `keep_warm=0` initially (~30s cold-start per first request after
   idle), revisit if traffic warrants. Or scale-to-zero with snapshot
   restore once Modal supports it for our image.
