# Per-material validation infrastructure

Status: **proposed**. Written 2026-04-23.

## Motivation

Training-time val loss (eval/tomat/loss, eval/tomat/bpb) is computed on
a held-out set of **random patches**. A 208M run hit val_loss=2.043 this
way, matching train loss. But this doesn't answer Betsy's question:

> It would be cool to piece together a full material by knitting
> sub-cubes together to validate if the performance is the same, or
> if there are some artifacts that emerge.

That is: **can the model reconstruct a *material* from its patches?**
Per-patch loss can look great while the patch boundaries have seams,
edge effects, or stitched-region inconsistencies.

This spec proposes infrastructure to answer that end-to-end.

## Scope

- Hold out a set of **whole materials** (not random patches) as the
  "full-material val set." Separate from the 256-seq random-patch val.
- For each material:
  1. Tile the grid with (disjoint or overlapping) patches.
  2. Decode each patch's density from the model's token predictions.
  3. Stitch patches back into a full (nx, ny, nz) density array.
  4. Compute **material-level NMAE** = mean(|ρ_pred − ρ_true|) / mean(|ρ_true|),
     plus **χ²** as the secondary metric (per spec 02 fidelity-sweep
     conventions).
- Report: per-mat NMAE distribution, aggregate mean/median/p99 NMAE,
  and visualize a few representative cases.

Crucially: this is **an eval**, not a training loss. The model is
already trained via the random-patch path. We're measuring downstream
reconstruction fidelity.

## Data split

Possibilities for the full-material val set:

1. **Carve out of val-full (4,305 mats)**: use ~500 mats as full-material
   val; retrain or re-tokenize without these. Smallest overhead.
2. **Use the ~4,400 unused Della mats** (86,192 total − 77,498 train −
   4,305 val = ~4,389). Probably the test split that Hananeh held out.
   Properly clean — never seen by training, never seen by random-patch val.
3. **Generate new held-out mats via DFT** — ideal long-term, but months
   of lead time. Skip for now.

**Recommendation:** start with **option 2** (the test split). Separate
tokenize pipeline, carved out once and reused.

## Reconstruction algorithms

Two modes (both worth supporting):

### 2a. Disjoint tiling
Cover the grid with non-overlapping P×P×P patches (ignore the remainder
past `⌊nx/P⌋*P`, `⌊ny/P⌋*P`, `⌊nz/P⌋*P` — typically ≤5% of grid
volume). Each voxel appears in exactly one patch → no averaging.
Fastest, cleanest semantics, tests "does the model handle each
patch in isolation?"

### 2b. Overlapping tiling
Stride of P/2 or P/4; each voxel appears in multiple patches.
Average the decoded densities (or pick-max, or Gaussian-weighted).
Tests "do patch boundaries align across shifted positions?" — the
symmetry Betsy cares about.

For balls (spec 10), only the overlapping mode applies since balls
don't tile.

## Model-in-the-loop

The model produces tokens, not densities. Reconstruction requires:

1. For each patch, construct the input preamble up to `[DENS_START]`.
2. **Greedy-decode** (argmax) the density tokens autoregressively for
   `V_patch` voxels × `tokens_per_voxel` positions. ~5k–7k decode
   steps per patch.
3. Decode the density-token stream back to floats via the codec.

At 77k val mats × ~500 patches (disjoint tiling of 128³) × 5k token
decode steps each, this is a lot of inference. Need a batched
inference path:

```python
# Pseudocode
def reconstruct_material(model, density_gt, tokenizer) -> np.ndarray:
    patches = tile_disjoint(density_gt, P)                # (N, P, P, P)
    preambles = [build_preamble(mat, patch_offset) for patch_offset in ...]  # (N, pre_len)
    # Batched KV-cached greedy decode:
    dens_tokens = model.decode_greedy_batch(preambles, max_new=5488)  # (N, 5488)
    dens_floats = tokenizer.decode_density_batch(dens_tokens)           # (N, P, P, P)
    return stitch_disjoint(dens_floats, density_gt.shape)               # (nx, ny, nz)
```

Levanter has a generation API (`levanter.inference.generate`); need to
check it supports batched decode with KV-cache. If not, we either write
our own or accept slower unbatched decode.

## Outputs

Per eval run:

```json
{
  "run_id": "train-full-tpu8-200M-...",
  "step": 1999,
  "val_set": "test-split-v1",
  "n_materials": 500,
  "tiling": "disjoint" | "overlapping_stride_7",
  "metrics": {
    "nmae_mean": 0.019,
    "nmae_median": 0.016,
    "nmae_p99": 0.087,
    "chi2_mean": 0.00012
  },
  "per_material": [
    {"mp_id": "mp-XXXXX", "nmae": 0.016, "chi2": 0.00008, "n_patches": 487},
    ...
  ]
}
```

Log to W&B as an artifact so it's pinned to the training step.

## Visualizations

For a handful (~5–10) representative materials:

- Slice-through 2D heatmaps: (ρ_true, ρ_pred, ρ_pred − ρ_true) triples.
- Maybe an elvis-style 3D iso-surface viz showing error clusters.
- Per-patch NMAE distribution plotted as a grid (shows if edge
  patches are systematically worse).

## Integration with training

Option A: **Post-training eval job.** `scripts/eval_reconstruction.py`
that loads a checkpoint from GCS, does the reconstruction pass,
writes metrics. Manual trigger.

Option B: **Mid-training eval callback.** Levanter supports custom
evaluation callbacks. Register one that runs reconstruction every N
steps (ideally at the same cadence as train-time val eval). Feeds
into the live W&B loss plot.

**Start with A, graduate to B** once the infra is stable. Reconstruction
eval is slow (~5 min for 100 mats even on v6e-8); doing it every
training eval would ~2× training cost.

## Open questions

- **Greedy vs temperature sampling**: greedy (argmax) is deterministic
  and matches reconstruction semantics best; but if the model is
  under-trained it may collapse to a bad mode. Temperature ε-sampling
  as a fallback?
- **Stride choice for overlapping**: P/2 gives 2^3 = 8× more patches;
  P/4 is 64× more. Start with P/2.
- **Atomic-info preamble**: does the model need to see *full* atomic
  inventory per patch, or just atoms within a neighborhood of the patch?
  Current training uses full inventory. Reconstruction should match.
- **Grid-dim > 1023 materials**: excluded from training (preamble
  overflow). Exclude from reconstruction eval too, or handle specially.

## Adjacent specs

- `specs/10-ball-patches.md` — ball ablation; per-mat eval is how we
  compare balls vs cubes at the right metric.
- `specs/12-tokenize-to-gcs.md` — per-material parquet sharding in the
  new tokenize pipeline makes per-mat holdouts cheap.
- `specs/done/02-fidelity-sweep.md` — prior NMAE/χ² evaluation on
  tokenizer schemes (fit-only, no LM training); reference for the
  metric definitions.
