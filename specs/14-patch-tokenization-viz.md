# Patch-tokenization viz (elvis-based)

Status: **proposed**. Written 2026-04-23.

## Motivation

Our tokenization pipeline is visually opaque: "a material becomes
tokens." For talks, paper figures, and debugging, we want an
interactive (or animated) viz that shows, for one chosen material:

1. The full 3D density cube with atomic positions overlaid.
2. A highlighted patch (cube or ball) being extracted.
3. The tokens that result — preamble (GRID, ATOMS, POS, SHAPE/RADIUS,
   OFFSET/CENTER) followed by the density token stream, expanded /
   collapsible.

This complements the existing decoded token preview in
`scripts/show_tokens.py` (text-only) and the
`tokens-example` block in `site/src/HomePage.tsx` (static).

## Candidate base: elvis (iframe/embed) + Plotly token panel

[elvis](https://elvis.oa.dev) already renders MP materials'
3D charge-density data interactively (iso-surface + atoms). Deck
slides and site pages already link to `elvis.oa.dev/?m=mp-…` for
individual materials.

**Architecture (revised — no r3f dependency):**

- **Full-material 3D density rendering**: lean on elvis. Either
  embed via iframe (simplest) or upstream a "patch overlay" query
  param into elvis itself (e.g. `elvis.oa.dev/?m=mp-X&patch=14,5,9,44`).
- **Patch highlight**: iframe message-passing lets the site's Plotly
  token panel coordinate with elvis's 3D view for cross-highlighting
  (click a density token → elvis highlights the corresponding voxel;
  drag the patch in elvis → token panel scrolls to that patch).
- **Token panel**: tomat-repo React component using our existing
  Plotly / HTML stack. No 3D code needed on the tomat side.
- **Ball-voxel viz** (spec 15) is a sibling piece — also Plotly,
  scatter3d for a few thousand voxels.

r3f (react-three-fiber) was the initial instinct for the 3D voxel
render but isn't needed: elvis already does the heavy 3D work
(iso-surfaces via marching cubes for millions of voxels), and our
sibling viz (spec 15) handles the smaller-voxel-count ball case with
Plotly scatter3d. Keeps the site bundle light and reuses existing
infra.

## Data pipeline

For any given material `mp-XXXXX`:

1. Fetch raw density from Modal volume (or cached copy) → numpy array.
2. Run `scripts/tokenize_patches.py` for one sample → list of tokens.
3. Decode the token stream with `scripts/show_tokens.py` → structured
   blocks (labels + token groups with IDs).
4. Emit JSON with three fields:
   ```json
   {
     "mp_id": "mp-2282417",
     "grid_shape": [64, 108, 108],
     "atoms": [{"Z": 39, "frac": [0.3,0.1,0.0]}, ...],
     "density": "path/to/density.bin or inline base64",
     "patches": [
       {"shape": "cube", "P": 14, "offset": [5, 9, 44],
        "tokens": [...], "annotations": {"preamble": [0,9], "density": [9,5497], ...}},
       ...
     ]
   }
   ```

Host the JSON + density binary on a CDN (cloudflare R2? GCS public
bucket?). Don't embed 10 MB of density data in the site bundle.

## Interaction design

### Main 3D view
- Iso-surface of the full density with slider to control level.
- Atomic positions as spheres (radius ∝ atomic number).
- Orbit controls.
- **Patch highlight overlay**: the P³ cube (wireframe box) or ball
  (semi-transparent sphere) showing where the selected patch sits.
- Small arrow / handle to drag the patch center around (or +/- steps
  for exact voxel offsets).

### Token panel (right side)
- Vertical stream of tokens, colored by block:
  - grey: specials (BOS, EOS, block delimiters)
  - blue: atom tokens (hover → element symbol)
  - green: int tokens (grid dims, shape, offset, bounds)
  - amber: position codec
  - purple: density codec
- Click a token → 3D view highlights corresponding voxel (only for
  density tokens) or atom (for atom tokens).

### Slider / toggle
- "Cube P=14" / "Cube P=15" / "Ball r²≤75" / "Ball r²≤86" modes
  (since these are the ablation variants we care about).
- "Disjoint tiling" mode: shows all ~360 disjoint patches at once
  (mini grid of wireframes).
- "Random offsets at M=32" / "M=256" toggle: compare coverage.

## Scope v1

Don't try to nail all of this at once. v1 MVP:

- One hardcoded material (mp-2282417 — the one we already feature in
  the site's token excerpt).
- Cube mode only, P=14, one fixed patch.
- Static token panel (no interactivity yet).
- Iso-surface + atoms + patch wireframe.

That alone is more compelling than the current text-only excerpt, and
ships as a single React route: `/#/patch-viz/mp-2282417`.

v2 adds patch dragging, cube/ball toggle, M coverage overlay.

## Implementation checklist (v1)

- [ ] `scripts/export_patch_viz_json.py` — one-shot emitter for a
      single mat's viz JSON (runs on laptop; reads from local mp-cache
      or fetches from Modal volume via gsutil).
- [ ] CDN host for viz JSON + density bin (1 material ≈ 2 MB; can
      fit in site/public if we're OK with it).
- [ ] `site/src/PatchTokenizationViz.tsx` — three.js scene + token
      panel. Start from a react-three-fiber template.
- [ ] Route wiring in `App.tsx` + site nav link.
- [ ] Legend describing each token color category.

## Reach items (v2+)

- Multi-material gallery.
- Ball-mode (spec 10).
- Per-patch NMAE heatmap overlay once spec 11 lands.
- "Click to generate": user picks mat + patch, backend tokenizes on
  demand.

## Adjacent

- `scripts/show_tokens.py` — existing text-only decoder.
- `site/src/HomePage.tsx` — current static example block.
- `elvis.oa.dev` — the viewer we'll eventually merge into.
