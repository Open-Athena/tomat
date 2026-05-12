# 3D ball voxel viz (Plotly-based, with R slider)

Status: **proposed**. Written 2026-04-23.

## Goal

Interactive viz on the tomat site that shows:

1. **3D scatter of the integer voxels inside a ball**, centered at
   origin, with an R slider that adjusts the squared-radius threshold.
2. **Brushed companion plot** showing V(R²) cumulative voxel count
   vs R², synced with the slider.

Use case: makes the ball-patch geometry tangible, lets users see *why*
R²=75 ≈ cube P=14, and gives a visual handle on the spiky r₃(n) shell
structure (OEIS A005875).

## Why Plotly, not r3f

Initial instinct was react-three-fiber. Overkill for this viz: each
ball has O(hundreds–low-thousands) voxels, and Plotly's `scatter3d`
handles that size natively with full interactivity (orbit, zoom,
hover-to-label, legend). Plotly is already a site dep.

r3f would be warranted for rendering a **full material's density cube**
(millions of voxels; needs iso-surface + volume shading) — but that
task belongs to elvis, not this viz.

## Layout

Single page at `/#/ball-viz` (or embedded as a section on the home).

```
  ┌──────────────────────────────────────────────────────────────┐
  │  R² slider [1 ────●──── 200]   |R|≈8.66   |V(R²)|=2,777       │
  ├───────────────────────────┬──────────────────────────────────┤
  │                           │                                  │
  │   3D scatter of voxels    │    V(R²) vs R² (brushed)         │
  │   coloured by r²          │                                  │
  │   marker size ~ small     │    ●  current R² (green dot)     │
  │                           │    —  r₃(n) bars below            │
  │                           │    …  (4/3)π R³ reference curve  │
  │                           │                                  │
  └───────────────────────────┴──────────────────────────────────┘
  Preset chips: [R²=75 (≈P=14)] [R²=86 (≈P=15)] [R²=138 (P=19)] [custom…]
```

## Data pipeline (client-side)

No backend. Precompute at build time:

```ts
// shared utility, runs at import
const MAX_R2 = 200;
const ballCache: Array<{ dx: number; dy: number; dz: number; r2: number }> = [];
for (let dx = -R; dx <= R; dx++)
  for (let dy = -R; dy <= R; dy++)
    for (let dz = -R; dz <= R; dz++) {
      const r2 = dx*dx + dy*dy + dz*dz;
      if (r2 <= MAX_R2) ballCache.push({ dx, dy, dz, r2 });
    }
ballCache.sort((a, b) => a.r2 - b.r2 || a.dy - b.dy || a.dx - b.dx || a.dz - b.dz);
```

At R² = N, render all voxels with `r2 ≤ N` as a Plotly `scatter3d`
trace. Slider updates trigger a layout update (not a full re-render).

## Component sketch

```tsx
// site/src/BallVoxelViz.tsx
import { useMemo, useState } from 'react'
import { Plot, useTheme } from 'pltly/react'

const BALL = precomputeBall(200)  // { dx, dy, dz, r2 }[]
const SHELL = buildShellCounts(200)           // r3(n)
const CUM = cumsum(SHELL)                      // V(n)

export function BallVoxelViz() {
  const [R2, setR2] = useState(75)
  const visible = useMemo(
    () => BALL.filter(v => v.r2 <= R2),
    [R2]
  )
  return (
    <>
      <div className="controls">
        <label>R² = {R2}  (R ≈ {Math.sqrt(R2).toFixed(2)}, V = {CUM[R2].toLocaleString()} voxels)</label>
        <input type="range" min={1} max={200} value={R2} onChange={e => setR2(+e.target.value)} />
        <div className="preset-chips">
          {[{r2:75, label:'R²=75 (≈P=14)'}, {r2:86, label:'R²=86 (≈P=15)'}, {r2:138, label:'R²=138 (P=19)'}].map(p =>
            <button key={p.r2} onClick={() => setR2(p.r2)}>{p.label}</button>
          )}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
        <Plot
          data={[{
            type: 'scatter3d', mode: 'markers',
            x: visible.map(v => v.dx),
            y: visible.map(v => v.dy),
            z: visible.map(v => v.dz),
            marker: {
              size: 4,
              color: visible.map(v => v.r2),
              colorscale: 'Viridis',
              showscale: true,
              colorbar: { title: { text: 'r²' }, thickness: 10 },
            },
            text: visible.map(v => `(${v.dx},${v.dy},${v.dz})<br>r²=${v.r2}`),
            hovertemplate: '%{text}<extra></extra>',
          }]}
          layout={{
            autosize: true, height: 500,
            scene: {
              xaxis: { range: [-15, 15] }, yaxis: { range: [-15, 15] }, zaxis: { range: [-15, 15] },
              aspectmode: 'cube',
            },
          }}
        />
        <Plot
          data={[
            // bars for shell count
            { type: 'bar', x: Array.from({length: 201}, (_, i) => i), y: SHELL, name: 'r₃(n)', marker: { color: '#3b6a9e' } },
            // cumulative curve (scaled for visual fit)
            { type: 'scatter', mode: 'lines', x: Array.from({length: 201}, (_, i) => i), y: CUM, name: 'V(n)', yaxis: 'y2', line: { color: '#d9471f', width: 2 } },
            // current R² marker
            { type: 'scatter', mode: 'markers', x: [R2], y: [CUM[R2]], yaxis: 'y2', marker: { color: '#2b8a3e', size: 14 }, showlegend: false },
          ]}
          layout={{
            xaxis: { title: { text: 'R²' } },
            yaxis: { title: { text: 'shell count r₃(n)' } },
            yaxis2: { overlaying: 'y', side: 'right', title: { text: 'cumulative V(n)' } },
          }}
        />
      </div>
    </>
  )
}
```

## Extensions (v2+)

- **Preset "cube comparison"**: overlay the matching cube P=? bounding
  box as a wireframe in the 3D view.
- **Shell-by-shell animation**: auto-play R² from 0 → 200, showing
  voxels snap on as each shell activates.
- **r₃(n)=0 spots highlighted** on the bar chart (Gauss's gap
  integers: `4ᵃ(8b+7)`).
- **Link-out to OEIS** (A005875, A117609) as hover footnotes.

## Scope / effort

- v1: ~1 hour React + styling. Reuses all existing Plotly infra.
- No backend work. No new dependency.
- No new asset uploads (all computed client-side).

## Adjacent

- `docs/ball-math.md` — the math this viz makes concrete.
- `scripts/plot_ball_voxels.py` — static matplotlib version (already
  live on site via `site/public/ball-voxel-counts.png`).
- `specs/10-ball-patches.md` — ball tokenizer design, consumer of this
  geometry.
