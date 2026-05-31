import { PlotlyProvider } from 'pltly/react'
import { DeckPage } from './DeckPage'
import { FilesPage } from './files/FilesPage'
import { HomePage } from './HomePage'
import { KbdShell } from './KbdSetup'
import { RunsPage } from './runs/RunsPage'
import { VoxelCorrPage } from './voxel-corr/VoxelCorrPage'
import { parseHash, useHash } from './useHash'

// Use the fork's `lib/index-basic.js` src-mode entry: bar + pie + calendars on
// top of `core` (scatter + components). Skips the `image` trace, so we don't
// pull `probe-image-size` / `buffer/` (Node-only, vite-browser can't resolve).
// Matches the pattern other ryan/runsascoded consumers use (hccs/hbt,
// hccs/household-vehicles); produces a ~1 MB plotly chunk vs ~3.5 MB full.
// `.then(m => m.default ?? m)` unwraps the ESM default export so
// pltly's `<Plot>` sees `P.react`, `P.purge`, etc. directly.
const plotlyLoader = () =>
  import('plotly.js/lib/index-basic.js').then(
    (m) => ((m as { default?: unknown }).default ?? m) as typeof import('plotly.js')
  )

export function App() {
  const hash = useHash()
  const parts = parseHash(hash)
  const route = parts[0] ?? ''

  return (
    <PlotlyProvider loader={plotlyLoader}>
      <KbdShell>
        {route === 'deck' ? (
          <DeckPage />
        ) : route === 'runs' ? (
          <RunsPage parts={parts.slice(1)} />
        ) : route === 'files' ? (
          <FilesPage />
        ) : route === 'voxel-corr' ? (
          <VoxelCorrPage parts={parts.slice(1)} />
        ) : (
          <HomePage />
        )}
      </KbdShell>
    </PlotlyProvider>
  )
}
