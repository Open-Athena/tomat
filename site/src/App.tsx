import { PlotlyProvider } from 'pltly/react'
import { DeckPage } from './DeckPage'
import { FilesPage } from './files/FilesPage'
import { HomePage } from './HomePage'
import { KbdShell } from './KbdSetup'
import { MpPage } from './mp/MpPage'
import { PostsPage } from './posts/PostsPage'
import { RunsPage } from './runs/RunsPage'
import { MEvalDrilldownPage } from './runs/MEvalDrilldownPage'
import { VoxelCorrPage } from './voxel-corr/VoxelCorrPage'
import { JointHistPage } from './viz/JointHistPage'
import { parseHash, useHash } from './useHash'

// Use the fork's `lib/index-basic.js` src-mode entry: bar + pie + calendars on
// top of `core` (scatter + components). Skips the `image` trace, so we don't
// pull `probe-image-size` / `buffer/` (Node-only, vite-browser can't resolve).
// Matches the pattern other ryan/runsascoded consumers use (hccs/hbt,
// hccs/household-vehicles); produces a ~1 MB plotly chunk vs ~3.5 MB full.
// `.then(m => m.default ?? m)` unwraps the ESM default export so
// pltly's `<Plot>` sees `P.react`, `P.purge`, etc. directly.
//
// `heatmap` is registered on top so the joint-hist viz (spec 59) can render
// 2-D density images. Adds ~150 KB minified.
const plotlyLoader = async () => {
  const [Pmod, hmMod] = await Promise.all([
    import('plotly.js/lib/index-basic.js'),
    import('plotly.js/lib/heatmap.js'),
  ])
  const P = ((Pmod as { default?: unknown }).default ?? Pmod) as typeof import('plotly.js')
  const heatmap = (hmMod as { default?: unknown }).default ?? hmMod
  // `register` is idempotent — calling more than once with the same trace
  // is a no-op. Safe across HMR re-loads.
  ;(P as { register: (mods: unknown[]) => void }).register([heatmap])
  return P
}

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
        ) : route === 'posts' ? (
          <PostsPage parts={parts.slice(1)} />
        ) : route === 'voxel-corr' ? (
          <VoxelCorrPage parts={parts.slice(1)} />
        ) : route === 'viz' && parts[1] === 'joint-hist' ? (
          <JointHistPage />
        ) : route === 'mp' && parts[1] ? (
          <MpPage mpId={parts[1]} />
        ) : (route === 'run' || route === 'runs')
              && parts[1] && parts[2] === 'm-eval'
              && parts[3] && parts[4] ? (
          // Spec 60: `#/run/<runId>/m-eval/<step>/<setKey>`. Accepts both
          // `run` and `runs` for the top segment — the rest of the app
          // routes runs under `#/runs/<label>`, and having the drilldown
          // reachable under the same prefix keeps back-navigation clean.
          <MEvalDrilldownPage
            runId={decodeURIComponent(parts[1])}
            step={parseInt(parts[3], 10)}
            setKey={decodeURIComponent(parts[4])}
          />
        ) : (
          <HomePage />
        )}
      </KbdShell>
    </PlotlyProvider>
  )
}
