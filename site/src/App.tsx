import { PlotlyProvider } from 'pltly/react'
import { DeckPage } from './DeckPage'
import { HomePage } from './HomePage'
import { RunsPage } from './runs/RunsPage'
import { parseHash, useHash } from './useHash'

const plotlyLoader = () =>
  import('plotly.js-basic-dist') as unknown as Promise<typeof import('plotly.js')>

export function App() {
  const hash = useHash()
  const parts = parseHash(hash)
  const route = parts[0] ?? ''

  return (
    <PlotlyProvider loader={plotlyLoader}>
      {route === 'deck' ? (
        <DeckPage />
      ) : route === 'runs' ? (
        <RunsPage parts={parts.slice(1)} />
      ) : (
        <HomePage />
      )}
    </PlotlyProvider>
  )
}
