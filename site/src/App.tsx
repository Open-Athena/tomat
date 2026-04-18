import { PlotlyProvider } from 'pltly/react'
import { DeckPage } from './DeckPage'
import { HomePage } from './HomePage'
import { parseHash, useHash } from './useHash'

const plotlyLoader = () =>
  import('plotly.js-basic-dist') as unknown as Promise<typeof import('plotly.js')>

export function App() {
  const hash = useHash()
  const parts = parseHash(hash)
  const route = parts[0] ?? ''

  return (
    <PlotlyProvider loader={plotlyLoader}>
      {route === 'deck' ? <DeckPage /> : <HomePage />}
    </PlotlyProvider>
  )
}
