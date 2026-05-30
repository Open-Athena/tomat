// Custom `use-prms` LocationStrategy for this app's HashRouter-with-query
// pattern.
//
// The app routes via `window.location.hash` (`#/runs`, `#/runs/<id>`, `#/deck`,
// …) but ALSO carries query-string state inside the hash:
//   `#/runs?tags=CE+~bunk&regex=cont33k`
//
// Neither built-in strategy fits:
//   • `queryStrategy` reads/writes `window.location.search` — the wrong place
//     when the app is hash-routed.
//   • `hashStrategy` treats the WHOLE hash as a `URLSearchParams` payload, so
//     it would parse `/runs?tags=CE` as `{ "/runs?tags": "CE" }` and clobber
//     the path on write.
//
// This strategy keeps the path segment (everything before `?`) intact and
// reads/writes only the post-`?` query string.
//
// Import this module for its side-effect ONCE at app start (in `main.tsx`),
// before any `useUrlState` hook renders.

import { setDefaultStrategy, parseMultiParams, serializeMultiParams } from 'use-prms'
import type { LocationStrategy, MultiEncoded } from 'use-prms'

const LOCATION_CHANGE_EVENT = 'use-prms:locationchange'

function splitHash(hash: string): { path: string; query: string } {
  // Normalise to a leading `#`; everything before `?` is the path (incl. `#`).
  const h = hash.startsWith('#') ? hash : `#${hash || '/'}`
  const qIdx = h.indexOf('?')
  if (qIdx < 0) return { path: h, query: '' }
  return { path: h.slice(0, qIdx), query: h.slice(qIdx + 1) }
}

export const hashRouterStrategy: LocationStrategy = {
  getRaw(): string {
    if (typeof window === 'undefined') return ''
    return window.location.hash
  },
  parse(): Record<string, MultiEncoded> {
    if (typeof window === 'undefined') return {}
    const { query } = splitHash(window.location.hash)
    return parseMultiParams(query)
  },
  buildUrl(base: URL, params: Record<string, MultiEncoded>): string {
    const { path } = splitHash(base.hash)
    const qs = serializeMultiParams(params)
    base.hash = qs ? `${path}?${qs}` : path
    return base.toString()
  },
  subscribe(callback: () => void): () => void {
    if (typeof window === 'undefined') return () => {}
    window.addEventListener('hashchange', callback)
    window.addEventListener('popstate', callback)
    window.addEventListener(LOCATION_CHANGE_EVENT, callback)
    return () => {
      window.removeEventListener('hashchange', callback)
      window.removeEventListener('popstate', callback)
      window.removeEventListener(LOCATION_CHANGE_EVENT, callback)
    }
  },
}

setDefaultStrategy(hashRouterStrategy)
