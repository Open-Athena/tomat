import { useEffect, useState } from 'react'

/** Subscribe to `location.hash`. Returns the raw value including the leading `#`. */
export function useHash() {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const handler = () => setHash(window.location.hash)
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])
  return hash
}

/** `#/deck` → `['deck']`; `#/deck/3` → `['deck', '3']`.
 *
 *  Strips the `?query` segment first so query state (e.g. `#/runs?tags=foo`)
 *  doesn't get glued onto the last path token — without this, the route token
 *  for `#/runs?tags=foo` is `'runs?tags=foo'`, which fails to match `'runs'`
 *  and falls through to the home page. */
export function parseHash(hash: string): string[] {
  const qIdx = hash.indexOf('?')
  const path = qIdx < 0 ? hash : hash.slice(0, qIdx)
  return path.replace(/^#\/?/, '').split('/').filter(Boolean)
}
