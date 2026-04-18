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

/** `#/deck` → `['deck']`; `#/deck/3` → `['deck', '3']`. */
export function parseHash(hash: string): string[] {
  return hash.replace(/^#\/?/, '').split('/').filter(Boolean)
}
