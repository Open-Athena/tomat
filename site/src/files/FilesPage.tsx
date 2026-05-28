// R2 file browser, powered by `@rdub/file-tree`.
//
// The tomat-runs-api Cloudflare Worker exposes the `tomat/` sub-tree of the
// `openathena` R2 bucket through the file-tree HTTP protocol
// (`/api/files/list`, `/api/files/get?path=…`). We point `HttpStore` at that
// API and let `<FileTree>` render the listing + viewers.
//
// Routing wrinkle: the rest of the site uses a hand-rolled hash router
// (`useHash` / `parseHash`), but `<FileTree>` (via `Breadcrumb` + `DirListing`)
// uses `react-router-dom`'s `<Link>` and `useLocation`. We mount a nested
// `<HashRouter>` here so both routers operate over the same `location.hash`
// — the parent's `route === 'files'` branch keeps us on this page, and the
// inner router controls everything past the `/files` segment.

import { useMemo } from 'react'
import { HashRouter, Route, Routes } from 'react-router-dom'
import { FileTree } from '@rdub/file-tree/react'
import { HttpStore } from '@rdub/file-tree/stores/http'
import { useUrlPersistedState } from '@rdub/file-tree/url-state'
import { ThemeToggle } from '../theme'
import { API_BASE } from '../runs/api'
import { ParquetViewer } from './ParquetViewer'

const FILES_API_BASE = `${API_BASE}/api/files`

export function FilesPage() {
  const store = useMemo(() => HttpStore(FILES_API_BASE), [])
  return (
    <HashRouter>
      <header>
        <h1>
          <a href="#/" style={{ color: 'inherit', textDecoration: 'none' }}>tomat 🍅</a>
          <span style={{ fontSize: '0.7em', opacity: 0.7 }}> · files</span>
        </h1>
        <nav style={{ marginLeft: '1rem', display: 'flex', gap: '0.75rem', fontSize: '0.9rem' }}>
          <a href="#/runs">runs</a>
          <a href="#/deck">deck</a>
        </nav>
        <ThemeToggle />
      </header>
      <p className="meta">
        Browsing R2{' '}
        <code>openathena/tomat/</code>{' '}
        — runs, eval artifacts, codecs, tokenized splits, …
      </p>
      <Routes>
        <Route
          path="/files/*"
          element={
            <FileTree
              store={store}
              routeBase="/files"
              rootPrefix="tomat/"
              title="openathena/tomat"
              className="files-tree"
              parquetRenderer={ParquetViewer}
              usePersistedState={useUrlPersistedState}
            />
          }
        />
        <Route
          path="*"
          element={
            <FileTree
              store={store}
              routeBase="/files"
              rootPrefix="tomat/"
              title="openathena/tomat"
              className="files-tree"
              parquetRenderer={ParquetViewer}
              usePersistedState={useUrlPersistedState}
            />
          }
        />
      </Routes>
    </HashRouter>
  )
}
