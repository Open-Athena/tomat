// Wire the use-kbd integration: a runs-search omnibar (opens on `/` or
// `Cmd+K`), a ShortcutsModal (`?`), and a lower-right SpeedDial widget with a
// GitHub link.
//
// The omnibar's results blend three sources:
//   - Run names (RUN_TAGS keys ∪ RUN_LINEAGE keys ∪ live snapshot's runs).
//     Selecting a run navigates to `#/runs/<name>`.
//   - Tag names (ALL_TAGS). Selecting a tag navigates to `#/runs?tags=<tag>`
//     (replaces any existing tag filter to make the omnibar's behavior
//     predictable — additive tagging is a job for the chip UI itself).
//   - A free-text "Filter /runs by …" entry that always appears. Picking it
//     navigates to `#/runs?regex=<query>` (reuses the existing name-filter
//     URL state — same input the plot's text box and the page reads).
import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  HotkeysProvider,
  Omnibar,
  ShortcutsModal,
  SpeedDial,
  useAction,
  useHotkeysContext,
  useOmnibarEndpoint,
} from 'use-kbd'
import 'use-kbd/styles.css'
import { fetchRunsSnapshot } from './runs/api'
import { ALL_TAGS, RUN_TAGS } from './runs/tags'
import { RUN_LINEAGE } from './runs/lineage'

// ── icons (inline SVG; matches use-kbd's own minimal-deps approach) ─────────
function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor"
      style={{ width: '1em', height: '1em' }} aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2c-3.2.69-3.87-1.36-3.87-1.36-.52-1.32-1.27-1.67-1.27-1.67-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.02 1.76 2.69 1.25 3.35.96.1-.74.4-1.25.72-1.54-2.55-.29-5.23-1.27-5.23-5.66 0-1.25.45-2.27 1.18-3.07-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.17.92-.26 1.91-.39 2.9-.39s1.98.13 2.9.39c2.21-1.49 3.17-1.17 3.17-1.17.63 1.59.23 2.76.11 3.05.74.8 1.18 1.82 1.18 3.07 0 4.4-2.68 5.36-5.24 5.65.41.36.78 1.06.78 2.13v3.16c0 .31.21.66.79.55C20.21 21.39 23.5 17.07 23.5 12 23.5 5.65 18.35.5 12 .5Z"/>
    </svg>
  )
}

// Replace any existing `tags=` param on `#/runs` with a single tag (the
// minimal "set the chip" UX). Other query params (regex, anc) are preserved.
function navigateRunsWithTag(tag: string) {
  const params = new URLSearchParams()
  params.set('tags', tag)
  window.location.hash = `#/runs?${params.toString()}`
}

function navigateRunsWithRegex(regex: string) {
  if (!regex.trim()) {
    window.location.hash = '#/runs'
    return
  }
  const params = new URLSearchParams()
  params.set('regex', regex)
  window.location.hash = `#/runs?${params.toString()}`
}

function navigateToRun(name: string) {
  window.location.hash = `#/runs/${name}`
}

// ── omnibar endpoints ───────────────────────────────────────────────────────
// Sync (filter-based) endpoints are instant — no debounce — and three of them
// cover the search shape we want. Priority orders the groups in the result
// list (higher first); we lead with run names, then tags, then the catch-all
// "filter the list" affordance.
function RunsOmnibarEndpoints() {
  // Live runs (whatever the dashboard sees) — re-uses TSQ cache shared with
  // RunsIndex, so this isn't a second poll. Falls back gracefully (empty) on
  // first paint or fetch failure.
  const snapshotQ = useQuery({
    queryKey: ['runs-snapshot'],
    queryFn: fetchRunsSnapshot,
    staleTime: 30_000,
  })

  // Union of all known run names: live snapshot (current truth) + curated
  // RUN_TAGS keys + RUN_LINEAGE keys (so historical/expired runs the snapshot
  // doesn't include any more remain searchable, since RUN_TAGS describes
  // every run that's appeared on the dashboard).
  const allRunNames = useMemo(() => {
    const s = new Set<string>()
    if (snapshotQ.data) for (const id of Object.keys(snapshotQ.data.runs)) s.add(id)
    for (const k of Object.keys(RUN_TAGS)) s.add(k)
    for (const k of Object.keys(RUN_LINEAGE)) s.add(k)
    return [...s].sort()
  }, [snapshotQ.data])

  useOmnibarEndpoint('runs', {
    group: 'Runs',
    priority: 100,
    minQueryLength: 0,
    filter: (query, pagination) => {
      const q = query.trim().toLowerCase()
      // Match if every whitespace-separated term is a substring of the name
      // (case-insensitive). Empty query → return everything (capped at page).
      const terms = q ? q.split(/\s+/).filter(Boolean) : []
      const matched = allRunNames.filter((name) => {
        const lower = name.toLowerCase()
        return terms.every((t) => lower.includes(t))
      })
      const start = pagination.offset
      const end = start + pagination.limit
      return {
        entries: matched.slice(start, end).map((name) => ({
          id: `run:${name}`,
          label: name,
          // Tag chips inlined into the result line so the user sees
          // objective/loss/status at a glance.
          description: (RUN_TAGS[name] ?? []).join(' · ') || undefined,
          handler: () => navigateToRun(name),
        })),
        total: matched.length,
        hasMore: end < matched.length,
      }
    },
  })

  useOmnibarEndpoint('tags', {
    group: 'Tags',
    priority: 50,
    minQueryLength: 0,
    filter: (query, pagination) => {
      const q = query.trim().toLowerCase()
      const matched = ALL_TAGS.filter((t) => !q || t.toLowerCase().includes(q))
      const start = pagination.offset
      const end = start + pagination.limit
      return {
        entries: matched.slice(start, end).map((tag) => {
          // How many curated runs carry this tag — a small hint of the
          // result size if the user picks it.
          let n = 0
          for (const tags of Object.values(RUN_TAGS)) if (tags.includes(tag)) n++
          return {
            id: `tag:${tag}`,
            label: `tag: ${tag}`,
            description: `${n} run${n === 1 ? '' : 's'} → /runs?tags=${tag}`,
            handler: () => navigateRunsWithTag(tag),
          }
        }),
        total: matched.length,
        hasMore: end < matched.length,
      }
    },
  })

  // A catch-all "Filter the runs list" entry that appears as long as the user
  // has typed something. This is the "Enter on free text → filter /runs"
  // behaviour from the spec — distinct from the per-name entries above (which
  // jump to a single run's detail page).
  useOmnibarEndpoint('runs-filter', {
    group: 'Filter',
    priority: 10,
    minQueryLength: 1,
    filter: (query) => {
      const q = query.trim()
      if (!q) return { entries: [] }
      return {
        entries: [{
          id: `filter:${q}`,
          label: `Filter /runs by "${q}"`,
          description: 'Apply this text as the runs-page name filter',
          handler: () => navigateRunsWithRegex(q),
        }],
        total: 1,
        hasMore: false,
      }
    },
  })

  return null
}

// ── extra hotkeys ───────────────────────────────────────────────────────────
// `Cmd+K` is wired by the Omnibar component itself; this adds `/` as a second
// trigger. Listed as an action (rather than a raw keydown listener) so it
// shows up in the ShortcutsModal alongside the rest.
function ExtraOmnibarHotkeys() {
  const ctx = useHotkeysContext()
  useAction('runs:omnibar-slash', {
    label: 'Open runs search',
    group: 'Search',
    defaultBindings: ['/'],
    handler: (e) => {
      // Default `/` would also start an in-page Quick Find in some browsers —
      // pre-empt that.
      e?.preventDefault?.()
      ctx.openOmnibar()
    },
  })
  // Navigation shortcuts (also appear in the ShortcutsModal).
  useAction('nav:runs', {
    label: 'Go to /runs',
    group: 'Navigation',
    defaultBindings: ['g r'],
    handler: () => { window.location.hash = '#/runs' },
  })
  useAction('nav:home', {
    label: 'Go to home',
    group: 'Navigation',
    defaultBindings: ['g h'],
    handler: () => { window.location.hash = '#/' },
  })
  return null
}

// ── top-level wrapper ───────────────────────────────────────────────────────
const GITHUB_URL = 'https://github.com/Open-Athena/tomat'

export function KbdShell({ children }: { children: React.ReactNode }) {
  return (
    <HotkeysProvider>
      {children}
      <RunsOmnibarEndpoints />
      <ExtraOmnibarHotkeys />
      <Omnibar placeholder="Search runs, tags, …" />
      <ShortcutsModal />
      <SpeedDial
        position={{ bottom: 16, right: 16 }}
        actions={[{
          key: 'github',
          label: 'GitHub',
          icon: <GitHubIcon />,
          href: GITHUB_URL,
        }]}
      />
    </HotkeysProvider>
  )
}
