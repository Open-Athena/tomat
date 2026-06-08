# Spec 57: Interactive components in posts (markdown-with-React)

We want posts to embed live tomat data — run plots, epoch breakdowns,
linked ckpt cards — without leaving markdown. Stay markdown-first
(authoring stays grep-friendly, lints in any editor); add a small
**registry of custom HTML tags** that `react-markdown` maps to React
components.

## Why not MDX

MDX gives us full JSX in posts, but at the cost of:
- Build config (rollup MDX loader, .mdx extension)
- Posts stop being pure markdown — can't edit in any markdown editor,
  can't `cat` and read coherently
- Component imports inline with prose (a wall of imports at the top)

Markdown + raw HTML + a tag-to-component mapping gets us 90% of the
expressive power with none of those costs. Posts stay `.md`. Authors
write `<elvis mat="mp-1788391" iso="0.6"/>` like any other inline HTML;
the renderer swaps the tag for the live component.

## Architecture

`react-markdown` already supports raw HTML when paired with
`rehype-raw`. Then its `components` prop lets us override how specific
tags are rendered:

```tsx
<ReactMarkdown
  remarkPlugins={[remarkGfm, remarkMath]}
  rehypePlugins={[rehypeRaw, rehypeKatex, ...]}
  components={{
    runplot: RunPlot,
    epochplot: EpochPlot,
    elvis: ElvisEmbed,
    runlink: RunLink,
    matlink: MatLink,
  }}
>
```

Tag names are lowercased in HTML; React component props come through
as lowercase strings. We marshal string→typed in each component (e.g.
`iso="0.6"` → `parseFloat`).

## Initial component set (minimum to validate the pattern)

### `<elvis>` — embed ELvis on a single material

Renders an iframe pointing at the elvis app's `?embed=1` mode (spec 58)
with a `?m=<mp_id>` + viz params.

```html
<elvis mat="mp-1788391" iso="0.6" height="500"/>
```

Props (all string in HTML, parsed in component):
- `mat` — `mp-…` material id (required)
- `iso` — isosurface threshold (default: elvis's own default)
- `height` — px or CSS height (default `400px`)
- `c` — camera position
- `rot` — rotation mode
- `hg` — hover-grid step (any param elvis accepts; we forward unknown
  props as URL params verbatim)

Auto-imports the iframe with `loading="lazy"` so a post with 5 embeds
doesn't slow first paint.

### `<runlink>` — canonical wandb run link

```html
<runlink>cont33k</runlink>
```

→ Looks up `cont33k` in `posts/runs-aliases.json` (small map of
short-aliases to full wandb run names), renders an `<a>` to the wandb
URL with the run name as link text. Falls back to literal text if the
alias isn't found, with a warning in dev.

Optional `wandb` prop for one-off cases without an alias:

```html
<runlink wandb="train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42">v4-epochwin</runlink>
```

### `<runplot>` — embed the dashboard's WallclockPlot trace

```html
<runplot runs="cont33k,v4-epochwin" metric="train/loss" xaxis="step" smooth="rolling:50"/>
```

Reuses `WallclockPlot` (or a stripped-down sibling) with a minimal
set of props. Internally fetches the same `raw.parquet` the dashboard
uses, so the post auto-updates when a new sync lands.

### `<matlink>` — link to elvis for a single material

```html
<matlink mat="mp-1788391"/>
```

→ `<a href="https://elvis.oa.dev/?m=mp-1788391">mp-1788391</a>` plus
optional MPDB metadata in the tooltip (formula, n_atoms) if we have
it. Light alternative to `<elvis>` when you don't want a full embed.

## Out of scope (deliberate)

- Counter-style state inside posts (`useState` in markdown). If a post
  needs that, it should become a proper React page under
  `site/src/pages/`, not a post.
- Inline math/expressions that compute against post content
  (`{2 + 2}`). Stay rehype-pluggable.
- Live editor in the dashboard. Posts are authored in `posts/*.md`,
  reviewed via PR.

## Implementation steps

1. Add `rehype-raw` to deps; thread into `PostsPage.tsx`'s
   `rehypePlugins` list (before katex/highlight so raw HTML is parsed
   first).
2. Wire `components` prop on `ReactMarkdown` with the tag map.
3. Implement components in `site/src/posts/embeds/`:
   - `Elvis.tsx`, `RunLink.tsx`, `RunPlot.tsx`, `MatLink.tsx`
4. Create `posts/runs-aliases.json` (`{ "cont33k": "train-mg-modal-...", ... }`)
   + a `useRunAliases()` hook that fetches it once.
5. Pilot use: add an `<elvis>` block to post 05 (which references
   `mp-1788391` heavily) and a `<runplot>` showing cont33k's
   TF-vs-FR-eval-gap from step-33k to step-79999.
6. Unit tests for the tag-to-component renderer (Jest/vitest).

## Files to touch

- `site/package.json` — add `rehype-raw`
- `site/src/posts/PostsPage.tsx` — thread plugins + components
- `site/src/posts/embeds/{Elvis,RunLink,RunPlot,MatLink}.tsx` — new
- `site/src/posts/embeds/index.ts` — barrel of the tag map
- `posts/runs-aliases.json` — new
- `posts/05-…md` — pilot embeds (defer if quicker to land plumbing alone)
- `site/src/posts/posts.css` — iframe + link styles

## Commits (suggest 3)

1. `posts: rehype-raw + components map for custom HTML tags`
2. `posts: <elvis> + <runlink> + <matlink> components`
3. `posts: <runplot> component (reuses WallclockPlot)`

(`<runplot>` is the trickiest because of the parquet data dependency.
Land 1+2 first; 3 can be its own follow-up if the parquet/transform
threading isn't trivial.)
