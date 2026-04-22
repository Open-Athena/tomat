# `oa-react-slides` — React port of the OA Slidev theme

Port [`/Users/ryan/c/oa/oa-slidev-theme`](../../oa-slidev-theme) (Vue/Slidev) to
a reusable React package so OA projects with React stacks (starting with
tomat's existing `site/src/slides.tsx` deck) can adopt the same visual identity
without switching frameworks.

## Motivation

- **Slidev adoption** is the obvious path for consistent OA presentations, but
  tomat's homepage is already React (interactive Plotly plots, TSQ, etc.).
  Running both a React SPA and a Slidev SPA under `tomat.oa.dev` duplicates
  build tooling and splits the deploy story.
- **There's no standard React equivalent of Slidev.** [Spectacle](
  https://github.com/FormidableLabs/spectacle) is the closest, but it's a
  full framework with its own opinions — adopting it means rewriting our
  deck *and* learning a new API. [mdx-deck](https://github.com/jxnblk/mdx-deck)
  is effectively unmaintained.
- Our existing `DeckPage.tsx` + `slides.tsx` is already 80% of a React slide
  framework. What's missing is a proper **theme** — the 14 OA Slidev layouts
  (cover, agenda, section, quote, split-*, *-cols, image, end) and the
  warm-beige / serif / copper design tokens.
- **OA expects tomat to be presented externally repeatedly**, so the theme
  investment amortizes. Once the React version exists, any other React-heavy
  OA project reuses it for free.

## Scope

In scope for v0.1:

- **Design tokens**: `--oa-beige*`, `--oa-dark*`, `--oa-copper*`, `--oa-green*`,
  `--oa-text-*`, `--oa-font-{serif,sans}` — copied verbatim from the Slidev
  theme's `styles/base.css`.
- **Fonts**: Castoro (serif titles) + Inter (body), bundled via `@fontsource`.
- **Layouts** as React components, one-per-Vue-layout:
  `Cover`, `Agenda`, `Section`, `Quote`, `Image`, `End`, `Default`,
  `TwoCols`, `ThreeCols`, `FourCols`,
  `SplitLeft`, `SplitRight`, `SplitLeftGreen`, `SplitRightGreen`.
- **Icons**: 149 SVGs under `public/icons/` exposed as an `<Icon name="…" />`
  component (or as direct imports for tree-shaking).
- **Reusable sub-components** from the theme: `Box`, `CornerDeco`, `Footnote`,
  `Num`.
- **Authoring model**: TSX per slide, mirroring tomat's current
  `slides: Slide[]` pattern. Consumers compose: `<Cover><h1>…</h1></Cover>`,
  `<SplitLeft><h2>…</h2>{/* left */}<div>{/* right */}</div></SplitLeft>`.

Out of scope for v0.1 (can land later):

- MDX-driven authoring (frontmatter `layout: cover` → component). Current TSX
  works; MDX is a follow-up if author ergonomics warrant it.
- Deck shell (scroll, keyboard nav, thumbnail drawer). Tomat already has one
  in `site/src/DeckPage.tsx`; consumers write their own or copy tomat's.
- Presenter mode, transitions, speaker notes, draw overlays. Add as needed.
- Shiki/syntax-highlighting config (the Slidev theme ships a `setup/shiki.ts`
  with a sepia color scheme).
- Print-mode / PDF export CSS. Tomat's `site/src/main.css` already has a
  `@media print` block that works against the generic deck layout; once layouts
  land, the same rules should apply.

## Architecture

### Stage 1 (now, in this repo)

```
tomat/
  oa-react-slides/          # standalone package — new in this PR
    package.json            # name: "oa-react-slides", version "0.1.0", type module
    tsconfig.json
    README.md
    src/
      index.ts              # barrel export
      layouts/
        Cover.tsx
        Agenda.tsx
        Section.tsx
        …
      components/
        Box.tsx
        CornerDeco.tsx
        Footnote.tsx
        Num.tsx
        Icon.tsx            # <Icon name="open-athena" />
      styles/
        base.css            # copied from oa-slidev-theme/styles/base.css
        index.ts            # imports fonts + base.css
      icons/                # bundled SVG assets, imported as React components
        open-athena.svg
        …
```

Tomat's `site/package.json` depends on it via local path:

```json
"dependencies": {
  "oa-react-slides": "file:../oa-react-slides"
}
```

### Stage 2 — separate repo

Move `oa-react-slides/` out to `~/c/oa/oa-react-slides/` as its own Git repo
under `Open-Athena/oa-react-slides`. Tomat's `site` switches its dep to a
GitHub URL (SHA-pinned): `github:Open-Athena/oa-react-slides#<sha>`.

### Stage 3 — GH dist branch via `npm-dist`

Add the [`npm-dist`](https://github.com/runsascoded/npm-dist) GH Action so
every push builds a `dist/` branch that downstream projects can consume
directly — no `pnpm build` at install time. Tomat's `site` consumes via
`pds gh oa-react-slides` (SHA-pinned dist branch).

### Stage 4 — npm publish

Publish `oa-react-slides` to npmjs.com under the `@openathena/` org (or
unscoped). Downstream consumers: `pnpm add oa-react-slides`.

## API sketch

```tsx
import 'oa-react-slides/styles'
import { Cover, Section, SplitLeft, Icon } from 'oa-react-slides'

export const slides = [
  {
    id: 'title',
    render: () => (
      <Cover logo={<img src="/logo.svg" />}>
        <h1>tomat 🍅</h1>
        <p>Tokenized materials — April 2026</p>
      </Cover>
    ),
  },
  {
    id: 'pivot',
    render: () => (
      <Section>
        <h1>Pivot: patches</h1>
        <p>Why we moved off full-grid tokenization</p>
      </Section>
    ),
  },
  {
    id: 'token-layout',
    render: () => (
      <SplitLeft>
        <h2><Icon name="code-braces" /> Patch token layout</h2>
        <pre>…</pre>
        {/* right slot (default children) */}
        <img src="/diagram.svg" />
      </SplitLeft>
    ),
  },
]
```

The consumer's Deck shell (tomat's `DeckPage.tsx`) stays unchanged — it just
renders each slide's output. Layouts are pure styled wrappers; no navigation
logic lives in the package.

## Migration plan for tomat

1. Land `oa-react-slides/` in this repo (stage 1). Tomat's site adds it as a
   file-path dep.
2. Rewrite `site/src/slides.tsx` to use the new layout components where
   obvious: title → `Cover`, status slide → `Section`, token layout /
   example → `SplitLeft` / `TwoCols`, links → `End`. The 3 Plotly slides
   stay as ad-hoc JSX inside `Default` for now.
3. Drop the tomat-specific `.slide-plot`, `.slide-title`, `.token-layout`,
   `.token-example` CSS from `main.css` — those get subsumed by OA layouts
   + a small `oa-react-slides/styles/extras.css` for code blocks.
4. Regenerate the PDF; confirm it matches the Slidev example gallery
   visually (allowing for the plots).

Not a hard requirement that *every* slide use a named OA layout — some slides
are unavoidably custom (the Pareto plot wants the whole frame). A `Default`
layout plus the design tokens gives those slides the right colors/fonts
without imposing a structure.

## Publishing roadmap checkboxes

- [ ] v0.1.0 (this PR): stage 1 — standalone dir in tomat, file-path dep.
- [ ] v0.2.0: stage 2 — move out to `Open-Athena/oa-react-slides` repo. Tomat
  flips to `github:Open-Athena/...` dep, SHA-pinned.
- [ ] v0.3.0: stage 3 — `npm-dist` GH Action on the new repo; tomat consumes
  via `pds gh`.
- [ ] v1.0.0: stage 4 — NPM publish under `@openathena/` or unscoped.

Stages 2–4 can lag tomat's usage; we only move forward when there's a second
consumer (another React-based OA project) or when the v0.1 API has stabilized
enough to merit a public release.

## Done criteria (v0.1)

- [ ] `oa-react-slides/` package builds cleanly (`tsc` + the CSS ships
  alongside as-is via the package's `exports` field).
- [ ] Tomat's `site` installs and imports from `oa-react-slides`.
- [ ] At least **3 layouts ported and in use** by tomat's deck (`Cover`,
  `Section`, `Default` — the rest can land in follow-up commits without
  blocking).
- [ ] `<Icon name="…" />` component resolves to a bundled SVG; at least the
  `open-athena` icon works.
- [ ] Deck's existing `DeckPage.tsx` shell unchanged (validates the theme
  doesn't assume its own navigation).
- [ ] Move this spec to `specs/done/` when tomat's homepage + deck both
  successfully consume from `oa-react-slides`.

## Open questions

- **Font delivery**: `@fontsource/*` packages the fonts into JS/CSS bundles.
  Works but adds ~300 kB to the bundle for two families. Alternatives: CDN
  link to Google Fonts (simpler, needs network at runtime), or host the woff2
  files directly in the package's `assets/`. Defer.
- **Dark mode**: the OA theme is light-mode only (`colorSchema: "light"`).
  Tomat's homepage supports dark mode already. Options: (a) the theme exports
  only light-mode tokens, tomat's homepage keeps its own theming; (b) we add
  a dark variant to the theme later. MVP is (a).
- **Scoped styles**: Vue's `<style scoped>` isolates layout styles. React
  doesn't have an equivalent out of the box. Options: CSS modules (`.module.css`),
  a `data-oa-layout="cover"` attribute + unscoped CSS, or
  CSS-in-JS. Going with **CSS modules** for MVP — build-time resolution, no
  runtime cost, plays nicely with Vite.
