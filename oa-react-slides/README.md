# oa-react-slides

React port of the [Open Athena Slidev theme](
https://github.com/Open-Athena/oa-slidev-theme) — warm beige + copper
tones, Castoro/Inter typography, 14 layouts (WIP), 149 icons.

**Status**: v0.1, bootstrapped from [spec 06](
https://github.com/Open-Athena/tomat/blob/main/specs/06-oa-react-slides.md).
Used in-repo at the moment; will factor out to its own repo once the API
stabilizes.

## Install

Local path (stage 1):

```jsonc
// site/package.json
"dependencies": {
  "oa-react-slides": "file:../oa-react-slides"
}
```

## Use

```tsx
import 'oa-react-slides/styles'
import { Cover, Section, Default, Icon } from 'oa-react-slides'

export const slides = [
  {
    render: () => (
      <Cover>
        <h1>tomat 🍅</h1>
        <p>Tokenized materials — April 2026</p>
      </Cover>
    ),
  },
  {
    render: () => (
      <Section>
        <h1>Pivot: patches</h1>
        <p>Why we moved off full-grid tokenization</p>
      </Section>
    ),
  },
  {
    render: () => (
      <Default>
        <h2><Icon name="open-athena" /> Pipeline</h2>
        <p>structure → tokens → Qwen3 → tokens → ρ(r)</p>
      </Default>
    ),
  },
]
```

The consumer provides the deck shell (scroll + keyboard + thumbnail drawer);
this package only ships layouts + theme CSS. Tomat's `site/src/DeckPage.tsx`
is a reference implementation of the shell.

## What's shipped today

- **Layouts**: `Default`, `Cover`, `Section`. More (`Agenda`, `Quote`,
  `Image`, `End`, `TwoCols`, `ThreeCols`, `FourCols`, `SplitLeft`,
  `SplitRight`, `SplitLeftGreen`, `SplitRightGreen`) land as follow-up
  commits.
- **Design tokens**: `--oa-beige*`, `--oa-dark*`, `--oa-copper*`,
  `--oa-green*`, `--oa-text-*`, `--oa-font-{serif,sans}`, copied verbatim
  from `oa-slidev-theme/styles/base.css`.
- **Fonts**: Castoro (serif) + Inter (body), bundled via `@fontsource`.
- **Icons**: `<Icon name="…" />` — 149 SVGs. See `ICON_NAMES` for the list
  (or the matching `icons.md` in the upstream Slidev theme).

## What's not here (yet)

- MDX-driven authoring (frontmatter `layout: cover` → component).
- Deck shell (scroll, keyboard, overview grid, presenter mode). Consumers
  BYO; tomat has one.
- Print-mode CSS for PDF export. Tomat's `site/src/main.css` has one that
  works against a generic shell; needs updating once OA layouts are in use.
- Syntax-highlighting (Shiki) theme.
