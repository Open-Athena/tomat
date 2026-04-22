/**
 * OA icon. 149 SVGs bundled under ../icons/.
 *
 * Usage:
 *   <Icon name="open-athena" />
 *   <Icon name="arrow-right" size={32} />
 *   <Icon name="gears" style={{ color: 'var(--oa-copper)' }} />
 *
 * SVGs use `fill="currentColor"` so `color` in the surrounding style cascades
 * through. Vite transforms the `?raw` import into the SVG source string at
 * build time; we parse attributes off and render inline so the icon inherits
 * font-size and color without an extra fetch.
 */
import type { CSSProperties } from 'react'

interface IconProps {
  name: string
  size?: number | string
  className?: string
  style?: CSSProperties
  'aria-label'?: string
}

// Vite: eager-import every icon as raw SVG source. Tree-shaking trims unused
// names if the consumer bundles with a sensible configuration.
const sources = import.meta.glob('../icons/*.svg', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>

const byName: Record<string, string> = Object.fromEntries(
  Object.entries(sources).map(([path, src]) => {
    const name = path.replace(/^.*\/([^/]+)\.svg$/, '$1')
    return [name, src]
  }),
)

export function Icon({ name, size, className, style, ...rest }: IconProps) {
  const src = byName[name]
  if (!src) {
    console.warn(`[oa-react-slides] Icon "${name}" not found`)
    return null
  }
  // Inject width/height/class onto the <svg> root without parsing the whole
  // string — the source already has `width="24" height="24"` set, so we
  // substitute only when the caller overrides. For size we just swap the first
  // occurrence of each attribute; for className/style, we rely on a wrapping
  // <span> so the consumer can target it without touching the SVG internals.
  let svg = src
  if (size != null) {
    const s = typeof size === 'number' ? String(size) : size
    svg = svg
      .replace(/width="[^"]*"/, `width="${s}"`)
      .replace(/height="[^"]*"/, `height="${s}"`)
  }
  return (
    <span
      className={className}
      style={{ display: 'inline-flex', verticalAlign: 'middle', ...style }}
      aria-label={rest['aria-label'] ?? name}
      // biome-ignore lint/security/noDangerouslySetInnerHtml: bundled icons are trusted
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

/** List every available icon name (useful for icons.md / theme docs). */
export const ICON_NAMES: string[] = Object.keys(byName).sort()
