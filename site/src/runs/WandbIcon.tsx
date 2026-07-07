// WandbIcon — crop-safe render of `/wandb.svg` (the full 1360×269
// horizontal lockup: icon + "Weights & Biases" wordmark + tagline).
//
// The SVG ships full-width because we don't have a mark-only asset, and
// naïvely rendering it at `width: 18` bleeds the leading "W" of the
// wordmark into the clip window (viewBox mark ends at x≈167, wordmark
// starts at x≈232 → at height=18 those land at rendered px 11.2 and
// 15.5 respectively, leaving ~2.5px of "W" leak).
//
// Clip the wrapper to a width tight enough to hide the wordmark but
// still show the yellow logo mark cleanly.
//
// Consumers pass through an optional `size` for the icon height; the
// wrapper width scales with the same viewBox ratio.
import type { CSSProperties, ReactElement } from 'react'

/** Rightmost x of the yellow logo mark in viewBox units (~167). Wordmark
 *  starts around x=232, so any wrapper width < 232*px-per-vbox is safe. */
const MARK_VBOX_WIDTH = 195
/** Full viewBox height. */
const VBOX_HEIGHT = 269

interface Props {
  size?: number
  style?: CSSProperties
  /** Native-tooltip fallback for cases where the caller isn't already
   *  wrapping this in a @floating-ui `<Tooltip>`. */
  title?: string
}

export function WandbIcon({ size = 18, style, title }: Props): ReactElement {
  const wrapperWidth = Math.round(size * (MARK_VBOX_WIDTH / VBOX_HEIGHT))
  return (
    <span title={title} style={{
      display: 'inline-block',
      width: wrapperWidth,
      height: size,
      overflow: 'hidden',
      verticalAlign: 'middle',
      ...style,
    }}>
      <img src="/wandb.svg" alt="wandb"
        style={{ height: size, width: 'auto', display: 'block' }} />
    </span>
  )
}
