import type { ReactNode } from 'react'

/**
 * Generic OA-themed slide — just applies the base palette + typography via
 * `.oa-slide`. Use for content that doesn't fit a named layout (cover,
 * section, split-*, etc.).
 */
export function Default({ children }: { children: ReactNode }) {
  return <div className="oa-slide">{children}</div>
}
