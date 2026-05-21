// Lightweight React tooltip (floating-ui) — replaces native `title=`, which
// is slow to appear and unstyleable / tiny.
//
// `children` must be a single element that accepts a ref + DOM props (a
// `<span>`, `<button>`, `<a>`, `<div>` …). It's cloned with the reference ref
// + hover/focus handlers — no extra wrapper DOM, so layout is unchanged.

import { cloneElement, useState } from 'react'
import type { ReactElement, ReactNode } from 'react'
import {
  autoUpdate, flip, offset, shift,
  FloatingPortal, useDismiss, useFloating, useFocus, useHover,
  useInteractions, useRole,
} from '@floating-ui/react'
import type { Placement } from '@floating-ui/react'

export function Tooltip({ content, placement = 'top', children }: {
  content: ReactNode
  placement?: Placement
  children: ReactElement
}) {
  const [open, setOpen] = useState(false)
  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: setOpen,
    placement,
    middleware: [offset(6), flip(), shift({ padding: 8 })],
    whileElementsMounted: autoUpdate,
  })
  // open: 80ms — far snappier than the browser's ~1s native-title delay.
  const hover = useHover(context, { move: false, delay: { open: 80, close: 0 } })
  const focus = useFocus(context)
  const dismiss = useDismiss(context)
  const role = useRole(context, { role: 'tooltip' })
  const { getReferenceProps, getFloatingProps } = useInteractions([hover, focus, dismiss, role])

  const empty = content == null || content === ''

  return (
    <>
      {cloneElement(
        children,
        { ref: refs.setReference, ...getReferenceProps() } as Record<string, unknown>,
      )}
      {open && !empty && (
        <FloatingPortal>
          <div
            ref={refs.setFloating}
            style={{
              ...floatingStyles,
              zIndex: 1000,
              maxWidth: 320,
              background: '#0f0f0f',
              color: '#ddd',
              border: '1px solid #383838',
              borderRadius: 5,
              padding: '5px 8px',
              fontSize: '0.74rem',
              lineHeight: 1.35,
              boxShadow: '0 4px 14px rgba(0,0,0,0.5)',
              pointerEvents: 'none',
            }}
            {...getFloatingProps()}
          >
            {content}
          </div>
        </FloatingPortal>
      )}
    </>
  )
}
