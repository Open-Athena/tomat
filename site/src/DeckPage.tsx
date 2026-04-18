import { useEffect, useMemo, useRef, useState } from 'react'
import { slides } from './slides'
import { useSweep } from './useSweep'
import { parseHash } from './useHash'
import { ThemeToggle } from './theme'

/** Parse `#/deck` → 0, `#/deck/3` → 3. */
function indexFromHash(hash: string, n: number): number {
  const parts = parseHash(hash)
  if (parts[0] !== 'deck') return 0
  const i = Number(parts[1])
  if (!Number.isFinite(i)) return 0
  return Math.max(0, Math.min(n - 1, i))
}

export function DeckPage() {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
  const { rows } = useSweep(`${base}/sweep-n50.csv`)
  const [active, setActive] = useState(() => indexFromHash(window.location.hash, slides.length))
  const slideRefs = useRef<(HTMLElement | null)[]>([])
  const mainRef = useRef<HTMLDivElement | null>(null)

  // Scroll the main container directly. Instant — native smooth-scroll fights
  // with scroll-snap in some engines, and a custom rAF loop dies when the tab
  // is backgrounded. Keystroke/click → snap to target. We also update `active`
  // + the URL hash directly, since programmatic scrollTop doesn't reliably fire
  // `scroll` events everywhere.
  const scrollToSlide = (i: number) => {
    const main = mainRef.current
    const el = slideRefs.current[i]
    if (!main || !el) return
    main.scrollTop = el.offsetTop
    setActive(i)
    const target = i === 0 ? '#/deck' : `#/deck/${i}`
    if (window.location.hash !== target) {
      history.replaceState(null, '', target)
    }
  }

  // Style #root to fill the viewport while on the deck route.
  useEffect(() => {
    document.body.classList.add('in-deck')
    return () => document.body.classList.remove('in-deck')
  }, [])

  // Track which slide is in view via a plain scroll handler — the slide whose
  // top is closest to (but not past) the viewport top, with a small threshold.
  useEffect(() => {
    const main = mainRef.current
    if (!main) return
    const update = () => {
      const y = main.scrollTop + main.clientHeight * 0.3
      let idx = 0
      for (let i = 0; i < slideRefs.current.length; i++) {
        const el = slideRefs.current[i]
        if (!el) continue
        if (el.offsetTop <= y) idx = i
      }
      setActive(prev => {
        if (prev === idx) return prev
        const target = idx === 0 ? '#/deck' : `#/deck/${idx}`
        if (window.location.hash !== target) {
          history.replaceState(null, '', target)
        }
        return idx
      })
    }
    main.addEventListener('scroll', update, { passive: true })
    update()
    return () => main.removeEventListener('scroll', update)
  }, [])

  // On mount, scroll to the slide referenced by the URL hash.
  useEffect(() => {
    scrollToSlide(indexFromHash(window.location.hash, slides.length))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Keyboard navigation: ↑/↓/←/→/Home/End/PageUp/PageDown.
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      let next: number | null = null
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {
        next = Math.min(slides.length - 1, active + 1)
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
        next = Math.max(0, active - 1)
      } else if (e.key === 'Home') {
        next = 0
      } else if (e.key === 'End') {
        next = slides.length - 1
      }
      if (next !== null && next !== active) {
        e.preventDefault()
        scrollToSlide(next)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [active])

  const ctx = useMemo(() => ({ rows }), [rows])

  return (
    <div className="deck">
      <aside className="deck-drawer" aria-label="Slide thumbnails">
        <div className="deck-drawer-head">
          <a className="deck-home" href="#/">← home</a>
          <ThemeToggle className="deck-theme-toggle" />
        </div>
        <ol className="deck-thumbs">
          {slides.map((s, i) => (
            <li key={s.id}>
              <button
                type="button"
                className={`deck-thumb ${i === active ? 'active' : ''}`}
                onClick={() => scrollToSlide(i)}
                aria-current={i === active ? 'true' : undefined}
              >
                <span className="deck-thumb-idx">{i + 1}</span>
                <span className="deck-thumb-label">{s.thumb}</span>
              </button>
            </li>
          ))}
        </ol>
      </aside>
      <div className="deck-main" ref={mainRef}>
        {slides.map((s, i) => (
          <section
            key={s.id}
            className="deck-slide"
            data-idx={i}
            ref={el => { slideRefs.current[i] = el }}
            aria-label={s.title}
          >
            <div className="deck-slide-inner">{s.render(ctx)}</div>
            <div className="deck-slide-footer">
              <span>{i + 1} / {slides.length}</span>
              <span>{s.title}</span>
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
