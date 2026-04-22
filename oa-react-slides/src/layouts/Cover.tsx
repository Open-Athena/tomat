import type { ReactNode } from 'react'
import styles from './Cover.module.css'

const COLS = 7
const ROWS = 8
// Per-row left-edge thresholds: staircase narrows then widens.
const THRESHOLDS = [1, 1, 2, 3, 4, 4, 3, 2]

function cellBackground(row: number, col: number): string {
  if (col < THRESHOLDS[row]) return 'transparent'
  const isCopper = (row + col) % 2 === 0
  return isCopper ? 'var(--oa-copper)' : 'var(--oa-beige)'
}

interface CoverProps {
  children: ReactNode
  logo?: ReactNode
}

/**
 * Title slide. Right 59% of the slide gets a copper-on-beige staircase
 * checkerboard; left side holds an optional logo + the slide content.
 *
 *     <Cover logo={<img src="/logo.svg" />}>
 *       <h1>tomat 🍅</h1>
 *       <p>Tokenized materials — April 2026</p>
 *     </Cover>
 */
export function Cover({ children, logo }: CoverProps) {
  const cells = []
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      cells.push(
        <div
          key={`${row}-${col}`}
          style={{ background: cellBackground(row, col) }}
        />,
      )
    }
  }
  return (
    <div className={`oa-slide ${styles.cover}`}>
      <div className={styles.checker}>{cells}</div>
      <div className={styles.content}>
        {logo ? <div className={styles.logo}>{logo}</div> : null}
        <div className={styles.text}>{children}</div>
      </div>
    </div>
  )
}
