import type { ReactNode } from 'react'
import styles from './Section.module.css'

/**
 * Dark section-divider slide. Gold serif title on OA-dark background with a
 * copper-to-transparent gradient strip at the bottom.
 *
 *     <Section>
 *       <h1>Pivot: patches</h1>
 *       <p>Why we moved off full-grid tokenization</p>
 *     </Section>
 */
export function Section({ children }: { children: ReactNode }) {
  return (
    <div className={`oa-slide ${styles.section}`}>
      <div className={styles.content}>{children}</div>
      <div className={styles.decoration} />
    </div>
  )
}
