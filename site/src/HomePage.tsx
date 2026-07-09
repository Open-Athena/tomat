import { useEffect, useState } from 'react'

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
}

interface PostMeta {
  slug: string
  title: string
}

// Hand-picked active runs to surface on the homepage. Update as new runs
// become the most-interesting ones to look at. Pure curation — the live
// /runs board has the full list.
const FEATURED_RUNS: { id: string; blurb: string }[] = [
  {
    id: 'train-mg-modal-h200x8-tz-v4-epochwin-bs128-seed42',
    blurb: 'v4-epochwin · 200M MaskGIT, currently 92k/100k steps on H200×8',
  },
  {
    id: 'train-mg-modal-h200x8-tz-fs-kl-s05-ts1-bs128-seed42',
    blurb: 'kl-s05 · the LF-ablation KL-Gauss(σ=0.5) winner (vs CE / CE+EMD / CRPS), step 20k',
  },
]

const POSTS_INDEX_URL = `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/posts-index.json`

export function HomePage() {
  const [posts, setPosts] = useState<PostMeta[] | null>(null)

  useEffect(() => {
    fetch(POSTS_INDEX_URL)
      .then((r) => r.ok ? r.json() as Promise<PostMeta[]> : Promise.reject(r.status))
      // Newest first (numeric prefix → reverse lex sort works).
      .then((list) => setPosts([...list].sort((a, b) => b.slug.localeCompare(a.slug))))
      .catch(() => setPosts([]))
  }, [])

  return (
    <>
      <header>
        <h1>MarinMat 🍅 — tokenized materials</h1>
        <p className="meta" style={{ fontSize: '0.85rem', marginTop: '-0.3rem' }}>
          f.k.a. <code>tomat</code> (still the CLI/package/URL name).
          Sibling to{' '}
          <ExtLink href="https://github.com/Open-Athena/MarinFold">MarinFold</ExtLink>
          {' + '}
          <ExtLink href="https://github.com/Open-Athena/marin-dna">MarinDNA</ExtLink>.
        </p>
      </header>

      <p className="meta">
        LLM-based electron-density prediction for periodic crystals. We
        tokenize ρ(r) directly and train a transformer over the resulting
        token stream — a sequence-model alternative to 3D-ResUNet approaches.{' '}
        <ExtLink href="https://github.com/Open-Athena/MarinMat">Open-Athena/MarinMat</ExtLink>.
      </p>

      <h2><a href="#/runs">→ Live runs dashboard</a></h2>
      <p className="meta">
        Every run's loss · step · MFU · per-checkpoint mat-NMAE/NEMD curves,
        lifecycle events, preemptions, MSRP — live from wandb + iris +
        Modal.
      </p>
      <ul style={{ marginTop: '0.5rem' }}>
        {FEATURED_RUNS.map((r) => (
          <li key={r.id}>
            <a href={`#/runs/${r.id}`} style={{ fontFamily: 'monospace' }}>{r.id}</a>
            {' — '}<span style={{ color: 'var(--fg-muted, #888)' }}>{r.blurb}</span>
          </li>
        ))}
      </ul>

      <h2><a href="#/posts">→ Posts</a></h2>
      <p className="meta">
        Long-form notes on tokenization, codecs, runs, and bugs.
      </p>
      {posts == null ? (
        <p className="meta">loading…</p>
      ) : posts.length === 0 ? (
        <p className="meta">no posts yet.</p>
      ) : (
        <ul style={{ marginTop: '0.5rem' }}>
          {posts.map((p) => (
            <li key={p.slug}>
              <a href={`#/posts/${p.slug}`}>{p.title || p.slug}</a>
            </li>
          ))}
        </ul>
      )}

      <h2>Status</h2>
      <p className="meta">
        Honest free-running mat-NMAE: <strong>not competitive yet</strong>.
        Current MaskGIT-from-scratch + LF-ablation work targets the
        AR-train / FR-eval gap directly (<a href="#/posts/13-why-pure-emd-collapses">post 13</a>).
        Reference: ChargE3Net (
        <ExtLink href="https://arxiv.org/abs/2312.05388">arxiv:2312.05388</ExtLink>
        ) is the published SOTA at 0.523 % FR mat-NMAE on Materials Project.
      </p>

      <div className="footer">
        <ExtLink href="https://github.com/Open-Athena/MarinMat/blob/main/OVERVIEW.md">
          <code>OVERVIEW.md</code>
        </ExtLink>{' · '}
        <ExtLink href="https://github.com/Open-Athena/MarinMat/tree/main/specs">
          <code>specs/</code>
        </ExtLink>{' · '}
        <ExtLink href="https://wandb.ai/open-athena/tomat-lmq-P19">W&amp;B project</ExtLink>
      </div>
    </>
  )
}
