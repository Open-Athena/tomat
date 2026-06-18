// Markdown posts pages. Posts live in `<repo>/posts/*.md` and are exposed at
// /posts/<slug>.md + /posts-index.json by the `tomat-posts` Vite plugin (see
// `vite.config.ts`).
//
// Routes:
//   #/posts                   → index (list of posts, most-recent first)
//   #/posts/<slug>            → detail (renders the markdown for that slug)

import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import rehypeSlug from 'rehype-slug'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark.css'
import './posts.css'

interface PostMeta {
  slug: string
  title: string
}

interface Props {
  parts: string[]
}

const INDEX_URL = `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/posts-index.json`
const postUrl = (slug: string) =>
  `${(import.meta.env.BASE_URL || '/').replace(/\/$/, '')}/posts/${slug}.md`

function PostsHeader({ subtitle }: { subtitle?: string }) {
  return (
    <header>
      <h1>
        <a href="#/" style={{ color: 'inherit', textDecoration: 'none' }}>tomat 🍅</a>
        <span style={{ fontSize: '0.7em', opacity: 0.7 }}> · posts{subtitle ? ` · ${subtitle}` : ''}</span>
      </h1>
      <nav style={{ marginLeft: '1rem', display: 'flex', gap: '0.75rem', fontSize: '0.9rem' }}>
        <a href="#/runs">runs</a>
        <a href="#/files">files</a>
        <a href="#/deck">deck</a>
      </nav>
    </header>
  )
}

export function PostsPage({ parts }: Props) {
  const slug = parts[0]
  return slug ? <PostDetail slug={slug} /> : <PostsIndex />
}

function PostsIndex() {
  const [posts, setPosts] = useState<PostMeta[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(INDEX_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`fetch ${INDEX_URL}: ${r.status}`)
        return r.json() as Promise<PostMeta[]>
      })
      // Newest first: posts are numerically-prefixed, so descending sort by
      // filename gives most-recent-first.
      .then((list) => setPosts([...list].sort((a, b) => b.slug.localeCompare(a.slug))))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  return (
    <>
      <PostsHeader />
      <p className="meta">
        Long-form notes on tokenization, codecs, runs, and bugs. Authored at{' '}
        <code>posts/</code> in the{' '}
        <a href="https://github.com/Open-Athena/tomat/tree/main/posts" target="_blank" rel="noopener noreferrer">
          tomat repo
        </a>.
      </p>
      {error && <p style={{ color: 'crimson' }}>error: {error}</p>}
      {!posts && !error && <p>loading…</p>}
      {posts && (
        <ul className="posts-index">
          {posts.map((p) => {
            const prefixMatch = p.slug.match(/^(\d+)-/)
            const num = prefixMatch ? prefixMatch[1] : null
            return (
              <li key={p.slug}>
                {num && <span className="posts-index-num">{num}</span>}
                <a href={`#/posts/${p.slug}`}>{p.title}</a>
              </li>
            )
          })}
        </ul>
      )}
    </>
  )
}

function PostDetail({ slug }: { slug: string }) {
  const [body, setBody] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setBody(null)
    setError(null)
    fetch(postUrl(slug))
      .then((r) => {
        if (!r.ok) throw new Error(`fetch ${postUrl(slug)}: ${r.status}`)
        return r.text()
      })
      .then(setBody)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [slug])

  return (
    <>
      <PostsHeader subtitle={slug} />
      <p className="meta" style={{ marginBottom: '1rem' }}>
        <a href="#/posts">← all posts</a>
        {' · '}
        <a
          href={`https://github.com/Open-Athena/tomat/blob/main/posts/${slug}.md`}
          target="_blank"
          rel="noopener noreferrer"
        >
          source on GitHub
        </a>
      </p>
      {error && <p style={{ color: 'crimson' }}>error: {error}</p>}
      {!body && !error && <p>loading…</p>}
      {body && (
        <article className="post">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[
              rehypeKatex,
              rehypeHighlight,
              // Each heading gets an `id` attribute from its slugified text
              // (e.g. `## Free-running eval` → `id="free-running-eval"`), so
              // anyone can permalink it via `#section-slug`. Then the
              // autolink plugin wraps each heading in an `<a>` that copies
              // the URL on click — visually a "#" icon next to the heading
              // styled in posts.css.
              rehypeSlug,
              [rehypeAutolinkHeadings, {
                behavior: 'append',
                properties: { className: ['heading-anchor'], ariaLabel: 'permalink' },
                content: { type: 'text', value: ' #' },
              }],
            ]}
          >
            {body}
          </ReactMarkdown>
        </article>
      )}
    </>
  )
}
