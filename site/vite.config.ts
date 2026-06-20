import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { readdirSync, readFileSync } from 'node:fs'
import { resolve, join } from 'node:path'

const allowedHosts = process.env.VITE_ALLOWED_HOSTS?.split(',') ?? []

// Markdown posts live in ../posts (sibling of site/) so they're authored next
// to the rest of the repo and not duplicated into public/. This plugin
// (a) serves /posts/<slug>.md + /posts-index.json in dev via middleware,
// (b) emits them as static assets at build time.
function postsPlugin(): Plugin {
  const postsDir = resolve(__dirname, '..', 'posts')

  const slugs = (): string[] =>
    readdirSync(postsDir)
      .filter((f) => f.endsWith('.md'))
      .sort()
      .map((f) => f.replace(/\.md$/, ''))

  const buildIndex = (): Array<{ slug: string; title: string }> =>
    slugs().map((slug) => {
      const md = readFileSync(join(postsDir, `${slug}.md`), 'utf8')
      const h1 = md.split('\n').find((l) => l.startsWith('# '))
      const title = h1 ? h1.replace(/^#\s+/, '').trim() : slug
      return { slug, title }
    })

  return {
    name: 'tomat-posts',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? ''
        if (url === '/posts-index.json') {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(buildIndex()))
          return
        }
        const m = url.match(/^\/posts\/([^/?#]+)\.md(?:\?.*)?$/)
        if (m) {
          const slug = m[1]
          try {
            const body = readFileSync(join(postsDir, `${slug}.md`), 'utf8')
            res.setHeader('Content-Type', 'text/markdown; charset=utf-8')
            res.end(body)
          } catch {
            res.statusCode = 404
            res.end(`post not found: ${slug}`)
          }
          return
        }
        next()
      })
    },
    generateBundle() {
      // Emit the index and each markdown file as static assets at
      // /posts-index.json and /posts/<slug>.md, matching the dev paths.
      this.emitFile({
        type: 'asset',
        fileName: 'posts-index.json',
        source: JSON.stringify(buildIndex()),
      })
      for (const slug of slugs()) {
        const source = readFileSync(join(postsDir, `${slug}.md`), 'utf8')
        this.emitFile({
          type: 'asset',
          fileName: `posts/${slug}.md`,
          source,
        })
      }
    },
  }
}

export default defineConfig({
  // Deployed at tomat.oa.dev (custom domain, root path).
  base: process.env.VITE_BASE ?? '/',
  server: {
    port: 4273,
    host: true,
    allowedHosts,
  },
  plugins: [react(), postsPlugin()]
})
