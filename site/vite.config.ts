import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const allowedHosts = process.env.VITE_ALLOWED_HOSTS?.split(',') ?? []

export default defineConfig({
  // Deployed at tomat.oa.dev (custom domain, root path).
  base: process.env.VITE_BASE ?? '/',
  server: {
    port: 4273,
    host: true,
    allowedHosts,
  },
  plugins: [react()]
})
