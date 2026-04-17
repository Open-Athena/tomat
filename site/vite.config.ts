import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const allowedHosts = process.env.VITE_ALLOWED_HOSTS?.split(',') ?? []

export default defineConfig({
  // Pages deploys to open-athena.github.io/tomat/ — override with VITE_BASE=/ in dev.
  base: process.env.VITE_BASE ?? '/tomat/',
  server: {
    port: 4273,
    host: true,
    allowedHosts,
  },
  plugins: [react()],
})
