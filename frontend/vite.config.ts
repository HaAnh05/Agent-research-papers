import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API remains a separate process in development. Production deployments may
// serve the generated bundle from FastAPI, so the browser only ever uses /api.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // This keeps local dev resilient when the workspace is nested under another
  // package tree. The production build still bundles all dependencies.
  optimizeDeps: {
    noDiscovery: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
