import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/editor/',
  build: {
    outDir: '../../gateway/editor',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/v1': 'http://localhost:30080',
      '/mcp': 'http://localhost:30080',
      '/forge': 'http://localhost:30080',
      '/health': 'http://localhost:30080',
      '/kimodo': 'http://localhost:30080',
      '/studio': 'http://localhost:30080',
      '/dashboard': 'http://localhost:30080',
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
    proxy: {
      '/v1': 'http://localhost:30080',
      '/mcp': 'http://localhost:30080',
      '/forge': 'http://localhost:30080',
      '/health': 'http://localhost:30080',
      '/kimodo': 'http://localhost:30080',
      '/studio': 'http://localhost:30080',
      '/dashboard': 'http://localhost:30080',
    },
  },
})
