import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// Ray Serve API URL — override with API_URL env var for remote dev
const API_URL = process.env.API_URL || 'http://localhost:30080'

const proxyConfig: Record<string, { target: string; ws: boolean }> = {
  // WebSocket paths (Viser 3D, ComfyUI)
  '/kimodo':  { target: API_URL, ws: true },
  '/comfyui': { target: API_URL, ws: true },
  // HTTP API paths
  '/v1':        { target: API_URL, ws: false },
  '/mcp':       { target: API_URL, ws: false },
  '/forge':     { target: API_URL, ws: false },
  '/health':    { target: API_URL, ws: false },
  '/studio':    { target: API_URL, ws: false },
  '/dashboard': { target: API_URL, ws: false },
  '/admin':     { target: API_URL, ws: false },
  '/llm':       { target: API_URL, ws: false },
}

export default defineConfig({
  resolve: { alias: { '@': '/src' } },
  plugins: [tailwindcss(),react()],
  base: '/editor/',
  build: {
    outDir: '../../gateway/editor',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: proxyConfig,
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
    proxy: proxyConfig,
  },
})
