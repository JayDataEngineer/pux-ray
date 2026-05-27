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
    port: 5173,
    proxy: {
      '/v1': 'http://localhost:30080',
      '/health': 'http://localhost:30080',
    },
  },
})
