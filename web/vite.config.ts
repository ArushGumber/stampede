import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { PRODUCT_NAME } from './src/config'

// The product name lives in src/config.ts only; index.html gets it at build time.
export default defineConfig({
  plugins: [react(), { name: 'product-name', transformIndexHtml: html => html.replace(/%PRODUCT_NAME%/g, PRODUCT_NAME) }],
  base: './',
  // The simulator worker loads Pyodide with a dynamic import, so it is built as an ES module worker.
  worker: { format: 'es' },
  server: { proxy: { '/api': 'http://localhost:8765', '/ws': { target: 'ws://localhost:8765', ws: true }, '/mcp': 'http://localhost:8765', '/data': 'http://localhost:8765' } },
})
