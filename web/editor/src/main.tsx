import './tailwind.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { App } from './App'
import { useAssetStore, nextAssetName } from './stores/assets'

// Expose stores/helpers on window in dev for Playwright e2e tests.  These are
// harmless in production (just references) and let tests assert naming logic
// without a separate test endpoint.
;(window as any).__assetStore = useAssetStore
;(window as any).__nextAssetName = nextAssetName

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
