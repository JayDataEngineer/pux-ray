import { useEffect } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toaster } from './components/Toaster'
import { useAssetStore } from './stores/assets'

export function App() {
  const initializeAssets = useAssetStore(s => s.initialize)

  useEffect(() => {
    // Initialize assets from localStorage + IndexedDB on app start
    initializeAssets()
  }, [initializeAssets])

  return (
    <ErrorBoundary>
      <TooltipProvider>
        <WorkspaceLayout />
        <Toaster />
      </TooltipProvider>
    </ErrorBoundary>
  )
}
