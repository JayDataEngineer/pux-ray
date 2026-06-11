import { useEffect, useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toaster } from './components/Toaster'
import { useAssetStore } from './stores/assets'
import { Loader2 } from 'lucide-react'

export function App() {
  const [loading, setLoading] = useState(true)
  const initializeAssets = useAssetStore(s => s.initialize)

  useEffect(() => {
    let mounted = true

    // Initialize assets from localStorage + IndexedDB on app start
    const init = async () => {
      try {
        await initializeAssets()
      } catch (err) {
        console.error('[App] Failed to initialize assets:', err)
      } finally {
        if (mounted) setLoading(false)
      }
    }

    init()

    return () => { mounted = false }
  }, [initializeAssets])

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading assets...</p>
        </div>
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <TooltipProvider>
        <WorkspaceLayout />
        <Toaster />
      </TooltipProvider>
    </ErrorBoundary>
  )
}
