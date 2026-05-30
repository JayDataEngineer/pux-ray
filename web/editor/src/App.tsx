import { TooltipProvider } from '@/components/ui/tooltip'
import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toaster } from './components/Toaster'

export function App() {
  return (
    <ErrorBoundary>
      <TooltipProvider>
        <WorkspaceLayout spec={null!} run={null} allSpecs={[]} onSpecChange={() => {}} />
      <Toaster />
      </TooltipProvider>
    </ErrorBoundary>
  )
}
