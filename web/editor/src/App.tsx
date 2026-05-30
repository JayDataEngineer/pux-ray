import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toaster } from './components/Toaster'

export function App() {
  return (
    <ErrorBoundary>
      <WorkspaceLayout spec={null!} run={null} allSpecs={[]} onSpecChange={() => {}} />
      <Toaster />
    </ErrorBoundary>
  )
}
