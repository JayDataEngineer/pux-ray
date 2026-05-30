import { useEffect, useState, useCallback } from 'react'
import { useWorkflowStore } from './stores/workflow'
import { useToastStore } from './stores/toast'
import { listSpecs, getSpec } from './api'
import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toaster } from './components/Toaster'

export function App() {
  const { spec, setSpec, loading } = useWorkflowStore()
  const toast = useToastStore((s) => s.addToast)
  const [allSpecs, setAllSpecs] = useState<{ name: string; description: string; steps: number }[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listSpecs()
      .then(async (specs) => {
        const filtered = specs.filter((s) => !s.name.startsWith('_'))
        setAllSpecs(filtered)
        if (filtered.length > 0) {
          const defaultSpec = filtered.find((s) => s.name === 'video_editor') || filtered[0]
          const s = await getSpec(defaultSpec.name)
          setSpec(s)
        }
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : 'Could not load pipeline specs')
        toast('error', e instanceof Error ? e.message : 'Failed to load')
      })
  }, [])

  const handleSpecChange = useCallback(async (name: string) => {
    try {
      const s = await getSpec(name)
      setSpec(s)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not load spec')
    }
  }, [setSpec, toast])

  if (!spec) {
    if (error) return (
      <div className="loading-screen">
        <p>Failed to load pipeline specs</p>
        <p className="error-detail">{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    )
    return <div className="loading-screen">Loading workspace...</div>
  }

  return (
    <ErrorBoundary>
      {loading && <div className="loading-overlay"><div className="spinner" /></div>}
      <WorkspaceLayout spec={spec} run={null} allSpecs={allSpecs} onSpecChange={handleSpecChange} />
      <Toaster />
    </ErrorBoundary>
  )
}
