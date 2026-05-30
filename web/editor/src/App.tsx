import { useEffect, useState, useCallback } from 'react'
import { useWorkflowStore } from './stores/workflow'
import { useToastStore } from './stores/toast'
import { listSpecs, getSpec, startRun, getRun } from './api'
import { useSSE } from './hooks/useSSE'
import { Layout } from './components/Layout'
import { WorkspaceLayout } from './components/workspaces/WorkspaceLayout'
import { Toaster } from './components/Toaster'
import type { SSEEvent } from './types'

export function App() {
  const { spec, run, setSpec, setRun, loading, updateStepState, reset, viewMode } = useWorkflowStore()
  const toast = useToastStore((s) => s.addToast)
  const specName = run?.spec_name ?? null
  const runId = run?.run_id ?? null

  const [allSpecs, setAllSpecs] = useState<{ name: string; description: string; steps: number }[]>([])
  const [error, setError] = useState<string | null>(null)

  const onSSEEvent = useCallback((event: SSEEvent) => {
    if (event.step_id) {
      if (event.event === 'step_started') {
        updateStepState(event.step_id, { status: 'running' })
      } else if (event.event === 'step_completed') {
        updateStepState(event.step_id, {
          status: 'completed',
          duration_ms: event.duration_ms,
          outputs: event.outputs,
        })
      } else if (event.event === 'step_failed') {
        updateStepState(event.step_id, { status: 'failed', error: event.error })
        toast('error', event.error || 'Step failed')
      } else if (event.event === 'step_waiting') {
        updateStepState(event.step_id, { status: 'waiting_input' })
      }
    }
    if (event.event === 'workflow_completed') {
      toast('success', 'Pipeline completed')
      if (specName && runId) getRun(specName, runId).then(setRun).catch(() => {})
    } else if (event.event === 'workflow_failed') {
      toast('error', event.error || 'Pipeline failed')
      if (specName && runId) getRun(specName, runId).then(setRun).catch(() => {})
    }
  }, [specName, runId, updateStepState, setRun, toast])

  useSSE(specName, runId, onSSEEvent)

  // Load specs and auto-create a run for the workspace
  useEffect(() => {
    listSpecs()
      .then(async (specs) => {
        const filtered = specs.filter((s) => !s.name.startsWith('_'))
        setAllSpecs(filtered)
        if (filtered.length > 0) {
          const defaultSpec = filtered.find((s) => s.name === 'video_editor') || filtered[0]
          const s = await getSpec(defaultSpec.name)
          setSpec(s)
          // Auto-create a run so workspace has context
          const result = await startRun(s.name, {}, true)
          const fullRun = await getRun(s.name, result.run_id)
          setRun(fullRun)
        }
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : 'Could not load pipeline specs')
        toast('error', e instanceof Error ? e.message : 'Failed to load')
      })
  }, [])

  const handleSpecChange = useCallback(async (name: string) => {
    reset()
    try {
      const s = await getSpec(name)
      setSpec(s)
      const result = await startRun(s.name, {}, true)
      const fullRun = await getRun(s.name, result.run_id)
      setRun(fullRun)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not load spec')
    }
  }, [reset, setSpec, setRun, toast])

  if (!spec || !run) {
    if (error) {
      return (
        <div className="loading-screen">
          <p>Failed to load pipeline specs</p>
          <p className="error-detail">{error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      )
    }
    return <div className="loading-screen">Loading workspace...</div>
  }

  // Use workspace layout for timeline/workspace modes, pipeline layout for pipeline mode
  if (viewMode === 'timeline' || viewMode === 'kimodo') {
    return (
      <>
        {loading && <div className="loading-overlay"><div className="spinner" /></div>}
        <WorkspaceLayout spec={spec} run={run} allSpecs={allSpecs} onSpecChange={handleSpecChange} />
        <Toaster />
      </>
    )
  }

  return (
    <>
      {loading && <div className="loading-overlay"><div className="spinner" /></div>}
      <Layout spec={spec} allSpecs={allSpecs} run={run}
        onStart={async () => {}} onNewRun={() => reset()} onSpecChange={handleSpecChange}
        onLoadRun={async (sn, rid) => {
          const s = await getSpec(sn); setSpec(s)
          const fr = await getRun(sn, rid); setRun(fr)
        }} />
      <Toaster />
    </>
  )
}
