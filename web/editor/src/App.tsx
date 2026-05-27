import { useEffect, useState, useCallback } from 'react'
import { useWorkflowStore } from './stores/workflow'
import { listSpecs, getSpec, startRun, getRun } from './api'
import { useSSE } from './hooks/useSSE'
import { Layout } from './components/Layout'
import type { SSEEvent } from './types'

export function App() {
  const { spec, run, setSpec, setRun, setLoading, setError, error, loading, updateStepState, reset } = useWorkflowStore()
  const specName = run?.spec_name ?? null
  const runId = run?.run_id ?? null

  const [allSpecs, setAllSpecs] = useState<{ name: string; description: string; steps: number }[]>([])

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
      } else if (event.event === 'step_waiting') {
        updateStepState(event.step_id, { status: 'waiting_input' })
      }
    }
    if (event.event === 'workflow_completed' || event.event === 'workflow_failed') {
      if (specName && runId) {
        getRun(specName, runId).then(setRun).catch(() => {})
      }
    }
  }, [specName, runId, updateStepState, setRun])

  useSSE(specName, runId, onSSEEvent)

  useEffect(() => {
    listSpecs()
      .then((specs) => {
        setAllSpecs(specs.filter((s) => !s.name.startsWith('_')))
        if (specs.length > 0) {
          const defaultSpec = specs.find((s) => s.name === 'video_editor') || specs[0]
          return getSpec(defaultSpec.name)
        }
        return null
      })
      .then((s) => { if (s) setSpec(s) })
      .catch(() => setError('Failed to load specs'))
  }, [setSpec, setError])

  const handleSpecChange = useCallback(async (name: string) => {
    reset()
    try {
      const s = await getSpec(name)
      setSpec(s)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load spec')
    }
  }, [reset, setSpec, setError])

  const handleStart = useCallback(async (inputs: Record<string, unknown>) => {
    if (!spec) return
    setLoading(true)
    setError(null)
    try {
      const result = await startRun(spec.name, inputs)
      const fullRun = await getRun(spec.name, result.run_id)
      setRun(fullRun)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start run')
    } finally {
      setLoading(false)
    }
  }, [spec, setRun, setLoading, setError])

  const handleLoadRun = useCallback(async (specName: string, existingRunId: string) => {
    setLoading(true)
    setError(null)
    try {
      const s = await getSpec(specName)
      setSpec(s)
      const fullRun = await getRun(specName, existingRunId)
      setRun(fullRun)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load run')
    } finally {
      setLoading(false)
    }
  }, [setSpec, setRun, setLoading, setError])

  const handleNewRun = useCallback(() => {
    reset()
    if (spec) getSpec(spec.name).then(setSpec).catch(() => {})
  }, [reset, spec, setSpec])

  if (!spec) {
    return <div className="loading-screen">Loading workflow specs...</div>
  }

  return (
    <>
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="btn btn-ghost btn-sm" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      {loading && <div className="loading-overlay"><div className="spinner" /></div>}
      <Layout
        spec={spec}
        allSpecs={allSpecs}
        run={run}
        onStart={handleStart}
        onNewRun={handleNewRun}
        onSpecChange={handleSpecChange}
        onLoadRun={handleLoadRun}
      />
    </>
  )
}
