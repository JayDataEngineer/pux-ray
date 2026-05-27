import { useEffect, useCallback } from 'react'
import { useWorkflowStore } from './stores/workflow'
import { getSpec, startRun, getRun } from './api'
import { useSSE } from './hooks/useSSE'
import { Layout } from './components/Layout'
import type { SSEEvent } from './types'

export function App() {
  const { spec, run, setSpec, setRun, setLoading, setError, updateStepState, reset } = useWorkflowStore()
  const specName = run?.spec_name ?? null
  const runId = run?.run_id ?? null

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
    getSpec('video_editor').then(setSpec).catch(() => setError('Failed to load spec'))
  }, [setSpec, setError])

  const handleStart = useCallback(async (inputs: Record<string, unknown>) => {
    setLoading(true)
    setError(null)
    try {
      const result = await startRun('video_editor', inputs)
      const fullRun = await getRun('video_editor', result.run_id)
      setRun(fullRun)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start run')
    } finally {
      setLoading(false)
    }
  }, [setRun, setLoading, setError])

  const handleNewRun = useCallback(() => {
    reset()
    getSpec('video_editor').then(setSpec).catch(() => {})
  }, [reset, setSpec])

  if (!spec) {
    return <div className="loading-screen">Loading workflow spec...</div>
  }

  return (
    <Layout
      spec={spec}
      run={run}
      onStart={handleStart}
      onNewRun={handleNewRun}
    />
  )
}
