import { useEffect, useState, useCallback } from 'react'
import { useWorkflowStore } from './stores/workflow'
import { useToastStore } from './stores/toast'
import { listSpecs, getSpec, startRun, getRun } from './api'
import { useSSE } from './hooks/useSSE'
import { Layout } from './components/Layout'
import { Toaster } from './components/Toaster'
import type { SSEEvent } from './types'

export function App() {
  const { spec, run, setSpec, setRun, setLoading, loading, updateStepState, reset } = useWorkflowStore()
  const toast = useToastStore((s) => s.addToast)
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
        toast('error', event.error || 'Step failed')
      } else if (event.event === 'step_waiting') {
        updateStepState(event.step_id, { status: 'waiting_input' })
      }
    }
    if (event.event === 'workflow_completed') {
      toast('success', 'Pipeline completed')
      if (specName && runId) {
        getRun(specName, runId).then(setRun).catch(() => {})
      }
    } else if (event.event === 'workflow_failed') {
      toast('error', event.error || 'Pipeline failed')
      if (specName && runId) {
        getRun(specName, runId).then(setRun).catch(() => {})
      }
    }
  }, [specName, runId, updateStepState, setRun, toast])

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
      .catch((e) => toast('error', 'Could not load pipeline specs'))
  }, [setSpec, toast])

  const handleSpecChange = useCallback(async (name: string) => {
    reset()
    try {
      const s = await getSpec(name)
      setSpec(s)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not load spec')
    }
  }, [reset, setSpec, toast])

  const handleStart = useCallback(async (inputs: Record<string, unknown>) => {
    if (!spec) return
    setLoading(true)
    try {
      const result = await startRun(spec.name, inputs)
      const fullRun = await getRun(spec.name, result.run_id)
      setRun(fullRun)
      toast('info', 'Pipeline started')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not start pipeline')
    } finally {
      setLoading(false)
    }
  }, [spec, setRun, setLoading, toast])

  const handleLoadRun = useCallback(async (specName: string, existingRunId: string) => {
    setLoading(true)
    try {
      const s = await getSpec(specName)
      setSpec(s)
      const fullRun = await getRun(specName, existingRunId)
      setRun(fullRun)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Could not load run')
    } finally {
      setLoading(false)
    }
  }, [setSpec, setRun, setLoading, toast])

  const handleNewRun = useCallback(() => {
    reset()
    if (spec) getSpec(spec.name).then(setSpec).catch(() => {})
  }, [reset, spec, setSpec])

  if (!spec) {
    return <div className="loading-screen">Loading workflow specs...</div>
  }

  return (
    <>
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
      <Toaster />
    </>
  )
}
