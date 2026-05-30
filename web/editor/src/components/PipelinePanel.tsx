import { useState, useEffect, useCallback } from 'react'
import type { WorkflowSpec, WorkflowRun } from '../types'
import { useWorkflowStore } from '../stores/workflow'
import { useToastStore } from '../stores/toast'
import { executeStep, getRun } from '../api'
import { StepCard } from './StepCard'
import { getSourceRefs } from '../utils/stepUtils'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onLoadRun: (specName: string, runId: string) => void
}

function canExecuteStep(stepId: string, spec: WorkflowSpec, run: WorkflowRun): boolean {
  const step = spec.steps.find((s) => s.id === stepId)
  if (!step) return false
  return step.depends_on.every((dep) => {
    const depState = run.step_states[dep]
    return depState?.status === 'completed'
  })
}

export function PipelinePanel({ spec, run, onLoadRun }: Props) {
  const selectedStepId = useWorkflowStore((s) => s.selectedStepId)
  const setSelectedStep = useWorkflowStore((s) => s.setSelectedStep)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)
  const [pastRuns, setPastRuns] = useState<{ run_id: string; status: string; created_at?: string }[]>([])
  const [executing, setExecuting] = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem(`runs_${spec.name}`)
    if (stored) {
      try { setPastRuns(JSON.parse(stored)) } catch { setPastRuns([]) }
    }
  }, [spec.name])

  useEffect(() => {
    if (run?.run_id) {
      const key = `runs_${spec.name}`
      const stored = JSON.parse(localStorage.getItem(key) || '[]') as { run_id: string; status: string; created_at?: string }[]
      if (!stored.find((r) => r.run_id === run.run_id)) {
        const updated = [{ run_id: run.run_id, status: run.status, created_at: new Date().toISOString() }, ...stored].slice(0, 20)
        localStorage.setItem(key, JSON.stringify(updated))
      } else {
        const updated = stored.map((r) => r.run_id === run.run_id ? { ...r, status: run.status } : r)
        localStorage.setItem(key, JSON.stringify(updated))
      }
    }
  }, [run?.run_id, run?.status, spec.name])

  const handleExecuteStep = useCallback(async (stepId: string) => {
    if (!run) return
    setExecuting(stepId)
    try {
      const result = await executeStep(run.spec_name, run.run_id, stepId) as Record<string, unknown>
      if (result.status === 'error') {
        toast('error', (result.error as string) || 'Step failed')
      }
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
    } catch (err) {
      toast('error', err instanceof Error ? err.message : 'Step execution failed')
    } finally {
      setExecuting(null)
    }
  }, [run, setRun, toast])

  const tabs = ['Steps', 'History'] as const
  const [activeTab, setActiveTab] = useState<typeof tabs[number]>('Steps')

  return (
    <div className="pipeline-panel">
      <div className="panel-header">
        <div className="panel-tabs">
          {tabs.map((tab) => (
            <button
              key={tab}
              className={`panel-tab ${activeTab === tab ? 'panel-tab--active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'Steps' ? (
        <div className="pipeline-steps">
          {spec.steps.map((step) => {
            const state = run?.step_states[step.id]
            const status = state?.status ?? 'pending'
            const isExecuting = executing === step.id

            const artifacts = run
              ? Object.entries(run.artifacts)
                  .filter(([k]) => k.startsWith(`${step.id}.`))
                  .map(([, v]) => v)
              : []

            const sourceRefs = getSourceRefs(step)
            const sourceArtifacts = sourceRefs.map((ref) => {
              const artKey = Object.keys(run?.artifacts || {}).find((k) => k.startsWith(`${ref.stepId}.${ref.outputKey}`))
              const art = artKey && run ? run.artifacts[artKey] : null
              const extMap: Record<string, string> = {
                'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp',
                'video/mp4': 'mp4', 'video/webm': 'webm',
                'audio/wav': 'wav', 'audio/mp3': 'mp3',
                'model/gltf-binary': 'glb',
              }
              const thumbnailUrl = art && run
                ? (() => {
                    const ext = extMap[art.media_type] || art.name.split('.').pop() || 'bin'
                    const filename = art.name.includes('.') ? art.name : `${art.name}.${ext}`
                    return `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${art.step_id}/${filename}`
                  })()
                : null
              return { stepId: ref.stepId, outputKey: ref.outputKey, thumbnailUrl }
            })

            const depsMet = run ? canExecuteStep(step.id, spec, run) : false

            return (
              <StepCard
                key={step.id}
                stepId={step.id}
                stepType={step.type}
                interaction={step.interaction}
                status={isExecuting ? 'running' : status}
                durationMs={state?.duration_ms ?? null}
                error={state?.error ?? null}
                artifacts={artifacts}
                sourceArtifacts={sourceArtifacts}
                specName={run?.spec_name ?? spec.name}
                runId={run?.run_id ?? ''}
                selected={selectedStepId === step.id}
                onClick={() => setSelectedStep(selectedStepId === step.id ? null : step.id)}
                onExecute={handleExecuteStep}
                canExecute={depsMet}
              />
            )
          })}
        </div>
      ) : (
        <div className="run-history">
          {pastRuns.length === 0 ? (
            <div className="control-empty">No past runs</div>
          ) : (
            pastRuns.map((pr) => (
              <div
                key={pr.run_id}
                className={`history-item ${pr.run_id === run?.run_id ? 'history-item--active' : ''}`}
                onClick={() => onLoadRun(spec.name, pr.run_id)}
              >
                <span className={`run-status run-status--${pr.status}`}>{pr.status}</span>
                <span className="history-id">{pr.run_id}</span>
                {pr.created_at && (
                  <span className="history-date">
                    {new Date(pr.created_at).toLocaleString()}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
