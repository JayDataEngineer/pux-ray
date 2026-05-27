import type { WorkflowSpec, WorkflowRun } from '../types'
import { useWorkflowStore } from '../stores/workflow'
import { StepCard } from './StepCard'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
}

export function PipelinePanel({ spec, run }: Props) {
  const selectedStepId = useWorkflowStore((s) => s.selectedStepId)
  const setSelectedStep = useWorkflowStore((s) => s.setSelectedStep)

  return (
    <div className="pipeline-panel">
      <div className="panel-header">Pipeline</div>
      <div className="pipeline-steps">
        {spec.steps.map((step) => {
          const state = run?.step_states[step.id]
          const artifacts = run
            ? Object.entries(run.artifacts)
                .filter(([k]) => k.startsWith(`${step.id}.`))
                .map(([, v]) => v)
            : []
          return (
            <StepCard
              key={step.id}
              stepId={step.id}
              stepType={step.type}
              status={state?.status ?? 'pending'}
              durationMs={state?.duration_ms ?? null}
              error={state?.error ?? null}
              artifacts={artifacts}
              specName={run?.spec_name ?? ''}
              runId={run?.run_id ?? ''}
              selected={selectedStepId === step.id}
              onClick={() => setSelectedStep(selectedStepId === step.id ? null : step.id)}
            />
          )
        })}
      </div>
    </div>
  )
}
