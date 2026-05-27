import type { WorkflowSpec, WorkflowRun } from '../types'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
}

export function TimelineBar({ spec, run }: Props) {
  return (
    <div className="timeline-bar">
      <div className="timeline-stages">
        {spec.steps.map((step) => {
          const state = run?.step_states[step.id]
          const status = state?.status ?? 'pending'
          return (
            <div key={step.id} className={`timeline-stage timeline-stage--${status}`}>
              <div className="stage-dot" />
              <div className="stage-label">{step.id.replace(/_/g, ' ')}</div>
              {status === 'running' && <div className="stage-pulse" />}
            </div>
          )
        })}
      </div>
      <div className="timeline-progress">
        {run && (() => {
          const completed = Object.values(run.step_states).filter((s) => s.status === 'completed').length
          const total = spec.steps.length
          return `${completed}/${total} steps`
        })()}
      </div>
    </div>
  )
}
