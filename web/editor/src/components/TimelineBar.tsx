import { useEffect } from 'react'
import type { WorkflowSpec, WorkflowRun } from '../types'
import { useTimelineStore } from '../stores/timeline'
import { TimelineEditor } from './timeline/TimelineEditor'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
}

export function TimelineBar({ spec, run }: Props) {
  const loadFromRun = useTimelineStore((s) => s.loadFromRun)
  const reset = useTimelineStore((s) => s.reset)

  // Sync timeline with workflow run
  useEffect(() => {
    if (run) {
      loadFromRun(run, spec)
    } else {
      reset()
    }
  }, [run, spec, loadFromRun, reset])

  // Show SimpleTimeline when no run is active
  if (!run) {
    return <SimpleTimeline spec={spec} />
  }

  // Show the Canvas timeline editor when we have a run
  return <TimelineEditor />
}

function SimpleTimeline({ spec }: { spec: WorkflowSpec }) {
  return (
    <div className="timeline-bar">
      <div className="timeline-stages">
        {spec.steps.map((step) => (
          <div key={step.id} className="timeline-stage timeline-stage--pending">
            <div className="stage-dot" />
            <div className="stage-label">{step.id.replace(/_/g, ' ')}</div>
          </div>
        ))}
      </div>
      <div className="timeline-progress">
        {spec.steps.length} steps
      </div>
    </div>
  )
}
