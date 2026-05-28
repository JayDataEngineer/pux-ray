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

  useEffect(() => {
    if (run) {
      loadFromRun(run, spec)
    } else {
      reset()
    }
  }, [run, spec, loadFromRun, reset])

  if (!run) {
    return <SimpleTimeline spec={spec} />
  }

  // Only show the full timeline editor when there's video or audio content
  const hasMedia = Object.values(run.artifacts).some(
    (a) => a.media_type?.startsWith('video/') || a.media_type?.startsWith('audio/'),
  )
  if (!hasMedia) {
    return <SimpleTimeline spec={spec} />
  }

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
