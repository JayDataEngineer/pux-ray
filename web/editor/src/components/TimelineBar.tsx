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

  return <TimelineEditor />
}
