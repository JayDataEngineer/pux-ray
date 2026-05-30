import { useMemo } from 'react'
import type { WorkflowSpec, WorkflowRun } from '../types'
import { Header } from './Header'
import { PipelinePanel } from './PipelinePanel'
import { PreviewPanel } from './PreviewPanel'
import { ControlPanel } from './ControlPanel'
import { TimelineBar } from './TimelineBar'
import { TimelineEditor } from './timeline/TimelineEditor'
import { useWorkflowStore } from '../stores/workflow'
import { useTimelineStore } from '../stores/timeline'

interface LayoutProps {
  spec: WorkflowSpec
  allSpecs: { name: string; description: string; steps: number }[]
  run: WorkflowRun | null
  onStart: (inputs: Record<string, unknown>) => void
  onNewRun: () => void
  onSpecChange: (name: string) => void
  onLoadRun: (specName: string, runId: string) => void
}

export function Layout({ spec, allSpecs, run, onStart, onNewRun, onSpecChange, onLoadRun }: LayoutProps) {
  const viewMode = useWorkflowStore((s) => s.viewMode)
  const loadFromRun = useTimelineStore((s) => s.loadFromRun)

  // Sync run results into timeline when run changes
  useMemo(() => {
    if (run && viewMode === 'timeline' && run.status === 'running') {
      loadFromRun(run, spec)
    }
  }, [run?.run_id, run?.status, viewMode])

  if (viewMode === 'timeline') {
    return (
      <div className="editor-layout">
        <Header spec={spec} allSpecs={allSpecs} run={run} onNewRun={onNewRun} onSpecChange={onSpecChange} />
        <div className="editor-body">
          <TimelineEditor />
          <PreviewPanel run={run} />
          <ControlPanel spec={spec} run={run} onStart={onStart} />
        </div>
      </div>
    )
  }

  return (
    <div className="editor-layout">
      <Header spec={spec} allSpecs={allSpecs} run={run} onNewRun={onNewRun} onSpecChange={onSpecChange} />
      <div className="editor-body">
        <PipelinePanel spec={spec} run={run} onLoadRun={onLoadRun} />
        <PreviewPanel run={run} />
        <ControlPanel spec={spec} run={run} onStart={onStart} />
      </div>
      <div className="editor-timeline">
        <TimelineBar spec={spec} run={run} />
      </div>
    </div>
  )
}
