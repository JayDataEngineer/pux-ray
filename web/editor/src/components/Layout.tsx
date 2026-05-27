import type { WorkflowSpec, WorkflowRun } from '../types'
import { Header } from './Header'
import { PipelinePanel } from './PipelinePanel'
import { PreviewPanel } from './PreviewPanel'
import { ControlPanel } from './ControlPanel'
import { TimelineBar } from './TimelineBar'

interface LayoutProps {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onStart: (inputs: Record<string, unknown>) => void
  onNewRun: () => void
}

export function Layout({ spec, run, onStart, onNewRun }: LayoutProps) {
  return (
    <div className="editor-layout">
      <Header spec={spec} run={run} onNewRun={onNewRun} />
      <div className="editor-body">
        <PipelinePanel spec={spec} run={run} />
        <PreviewPanel run={run} />
        <ControlPanel spec={spec} run={run} onStart={onStart} />
      </div>
      <TimelineBar spec={spec} run={run} />
    </div>
  )
}
