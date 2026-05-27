import type { WorkflowSpec, WorkflowRun } from '../types'
import { Header } from './Header'
import { PipelinePanel } from './PipelinePanel'
import { PreviewPanel } from './PreviewPanel'
import { ControlPanel } from './ControlPanel'
import { TimelineBar } from './TimelineBar'

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
  return (
    <div className="editor-layout">
      <Header spec={spec} allSpecs={allSpecs} run={run} onNewRun={onNewRun} onSpecChange={onSpecChange} />
      <div className="editor-body">
        <PipelinePanel spec={spec} run={run} onLoadRun={onLoadRun} />
        <PreviewPanel run={run} />
        <ControlPanel spec={spec} run={run} onStart={onStart} />
      </div>
      <TimelineBar spec={spec} run={run} />
    </div>
  )
}
