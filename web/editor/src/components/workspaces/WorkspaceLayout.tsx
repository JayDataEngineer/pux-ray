import { useState } from 'react'
import { AudioWorkspace } from './AudioWorkspace'
import { VisualWorkspace } from './VisualWorkspace'
import { VideoWorkspace } from './VideoWorkspace'
import { AssetSidebar } from './AssetSidebar'
import { useWorkflowStore } from '../../stores/workflow'
import { useToastStore } from '../../stores/toast'
import { getSpec, getRun } from '../../api'
import type { WorkflowRun, WorkflowSpec } from '../../types'

type WorkspaceTab = 'audio' | 'visuals' | 'video'

interface Props {
  spec: WorkflowSpec
  run: WorkflowRun | null
  onSpecChange: (name: string) => void
  allSpecs: { name: string; description: string; steps: number }[]
}

export function WorkspaceLayout({ spec, run, onSpecChange, allSpecs }: Props) {
  const [tab, setTab] = useState<WorkspaceTab>('visuals')
  const setSpec = useWorkflowStore((s) => s.setSpec)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)

  const handleNavigateHistory = async (specName: string, runId: string) => {
    try {
      const s = await getSpec(specName)
      setSpec(s)
      const r = await getRun(specName, runId)
      setRun(r)
    } catch {
      toast('error', 'Could not load past run')
    }
  }

  const tabs: { id: WorkspaceTab; label: string }[] = [
    { id: 'audio', label: 'Audio' },
    { id: 'visuals', label: 'Visuals' },
    { id: 'video', label: 'Video' },
  ]

  return (
    <div className="workspace-layout">
      <header className="workspace-header">
        <div className="workspace-header-left">
          <span className="workspace-logo">TECH NOIR</span>
          <nav className="workspace-tabs">
            {tabs.map((t) => (
              <button
                key={t.id}
                className={`workspace-tab ${tab === t.id ? 'workspace-tab--active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="workspace-header-right">
          <button className="btn btn-ghost btn-sm" onClick={() => window.open('/studio/', '_blank')}>
            Studio
          </button>
        </div>
      </header>
      <div className="workspace-body">
        <AssetSidebar run={run} onNavigateHistory={handleNavigateHistory} />
        {tab === 'audio' && <AudioWorkspace run={run} />}
        {tab === 'visuals' && <VisualWorkspace spec={spec} run={run} allSpecs={allSpecs} onSpecChange={onSpecChange} />}
        {tab === 'video' && <VideoWorkspace run={run} />}
      </div>
    </div>
  )
}
