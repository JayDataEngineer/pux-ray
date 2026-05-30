import { useState, useEffect, useRef } from 'react'
import { AudioWorkspace } from './AudioWorkspace'
import { VisualWorkspace } from './VisualWorkspace'
import { VideoWorkspace } from './VideoWorkspace'
import { AssetSidebar } from './AssetSidebar'
import { useAssetStore } from '../../stores/assets'
import { useWorkflowStore } from '../../stores/workflow'
import { useToastStore } from '../../stores/toast'
import { getSpec, getRun } from '../../api'
import type { WorkflowRun, WorkflowSpec } from '../../types'

function extForMedia(mediaType: string): string {
  const m: Record<string, string> = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'video/mp4': 'mp4', 'audio/wav': 'wav', 'audio/mp3': 'mp3' }
  return m[mediaType] || 'bin'
}

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
  const addAsset = useAssetStore((s) => s.addAsset)
  const existingAssetIds = useRef(new Set<string>())

  // Sync generated artifacts into the persistent asset store
  useEffect(() => {
    if (!run) return
    for (const [, art] of Object.entries(run.artifacts)) {
      const assetId = `${run.run_id}:${art.step_id}:${art.name}`
      if (existingAssetIds.current.has(assetId)) continue
      existingAssetIds.current.add(assetId)
      const ext = extForMedia(art.media_type)
      const filename = art.name.includes('.') ? art.name : `${art.name}.${ext}`
      const url = `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${art.step_id}/${filename}`
      const type = art.media_type.startsWith('image/') ? 'image' as const
        : art.media_type.startsWith('audio/') ? 'audio' as const
        : art.media_type.startsWith('video/') ? 'video' as const
        : 'other' as const
      addAsset({
        name: `${art.step_id.replace(/_/g, ' ')} (${run.run_id.slice(0, 6)})`,
        type, mediaType: art.media_type, url,
        sizeBytes: art.size_bytes, source: 'generated',
        sourceRunId: run.run_id, sourceStepId: art.step_id,
      })
    }
  }, [run?.run_id, run?.artifacts])

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
