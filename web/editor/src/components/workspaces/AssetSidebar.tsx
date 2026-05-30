import { useState, useRef, useCallback } from 'react'
import { FolderOpen, History, Plus, Music, Trash2, Play } from 'lucide-react'
import { useAssetStore } from '../../stores/assets'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'
import type { WorkflowRun } from '../../types'

type SidebarTab = 'assets' | 'history'

interface Props {
  run: WorkflowRun | null
  onNavigateHistory?: (specName: string, runId: string) => void
}

export function AssetSidebar({ run: _run, onNavigateHistory }: Props) {
  const [activeTab, setActiveTab] = useState<SidebarTab>('assets')
  const [playingId, setPlayingId] = useState<string | null>(null)
  const assets = useAssetStore((s) => s.assets)
  const addAsset = useAssetStore((s) => s.addAsset)
  const removeAsset = useAssetStore((s) => s.removeAsset)
  const [pastRuns] = useState<{ run_id: string; spec_name: string; status: string }[]>(() => {
    try { return JSON.parse(localStorage.getItem('past_runs') || '[]') } catch { return [] }
  })
  const fileRef = useRef<HTMLInputElement>(null)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const toast = useToastStore((s) => s.addToast)

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    const type = file.type.startsWith('image/') ? 'image' as const
      : file.type.startsWith('audio/') ? 'audio' as const
      : file.type.startsWith('video/') ? 'video' as const
      : 'other' as const
    addAsset({ name: file.name, type, mediaType: file.type, url, sizeBytes: file.size, source: 'uploaded' })
    if (type === 'audio') {
      addAudioCue({ track: 'sfx', start: 0, duration: 5, label: file.name.replace(/\.[^.]+$/, ''), audioUrl: url, volume: 0.8, waveformPeaks: null, sourceStepId: null })
    }
    toast('info', `"${file.name}" added to Assets`)
    if (fileRef.current) fileRef.current.value = ''
  }, [addAsset, addAudioCue, toast])

  const handlePlay = (asset: { url: string; id: string }) => {
    setPlayingId((prev) => prev === asset.id ? null : asset.id)
  }

  const onDragStart = (e: React.DragEvent, asset: { url: string; id: string; type: string; name: string }) => {
    e.dataTransfer.setData('application/tech-noir-asset', JSON.stringify({ url: asset.url, type: asset.type, name: asset.name, id: asset.id }))
    e.dataTransfer.setData('text/plain', asset.url)
    e.dataTransfer.effectAllowed = 'copy'
  }

  const onDoubleClick = (url: string) => {
    // Open full-size in a module-style overlay
    const overlay = document.createElement('div')
    overlay.className = 'asset-focus-overlay'
    overlay.onclick = () => overlay.remove()
    const img = document.createElement('img')
    img.src = url
    img.className = 'asset-focus-img'
    overlay.appendChild(img)
    document.body.appendChild(overlay)
  }

  return (
    <aside className="workspace-sidebar">
      <div className="sidebar-section sidebar-header-row">
        <div className="sidebar-title">ASSETS</div>
        <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*,audio/*,video/*" onChange={handleImport} />
        <button className="btn-icon" onClick={() => fileRef.current?.click()} title="Import file">
          <Plus size={16} />
        </button>
      </div>

      <nav className="sidebar-nav">
        <a className={`sidebar-link ${activeTab === 'assets' ? 'sidebar-link--active' : ''}`} onClick={() => setActiveTab('assets')}>
          <FolderOpen size={14} className="sidebar-icon" />{assets.length > 0 ? `Assets (${assets.length})` : 'Assets'}
        </a>
        <a className={`sidebar-link ${activeTab === 'history' ? 'sidebar-link--active' : ''}`} onClick={() => setActiveTab('history')}>
          <History size={14} className="sidebar-icon" />History
        </a>
      </nav>

      {activeTab === 'assets' && (
        <>
          {assets.length === 0 && (
            <div className="sidebar-section">
              <div className="sidebar-empty">No assets yet<br />Generate or Import</div>
            </div>
          )}
          {assets.filter(a => a.type === 'image').length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">Images ({assets.filter(a => a.type === 'image').length})</div>
              <div className="sidebar-thumb-grid">
                {assets.filter(a => a.type === 'image').map((a) => (
                  <div key={a.id} className="sidebar-thumb" draggable
                    onDragStart={(e) => onDragStart(e, a)}
                    onDoubleClick={() => onDoubleClick(a.url)}>
                    <img src={a.url} alt={a.name} draggable={false} />
                    <span>{a.name.slice(0, 16)}</span>
                    <button className="sidebar-thumb-delete" onClick={(e) => { e.stopPropagation(); removeAsset(a.id) }} title="Remove">
                      <Trash2 size={10} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          {assets.filter(a => a.type === 'audio').length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">Audio ({assets.filter(a => a.type === 'audio').length})</div>
              {assets.filter(a => a.type === 'audio').map((a) => (
                <div key={a.id} className={`sidebar-clip ${playingId === a.id ? 'sidebar-clip--active' : ''}`}>
                  <button className="btn btn-ghost btn-sm" onClick={() => handlePlay(a)}>
                    {playingId === a.id ? '⏸' : <Play size={12} />}
                  </button>
                  <Music size={14} className="sidebar-icon" />
                  <span className="sidebar-clip-label">{a.name}</span>
                  <button className="btn btn-ghost btn-sm" onClick={() => removeAsset(a.id)} title="Remove">
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {activeTab === 'history' && (
        <div className="sidebar-section">
          {pastRuns.length === 0 ? (
            <div className="sidebar-empty">No run history</div>
          ) : (
            pastRuns.slice(0, 20).map((pr) => (
              <div key={pr.run_id} className="sidebar-clip"
                onClick={() => onNavigateHistory?.(pr.spec_name, pr.run_id)}>
                <span className="sidebar-clip-label">{pr.run_id.slice(0, 8)}</span>
                <span className={`seg-status seg-status--${pr.status === 'completed' ? 'ready' : 'failed'}`}>{pr.status}</span>
              </div>
            ))
          )}
        </div>
      )}
    </aside>
  )
}
