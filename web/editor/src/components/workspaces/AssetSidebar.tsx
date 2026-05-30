import { useState, useRef, useCallback } from 'react'
import { FolderOpen, History, Plus, Music, Trash2, Play, Upload } from 'lucide-react'
import { useAssetStore } from '../../stores/assets'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'
import type { WorkflowRun } from '../../types'

type SidebarTab = 'assets' | 'uploads' | 'history'

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
  const images = assets.filter((a) => a.type === 'image')
  const audio = assets.filter((a) => a.type === 'audio')
  const uploaded = assets.filter((a) => a.source === 'uploaded')
  const uploadedImages = uploaded.filter((a) => a.type === 'image')
  const uploadedAudio = uploaded.filter((a) => a.type === 'audio')
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

  return (
    <aside className="workspace-sidebar">
      <div className="sidebar-section">
        <div className="sidebar-title">ASSETS</div>
        <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*,audio/*,video/*" onChange={handleImport} />
        <button className="btn btn-primary btn-block" onClick={() => fileRef.current?.click()}>
          <Plus size={14} /> IMPORT
        </button>
      </div>

      <nav className="sidebar-nav">
        <a className={`sidebar-link ${activeTab === 'assets' ? 'sidebar-link--active' : ''}`} onClick={() => setActiveTab('assets')}>
          <FolderOpen size={14} className="sidebar-icon" />{assets.length > 0 ? `Assets (${assets.length})` : 'Assets'}
        </a>
        <a className={`sidebar-link ${activeTab === 'uploads' ? 'sidebar-link--active' : ''}`} onClick={() => setActiveTab('uploads')}>
          <Upload size={14} className="sidebar-icon" />{uploaded.length > 0 ? `Uploads (${uploaded.length})` : 'Uploads'}
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
          {images.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">Images ({images.length})</div>
              <div className="sidebar-thumb-grid">
                {images.map((a) => (
                  <div key={a.id} className="sidebar-thumb">
                    <img src={a.url} alt={a.name} />
                    <span>{a.name.slice(0, 16)}</span>
                    <button className="sidebar-thumb-delete" onClick={(e) => { e.stopPropagation(); removeAsset(a.id) }} title="Remove">
                      <Trash2 size={10} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          {audio.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">Audio ({audio.length})</div>
              {audio.map((a) => (
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

      {activeTab === 'uploads' && (
        <>
          {uploaded.length === 0 ? (
            <div className="sidebar-section">
              <div className="sidebar-empty">Nothing uploaded yet<br />Use IMPORT to add files</div>
            </div>
          ) : (
            <>
              {uploadedImages.length > 0 && (
                <div className="sidebar-section">
                  <div className="sidebar-subtitle">Images ({uploadedImages.length})</div>
                  <div className="sidebar-thumb-grid">
                    {uploadedImages.map((a) => (
                      <div key={a.id} className="sidebar-thumb">
                        <img src={a.url} alt={a.name} />
                        <span>{a.name.slice(0, 16)}</span>
                        <button className="sidebar-thumb-delete" onClick={(e) => { e.stopPropagation(); removeAsset(a.id) }}>
                          <Trash2 size={10} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {uploadedAudio.length > 0 && (
                <div className="sidebar-section">
                  <div className="sidebar-subtitle">Audio ({uploadedAudio.length})</div>
                  {uploadedAudio.map((a) => (
                    <div key={a.id} className={`sidebar-clip ${playingId === a.id ? 'sidebar-clip--active' : ''}`}>
                      <button className="btn btn-ghost btn-sm" onClick={() => handlePlay(a)}>
                        {playingId === a.id ? '⏸' : <Play size={12} />}
                      </button>
                      <Music size={14} className="sidebar-icon" />
                      <span className="sidebar-clip-label">{a.name}</span>
                      <button className="btn btn-ghost btn-sm" onClick={() => removeAsset(a.id)}>
                        <Trash2 size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </>
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
