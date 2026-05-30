import { useState, useRef, useCallback } from 'react'
import { FolderOpen, Upload, History, Library, Plus, Music } from 'lucide-react'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'
import type { WorkflowRun, ArtifactRef } from '../../types'

type SidebarTab = 'assets' | 'uploads' | 'history' | 'library'

const TAB_ICONS: Record<SidebarTab, typeof FolderOpen> = {
  assets: FolderOpen, uploads: Upload, history: History, library: Library,
}

interface Props {
  run: WorkflowRun | null
  onNavigateHistory?: (specName: string, runId: string) => void
}

export function AssetSidebar({ run, onNavigateHistory }: Props) {
  const [activeTab, setActiveTab] = useState<SidebarTab>('assets')
  const [pastRuns] = useState<{ run_id: string; spec_name: string; status: string; created_at?: string }[]>(() => {
    try { return JSON.parse(localStorage.getItem('past_runs') || '[]') } catch { return [] }
  })
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; url: string; type: string }[]>([])
  const toast = useToastStore((s) => s.addToast)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setUploadedFiles((prev) => [{ name: file.name, url, type: file.type }, ...prev])
    if (file.type.startsWith('audio/')) {
      addAudioCue({
        track: 'sfx', start: 0, duration: 5,
        label: file.name.replace(/\.[^.]+$/, ''),
        audioUrl: url, volume: 0.8, waveformPeaks: null, sourceStepId: null,
      })
      toast('info', `Added "${file.name}" to audio tracks`)
    }
    if (fileRef.current) fileRef.current.value = ''
  }, [addAudioCue, toast])

  const artifacts: ArtifactRef[] = run ? Object.values(run.artifacts) : []
  const imageArtifacts = artifacts.filter((a) => a.media_type.startsWith('image/'))
  const audioArtifacts = artifacts.filter((a) => a.media_type.startsWith('audio/'))

  const navItems: { id: SidebarTab; label: string }[] = [
    { id: 'assets', label: 'Assets' },
    { id: 'uploads', label: 'Uploads' },
    { id: 'history', label: 'History' },
    { id: 'library', label: 'Library' },
  ]

  return (
    <aside className="workspace-sidebar">
      <div className="sidebar-section">
        <div className="sidebar-title">ASSET EXPLORER</div>
        <input ref={fileRef} type="file" style={{ display: 'none' }}
          accept="image/*,audio/*,video/*" onChange={handleFileUpload} />
        <button className="btn btn-primary btn-block" onClick={() => fileRef.current?.click()}>
          <Plus size={14} /> IMPORT
        </button>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = TAB_ICONS[item.id]
          return (
            <a key={item.id}
              className={`sidebar-link ${activeTab === item.id ? 'sidebar-link--active' : ''}`}
              onClick={() => setActiveTab(item.id)}>
              <Icon size={14} className="sidebar-icon" />{item.label}
            </a>
          )
        })}
        {activeTab === 'uploads' && (
          <button className="sidebar-link" onClick={() => fileRef.current?.click()}>
            <Plus size={14} className="sidebar-icon" />Import File
          </button>
        )}
      </nav>

      {activeTab === 'assets' && (
        <>
          {imageArtifacts.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">IMAGES ({imageArtifacts.length})</div>
              <div className="sidebar-thumb-grid">
                {imageArtifacts.map((art) => {
                  const ext = art.media_type === 'image/png' ? 'png' : art.name.includes('.') ? art.name.split('.').pop() : 'png'
                  const url = run ? `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${art.step_id}/${art.name.includes('.') ? art.name : art.name + '.' + ext}` : ''
                  return (
                    <div key={art.name + art.step_id} className="sidebar-thumb">
                      <img src={url} alt={art.step_id} />
                      <span>{art.step_id.replace(/_/g, ' ')}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
          {audioArtifacts.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-subtitle">AUDIO ({audioArtifacts.length})</div>
              {audioArtifacts.map((art) => {
                const ext = 'wav'
                const url = run ? `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${art.step_id}/${art.name}.${ext}` : ''
                return (
                  <div key={art.name + art.step_id} className="sidebar-clip">
                    <Music size={14} className="sidebar-icon" />
                    <span className="sidebar-clip-label">{art.step_id.replace(/_/g, ' ')}</span>
                    <audio src={url} controls className="sidebar-audio" />
                  </div>
                )
              })}
            </div>
          )}
          {artifacts.length === 0 && (
            <div className="sidebar-section">
              <div className="sidebar-empty">No assets yet — generate or import</div>
            </div>
          )}
        </>
      )}

      {activeTab === 'uploads' && (
        <div className="sidebar-section">
          <div className="sidebar-subtitle">UPLOADED</div>
          {uploadedFiles.length === 0 ? (
            <div className="sidebar-empty">Drop files or click IMPORT</div>
          ) : (
            uploadedFiles.map((f) => (
              <div key={f.name} className="sidebar-clip">
                <span className="sidebar-icon">{f.type.startsWith('image/') ? 'IMG' : f.type.startsWith('audio/') ? 'AUD' : 'FILE'}</span>
                <span className="sidebar-clip-label">{f.name}</span>
                <a href={f.url} download className="sidebar-clip-dur">↓</a>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'history' && (
        <div className="sidebar-section">
          <div className="sidebar-subtitle">PAST RUNS</div>
          {pastRuns.length === 0 ? (
            <div className="sidebar-empty">No history yet</div>
          ) : (
            pastRuns.slice(0, 20).map((pr) => (
              <div key={pr.run_id} className="sidebar-clip"
                onClick={() => onNavigateHistory?.(pr.spec_name, pr.run_id)}>
                <span className={`seg-status seg-status--${pr.status === 'completed' ? 'ready' : pr.status === 'failed' ? 'failed' : 'generating'}`}>●</span>
                <span className="sidebar-clip-label">{pr.run_id.slice(0, 8)}</span>
                <span className="sidebar-clip-dur">{pr.status}</span>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'library' && (
        <div className="sidebar-section">
          <div className="sidebar-subtitle">ALL FILES</div>
          {artifacts.length === 0 ? (
            <div className="sidebar-empty">No files in this run</div>
          ) : (
            artifacts.map((art) => (
              <div key={art.name + art.step_id} className="sidebar-clip">
                <span className="sidebar-icon">{art.media_type.startsWith('image/') ? 'IMG' : art.media_type.startsWith('audio/') ? 'AUD' : art.media_type.startsWith('video/') ? 'VID' : 'FILE'}</span>
                <span className="sidebar-clip-label">{art.step_id}/{art.name}</span>
                <span className="sidebar-clip-dur">{(art.size_bytes / 1024).toFixed(0)}KB</span>
              </div>
            ))
          )}
        </div>
      )}
    </aside>
  )
}
