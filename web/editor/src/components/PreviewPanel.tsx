import { useRef, lazy, Suspense } from 'react'
import type { WorkflowRun } from '../types'
import { useWorkflowStore } from '../stores/workflow'
import { AudioPreview } from './AudioPreview'
import { KimodoEmbed } from './KimodoEmbed'

const Preview3D = lazy(() => import('./Preview3D').then((m) => ({ default: m.Preview3D })))

interface Props {
  run: WorkflowRun | null
}

export function PreviewPanel({ run }: Props) {
  const selectedStepId = useWorkflowStore((s) => s.selectedStepId)
  const videoRef = useRef<HTMLVideoElement>(null)

  if (!run || !selectedStepId) {
    return (
      <div className="preview-panel">
        <div className="panel-header">Preview</div>
        <div className="preview-empty">
          {run ? 'Select a step to preview its output' : 'Start a run to see previews'}
        </div>
      </div>
    )
  }

  const stepState = run.step_states[selectedStepId]
  const artifacts = Object.entries(run.artifacts)
    .filter(([k]) => k.startsWith(`${selectedStepId}.`))
    .map(([, v]) => v)

  if (!stepState || stepState.status !== 'completed' || artifacts.length === 0) {
    const showKimodo = selectedStepId === 'mesh_pose' && stepState?.status === 'waiting_input'
    return (
      <div className="preview-panel">
        <div className="panel-header">
          Preview — {selectedStepId.replace(/_/g, ' ')}
          {showKimodo && <span className="preview-meta">Kimodo Director</span>}
        </div>
        {showKimodo ? (
          <KimodoEmbed />
        ) : (
          <div className="preview-empty">
            {stepState?.status === 'running' ? 'Generating...' :
             stepState?.status === 'waiting_input' ? 'Waiting for input' :
             stepState?.status === 'failed' ? `Failed: ${stepState.error}` :
             'No output yet'}
          </div>
        )}
      </div>
    )
  }

  const artifact = artifacts[0]
  const artifactUrl = `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${artifact.step_id}/${artifact.name.includes('.') ? artifact.name : artifact.name + '.bin'}`
  const mediaType = artifact.media_type

  return (
    <div className="preview-panel">
      <div className="panel-header">
        Preview — {selectedStepId.replace(/_/g, ' ')}
        <div className="preview-actions">
          <span className="preview-meta">
            {media_type_label(mediaType)} &middot; {formatBytes(artifact.size_bytes)}
          </span>
          <a href={artifactUrl} download className="btn btn-ghost btn-sm" title="Download">
            Download
          </a>
          <a href={artifactUrl} target="_blank" rel="noreferrer" className="btn btn-ghost btn-sm" title="Open in new tab">
            Open
          </a>
        </div>
      </div>
      <div className="preview-content">
        {mediaType.startsWith('image/') && (
          <img src={artifactUrl} alt={selectedStepId} className="preview-image" />
        )}
        {mediaType.startsWith('video/') && (
          <video
            ref={videoRef}
            src={artifactUrl}
            controls
            autoPlay
            loop
            className="preview-video"
          />
        )}
        {mediaType.startsWith('audio/') && (
          <AudioPreview url={artifactUrl} />
        )}
        {(mediaType === 'model/gltf-binary' || mediaType === 'model/glb' || artifact.name.endsWith('.glb')) && (
          <Suspense fallback={<div className="preview-empty">Loading 3D viewer...</div>}>
            <Preview3D url={artifactUrl} />
          </Suspense>
        )}
        {!mediaType.startsWith('image/') && !mediaType.startsWith('video/') && !mediaType.startsWith('audio/') && mediaType !== 'model/gltf-binary' && mediaType !== 'model/glb' && !artifact.name.endsWith('.glb') && (
          <div className="preview-unknown">
            <p>Media type: {mediaType}</p>
            <div style={{ display: 'flex', gap: 8 }}>
              <a href={artifactUrl} download className="btn btn-primary">Download</a>
              <a href={artifactUrl} target="_blank" rel="noreferrer" className="btn btn-secondary">Open</a>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function media_type_label(mt: string): string {
  const labels: Record<string, string> = {
    'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WebP',
    'video/mp4': 'MP4', 'video/webm': 'WebM',
    'audio/wav': 'WAV', 'audio/mp3': 'MP3', 'audio/ogg': 'OGG',
    'application/json': 'JSON', 'application/octet-stream': 'Binary',
    'model/gltf-binary': 'GLB',
  }
  return labels[mt] || mt
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
