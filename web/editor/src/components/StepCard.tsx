import { useRef, useState } from 'react'
import type { ArtifactRef } from '../types'
import { approveStep, continueStep, getRun, loadKimodo, kimodoUrl } from '../api'
import { useWorkflowStore } from '../stores/workflow'

const STATUS_ICONS: Record<string, string> = {
  pending: '○',
  running: '●',
  waiting_input: '⏳',
  completed: '✓',
  failed: '✗',
  skipped: '—',
}

const STEP_LABELS: Record<string, string> = {
  generate_character: 'Character',
  mesh_pose: 'Mesh Pose',
  scene_compose: 'Scene',
  generate_video: 'Video',
  voice: 'Voice',
  sound_fx: 'SFX',
  music: 'Music',
  mix_audio: 'Mix Audio',
  lipsync: 'Lip Sync',
  video_edit: 'Edit',
  upscale: 'Upscale',
}

export interface SourceRef {
  stepId: string
  outputKey: string
  thumbnailUrl: string | null
}

interface Props {
  stepId: string
  stepType: string
  interaction?: string | null
  status: string
  durationMs: number | null
  error: string | null
  artifacts: ArtifactRef[]
  sourceArtifacts: SourceRef[]
  specName: string
  runId: string
  selected: boolean
  onClick: () => void
  onExecute?: (stepId: string) => void
  canExecute?: boolean
}

export function StepCard({ stepId, stepType, interaction, status, durationMs, error, artifacts, sourceArtifacts, specName, runId, selected, onClick, onExecute, canExecute }: Props) {
  const label = STEP_LABELS[stepId] || stepId.replace(/_/g, ' ')
  const icon = STATUS_ICONS[status] || '○'
  const hasArtifact = artifacts.length > 0
  const artifact = hasArtifact ? artifacts[0] : null
  const extMap: Record<string, string> = {
    'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp',
    'video/mp4': 'mp4', 'video/webm': 'webm',
    'audio/wav': 'wav', 'audio/mp3': 'mp3',
    'model/gltf-binary': 'glb', 'application/json': 'json',
  }
  const ext = artifact ? (extMap[artifact.media_type] || 'bin') : 'bin'
  const thumbUrl = artifact && runId
    ? `/v1/wf/${specName}/runs/${runId}/artifacts/${artifact.step_id}/${artifact.name.includes('.') ? artifact.name : artifact.name + '.' + ext}`
    : null
  const isReview = interaction === 'review'
  const hasSourceThumbs = sourceArtifacts.some((s) => s.thumbnailUrl !== null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [continuing, setContinuing] = useState(false)
  const [kimodoLoading, setKimodoLoading] = useState(false)
  const setRun = useWorkflowStore((s) => s.setRun)

  const isExternalTool = stepType === 'external_wait'
  const isKimodo = stepId === 'mesh_pose'

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !runId) return
    setUploading(true)
    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          const dataUrl = reader.result as string
          resolve(dataUrl.split(',')[1] || '')
        }
        reader.onerror = reject
        reader.readAsDataURL(file)
      })
      await approveStep(specName, runId, stepId, {
        file_data: base64,
        name: file.name,
        media_type: file.type || 'application/octet-stream',
      })
      const updated = await getRun(specName, runId)
      setRun(updated)
    } catch (err) {
      console.error('Upload failed:', err)
    } finally {
      setUploading(false)
    }
  }

  const handleContinue = async () => {
    if (!runId) return
    setContinuing(true)
    try {
      await continueStep(specName, runId, stepId)
      const updated = await getRun(specName, runId)
      setRun(updated)
    } catch (err) {
      console.error('Continue failed:', err)
    } finally {
      setContinuing(false)
    }
  }

  const handleLaunchKimodo = async () => {
    setKimodoLoading(true)
    try {
      await loadKimodo()
      window.open(kimodoUrl(), '_blank')
    } catch (err) {
      console.error('Kimodo launch failed:', err)
    } finally {
      setKimodoLoading(false)
    }
  }

  return (
    <div
      className={`step-card step-card--${status} ${selected ? 'step-card--selected' : ''}`}
      onClick={onClick}
    >
      <div className="step-card-header">
        <span className={`step-icon step-icon--${status}`}>{icon}</span>
        <span className="step-label">{label}</span>
        {durationMs != null && <span className="step-duration">{(durationMs / 1000).toFixed(1)}s</span>}
      </div>

      {hasSourceThumbs && !thumbUrl && (
        <div className="step-chain">
          {sourceArtifacts.filter((s) => s.thumbnailUrl).map((src) => (
            <div key={`${src.stepId}.${src.outputKey}`} className="chain-thumb" title={`from ${STEP_LABELS[src.stepId] || src.stepId}`}>
              <img src={src.thumbnailUrl!} alt={src.stepId} />
            </div>
          ))}
        </div>
      )}

      {thumbUrl && (
        <div className="step-thumbnail">
          <img src={thumbUrl} alt={stepId} loading="lazy" />
        </div>
      )}

      {status === 'failed' && error && (
        <div className="step-error">{error}</div>
      )}

      {status === 'pending' && canExecute && runId && (
        <div className="step-upload">
          <button
            className="btn btn-primary btn-sm"
            onClick={(e) => { e.stopPropagation(); onExecute?.(stepId) }}
          >
            Run Step
          </button>
        </div>
      )}

      {status === 'waiting_input' && runId && (
        <div className="step-upload">
          {isKimodo && (
            <button
              className="btn btn-primary btn-sm"
              disabled={kimodoLoading}
              onClick={(e) => { e.stopPropagation(); handleLaunchKimodo() }}
            >
              {kimodoLoading ? 'Starting Kimodo...' : 'Open Kimodo Director'}
            </button>
          )}
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            onChange={handleUpload}
            accept="image/*,video/*,audio/*,.json,.npz,.bvh,.glb"
          />
          {isReview ? (
            <>
              <button
                className="btn btn-primary btn-sm"
                disabled={continuing}
                onClick={(e) => { e.stopPropagation(); handleContinue() }}
              >
                {continuing ? 'Continuing...' : 'Approve & Continue'}
              </button>
              <button
                className="btn btn-ghost btn-sm"
                disabled={uploading}
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
              >
                {uploading ? 'Uploading...' : 'Upload Custom'}
              </button>
            </>
          ) : (
            <>
              {!isKimodo && (
                <button
                  className="btn btn-primary btn-sm"
                  disabled={uploading}
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                >
                  {uploading ? 'Uploading...' : 'Upload File'}
                </button>
              )}
              {isExternalTool && (
                <button
                  className="btn btn-secondary btn-sm"
                  disabled={uploading}
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                >
                  {uploading ? 'Uploading...' : 'Upload result'}
                </button>
              )}
              {isKimodo && (
                <span className="upload-hint">Pose the character in Kimodo, then upload the result image</span>
              )}
              {isExternalTool && !isKimodo && (
                <span className="upload-hint">Upload output from external tool</span>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
