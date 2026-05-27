import { useRef, useState } from 'react'
import type { ArtifactRef } from '../types'
import { approveStep, continueStep, getRun } from '../api'
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

interface Props {
  stepId: string
  stepType: string
  interaction?: string | null
  status: string
  durationMs: number | null
  error: string | null
  artifacts: ArtifactRef[]
  specName: string
  runId: string
  selected: boolean
  onClick: () => void
}

export function StepCard({ stepId, stepType, interaction, status, durationMs, error, artifacts, specName, runId, selected, onClick }: Props) {
  const label = STEP_LABELS[stepId] || stepId.replace(/_/g, ' ')
  const icon = STATUS_ICONS[status] || '○'
  const hasArtifact = artifacts.length > 0
  const artifact = hasArtifact ? artifacts[0] : null
  const thumbUrl = artifact && runId
    ? `/v1/wf/${specName}/runs/${runId}/artifacts/${artifact.step_id}/${artifact.name.includes('.') ? artifact.name : artifact.name + '.png'}`
    : null
  const isReview = interaction === 'review'

  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [continuing, setContinuing] = useState(false)
  const setRun = useWorkflowStore((s) => s.setRun)

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !runId) return
    setUploading(true)
    try {
      const data = await file.arrayBuffer()
      const base64 = btoa(String.fromCharCode(...new Uint8Array(data)))
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
      {thumbUrl && (
        <div className="step-thumbnail">
          <img src={thumbUrl} alt={stepId} loading="lazy" />
        </div>
      )}
      {status === 'failed' && error && (
        <div className="step-error">{error}</div>
      )}
      {status === 'waiting_input' && runId && (
        <div className="step-upload">
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
              <button
                className="btn btn-primary btn-sm"
                disabled={uploading}
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
              >
                {uploading ? 'Uploading...' : 'Upload File'}
              </button>
              {stepType === 'external_wait' && (
                <span className="upload-hint">External tool output</span>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
