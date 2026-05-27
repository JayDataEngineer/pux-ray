import type { ArtifactRef } from '../types'

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
  status: string
  durationMs: number | null
  error: string | null
  artifacts: ArtifactRef[]
  specName: string
  runId: string
  selected: boolean
  onClick: () => void
}

export function StepCard({ stepId, status, durationMs, error, artifacts, specName, runId, selected, onClick }: Props) {
  const label = STEP_LABELS[stepId] || stepId
  const icon = STATUS_ICONS[status] || '○'
  const hasArtifact = artifacts.length > 0 && status === 'completed'
  const artifact = hasArtifact ? artifacts[0] : null
  const thumbUrl = artifact
    ? `/v1/wf/${specName}/runs/${runId}/artifacts/${artifact.step_id}/${artifact.name.includes('.') ? artifact.name : artifact.name + '.png'}`
    : null

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
      {status === 'waiting_input' && (
        <div className="step-waiting">Waiting for input...</div>
      )}
    </div>
  )
}
