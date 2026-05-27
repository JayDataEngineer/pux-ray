import { useRef } from 'react'
import type { WorkflowRun } from '../types'
import { useWorkflowStore } from '../stores/workflow'
import { AudioPreview } from './AudioPreview'

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
    return (
      <div className="preview-panel">
        <div className="panel-header">Preview — {selectedStepId}</div>
        <div className="preview-empty">
          {stepState?.status === 'running' ? 'Generating...' :
           stepState?.status === 'waiting_input' ? 'Waiting for input' :
           stepState?.status === 'failed' ? `Failed: ${stepState.error}` :
           'No output yet'}
        </div>
      </div>
    )
  }

  const artifact = artifacts[0]
  const artifactUrl = `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${artifact.step_id}/${artifact.name.includes('.') ? artifact.name : artifact.name + '.bin'}`
  const mediaType = artifact.media_type

  return (
    <div className="preview-panel">
      <div className="panel-header">
        Preview — {selectedStepId}
        <span className="preview-meta">
          {mediaType} &middot; {(artifact.size_bytes / 1024).toFixed(0)} KB
        </span>
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
        {!mediaType.startsWith('image/') && !mediaType.startsWith('video/') && !mediaType.startsWith('audio/') && (
          <div className="preview-unknown">
            <p>Media type: {mediaType}</p>
            <a href={artifactUrl} target="_blank" rel="noreferrer" className="btn btn-primary">
              Download
            </a>
          </div>
        )}
      </div>
    </div>
  )
}
