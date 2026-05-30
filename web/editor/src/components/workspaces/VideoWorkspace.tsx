import { useState, useCallback } from 'react'
import { useTimelineStore } from '../../stores/timeline'
import { useWorkflowStore } from '../../stores/workflow'
import { useToastStore } from '../../stores/toast'
import { executeStep, getRun } from '../../api'
import type { WorkflowRun } from '../../types'

export function VideoWorkspace({ run }: { run: WorkflowRun | null }) {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const removeSegment = useTimelineStore((s) => s.removeSegment)
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)

  const [model, setModel] = useState<'ltx' | 'wan'>('ltx')
  const [duration, setDuration] = useState(12.5)
  const [motion, setMotion] = useState(75)
  const [resolution, setResolution] = useState('1080p')
  const [generating, setGenerating] = useState(false)
  const [showBlendEditor, setShowBlendEditor] = useState(false)

  const selectedSegment = segments.find((s) => s.id === selectedSegmentId)

  const handleAddKeyframe = useCallback(() => {
    const seg = addSegment({ duration: 5, prompt: '', status: 'empty' })
    setSelectedSegment(seg.id)
  }, [addSegment, setSelectedSegment])

  const handleUploadKeyImage = useCallback((segId: string, field: 'firstFrameB64' | 'lastFrameB64') => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => updateSegment(segId, { [field]: reader.result as string })
    reader.readAsDataURL(file)
  }, [updateSegment])

  const handleGenerateSegment = useCallback(async (segId: string) => {
    const seg = segments.find((s) => s.id === segId)
    if (!seg || !run) return
    setGenerating(true)
    updateSegment(segId, { status: 'generating' })
    try {
      const resolutionMap: Record<string, { w: number; h: number }> = {
        '720p': { w: 1280, h: 720 },
        '1080p': { w: 1920, h: 1080 },
        '4K_UHD': { w: 3840, h: 2160 },
      }
      const res = resolutionMap[resolution] || { w: 1920, h: 1080 }
      const result = await executeStep(run.spec_name, run.run_id, 'generate_video', {
        input_prompt: seg.prompt,
        image_b64: seg.firstFrameB64,
        image_end_b64: seg.lastFrameB64,
        seed: seg.params.seed,
        fps: seg.params.fps,
        frame_num: Math.round(seg.duration * seg.params.fps),
        guide_scale: seg.params.guideScale,
        width: res.w,
        height: res.h,
      }) as Record<string, unknown>
      if (result.status === 'error') {
        updateSegment(segId, { status: 'failed' })
        toast('error', String(result.error || 'Generation failed'))
        return
      }
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      const art = Object.values(updated.artifacts).find((a) => a.step_id === 'generate_video' && a.media_type.startsWith('video/'))
      if (art) {
        const url = `/v1/wf/${updated.spec_name}/runs/${updated.run_id}/artifacts/${art.step_id}/${art.name}.mp4`
        updateSegment(segId, { videoUrl: url, status: 'ready' })
      }
      toast('success', 'Segment generated')
    } catch (e) {
      updateSegment(segId, { status: 'failed' })
      toast('error', e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }, [segments, run, resolution, updateSegment, setRun, toast])

  const handleGenerateAll = useCallback(async () => {
    for (const seg of segments.filter((s) => s.firstFrameB64)) {
      await handleGenerateSegment(seg.id)
    }
  }, [segments, handleGenerateSegment])

  return (
    <div className="video-workspace">
      {/* Preview + Timeline */}
      <main className="video-main">
        <div className="video-preview">
          {selectedSegment?.videoUrl ? (
            <video src={selectedSegment.videoUrl} controls className="video-player" />
          ) : selectedSegment?.thumbnailUrl ? (
            <img src={selectedSegment.thumbnailUrl} alt="Preview" className="video-preview-img" />
          ) : (
            <div className="video-preview-empty">Add key frames and generate to preview video</div>
          )}
        </div>

        {/* Timeline */}
        <div className="video-timeline">
          <div className="timeline-header">
            <span>TIMELINE</span>
            <div className="timeline-zoom">
              <button className="btn btn-ghost btn-sm">−</button>
              <button className="btn btn-ghost btn-sm">+</button>
            </div>
          </div>
          <div className="timeline-tracks">
            {/* Video track */}
            <div className="timeline-track">
              <span className="track-label">V1_MASTER</span>
              <div className="track-segments">
                {segments.map((seg) => (
                  <div key={seg.id} className={`track-segment ${seg.id === selectedSegmentId ? 'track-segment--active' : ''} track-segment--${seg.status}`}
                    style={{ width: `${seg.duration * 60}px` }}
                    onClick={() => setSelectedSegment(seg.id)}>
                    {seg.thumbnailUrl && <img src={seg.thumbnailUrl} alt="" />}
                    <span className="seg-label">K_{String(seg.order + 1).padStart(2, '0')}</span>
                  </div>
                ))}
                <div className="track-segment track-segment--add" onClick={handleAddKeyframe}>+</div>
              </div>
            </div>
            {/* Audio track */}
            {audioCues.length > 0 && (
              <div className="timeline-track">
                <span className="track-label">A1_SFX</span>
                <div className="track-segments">
                  {audioCues.map((cue) => (
                    <div key={cue.id} className={`track-segment track-segment--audio track-segment--${cue.track}`}
                      style={{ width: `${cue.duration * 60}px`, left: `${cue.start * 60}px`, position: 'relative' }}>
                      <span className="seg-label">{cue.label.slice(0, 15)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Controls Panel */}
      <aside className="workspace-panel">
        <div className="panel-title">SEQUENCE EDITOR</div>

        {/* Model Selection */}
        <div className="panel-section">
          <label className="panel-label">GENERATOR MODEL</label>
          <div className="model-selector">
            <label className={`model-option ${model === 'ltx' ? 'model-option--active' : ''}`}>
              <input type="radio" name="vidmodel" checked={model === 'ltx'} onChange={() => setModel('ltx')} />
              <span>LTX Video (I2V)</span>
            </label>
            <label className={`model-option ${model === 'wan' ? 'model-option--active' : ''}`}>
              <input type="radio" name="vidmodel" checked={model === 'wan'} onChange={() => setModel('wan')} />
              <span>WAN (Generation)</span>
            </label>
          </div>
        </div>

        {/* Segment Detail */}
        {selectedSegment && (
          <div className="panel-section">
            <label className="panel-label">SEGMENT {selectedSegment.order + 1}</label>
            <div className="form-group">
              <label className="form-label">Prompt</label>
              <textarea className="form-input form-textarea" value={selectedSegment.prompt}
                onChange={(e) => updateSegment(selectedSegment.id, { prompt: e.target.value })}
                rows={3} placeholder="Describe the video segment..." />
            </div>
            <div className="form-row">
              <div className="form-group form-group-half">
                <label className="form-label">First frame</label>
                <input type="file" accept="image/*" style={{ display: 'none' }} id={`ff-${selectedSegment.id}`} onChange={handleUploadKeyImage(selectedSegment.id, 'firstFrameB64')} />
                <label htmlFor={`ff-${selectedSegment.id}`} className="btn btn-ghost btn-sm btn-block" style={{ cursor: 'pointer' }}>
                  {selectedSegment.firstFrameB64 ? 'Change' : 'Upload'}
                </label>
              </div>
              <div className="form-group form-group-half">
                <label className="form-label">Last frame</label>
                <input type="file" accept="image/*" style={{ display: 'none' }} id={`lf-${selectedSegment.id}`} onChange={handleUploadKeyImage(selectedSegment.id, 'lastFrameB64')} />
                <label htmlFor={`lf-${selectedSegment.id}`} className="btn btn-ghost btn-sm btn-block" style={{ cursor: 'pointer' }}>
                  {selectedSegment.lastFrameB64 ? 'Change' : 'Upload'}
                </label>
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Duration (s)</label>
              <input type="number" className="form-input" min={1} max={30} value={selectedSegment.duration}
                onChange={(e) => updateSegment(selectedSegment.id, { duration: parseFloat(e.target.value) || 5 })} />
            </div>
            <button className="btn btn-primary btn-block" disabled={generating || !selectedSegment.prompt || !selectedSegment.firstFrameB64}
              onClick={() => handleGenerateSegment(selectedSegment.id)}>
              {selectedSegment.status === 'generating' ? 'GENERATING...' : 'GENERATE SEGMENT'}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => removeSegment(selectedSegment.id)}>
              Delete segment
            </button>
          </div>
        )}

        {/* Sequence Controls */}
        <div className="panel-section">
          <label className="panel-label">SEQUENCE DURATION</label>
          <div className="param-row">
            <span>{duration}s</span>
            <input type="range" min={1} max={60} step={0.5} value={duration} onChange={(e) => setDuration(parseFloat(e.target.value))} />
          </div>
        </div>

        <div className="panel-section">
          <label className="panel-label">MOTION SCALE</label>
          <div className="param-row">
            <span>{motion}</span>
            <input type="range" min={0} max={100} value={motion} onChange={(e) => setMotion(parseInt(e.target.value))} />
          </div>
        </div>

        <div className="panel-section">
          <label className="panel-label">OUTPUT RESOLUTION</label>
          <div className="resolution-grid">
            {['720p', '1080p', '4K_UHD'].map((r) => (
              <button key={r} className={`btn btn-sm ${resolution === r ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setResolution(r)}>
                {r}
              </button>
            ))}
          </div>
        </div>

        {/* Post Processing */}
        <div className="panel-section">
          <label className="panel-label">POST-PROCESSING</label>
          <button className="btn btn-ghost btn-block" onClick={() => setShowBlendEditor(!showBlendEditor)}>
            Blends/Cuts
          </button>
          <button className="btn btn-ghost btn-block" onClick={() => toast('info', 'Color grading coming soon')}>
            Color Grading
          </button>
        </div>

        {/* Render */}
        <div className="panel-section">
          <button className="btn btn-primary btn-block" disabled={generating || segments.length === 0} onClick={handleGenerateAll}>
            RENDER SEQUENCE
          </button>
        </div>
      </aside>
    </div>
  )
}
