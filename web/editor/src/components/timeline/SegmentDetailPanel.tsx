import { useState, useRef, useCallback } from 'react'
import { useTimelineStore } from '../../stores/timeline'
import { executeStep, getRun } from '../../api'
import { useWorkflowStore } from '../../stores/workflow'
import { useToastStore } from '../../stores/toast'

interface Props {
  segmentId: string
}

type Tab = 'prompt' | 'edit' | 'sync'

export function SegmentDetailPanel({ segmentId }: Props) {
  const segment = useTimelineStore((s) => s.segments.find((seg) => seg.id === segmentId))
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const removeSegment = useTimelineStore((s) => s.removeSegment)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const run = useWorkflowStore((s) => s.run)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)
  const audioCues = useTimelineStore((s) => s.audioCues)

  const [tab, setTab] = useState<Tab>('prompt')
  const [editPrompt, setEditPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const handlePromptChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    updateSegment(segmentId, { prompt: e.target.value })
  }, [segmentId, updateSegment])

  const handleImageUpload = useCallback((field: 'firstFrameB64' | 'lastFrameB64') => (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result as string
      updateSegment(segmentId, { [field]: base64 })
    }
    reader.readAsDataURL(file)
  }, [segmentId, updateSegment])

  const handleGenerate = useCallback(async () => {
    if (!segment || !run) return
    setLoading(true)
    updateSegment(segmentId, { status: 'generating' })
    try {
      const result = await executeStep(run.spec_name, run.run_id, 'generate_video', {
        input_prompt: segment.prompt,
        image_b64: segment.firstFrameB64,
        image_end_b64: segment.lastFrameB64,
        seed: segment.params.seed,
        fps: segment.params.fps,
        frame_num: segment.params.frames,
        guide_scale: segment.params.guideScale,
        width: segment.params.width,
        height: segment.params.height,
      }) as Record<string, unknown>
      if (result.status !== 'error') {
        const updated = await getRun(run.spec_name, run.run_id)
        setRun(updated)
        toast('success', 'Segment generated')
      }
    } catch (err) {
      updateSegment(segmentId, { status: 'failed' })
      toast('error', err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setLoading(false)
    }
  }, [segment, run, segmentId])

  const handleLanceEdit = useCallback(async () => {
    if (!segment?.videoUrl || !run || !editPrompt) return
    setLoading(true)
    try {
      await executeStep(run.spec_name, run.run_id, 'video_edit', {
        video: segment.videoUrl,
        prompt: editPrompt,
      })
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      toast('success', 'Video edit applied')
    } catch (err) {
      toast('error', err instanceof Error ? err.message : 'Edit failed')
    } finally {
      setLoading(false)
    }
  }, [segment, run, editPrompt])

  const handleLipSync = useCallback(async (cueId: string) => {
    if (!segment?.videoUrl || !run) return
    const cue = audioCues.find((c) => c.id === cueId)
    if (!cue?.audioUrl) return
    setLoading(true)
    try {
      await executeStep(run.spec_name, run.run_id, 'lipsync', {
        video: segment.videoUrl,
        audio: cue.audioUrl,
        mode: 'lipsync',
      })
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      toast('success', 'Lip sync applied')
    } catch (err) {
      toast('error', err instanceof Error ? err.message : 'Lip sync failed')
    } finally {
      setLoading(false)
    }
  }, [segment, run, audioCues])

  if (!segment) return null

  const hasVideo = !!segment.videoUrl
  const hasAudio = audioCues.some((c) => c.audioUrl)

  return (
    <div className="segment-detail">
      <div className="segment-detail-header">
        <span>Segment {segment.order + 1}</span>
        <button className="btn btn-ghost btn-sm" onClick={() => setSelectedSegment(null)}>✕</button>
      </div>

      <div className="segment-detail-tabs">
        <button className={`tab-btn ${tab === 'prompt' ? 'active' : ''}`} onClick={() => setTab('prompt')}>Prompt</button>
        {hasVideo && <button className={`tab-btn ${tab === 'edit' ? 'active' : ''}`} onClick={() => setTab('edit')}>Edit</button>}
        {hasVideo && hasAudio && <button className={`tab-btn ${tab === 'sync' ? 'active' : ''}`} onClick={() => setTab('sync')}>Sync</button>}
      </div>

      <div className="segment-detail-body">
        {tab === 'prompt' && (
          <>
            <div className="form-group">
              <label className="form-label">Prompt</label>
              <textarea
                className="form-input form-textarea"
                value={segment.prompt}
                onChange={handlePromptChange}
                rows={3}
                placeholder="Describe the video..."
              />
            </div>
            <div className="form-row">
              <div className="form-group form-group-half">
                <label className="form-label">First frame</label>
                <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*" onChange={handleImageUpload('firstFrameB64')} />
                <button className="btn btn-ghost btn-sm btn-block" onClick={() => fileRef.current?.click()}>
                  {segment.firstFrameB64 ? 'Change' : 'Upload'}
                </button>
              </div>
              <div className="form-group form-group-half">
                <label className="form-label">Last frame</label>
                <input ref={fileRef} type="file" style={{ display: 'none' }} accept="image/*" onChange={handleImageUpload('lastFrameB64')} />
                <button className="btn btn-ghost btn-sm btn-block" onClick={() => fileRef.current?.click()}>
                  {segment.lastFrameB64 ? 'Change' : 'Upload'}
                </button>
              </div>
            </div>
            <div className="form-row">
              <div className="form-group form-group-half">
                <label className="form-label">Duration (s)</label>
                <input
                  className="form-input"
                  type="number"
                  min={1}
                  max={30}
                  step={0.5}
                  value={segment.duration}
                  onChange={(e) => updateSegment(segmentId, { duration: parseFloat(e.target.value) || 5 })}
                />
              </div>
              <div className="form-group form-group-half">
                <label className="form-label">Seed</label>
                <input
                  className="form-input"
                  type="number"
                  value={segment.params.seed}
                  onChange={(e) => updateSegment(segmentId, { params: { ...segment.params, seed: parseInt(e.target.value) || 42 } })}
                />
              </div>
            </div>
            <div className="form-actions">
              <button
                className="btn btn-primary btn-block"
                disabled={loading || !segment.prompt}
                onClick={handleGenerate}
              >
                {loading ? 'Generating...' : 'Generate'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => removeSegment(segmentId)}>
                Delete segment
              </button>
            </div>
          </>
        )}

        {tab === 'edit' && hasVideo && (
          <>
            <div className="form-group">
              <label className="form-label">Edit instruction</label>
              <textarea
                className="form-input form-textarea"
                value={editPrompt}
                onChange={(e) => setEditPrompt(e.target.value)}
                rows={3}
                placeholder="e.g. enhance visual quality, fix artifacts..."
              />
            </div>
            <button
              className="btn btn-primary btn-block"
              disabled={loading || !editPrompt}
              onClick={handleLanceEdit}
            >
              {loading ? 'Applying...' : 'Run Lance Edit'}
            </button>
          </>
        )}

        {tab === 'sync' && hasVideo && hasAudio && (
          <>
            <div className="form-group">
              <label className="form-label">Audio source</label>
              <select
                className="form-input"
                onChange={(e) => {
                  if (e.target.value) handleLipSync(e.target.value)
                }}
              >
                <option value="">Select audio cue...</option>
                {audioCues.filter((c) => c.audioUrl).map((cue) => (
                  <option key={cue.id} value={cue.id}>
                    {cue.label} ({cue.track})
                  </option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-secondary btn-block"
              disabled={loading}
              onClick={() => {
                const firstCue = audioCues.find((c) => c.audioUrl)
                if (firstCue) handleLipSync(firstCue.id)
              }}
            >
              {loading ? 'Syncing...' : 'Auto Lip Sync'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}
