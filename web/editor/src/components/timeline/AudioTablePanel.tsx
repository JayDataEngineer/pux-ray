import { useCallback, useRef, useState } from 'react'
import { useTimelineStore } from '../../stores/timeline'
import { useWorkflowStore } from '../../stores/workflow'
import { useToastStore } from '../../stores/toast'
import { executeStep, getRun } from '../../api'

export function AudioTablePanel() {
  const audioCues = useTimelineStore((s) => s.audioCues)
  const audioTracks = useTimelineStore((s) => s.audioTracks)
  const updateAudioCue = useTimelineStore((s) => s.updateAudioCue)
  const removeAudioCue = useTimelineStore((s) => s.removeAudioCue)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const setSelectedAudioCue = useTimelineStore((s) => s.setSelectedAudioCue)
  const selectedAudioCueId = useTimelineStore((s) => s.selectedAudioCueId)
  const run = useWorkflowStore((s) => s.run)
  const setRun = useWorkflowStore((s) => s.setRun)
  const toast = useToastStore((s) => s.addToast)

  const [genPrompt, setGenPrompt] = useState('')
  const [genTrack, setGenTrack] = useState('sfx')
  const [genDuration, setGenDuration] = useState(5)
  const [generating, setGenerating] = useState(false)

  const fileRef = useRef<HTMLInputElement>(null)

  const handleGenerateAudio = useCallback(async () => {
    if (!run || !genPrompt) return
    setGenerating(true)
    try {
      const stepMap: Record<string, { step: string; model: string }> = {
        sfx: { step: 'sound_fx', model: 'moss/moss-soundeffect' },
        music: { step: 'music', model: 'tts/ace_step_v1_5' },
        voice: { step: 'voice', model: 'kokoro' },
      }
      const cfg = stepMap[genTrack] || stepMap.sfx
      const params: Record<string, unknown> = {
        input_prompt: genPrompt,
        duration_seconds: genDuration,
      }
      if (genTrack === 'voice') {
        params.text = genPrompt
        params.voice = 'af_bella'
      }
      const result = await executeStep(run.spec_name, run.run_id, cfg.step, params) as Record<string, unknown>
      if (result.status === 'error') {
        toast('error', String(result.error || 'Generation failed'))
        return
      }
      const updated = await getRun(run.spec_name, run.run_id)
      setRun(updated)
      // Find the new audio artifact and add as cue
      const artKey = `${cfg.step}.output`
      const art = updated.artifacts[artKey]
      if (art) {
        const ext = art.media_type.startsWith('audio/') ? '.wav' : '.bin'
        const filename = art.name.includes('.') ? art.name : art.name + ext
        const audioUrl = `/v1/wf/${updated.spec_name}/runs/${updated.run_id}/artifacts/${cfg.step}/${filename}`
        addAudioCue({
          track: genTrack,
          start: 0,
          duration: genDuration,
          label: genTrack === 'sfx' ? `SFX: ${genPrompt.slice(0, 30)}` :
                 genTrack === 'music' ? `Music: ${genPrompt.slice(0, 30)}` :
                 `Voice: ${genPrompt.slice(0, 30)}`,
          audioUrl,
          volume: genTrack === 'music' ? 0.4 : 0.8,
          waveformPeaks: null,
          sourceStepId: cfg.step,
        })
        toast('success', `${cfg.step} generated`)
        setGenPrompt('')
      }
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Audio generation failed')
    } finally {
      setGenerating(false)
    }
  }, [run, genPrompt, genTrack, genDuration, addAudioCue, setRun, toast])

  const handleVolumeChange = useCallback((id: string, volume: number) => {
    updateAudioCue(id, { volume })
  }, [updateAudioCue])

  const handleTimeChange = useCallback((id: string, start: number) => {
    updateAudioCue(id, { start: Math.max(0, start) })
  }, [updateAudioCue])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    addAudioCue({
      track: genTrack,
      start: 0,
      duration: 5,
      label: file.name.replace(/\.[^.]+$/, ''),
      audioUrl: url,
      volume: 0.8,
      waveformPeaks: null,
      sourceStepId: null,
    })
  }, [addAudioCue, genTrack])

  return (
    <div className="audio-table">
      <div className="audio-table-header">
        <span className="audio-table-title">Audio</span>
        <div className="audio-gen-row">
          <select className="form-input audio-track-select" value={genTrack} onChange={(e) => setGenTrack(e.target.value)}>
            <option value="sfx">SFX (MOSS)</option>
            <option value="music">Music (ACE-Step)</option>
            <option value="voice">Voice (Kokoro)</option>
          </select>
          <input
            className="form-input"
            placeholder={genTrack === 'sfx' ? 'e.g. rain and thunder' : genTrack === 'music' ? 'e.g. epic orchestral' : 'e.g. hello world'}
            value={genPrompt}
            onChange={(e) => setGenPrompt(e.target.value)}
          />
          <input
            className="form-input"
            type="number" min={1} max={30} value={genDuration}
            onChange={(e) => setGenDuration(parseInt(e.target.value) || 5)}
            title="Duration in seconds"
          />
          <button
            className="btn btn-primary btn-sm"
            disabled={generating || !genPrompt}
            onClick={handleGenerateAudio}
          >
            {generating ? '...' : 'Generate'}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => fileRef.current?.click()}>
            Upload
          </button>
          <input ref={fileRef} type="file" style={{ display: 'none' }} accept="audio/*" onChange={handleFileUpload} />
        </div>
      </div>
      {audioCues.length === 0 ? (
        <div className="audio-empty">No audio cues yet — generate or upload above</div>
      ) : (
      <table className="audio-table-grid">
        <thead>
          <tr>
            <th>Time</th>
            <th>Track</th>
            <th>Label</th>
            <th>Duration</th>
            <th>Volume</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {audioCues.map((cue) => {
            const track = audioTracks.find((t) => t.id === cue.track)
            const isSelected = cue.id === selectedAudioCueId
            return (
              <tr
                key={cue.id}
                className={`audio-table-row ${isSelected ? 'selected' : ''}`}
                onClick={() => setSelectedAudioCue(cue.id)}
              >
                <td>
                  <input
                    className="form-input audio-time-input"
                    type="number"
                    min={0}
                    step={0.1}
                    value={cue.start.toFixed(1)}
                    onChange={(e) => handleTimeChange(cue.id, parseFloat(e.target.value) || 0)}
                  />
                </td>
                <td>
                  <span className="audio-track-badge" style={{ color: track?.color }}>
                    {track?.label || cue.track}
                  </span>
                </td>
                <td>{cue.label}</td>
                <td>{cue.duration.toFixed(1)}s</td>
                <td>
                  <input
                    type="range"
                    className="audio-volume-slider"
                    min={0}
                    max={1}
                    step={0.05}
                    value={cue.volume}
                    onChange={(e) => handleVolumeChange(cue.id, parseFloat(e.target.value))}
                  />
                </td>
                <td>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={(e) => { e.stopPropagation(); removeAudioCue(cue.id) }}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      )}
    </div>
  )
}
