import { useCallback, useRef } from 'react'
import { useTimelineStore } from '../../stores/timeline'

export function AudioTablePanel() {
  const audioCues = useTimelineStore((s) => s.audioCues)
  const audioTracks = useTimelineStore((s) => s.audioTracks)
  const updateAudioCue = useTimelineStore((s) => s.updateAudioCue)
  const removeAudioCue = useTimelineStore((s) => s.removeAudioCue)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const setSelectedAudioCue = useTimelineStore((s) => s.setSelectedAudioCue)
  const selectedAudioCueId = useTimelineStore((s) => s.selectedAudioCueId)

  const fileRef = useRef<HTMLInputElement>(null)
  const [addTrack] = [audioTracks[0]?.id || 'sfx']

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
      track: addTrack,
      start: 0,
      duration: 5,
      label: file.name.replace(/\.[^.]+$/, ''),
      audioUrl: url,
      volume: 0.8,
      waveformPeaks: null,
      sourceStepId: null,
    })
  }, [addAudioCue, addTrack])

  if (audioCues.length === 0) return null

  return (
    <div className="audio-table">
      <div className="audio-table-header">
        <span className="audio-table-title">Audio Cues</span>
        <button className="btn btn-ghost btn-sm" onClick={() => fileRef.current?.click()}>
          + Add Audio
        </button>
        <input ref={fileRef} type="file" style={{ display: 'none' }} accept="audio/*" onChange={handleFileUpload} />
      </div>
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
    </div>
  )
}
