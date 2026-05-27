import { useCallback } from 'react'
import { useTimelineStore } from '../../stores/timeline'
import { useAudioPlayback } from '../../hooks/useAudioPlayback'

export function TimelineToolbar() {
  const viewport = useTimelineStore((s) => s.viewport)
  const playback = useTimelineStore((s) => s.playback)
  const segments = useTimelineStore((s) => s.segments)
  const setViewport = useTimelineStore((s) => s.setViewport)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const { play, pause } = useAudioPlayback()

  const handleZoom = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setViewport({ pixelsPerSecond: Number(e.target.value) })
  }, [setViewport])

  const handleZoomIn = useCallback(() => {
    setViewport({ pixelsPerSecond: Math.min(400, viewport.pixelsPerSecond * 1.3) })
  }, [viewport.pixelsPerSecond, setViewport])

  const handleZoomOut = useCallback(() => {
    setViewport({ pixelsPerSecond: Math.max(20, viewport.pixelsPerSecond / 1.3) })
  }, [viewport.pixelsPerSecond, setViewport])

  const handleFitAll = useCallback(() => {
    if (segments.length === 0) return
    const total = segments.reduce((t, s) => Math.max(t, s.start + s.duration), 0)
    const canvasW = viewport.canvasWidth - 80 - 20 // minus label width and padding
    if (total > 0 && canvasW > 0) {
      setViewport({ pixelsPerSecond: canvasW / total, scrollX: 0 })
    }
  }, [segments, viewport.canvasWidth, setViewport])

  const handlePlayPause = useCallback(() => {
    if (playback.isPlaying) {
      pause()
    } else {
      play()
    }
  }, [playback.isPlaying, play, pause])

  const handleAddSegment = useCallback(() => {
    addSegment()
  }, [addSegment])

  return (
    <div className="timeline-toolbar">
      <div className="timeline-toolbar-left">
        <button className="btn btn-ghost btn-sm" onClick={handlePlayPause} title={playback.isPlaying ? 'Pause' : 'Play'}>
          {playback.isPlaying ? '⏸' : '▶'}
        </button>
        <span className="timeline-time-display">
          {formatTime(playback.currentTime)} / {formatTime(playback.totalDuration)}
        </span>
      </div>
      <div className="timeline-toolbar-center">
        <span className="timeline-toolbar-label">{segments.length} segment{segments.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="timeline-toolbar-right">
        <button className="btn btn-ghost btn-sm" onClick={handleAddSegment}>+ Segment</button>
        <button className="btn btn-ghost btn-sm" onClick={handleZoomOut} title="Zoom out">−</button>
        <input
          type="range"
          className="timeline-zoom-slider"
          min={20}
          max={400}
          value={viewport.pixelsPerSecond}
          onChange={handleZoom}
        />
        <button className="btn btn-ghost btn-sm" onClick={handleZoomIn} title="Zoom in">+</button>
        <button className="btn btn-ghost btn-sm" onClick={handleFitAll} title="Fit all">⊞</button>
      </div>
    </div>
  )
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  const ms = Math.floor((s % 1) * 10)
  return m > 0 ? `${m}:${String(sec).padStart(2, '0')}.${ms}` : `${sec}.${ms}s`
}
