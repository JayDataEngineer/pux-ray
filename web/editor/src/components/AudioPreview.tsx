import { useEffect, useRef } from 'react'
import WaveSurfer from 'wavesurfer.js'

interface Props {
  url: string
}

export function AudioPreview({ url }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#818cf8',
      progressColor: '#6366f1',
      cursorColor: '#a78bfa',
      height: 128,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    })
    wsRef.current = ws
    ws.load(url)

    return () => {
      ws.destroy()
    }
  }, [url])

  return (
    <div className="audio-preview">
      <div ref={containerRef} className="audio-waveform" />
      <div className="audio-controls">
        <button
          className="btn btn-primary btn-sm"
          onClick={() => wsRef.current?.playPause()}
        >
          Play / Pause
        </button>
      </div>
    </div>
  )
}
