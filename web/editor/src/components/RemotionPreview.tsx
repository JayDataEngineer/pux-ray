import { useRef, useEffect, useState, useCallback } from 'react'
import { Player, type PlayerRef } from '@remotion/player'
import { AbsoluteFill, Html5Video, Html5Audio, Sequence } from 'remotion'

export interface AudioTrack {
  url: string
  startFrame: number
  durationFrames: number
  volume?: number
}

export interface VideoCompositionProps {
  videoUrl: string
  audioTracks: AudioTrack[]
}

function VideoComposition({ videoUrl, audioTracks }: VideoCompositionProps) {
  return (
    <AbsoluteFill style={{ backgroundColor: '#000' }}>
      <Html5Video
        src={videoUrl}
        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
      />
      {audioTracks.map((track, i) => (
        <Sequence key={i} from={track.startFrame} durationInFrames={track.durationFrames}>
          <Html5Audio src={track.url} volume={track.volume ?? 1} />
        </Sequence>
      ))}
    </AbsoluteFill>
  )
}

interface UseVideoMetaResult {
  duration: number
  fps: number
  width: number
  height: number
  loading: boolean
}

function useVideoMeta(url: string): UseVideoMetaResult {
  const [meta, setMeta] = useState<UseVideoMetaResult>({ duration: 0, fps: 24, width: 1920, height: 1080, loading: true })

  useEffect(() => {
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.src = url
    v.onloadedmetadata = () => {
      setMeta({
        duration: v.duration,
        fps: 24,
        width: v.videoWidth || 1920,
        height: v.videoHeight || 1080,
        loading: false,
      })
      v.remove()
    }
    v.onerror = () => setMeta((m) => ({ ...m, loading: false }))
    return () => v.remove()
  }, [url])

  return meta
}

export function useCurrentPlayerFrame(ref: React.RefObject<PlayerRef | null>): number {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const el = ref.current
      if (!el) return () => {}
      const handler = () => onChange()
      el.addEventListener('frameupdate', handler)
      return () => el.removeEventListener('frameupdate', handler)
    },
    [ref],
  )
  const getFrame = useCallback(() => ref.current?.getCurrentFrame() ?? 0, [ref])
  return React.useSyncExternalStore(subscribe, getFrame, getFrame)
}

import React from 'react'

interface Props {
  videoUrl: string
  audioTracks?: AudioTrack[]
  fps?: number
}

export function RemotionPreview({ videoUrl, audioTracks = [], fps: inputFps }: Props) {
  const playerRef = useRef<PlayerRef>(null)
  const meta = useVideoMeta(videoUrl)
  const [frame, setFrame] = useState(0)

  const fps = inputFps ?? meta.fps
  const totalFrames = Math.ceil(meta.duration * fps) || 1

  useEffect(() => {
    const el = playerRef.current
    if (!el) return
    const onUpdate = (e: { detail: { frame: number } }) => setFrame(e.detail.frame)
    el.addEventListener('frameupdate', onUpdate)
    return () => { el.removeEventListener('frameupdate', onUpdate) }
  }, [meta.loading])

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = Math.round(Number(e.target.value))
    playerRef.current?.seekTo(f)
  }, [])

  if (meta.loading) {
    return <div className="preview-empty">Loading video metadata...</div>
  }

  const time = frame / fps

  return (
    <div className="remotion-preview">
      <Player
        ref={playerRef}
        component={VideoComposition}
        inputProps={{ videoUrl, audioTracks }}
        durationInFrames={totalFrames}
        fps={fps}
        compositionWidth={meta.width}
        compositionHeight={meta.height}
        controls
        style={{ width: '100%' }}
      />
      <div className="video-scrubber">
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={frame}
          onChange={handleScrub}
          className="scrubber-range"
        />
        <div className="scrubber-info">
          <span>{formatTime(time)} / {formatTime(meta.duration)}</span>
          <span className="scrubber-frame">Frame {frame}</span>
        </div>
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
