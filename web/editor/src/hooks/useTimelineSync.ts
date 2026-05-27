import { useEffect, useCallback } from 'react'
import { useTimelineStore } from '../stores/timeline'

export function useTimelineSync(playerSeekTo?: (frame: number) => void, fps: number = 24) {
  const setPlayback = useTimelineStore((s) => s.setPlayback)
  const playback = useTimelineStore((s) => s.playback)

  const seekTimeline = useCallback((time: number) => {
    setPlayback({ currentTime: time })
  }, [setPlayback])

  // Sync timeline playhead → video preview
  useEffect(() => {
    if (playerSeekTo && playback.currentTime >= 0) {
      const frame = Math.round(playback.currentTime * fps)
      playerSeekTo(frame)
    }
  }, [playback.currentTime, playerSeekTo, fps])

  return { seekTimeline }
}
