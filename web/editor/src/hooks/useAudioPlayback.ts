import { useCallback, useRef, useEffect } from 'react'
import { useTimelineStore } from '../stores/timeline'

export function useAudioPlayback() {
  const ctxRef = useRef<AudioContext | null>(null)
  const sourcesRef = useRef<Map<string, { source: AudioBufferSourceNode; gain: GainNode }>>(new Map())
  const buffersRef = useRef<Map<string, AudioBuffer>>(new Map())
  const rafRef = useRef<number>(0)

  const audioCues = useTimelineStore((s) => s.audioCues)
  const playback = useTimelineStore((s) => s.playback)
  const setPlayback = useTimelineStore((s) => s.setPlayback)

  const getCtx = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext()
    }
    return ctxRef.current
  }, [])

  // Pre-load audio buffers when cues change
  useEffect(() => {
    const ctx = getCtx()
    for (const cue of audioCues) {
      if (!cue.audioUrl || buffersRef.current.has(cue.id)) continue
      fetch(cue.audioUrl)
        .then((r) => r.arrayBuffer())
        .then((data) => ctx.decodeAudioData(data))
        .then((buffer) => {
          buffersRef.current.set(cue.id, buffer)
          // Compute waveform peaks
          const channel = buffer.getChannelData(0)
          const peaks = downsample(channel, 200)
          useTimelineStore.getState().updateAudioCue(cue.id, { waveformPeaks: peaks })
        })
        .catch(() => {})
    }
  }, [audioCues, getCtx])

  const play = useCallback(() => {
    const ctx = getCtx()
    // Stop any existing sources
    stopSources()

    // Start all cues
    for (const cue of audioCues) {
      const buffer = buffersRef.current.get(cue.id)
      if (!buffer) continue

      const source = ctx.createBufferSource()
      source.buffer = buffer

      const gain = ctx.createGain()
      gain.gain.value = cue.volume

      source.connect(gain)
      gain.connect(ctx.destination)

      // Start at the cue's offset relative to the playhead
      const offset = Math.max(0, playback.currentTime - cue.start)
      if (offset < cue.duration) {
        source.start(0, offset)
      }

      sourcesRef.current.set(cue.id, { source, gain })
    }

    // Animate playhead
    const startTime = ctx.currentTime
    const startOffset = playback.currentTime
    function tick() {
      const elapsed = ctx.currentTime - startTime
      const newTime = startOffset + elapsed
      const totalDuration = playback.totalDuration
      if (newTime >= totalDuration) {
        setPlayback({ isPlaying: false, currentTime: 0 })
        stopSources()
        return
      }
      setPlayback({ currentTime: newTime })
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)

    setPlayback({ isPlaying: true })
  }, [audioCues, playback, setPlayback, getCtx])

  const pause = useCallback(() => {
    stopSources()
    cancelAnimationFrame(rafRef.current)
    setPlayback({ isPlaying: false })
  }, [setPlayback])

  const seek = useCallback((time: number) => {
    const wasPlaying = playback.isPlaying
    if (wasPlaying) stopSources()
    cancelAnimationFrame(rafRef.current)
    setPlayback({ currentTime: time })
    if (wasPlaying) play()
  }, [playback.isPlaying, setPlayback, play])

  function stopSources() {
    for (const [, { source }] of sourcesRef.current) {
      try { source.stop() } catch {}
    }
    sourcesRef.current.clear()
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current)
      stopSources()
      ctxRef.current?.close()
    }
  }, [])

  return { play, pause, seek }
}

function downsample(data: Float32Array, numPoints: number): number[] {
  const blockSize = Math.floor(data.length / numPoints)
  const peaks: number[] = []
  for (let i = 0; i < numPoints; i++) {
    let max = 0
    const start = i * blockSize
    for (let j = 0; j < blockSize; j++) {
      const abs = Math.abs(data[start + j] || 0)
      if (abs > max) max = abs
    }
    peaks.push(max)
  }
  return peaks
}
