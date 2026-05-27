import { useRef, useEffect, useCallback } from 'react'
import { useTimelineStore } from '../../stores/timeline'
import { render, totalCanvasHeight, type RenderState } from './timelineRenderer'
import { hitTest, computeDragGhostOrder, applyCollisionPhysics } from './timelineInteraction'

export function TimelineCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const thumbCache = useRef(new Map<string, HTMLImageElement>())

  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const audioTracks = useTimelineStore((s) => s.audioTracks)
  const viewport = useTimelineStore((s) => s.viewport)
  const playback = useTimelineStore((s) => s.playback)
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)
  const selectedAudioCueId = useTimelineStore((s) => s.selectedAudioCueId)
  const drag = useTimelineStore((s) => s.drag)

  const setViewport = useTimelineStore((s) => s.setViewport)
  const setPlayback = useTimelineStore((s) => s.setPlayback)
  const setDrag = useTimelineStore((s) => s.setDrag)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const setSelectedAudioCue = useTimelineStore((s) => s.setSelectedAudioCue)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const reorderSegments = useTimelineStore((s) => s.reorderSegments)

  // Load thumbnails when segments change
  useEffect(() => {
    for (const seg of segments) {
      if (seg.thumbnailUrl && !thumbCache.current.has(seg.thumbnailUrl)) {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
          thumbCache.current.set(seg.thumbnailUrl!, img)
          // Trigger re-render
          const canvas = canvasRef.current
          if (canvas) {
            const ctx = canvas.getContext('2d')
            if (ctx) draw(ctx)
          }
        }
        img.src = seg.thumbnailUrl
      }
    }
  }, [segments])

  function draw(ctx: CanvasRenderingContext2D) {
    const state: RenderState = {
      segments,
      audioCues,
      audioTracks,
      viewport,
      playback,
      selectedSegmentId,
      selectedAudioCueId,
      dragSegmentId: drag.segmentId,
      dragGhostOrder: drag.ghostOrder,
      isDragging: drag.isDragging,
      thumbnails: thumbCache.current,
    }
    render(ctx, state)
  }

  // Resize observer
  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return

    const observer = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      const w = wrap.clientWidth
      const h = Math.max(totalCanvasHeight(audioTracks), wrap.clientHeight)
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      setViewport({ canvasWidth: w, canvasHeight: h })

      const ctx = canvas.getContext('2d')
      if (ctx) draw(ctx)
    })
    observer.observe(wrap)
    return () => observer.disconnect()
  }, [audioTracks, setViewport])

  // Re-render on state changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (ctx) draw(ctx)
  }, [segments, audioCues, viewport, playback, selectedSegmentId, selectedAudioCueId, drag])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const target = hitTest(x, y, segments, audioCues, audioTracks, viewport)

    if (target.type === 'ruler') {
      setPlayback({ currentTime: target.time })
    } else if (target.type === 'segment') {
      setSelectedSegment(target.segmentId)
      // Start drag
      setDrag({
        isDragging: true,
        segmentId: target.segmentId,
        mouseX: x,
        mouseY: y,
        originalOrder: segments.find((s) => s.id === target.segmentId)?.order ?? 0,
        ghostOrder: segments.find((s) => s.id === target.segmentId)?.order ?? 0,
      })
    } else if (target.type === 'audioCue') {
      setSelectedAudioCue(target.cueId)
      setSelectedSegment(null)
    } else {
      setSelectedSegment(null)
      setSelectedAudioCue(null)
    }
  }, [segments, audioCues, audioTracks, viewport])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drag.isDragging || !drag.segmentId) return
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left

    const ghostOrder = computeDragGhostOrder(x, segments, drag.segmentId, viewport)
    setDrag({ mouseX: x, ghostOrder })
  }, [drag.isDragging, drag.segmentId, segments, viewport])

  const handleMouseUp = useCallback(() => {
    if (!drag.isDragging || !drag.segmentId) return

    const reordered = applyCollisionPhysics(segments, drag.segmentId, drag.ghostOrder)
    const orderedIds = [...reordered].sort((a, b) => a.order - b.order).map((s) => s.id)
    reorderSegments(orderedIds)

    // Update segment positions
    for (const seg of reordered) {
      updateSegment(seg.id, { start: seg.start, order: seg.order })
    }

    setDrag({ isDragging: false, segmentId: null })
  }, [drag.isDragging, drag.segmentId, drag.ghostOrder, segments])

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (e.ctrlKey || e.metaKey) {
      // Zoom
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      const newPps = Math.max(20, Math.min(400, viewport.pixelsPerSecond * delta))
      setViewport({ pixelsPerSecond: newPps })
    } else {
      // Horizontal scroll
      const delta = e.deltaX || e.deltaY
      const scrollDelta = delta / viewport.pixelsPerSecond
      const newScrollX = Math.max(0, viewport.scrollX + scrollDelta)
      setViewport({ scrollX: newScrollX })
    }
  }, [viewport])

  return (
    <div ref={wrapRef} className="timeline-canvas-wrap">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
    </div>
  )
}
