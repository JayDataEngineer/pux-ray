import type { TimelineSegment, AudioCue, AudioTrackDef, TimelineViewport, PlaybackState } from '../../types/timeline'

const RULER_HEIGHT = 24
const IMAGE_TRACK_HEIGHT = 160
const LABEL_WIDTH = 80
const SEGMENT_GAP = 4
const SEGMENT_RADIUS = 6

const COLORS = {
  bg: '#18181b',
  bgAlt: '#1c1c20',
  border: '#27272a',
  borderFocus: '#3f3f46',
  textPrimary: '#fafafa',
  textSecondary: '#a1a1aa',
  textMuted: '#71717a',
  accent: '#818cf8',
  accentDim: '#6366f1',
  green: '#4ade80',
  yellow: '#facc15',
  red: '#f87171',
  blue: '#60a5fa',
  orange: '#fb923c',
  segmentBg: '#22222a',
  segmentBgReady: '#2a2a38',
  ghostBg: 'rgba(129, 140, 248, 0.15)',
  ghostBorder: 'rgba(129, 140, 248, 0.5)',
}

export const LAYOUT = {
  RULER_HEIGHT,
  IMAGE_TRACK_HEIGHT,
  LABEL_WIDTH,
  SEGMENT_GAP,
  SEGMENT_RADIUS,
}

export interface RenderState {
  segments: TimelineSegment[]
  audioCues: AudioCue[]
  audioTracks: AudioTrackDef[]
  viewport: TimelineViewport
  playback: PlaybackState
  selectedSegmentId: string | null
  selectedAudioCueId: string | null
  dragSegmentId: string | null
  dragGhostOrder: number
  isDragging: boolean
  thumbnails: Map<string, HTMLImageElement>
}

export function totalCanvasHeight(audioTracks: AudioTrackDef[]): number {
  return RULER_HEIGHT + IMAGE_TRACK_HEIGHT + audioTracks.reduce((h, t) => h + t.height, 0)
}

function toPixelX(time: number, viewport: TimelineViewport): number {
  return LABEL_WIDTH + (time - viewport.scrollX) * viewport.pixelsPerSecond
}

function toTime(pixelX: number, viewport: TimelineViewport): number {
  return (pixelX - LABEL_WIDTH) / viewport.pixelsPerSecond + viewport.scrollX
}

export { toPixelX, toTime }

export function drawBackground(ctx: CanvasRenderingContext2D, width: number, height: number, viewport: TimelineViewport, audioTracks: AudioTrackDef[]) {
  ctx.fillStyle = COLORS.bg
  ctx.fillRect(0, 0, width, height)

  // Vertical grid lines at 1-second intervals
  ctx.strokeStyle = COLORS.border
  ctx.lineWidth = 1
  const startSec = Math.floor(viewport.scrollX)
  const endSec = Math.ceil(toTime(width, viewport))
  for (let s = startSec; s <= endSec; s++) {
    const x = Math.round(toPixelX(s, viewport)) + 0.5
    if (x < LABEL_WIDTH) continue
    ctx.beginPath()
    ctx.moveTo(x, RULER_HEIGHT)
    ctx.lineTo(x, height)
    ctx.stroke()
  }

  // Label column background
  ctx.fillStyle = COLORS.bgAlt
  ctx.fillRect(0, 0, LABEL_WIDTH, height)

  // Horizontal dividers
  ctx.strokeStyle = COLORS.border
  const imageTrackEnd = RULER_HEIGHT + IMAGE_TRACK_HEIGHT
  ctx.beginPath()
  ctx.moveTo(0, imageTrackEnd + 0.5)
  ctx.lineTo(width, imageTrackEnd + 0.5)
  ctx.stroke()

  let y = imageTrackEnd
  for (const track of audioTracks) {
    y += track.height
    ctx.beginPath()
    ctx.moveTo(0, y + 0.5)
    ctx.lineTo(width, y + 0.5)
    ctx.stroke()
  }
}

export function drawRuler(ctx: CanvasRenderingContext2D, width: number, viewport: TimelineViewport) {
  // Ruler background
  ctx.fillStyle = COLORS.bgAlt
  ctx.fillRect(0, 0, width, RULER_HEIGHT)

  // Bottom border
  ctx.strokeStyle = COLORS.border
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(0, RULER_HEIGHT + 0.5)
  ctx.lineTo(width, RULER_HEIGHT + 0.5)
  ctx.stroke()

  const startSec = Math.floor(viewport.scrollX)
  const endSec = Math.ceil(toTime(width, viewport))

  ctx.font = '10px monospace'
  ctx.fillStyle = COLORS.textMuted
  ctx.textAlign = 'center'

  for (let s = startSec; s <= endSec; s++) {
    const x = toPixelX(s, viewport)
    if (x < LABEL_WIDTH) continue

    // Major tick
    ctx.strokeStyle = COLORS.textMuted
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(x + 0.5, RULER_HEIGHT - 8)
    ctx.lineTo(x + 0.5, RULER_HEIGHT)
    ctx.stroke()

    // Label
    const label = s >= 60 ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : `${s}s`
    ctx.fillText(label, x, RULER_HEIGHT - 10)

    // Sub-ticks at 0.5s if zoomed in enough
    if (viewport.pixelsPerSecond >= 40) {
      const halfX = toPixelX(s + 0.5, viewport)
      ctx.strokeStyle = COLORS.border
      ctx.beginPath()
      ctx.moveTo(halfX + 0.5, RULER_HEIGHT - 4)
      ctx.lineTo(halfX + 0.5, RULER_HEIGHT)
      ctx.stroke()
    }
  }
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.arcTo(x + w, y, x + w, y + r, r)
  ctx.lineTo(x + w, y + h - r)
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
  ctx.lineTo(x + r, y + h)
  ctx.arcTo(x, y + h, x, y + h - r, r)
  ctx.lineTo(x, y + r)
  ctx.arcTo(x, y, x + r, y, r)
  ctx.closePath()
}

export function drawSegments(ctx: CanvasRenderingContext2D, state: RenderState) {
  const { segments, viewport, selectedSegmentId, thumbnails, isDragging, dragSegmentId, dragGhostOrder } = state
  const trackY = RULER_HEIGHT
  const segH = IMAGE_TRACK_HEIGHT - SEGMENT_GAP * 2

  // Draw ghost preview if dragging
  if (isDragging && dragSegmentId) {
    const ghostX = LABEL_WIDTH + dragGhostOrder * (viewport.pixelsPerSecond * 5) + SEGMENT_GAP
    const ghostW = viewport.pixelsPerSecond * 5 - SEGMENT_GAP * 2
    roundRect(ctx, ghostX, trackY + SEGMENT_GAP, ghostW, segH, SEGMENT_RADIUS)
    ctx.fillStyle = COLORS.ghostBg
    ctx.fill()
    ctx.strokeStyle = COLORS.ghostBorder
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4])
    ctx.stroke()
    ctx.setLineDash([])
  }

  // Draw segments in order
  const sorted = [...segments].sort((a, b) => a.order - b.order)
  for (const seg of sorted) {
    const isDragSource = isDragging && seg.id === dragSegmentId
    if (isDragSource) continue // Don't draw the dragged segment at its original position

    const x = toPixelX(seg.start, viewport) + SEGMENT_GAP
    const w = seg.duration * viewport.pixelsPerSecond - SEGMENT_GAP * 2

    if (x + w < LABEL_WIDTH || x > viewport.canvasWidth) continue // Off-screen

    const isSelected = seg.id === selectedSegmentId

    // Segment background
    const bgColor = seg.status === 'ready' ? COLORS.segmentBgReady : COLORS.segmentBg
    roundRect(ctx, x, trackY + SEGMENT_GAP, w, segH, SEGMENT_RADIUS)
    ctx.fillStyle = bgColor
    ctx.fill()

    // Border
    ctx.strokeStyle = isSelected ? COLORS.accent : COLORS.borderFocus
    ctx.lineWidth = isSelected ? 2 : 1
    ctx.stroke()

    // Thumbnail image
    if (seg.thumbnailUrl && thumbnails.has(seg.thumbnailUrl)) {
      const img = thumbnails.get(seg.thumbnailUrl)!
      const imgPad = 4
      const imgH = segH - imgPad * 2 - 20 // Leave room for text
      const imgW = imgH * (img.naturalWidth / img.naturalHeight || 1)
      const imgX = x + imgPad
      const imgY = trackY + SEGMENT_GAP + imgPad
      ctx.save()
      roundRect(ctx, imgX, imgY, imgW, imgH, 4)
      ctx.clip()
      ctx.drawImage(img, imgX, imgY, imgW, imgH)
      ctx.restore()
    } else {
      // Status icon
      const statusColors: Record<string, string> = {
        empty: COLORS.textMuted,
        pending: COLORS.yellow,
        generating: COLORS.blue,
        ready: COLORS.green,
        failed: COLORS.red,
      }
      ctx.fillStyle = statusColors[seg.status] || COLORS.textMuted
      ctx.beginPath()
      ctx.arc(x + w / 2, trackY + SEGMENT_GAP + segH / 2 - 10, 12, 0, Math.PI * 2)
      ctx.fill()

      ctx.font = 'bold 14px sans-serif'
      ctx.fillStyle = COLORS.bg
      ctx.textAlign = 'center'
      const statusIcons: Record<string, string> = { empty: '+', pending: '...', generating: '●', ready: '✓', failed: '✗' }
      ctx.fillText(statusIcons[seg.status] || '?', x + w / 2, trackY + SEGMENT_GAP + segH / 2 - 6)
    }

    // Prompt text
    ctx.font = '11px sans-serif'
    ctx.fillStyle = COLORS.textSecondary
    ctx.textAlign = 'left'
    const maxTextW = w - 8
    const promptText = seg.prompt || '(no prompt)'
    const truncated = truncateText(ctx, promptText, maxTextW)
    ctx.fillText(truncated, x + 4, trackY + SEGMENT_GAP + segH - 8)

    // Duration label
    ctx.font = '10px monospace'
    ctx.fillStyle = COLORS.textMuted
    ctx.textAlign = 'right'
    ctx.fillText(`${seg.duration.toFixed(1)}s`, x + w - 4, trackY + SEGMENT_GAP + segH - 8)
  }
}

export function drawAudioTracks(ctx: CanvasRenderingContext2D, state: RenderState) {
  const { audioCues, audioTracks, viewport, selectedAudioCueId } = state
  const trackStartY = RULER_HEIGHT + IMAGE_TRACK_HEIGHT

  let trackY = trackStartY
  for (const track of audioTracks) {
    // Track label
    ctx.fillStyle = COLORS.textMuted
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(track.label, LABEL_WIDTH / 2, trackY + track.height / 2 + 4)

    // Draw cues for this track
    const trackCues = audioCues.filter((c) => c.track === track.id)
    for (const cue of trackCues) {
      const x = toPixelX(cue.start, viewport) + 2
      const w = cue.duration * viewport.pixelsPerSecond - 4
      const cueY = trackY + 4
      const cueH = track.height - 8

      if (x + w < LABEL_WIDTH || x > viewport.canvasWidth) continue

      const isSelected = cue.id === selectedAudioCueId

      // Cue background
      ctx.fillStyle = track.color + (isSelected ? '60' : '30')
      roundRect(ctx, x, cueY, w, cueH, 4)
      ctx.fill()

      // Border
      ctx.strokeStyle = isSelected ? track.color : track.color + '80'
      ctx.lineWidth = isSelected ? 2 : 1
      ctx.stroke()

      // Waveform
      if (cue.waveformPeaks && cue.waveformPeaks.length > 0) {
        drawWaveform(ctx, x, cueY, w, cueH, cue.waveformPeaks, track.color)
      }

      // Label
      ctx.font = '10px sans-serif'
      ctx.fillStyle = track.color
      ctx.textAlign = 'left'
      ctx.fillText(cue.label, x + 4, cueY + cueH - 4)
    }

    trackY += track.height
  }
}

function drawWaveform(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  peaks: number[], color: string,
) {
  const barW = Math.max(1, w / peaks.length)
  const midY = y + h / 2

  ctx.fillStyle = color + '80'
  for (let i = 0; i < peaks.length; i++) {
    const barH = peaks[i] * (h / 2 - 2)
    const barX = x + (i / peaks.length) * w
    ctx.fillRect(barX, midY - barH, Math.max(1, barW - 0.5), barH * 2)
  }
}

export function drawPlayhead(ctx: CanvasRenderingContext2D, state: RenderState) {
  const { playback, viewport } = state
  const x = toPixelX(playback.currentTime, viewport)

  if (x < LABEL_WIDTH) return

  // Triangle indicator on ruler
  ctx.fillStyle = COLORS.accent
  ctx.beginPath()
  ctx.moveTo(x - 5, 0)
  ctx.lineTo(x + 5, 0)
  ctx.lineTo(x, RULER_HEIGHT)
  ctx.closePath()
  ctx.fill()

  // Vertical line
  ctx.strokeStyle = COLORS.accent
  ctx.lineWidth = 1.5
  ctx.setLineDash([4, 3])
  ctx.beginPath()
  ctx.moveTo(x + 0.5, RULER_HEIGHT)
  ctx.lineTo(x + 0.5, ctx.canvas.height)
  ctx.stroke()
  ctx.setLineDash([])
}

export function render(ctx: CanvasRenderingContext2D, state: RenderState) {
  const { viewport, audioTracks } = state
  const dpr = window.devicePixelRatio || 1
  const w = viewport.canvasWidth
  const h = totalCanvasHeight(audioTracks)

  ctx.save()
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  drawBackground(ctx, w, h, viewport, audioTracks)
  drawRuler(ctx, w, viewport)
  drawSegments(ctx, state)
  drawAudioTracks(ctx, state)
  drawPlayhead(ctx, state)

  ctx.restore()
}

function truncateText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text
  let t = text
  while (t.length > 0 && ctx.measureText(t + '...').width > maxWidth) {
    t = t.slice(0, -1)
  }
  return t + '...'
}
