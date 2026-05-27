import type { TimelineSegment, AudioCue, AudioTrackDef, TimelineViewport } from '../../types/timeline'
import { LAYOUT } from './timelineRenderer'
import { toTime } from './timelineRenderer'

const { RULER_HEIGHT, IMAGE_TRACK_HEIGHT } = LAYOUT

export type HitTarget =
  | { type: 'ruler'; time: number }
  | { type: 'segment'; segmentId: string }
  | { type: 'audioCue'; cueId: string }
  | { type: 'empty' }

export function hitTest(
  pixelX: number, pixelY: number,
  segments: TimelineSegment[],
  audioCues: AudioCue[],
  audioTracks: AudioTrackDef[],
  viewport: TimelineViewport,
): HitTarget {
  if (pixelY < RULER_HEIGHT) {
    return { type: 'ruler', time: toTime(pixelX, viewport) }
  }

  if (pixelY < RULER_HEIGHT + IMAGE_TRACK_HEIGHT) {
    const time = toTime(pixelX, viewport)
    const sorted = [...segments].sort((a, b) => a.order - b.order)
    for (let i = sorted.length - 1; i >= 0; i--) {
      const seg = sorted[i]
      if (time >= seg.start && time <= seg.start + seg.duration) {
        return { type: 'segment', segmentId: seg.id }
      }
    }
    return { type: 'empty' }
  }

  // Audio tracks
  const trackStartY = RULER_HEIGHT + IMAGE_TRACK_HEIGHT
  let trackY = trackStartY
  for (const track of audioTracks) {
    if (pixelY >= trackY && pixelY < trackY + track.height) {
      const time = toTime(pixelX, viewport)
      const trackCues = audioCues.filter((c) => c.track === track.id)
      for (const cue of trackCues) {
        if (time >= cue.start && time <= cue.start + cue.duration) {
          return { type: 'audioCue', cueId: cue.id }
        }
      }
    }
    trackY += track.height
  }

  return { type: 'empty' }
}

export function computeDragGhostOrder(
  mouseX: number,
  segments: TimelineSegment[],
  draggedId: string,
  viewport: TimelineViewport,
): number {
  const time = toTime(mouseX, viewport)
  const sorted = [...segments].sort((a, b) => a.order - b.order)
  const dragged = sorted.find((s) => s.id === draggedId)
  if (!dragged) return 0

  // Compute ghost order based on time position
  let ghostOrder = 0
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].id === draggedId) continue
    const segMid = sorted[i].start + sorted[i].duration / 2
    if (time > segMid) {
      ghostOrder = i + 1
    }
  }

  // Clamp
  const otherCount = sorted.length - 1
  ghostOrder = Math.max(0, Math.min(ghostOrder, otherCount))

  return ghostOrder
}

export function applyCollisionPhysics(
  segments: TimelineSegment[],
  draggedId: string,
  ghostOrder: number,
): TimelineSegment[] {
  const sorted = [...segments].sort((a, b) => a.order - b.order)
  const dragged = sorted.find((s) => s.id === draggedId)
  if (!dragged) return segments

  // Remove dragged from list
  const others = sorted.filter((s) => s.id !== draggedId)

  // Insert at ghost position
  others.splice(ghostOrder, 0, dragged)

  // Recompute positions
  let cursor = 0
  return others.map((seg, i) => {
    const newSeg = { ...seg, order: i, start: cursor }
    cursor += seg.duration
    return newSeg
  })
}
