/** A video segment on the timeline — the core editing unit */
export interface TimelineSegment {
  id: string
  order: number
  start: number
  duration: number
  prompt: string
  thumbnailUrl: string | null
  videoUrl: string | null
  firstFrameB64: string | null
  lastFrameB64: string | null
  params: SegmentParams
  trimStart: number
  sourceStepId: string | null
  status: 'empty' | 'pending' | 'generating' | 'ready' | 'failed'
}

export interface SegmentParams {
  seed: number
  width: number
  height: number
  frames: number
  fps: number
  guideScale: number
}

/** An audio cue placed at a specific time on a named track */
export interface AudioCue {
  id: string
  track: string
  start: number
  duration: number
  label: string
  audioUrl: string | null
  volume: number
  waveformPeaks: number[] | null
  sourceStepId: string | null
}

export interface AudioTrackDef {
  id: string
  label: string
  color: string
  height: number
}

export interface TimelineViewport {
  scrollX: number
  pixelsPerSecond: number
  canvasWidth: number
  canvasHeight: number
}

export interface PlaybackState {
  isPlaying: boolean
  currentTime: number
  totalDuration: number
}

export interface DragState {
  isDragging: boolean
  segmentId: string | null
  mouseX: number
  mouseY: number
  originalOrder: number
  ghostOrder: number
}

export const AUDIO_TRACKS: AudioTrackDef[] = [
  { id: 'voice', label: 'Voice', color: '#4ade80', height: 48 },
  { id: 'sfx', label: 'SFX', color: '#facc15', height: 48 },
  { id: 'music', label: 'Music', color: '#fb923c', height: 48 },
]

export const DEFAULT_SEGMENT_PARAMS: SegmentParams = {
  seed: 42,
  width: 768,
  height: 512,
  frames: 97,
  fps: 24,
  guideScale: 3.0,
}
