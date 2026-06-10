/** A video segment on the timeline — the core editing unit */
export interface TimelineSegment {
  id: string
  order: number
  start: number
  duration: number
  prompt: string
  negativePrompt: string
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
  samplingSteps: number
  model: string
  /** 2-phase guidance (1 or 2) */
  guidePhases: number
  /** Prompt relay boundary sharpness */
  epsilon: number
  /** Control strength for video conditioning */
  denoisingStrength: number
  /** Spatial 2x upscale after generation */
  spatialUpscale: boolean
  /** Comma-separated LoRA filenames */
  loras: string
  /** Perturbation mode: 0=off, 1=skip layer, 2=skip self-attn */
  perturbationSwitch: number
}

/** An audio cue placed at a specific time on a named track */
export interface AudioCue {
  id: string
  track: string
  start: number
  duration: number
  label: string
  audioUrl: string | null
  audioB64: string | null
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

/** Available Wan2GP video models */
export const VIDEO_MODELS = [
  { id: 'wan/t2v_1.3B', label: 'Wan 1.3B (fast)', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  { id: 'wan/t2v', label: 'Wan 14B (quality)', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  { id: 'wan/i2v', label: 'Wan I2V 14B', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  { id: 'ltx2', label: 'LTX Video', defaultFrames: 121, defaultFps: 24, defaultWidth: 768, defaultHeight: 512 },
] as const

export const DEFAULT_SEGMENT_PARAMS: SegmentParams = {
  seed: 42,
  width: 1280,
  height: 720,
  frames: 81,
  fps: 16,
  guideScale: 5.0,
  samplingSteps: 20,
  model: 'wan/t2v_1.3B',
  guidePhases: 2,
  epsilon: 0.001,
  denoisingStrength: 1.0,
  spatialUpscale: false,
  loras: '',
  perturbationSwitch: 0,
}
