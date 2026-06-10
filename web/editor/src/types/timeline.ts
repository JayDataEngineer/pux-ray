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
  /** Last error message if status is 'failed' */
  error: string | null
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
  /** Camera pan X offset (-1 to 1) */
  cameraPanX: number
  /** Camera pan Y offset (-1 to 1) */
  cameraPanY: number
  /** Camera zoom (1.0 = none, >1 = zoom in) */
  cameraZoom: number
  /** Resize/fit method: stretch, fit, crop, pad */
  resizeMethod: 'stretch' | 'fit' | 'crop' | 'pad'
  /** Use distilled mode (loads distilled LoRA, fewer steps) */
  distilledMode: boolean
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

/** Colors for auto-assigned audio tracks */
export const TRACK_COLORS = [
  '#4ade80', '#facc15', '#fb923c', '#60a5fa', '#f472b6',
  '#a78bfa', '#34d399', '#fbbf24', '#f87171', '#38bdf8',
]

/** Available video models */
export const VIDEO_MODELS = [
  // ── Wan Video (text-to-video, image-to-video) ──
  { id: 'wan/t2v_1.3B', label: 'Wan 1.3B (fast)', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  { id: 'wan/t2v', label: 'Wan 14B (quality)', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  { id: 'wan/i2v', label: 'Wan I2V 14B', defaultFrames: 81, defaultFps: 16, defaultWidth: 1280, defaultHeight: 720 },
  // ── LTX-Video (Director prompt relay, FFLF, audio conditioning) ──
  { id: 'ltx2', label: 'LTX 2.3 22B (dev)', defaultFrames: 121, defaultFps: 24, defaultWidth: 768, defaultHeight: 512 },
  { id: 'ltx2_19B', label: 'LTX 2.0 19B (dev)', defaultFrames: 121, defaultFps: 24, defaultWidth: 768, defaultHeight: 512 },
  { id: 'ltxv_098_13b', label: 'LTX-Video 0.9.8 13B', defaultFrames: 97, defaultFps: 24, defaultWidth: 768, defaultHeight: 512 },
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
  cameraPanX: 0,
  cameraPanY: 0,
  cameraZoom: 1.0,
  resizeMethod: 'fit',
  distilledMode: false,
}
