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
  /** Asset ID for generated video (references asset store) */
  assetId?: string
  firstFrameB64: string | null
  lastFrameB64: string | null
  params: SegmentParams
  trimStart: number
  sourceStepId: string | null
  status: 'empty' | 'pending' | 'generating' | 'ready' | 'failed'
  /** Last error message if status is 'failed' */
  error: string | null
  /** Control video URL for IC-LoRA conditioning (pose/depth/canny transfer) */
  controlVideoUrl: string | null
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
  /** Start image / source strength (lower = more motion freedom) */
  inputVideoStrength: number
  /** Spatial 2x upscale after generation */
  spatialUpscale: boolean
  /** Comma-separated LoRA filenames */
  loras: string
  /** Perturbation mode: 0=off, 1=skip layer, 2=skip self-attn */
  perturbationSwitch: number
  /** Which transformer layers to perturb */
  perturbationLayers: number[]
  /** Perturbation start percentage (0-100) */
  perturbationStartPerc: number
  /** Perturbation end percentage (0-100) */
  perturbationEndPerc: number
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
  /** Distilled LoRA strength (matches WDC ComfyUI default 0.5) */
  distilledLoraStrength: number

  // ── Advanced guidance ──
  /** APG (Adaptive Projected Guidance) — dev only */
  apgSwitch: boolean
  /** CFG Star rescaling — dev only */
  cfgStarSwitch: boolean
  /** NAG scale — distilled only */
  nagScale: number
  /** NAG tau — distilled only */
  nagTau: number
  /** NAG alpha — distilled only */
  nagAlpha: number
  /** Alt/modality guidance scale — dev only */
  altGuideScale: number
  /** Alt guidance rescale — dev only */
  altScale: number
  /** Audio guidance scale — dev only */
  audioGuideScale: number
  /** Audio CFG scale */
  audioCfgScale: number
  /** Sample solver: euler, res2s (22B dev only) */
  sampleSolver: string

  // ── Self-Refiner ──
  /** Enable self refiner */
  selfRefinerSetting: number
  /** Refiner plan string e.g. "2-8:3" */
  selfRefinerPlan: string
  /** Frame uncertainty threshold */
  selfRefinerFUncertainty: number
  /** Certain percentage threshold */
  selfRefinerCertainPercentage: number

  // ── Sliding window ──
  /** Enable sliding window for long videos */
  slidingWindow: boolean
  /** Window size in frames (default 241) */
  slidingWindowSize: number
  /** Overlap between windows (default 9) */
  slidingWindowOverlap: number

  // ── Control video / IC-LoRA (distilled only) ──
  /** Control video mode: "" | "PVG" | "OVG" | "DVG" | "EVG" | "VG" | "V&G" | "KFI" */
  videoPromptType: string
  /** Masking strength for control video */
  maskingStrength: number
  /** Mask preprocessing: "" | "A" | "NA" | "XA" | "XNA" */
  maskingSource: string
  /** Outpainting enabled */
  outpaintingEnabled: boolean
  /** Outpaint ratio e.g. "16:9" */
  outpaintingRatio: string

  // ── Audio conditioning ──
  /** Audio prompt type: "" | "A" | "A1OF" | "K" | "2" */
  audioPromptType: string

  // ── Prompt enhancement ──
  /** Use built-in LTX2 prompt enhancer */
  enhancePrompt: boolean
}

/** Control video modes for IC-LoRA (distilled only) */
export const CONTROL_VIDEO_MODES = [
  { id: 'off', label: 'None' },
  { id: 'VG', label: 'Raw Control Video' },
  { id: 'PVG', label: 'Transfer Human Motion' },
  { id: 'OVG', label: 'Transfer Motion + Pose Alignment' },
  { id: 'DVG', label: 'Transfer Depth' },
  { id: 'EVG', label: 'Transfer Canny Edges' },
  { id: 'V&G', label: 'SDR → HDR (22B distilled only)' },
  { id: 'KFI', label: 'Inject Frames' },
] as const

/** Audio conditioning modes */
export const AUDIO_PROMPT_MODES = [
  { id: 'off', label: 'Text Only (video + optional soundtrack)' },
  { id: 'A', label: 'Audio-Driven (video from soundtrack)' },
  { id: 'A1OF', label: 'Reference Voice + ID-LoRA (dev)' },
  { id: 'K', label: 'Control Video Audio (distilled)' },
  { id: '2', label: 'Generate Audio from Video (distilled)' },
] as const

/** Mask preprocessing modes */
export const MASK_MODES = [
  { id: 'off', label: 'None' },
  { id: 'A', label: 'Auto Mask' },
  { id: 'NA', label: 'No Auto Mask' },
  { id: 'XA', label: 'Extended Auto' },
  { id: 'XNA', label: 'Extended No Auto' },
] as const

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
  width: 768,
  height: 512,
  frames: 121,
  fps: 24,
  guideScale: 1.0,     // WDC ComfyUI distilled default (CFG 1.0)
  samplingSteps: 8,     // distilled
  model: 'ltx2',
  guidePhases: 2,       // >=2 enables stage 2 spatial upscale (WDC always runs it)
  epsilon: 0.001,
  denoisingStrength: 1.0,
  inputVideoStrength: 1.0,
  spatialUpscale: false,
  loras: '',
  perturbationSwitch: 0,
  perturbationLayers: [28],
  perturbationStartPerc: 0,
  perturbationEndPerc: 100,
  cameraPanX: 0,
  cameraPanY: 0,
  cameraZoom: 1.0,
  resizeMethod: 'crop',
  distilledMode: true,
  distilledLoraStrength: 0.5,  // WDC ComfyUI default

  // Advanced guidance
  apgSwitch: false,
  cfgStarSwitch: false,
  nagScale: 1.0,
  nagTau: 3.5,
  nagAlpha: 0.5,
  altGuideScale: 1.0,
  altScale: 0.0,
  audioGuideScale: 1.0,
  audioCfgScale: 1.0,
  sampleSolver: 'euler',

  // Self-Refiner
  selfRefinerSetting: 0,
  selfRefinerPlan: '',
  selfRefinerFUncertainty: 0.1,
  selfRefinerCertainPercentage: 0.999,

  // Sliding window
  slidingWindow: false,
  slidingWindowSize: 241,
  slidingWindowOverlap: 9,

  // Control video
  videoPromptType: 'off',
  maskingStrength: 0.0,
  maskingSource: 'off',
  outpaintingEnabled: false,
  outpaintingRatio: '',

  // Audio
  audioPromptType: 'off',

  // Prompt enhancement
  enhancePrompt: false,
}
