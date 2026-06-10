import { create } from 'zustand'
import type { WorkflowRun, WorkflowSpec } from '../types'
import type { TimelineSegment, AudioCue, AudioTrackDef, TimelineViewport, PlaybackState, DragState } from '../types/timeline'
import { TRACK_COLORS, DEFAULT_SEGMENT_PARAMS } from '../types/timeline'

let _nextId = 1
function uid(): string {
  return `seg_${_nextId++}_${Date.now().toString(36)}`
}

// ── Undo/Redo history ─────────────────────────────────────────────────────────
const _history: { segments: TimelineSegment[]; audioCues: AudioCue[] }[] = []
let _historyIdx = -1
const MAX_HISTORY = 50

function pushHistory(segments: TimelineSegment[], audioCues: AudioCue[]) {
  _history.splice(_historyIdx + 1)
  _history.push({
    segments: segments.map(s => ({ ...s, params: { ...s.params } })),
    audioCues: audioCues.map(c => ({ ...c })),
  })
  if (_history.length > MAX_HISTORY) _history.shift()
  _historyIdx = _history.length - 1
}

function getArtifactThumb(run: WorkflowRun, stepId: string): string | null {
  for (const [key, art] of Object.entries(run.artifacts)) {
    if (!key.startsWith(`${stepId}.`)) continue
    if (art.media_type.startsWith('image/')) {
      const name = art.name.includes('.') ? art.name : art.name + '.png'
      return `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${stepId}/${name}`
    }
  }
  return null
}

function getArtifactVideo(run: WorkflowRun, stepId: string): string | null {
  for (const [key, art] of Object.entries(run.artifacts)) {
    if (!key.startsWith(`${stepId}.`)) continue
    if (art.media_type.startsWith('video/')) {
      const name = art.name.includes('.') ? art.name : art.name + '.mp4'
      return `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${stepId}/${name}`
    }
  }
  return null
}

function getArtifactAudio(run: WorkflowRun, stepId: string): string | null {
  for (const [key, art] of Object.entries(run.artifacts)) {
    if (!key.startsWith(`${stepId}.`)) continue
    if (art.media_type.startsWith('audio/')) {
      const name = art.name.includes('.') ? art.name : art.name + '.wav'
      return `/v1/wf/${run.spec_name}/runs/${run.run_id}/artifacts/${stepId}/${name}`
    }
  }
  return null
}



interface TimelineStore {
  segments: TimelineSegment[]
  audioCues: AudioCue[]
  audioTracks: AudioTrackDef[]
  addAudioTrack: (label?: string) => AudioTrackDef
  removeAudioTrack: (id: string) => void
  viewport: TimelineViewport
  playback: PlaybackState
  drag: DragState
  selectedSegmentId: string | null
  selectedAudioCueId: string | null

  // Relay state — the single continuous video from Director prompt relay
  relayVideoUrl: string | null
  relaySegmentIds: string[]

  addSegment: (partial?: Partial<TimelineSegment>) => TimelineSegment
  removeSegment: (id: string) => void
  updateSegment: (id: string, patch: Partial<TimelineSegment>) => void
  reorderSegments: (orderedIds: string[]) => void
  undo: () => void
  redo: () => void

  addAudioCue: (cue: Omit<AudioCue, 'id'>) => void
  removeAudioCue: (id: string) => void
  updateAudioCue: (id: string, patch: Partial<AudioCue>) => void

  setRelayVideo: (url: string | null, segmentIds: string[]) => void
  setViewport: (patch: Partial<TimelineViewport>) => void
  setPlayback: (patch: Partial<PlaybackState>) => void
  setDrag: (patch: Partial<DragState>) => void
  setSelectedSegment: (id: string | null) => void
  setSelectedAudioCue: (id: string | null) => void

  loadFromRun: (run: WorkflowRun, spec: WorkflowSpec) => void
  reset: () => void
}

const DEFAULT_VIEWPORT: TimelineViewport = {
  scrollX: 0,
  pixelsPerSecond: 60,
  canvasWidth: 1200,
  canvasHeight: 400,
}

const DEFAULT_PLAYBACK: PlaybackState = {
  isPlaying: false,
  currentTime: 0,
  totalDuration: 5,
}

const DEFAULT_DRAG: DragState = {
  isDragging: false,
  segmentId: null,
  mouseX: 0,
  mouseY: 0,
  originalOrder: 0,
  ghostOrder: 0,
}

export const useTimelineStore = create<TimelineStore>((set, get) => ({
  segments: [],
  audioCues: [],
  audioTracks: [],
  viewport: { ...DEFAULT_VIEWPORT },
  playback: { ...DEFAULT_PLAYBACK },
  drag: { ...DEFAULT_DRAG },
  selectedSegmentId: null,
  selectedAudioCueId: null,
  relayVideoUrl: null,
  relaySegmentIds: [],

  addSegment: (partial) => {
    const state = get()
    const order = state.segments.length
    const seg: TimelineSegment = {
      id: uid(),
      order,
      start: order * 5,
      duration: 5,
      prompt: '',
      negativePrompt: '',
      thumbnailUrl: null,
      videoUrl: null,
      firstFrameB64: null,
      lastFrameB64: null,
      params: { ...DEFAULT_SEGMENT_PARAMS },
      trimStart: 0,
      sourceStepId: null,
      status: 'empty',
      error: null,
      ...partial,
    }
    set((s) => {
      const segments = [...s.segments, seg]
      const totalDuration = segments.reduce((t, seg) => Math.max(t, seg.start + seg.duration), 0)
      pushHistory(segments, s.audioCues)
      return { segments, playback: { ...s.playback, totalDuration } }
    })
    return seg
  },

  removeSegment: (id) => set((s) => {
    const segments = s.segments.filter((seg) => seg.id !== id).map((seg, i) => ({ ...seg, order: i }))
    pushHistory(segments, s.audioCues)
    // Invalidate relay if removed segment was part of it
    const relayActive = s.relaySegmentIds.includes(id)
    return {
      segments,
      selectedSegmentId: s.selectedSegmentId === id ? null : s.selectedSegmentId,
      ...(relayActive ? { relayVideoUrl: null, relaySegmentIds: [] } : {}),
    }
  }),

  updateSegment: (id, patch) => set((s) => {
    const segments = s.segments.map((seg) => seg.id === id ? { ...seg, ...patch } : seg)
    pushHistory(segments, s.audioCues)
    return { segments }
  }),

  reorderSegments: (orderedIds) => set((s) => {
    const map = new Map(s.segments.map((seg) => [seg.id, seg]))
    let cursor = 0
    const segments = orderedIds.map((id, i) => {
      const seg = map.get(id)
      if (!seg) return null
      const newSeg = { ...seg, order: i, start: cursor }
      cursor += seg.duration
      return newSeg
    }).filter(Boolean) as TimelineSegment[]
    const totalDuration = segments.reduce((t, seg) => Math.max(t, seg.start + seg.duration), 0)
    pushHistory(segments, s.audioCues)
    return { segments, playback: { ...s.playback, totalDuration } }
  }),

  addAudioTrack: (label) => {
    const state = get()
    const idx = state.audioTracks.length
    const color = TRACK_COLORS[idx % TRACK_COLORS.length]
    const track: AudioTrackDef = {
      id: uid(),
      label: label || `Audio ${idx + 1}`,
      color,
      height: 48,
    }
    set((s) => ({ audioTracks: [...s.audioTracks, track] }))
    return track
  },

  removeAudioTrack: (id) => set((s) => ({
    audioTracks: s.audioTracks.filter((t) => t.id !== id),
    audioCues: s.audioCues.filter((c) => c.track !== id),
    selectedAudioCueId: s.selectedAudioCueId && s.audioCues.find(c => c.id === s.selectedAudioCueId)?.track === id ? null : s.selectedAudioCueId,
  })),

  addAudioCue: (cue) => {
    const full: AudioCue = { ...cue, id: uid() }
    set((s) => ({ audioCues: [...s.audioCues, full] }))
  },

  removeAudioCue: (id) => set((s) => ({
    audioCues: s.audioCues.filter((c) => c.id !== id),
    selectedAudioCueId: s.selectedAudioCueId === id ? null : s.selectedAudioCueId,
  })),

  updateAudioCue: (id, patch) => set((s) => ({
    audioCues: s.audioCues.map((c) => c.id === id ? { ...c, ...patch } : c),
  })),

  setRelayVideo: (url, segmentIds) => set({ relayVideoUrl: url, relaySegmentIds: segmentIds }),

  setViewport: (patch) => set((s) => ({ viewport: { ...s.viewport, ...patch } })),
  setPlayback: (patch) => set((s) => ({ playback: { ...s.playback, ...patch } })),
  setDrag: (patch) => set((s) => ({ drag: { ...s.drag, ...patch } })),
  setSelectedSegment: (id) => set({ selectedSegmentId: id }),
  setSelectedAudioCue: (id) => set({ selectedAudioCueId: id }),

  loadFromRun: (run, _spec) => {
    const segments: TimelineSegment[] = []
    const audioCues: AudioCue[] = []

    let cursor = 0
    const fps = typeof run.inputs?.video_fps === 'number' ? run.inputs.video_fps : 24
    const frames = typeof run.inputs?.video_frames === 'number' ? run.inputs.video_frames : 121
    const defaultDuration = frames / fps

    // Find the primary video output (prefer upscale > video_edit > lipsync > generate_video)
    const videoStepPriority = ['upscale', 'video_edit', 'lipsync', 'generate_video']
    let primaryVideoStep: string | null = null
    for (const stepId of videoStepPriority) {
      const state = run.step_states[stepId]
      if (state?.status === 'completed' && getArtifactVideo(run, stepId)) {
        primaryVideoStep = stepId
        break
      }
    }

    if (primaryVideoStep) {
      const videoUrl = getArtifactVideo(run, primaryVideoStep)!
      const thumbnailUrl = getArtifactThumb(run, primaryVideoStep) || getArtifactThumb(run, 'scene_compose')
      segments.push({
        id: uid(),
        order: 0,
        start: 0,
        duration: defaultDuration,
        prompt: (run.inputs?.video_prompt as string) || '',
        thumbnailUrl,
        videoUrl,
        firstFrameB64: null,
        lastFrameB64: null,
        params: { ...DEFAULT_SEGMENT_PARAMS, fps, frames },
        trimStart: 0,
        sourceStepId: primaryVideoStep,
        status: 'ready',
      })
      cursor = defaultDuration
    }

    // Audio cues from pipeline steps — create tracks dynamically
    const audioStepIds = ['voice', 'sound_fx', 'music']
    const createdTracks: AudioTrackDef[] = []
    for (const stepId of audioStepIds) {
      const stepState = run.step_states[stepId]
      if (stepState?.status !== 'completed') continue
      const audioUrl = getArtifactAudio(run, stepId)
      if (!audioUrl) continue

      const tIdx = createdTracks.length
      const track: AudioTrackDef = {
        id: uid(),
        label: stepId.replace(/_/g, ' '),
        color: TRACK_COLORS[tIdx % TRACK_COLORS.length],
        height: 48,
      }
      createdTracks.push(track)

      audioCues.push({
        id: uid(),
        track: track.id,
        start: 0,
        duration: defaultDuration,
        label: stepId.replace(/_/g, ' '),
        audioUrl,
        volume: stepId === 'music' ? 0.4 : 1.0,
        waveformPeaks: null,
        sourceStepId: stepId,
      })
    }

    const totalDuration = cursor || defaultDuration
    set({
      segments,
      audioCues,
      audioTracks: createdTracks,
      viewport: { ...DEFAULT_VIEWPORT },
      playback: { isPlaying: false, currentTime: 0, totalDuration },
      drag: { ...DEFAULT_DRAG },
      selectedSegmentId: null,
      selectedAudioCueId: null,
    })
  },

  reset: () => set({
    segments: [],
    audioCues: [],
    audioTracks: [],
    viewport: { ...DEFAULT_VIEWPORT },
    playback: { ...DEFAULT_PLAYBACK },
    drag: { ...DEFAULT_DRAG },
    selectedSegmentId: null,
    selectedAudioCueId: null,
    relayVideoUrl: null,
    relaySegmentIds: [],
  }),

  undo: () => {
    if (_historyIdx <= 0) return
    _historyIdx--
    const snap = _history[_historyIdx]
    if (snap) set({ segments: snap.segments, audioCues: snap.audioCues })
  },

  redo: () => {
    if (_historyIdx >= _history.length - 1) return
    _historyIdx++
    const snap = _history[_historyIdx]
    if (snap) set({ segments: snap.segments, audioCues: snap.audioCues })
  },
}))
