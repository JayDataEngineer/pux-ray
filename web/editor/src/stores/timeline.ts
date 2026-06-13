import { create } from 'zustand'
import type { WorkflowRun, WorkflowSpec } from '../types'
import type { TimelineSegment, AudioCue, AudioTrackDef, TimelineViewport, PlaybackState, DragState } from '../types/timeline'
import { TRACK_COLORS, DEFAULT_SEGMENT_PARAMS } from '../types/timeline'

let _nextId = 1
function uid(): string {
  return `seg_${_nextId++}_${Date.now().toString(36)}`
}

// ═══════════════════════════════════════════════════════════════════════════
// IndexedDB for large timeline data (video URLs, frame b64, relay video)
// ═══════════════════════════════════════════════════════════════════════════
const TL_DB_NAME = 'TechNoirTimelineDB'
const TL_DB_VERSION = 1
const TL_STORE = 'blobs'

let tlDbPromise: Promise<IDBDatabase> | null = null

function getTLDB(): Promise<IDBDatabase> {
  if (tlDbPromise) return tlDbPromise
  tlDbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(TL_DB_NAME, TL_DB_VERSION)
    req.onerror = () => reject(req.error)
    req.onsuccess = () => resolve(req.result)
    req.onupgradeneeded = (ev) => {
      const db = (ev.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(TL_STORE)) {
        db.createObjectStore(TL_STORE)
      }
    }
  })
  return tlDbPromise
}

async function tlSave(key: string, value: string): Promise<void> {
  const db = await getTLDB()
  const tx = db.transaction(TL_STORE, 'readwrite')
  tx.objectStore(TL_STORE).put(value, key)
  await new Promise<void>((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error) })
}

async function tlLoad(key: string): Promise<string | null> {
  try {
    const db = await getTLDB()
    const tx = db.transaction(TL_STORE, 'readonly')
    const req = tx.objectStore(TL_STORE).get(key)
    return new Promise((res, rej) => { req.onsuccess = () => res(req.result || null); req.onerror = () => rej(req.error) })
  } catch { return null }
}

export async function _tlDelete(key: string): Promise<void> {
  const db = await getTLDB()
  const tx = db.transaction(TL_STORE, 'readwrite')
  tx.objectStore(TL_STORE).delete(key)
  await new Promise<void>((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error) })
}

async function tlClearAll(): Promise<void> {
  const db = await getTLDB()
  const tx = db.transaction(TL_STORE, 'readwrite')
  tx.objectStore(TL_STORE).clear()
  await new Promise<void>((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error) })
}

// ═══════════════════════════════════════════════════════════════════════════
// Timeline persistence helpers
// ═══════════════════════════════════════════════════════════════════════════

// Fields that may contain large base64 data URLs — stored in IndexedDB
const BLOB_FIELDS = ['videoUrl', 'firstFrameB64', 'lastFrameB64', 'thumbnailUrl'] as const

function isDataUrl(val: unknown): val is string {
  return typeof val === 'string' && val.startsWith('data:')
}

// Blob keys use "TLB:" prefix to avoid collision with segment IDs ("seg_")
const BLOB_PREFIX = 'TLB:'

function blobKey(segId: string, field: string): string {
  return `${BLOB_PREFIX}${segId}_${field}`
}

function isBlobRef(val: unknown): val is string {
  return typeof val === 'string' && val.startsWith(BLOB_PREFIX)
}

// Save: strip large blobs from segments, store them in IndexedDB
async function persistTimeline(
  segments: TimelineSegment[],
  audioCues: AudioCue[],
  audioTracks: AudioTrackDef[],
  relayVideoUrl: string | null,
  relaySegmentIds: string[],
  relayAssetId: string | null,
) {
  try {
    // Collect blob saves
    const blobSaves: Promise<void>[] = []

    // Strip blobs from segments, replace with keys
    const strippedSegments = segments.map(seg => {
      const clone = { ...seg, params: { ...seg.params } }
      for (const field of BLOB_FIELDS) {
        const val = clone[field] as string | null
        if (isDataUrl(val)) {
          const key = blobKey(seg.id, field)
          blobSaves.push(tlSave(key, val))
          ;(clone as any)[field] = key // placeholder
        }
      }
      // Also handle controlVideoUrl
      if (isDataUrl(clone.controlVideoUrl)) {
        const key = blobKey(seg.id, 'controlVideoUrl')
        blobSaves.push(tlSave(key, clone.controlVideoUrl!))
        clone.controlVideoUrl = key
      }
      return clone
    })

    // Strip blobs from audio cues
    const strippedCues = audioCues.map(cue => {
      const clone = { ...cue }
      if (isDataUrl(clone.audioUrl)) {
        const key = `${BLOB_PREFIX}cue_${cue.id}_audioUrl`
        blobSaves.push(tlSave(key, clone.audioUrl!))
        clone.audioUrl = key
      }
      if (isDataUrl(clone.audioB64)) {
        const key = `${BLOB_PREFIX}cue_${cue.id}_audioB64`
        blobSaves.push(tlSave(key, clone.audioB64!))
        clone.audioB64 = key
      }
      return clone
    })

    // Relay video blob
    let relayBlobKey: string | null = null
    if (isDataUrl(relayVideoUrl)) {
      relayBlobKey = `${BLOB_PREFIX}relay_video`
      blobSaves.push(tlSave(relayBlobKey, relayVideoUrl))
    }

    // Save metadata to localStorage
    const metadata = {
      segments: strippedSegments,
      audioCues: strippedCues,
      audioTracks,
      relayVideoUrl: relayBlobKey || relayVideoUrl,
      relaySegmentIds,
      relayAssetId,
    }
    localStorage.setItem('tech_noir_timeline', JSON.stringify(metadata))

    // Fire-and-forget blob saves
    Promise.all(blobSaves).catch(err =>
      console.error('[Timeline] Failed to save blobs to IndexedDB:', err)
    )

    console.log('[Timeline] Persisted', segments.length, 'segments,', audioCues.length, 'cues,', blobSaves.length, 'blobs')
  } catch (e) {
    console.error('[Timeline] Failed to persist:', e)
  }
}

// Load: read metadata from localStorage, restore blobs from IndexedDB
async function loadPersistedTimeline(): Promise<{
  segments: TimelineSegment[]
  audioCues: AudioCue[]
  audioTracks: AudioTrackDef[]
  relayVideoUrl: string | null
  relaySegmentIds: string[]
  relayAssetId: string | null
} | null> {
  try {
    const raw = localStorage.getItem('tech_noir_timeline')
    if (!raw) return null

    const meta = JSON.parse(raw)
    if (!meta.segments) return null

    // Restore segment blobs
    const segments: TimelineSegment[] = []
    for (const seg of meta.segments) {
      const clone = { ...seg, params: { ...seg.params } }
      for (const field of BLOB_FIELDS) {
        const val = clone[field] as string | null
        if (isBlobRef(val)) {
          const data = await tlLoad(val)
          ;(clone as any)[field] = data
        }
      }
      // Restore controlVideoUrl
      if (isBlobRef(clone.controlVideoUrl)) {
        clone.controlVideoUrl = await tlLoad(clone.controlVideoUrl)
      }
      // Migrate old empty-string params to 'off' (ShadCN Select can't render '')
      if (clone.params.videoPromptType === '') clone.params.videoPromptType = 'off'
      if (clone.params.audioPromptType === '') clone.params.audioPromptType = 'off'
      if (clone.params.maskingSource === '') clone.params.maskingSource = 'off'
      segments.push(clone)
    }

    // Restore audio cue blobs
    const audioCues: AudioCue[] = []
    for (const cue of meta.audioCues || []) {
      const clone = { ...cue }
      if (isBlobRef(clone.audioUrl)) {
        clone.audioUrl = await tlLoad(clone.audioUrl)
      }
      if (isBlobRef(clone.audioB64)) {
        clone.audioB64 = await tlLoad(clone.audioB64)
      }
      audioCues.push(clone)
    }

    // Restore relay video
    let relayVideoUrl = meta.relayVideoUrl || null
    if (isBlobRef(relayVideoUrl)) {
      relayVideoUrl = await tlLoad(relayVideoUrl)
    }

    console.log('[Timeline] Loaded', segments.length, 'segments,', audioCues.length, 'cues from localStorage + IndexedDB')

    return {
      segments,
      audioCues,
      audioTracks: meta.audioTracks || [],
      relayVideoUrl,
      relaySegmentIds: meta.relaySegmentIds || [],
      relayAssetId: meta.relayAssetId || null,
    }
  } catch (e) {
    console.error('[Timeline] Failed to load persisted state:', e)
    return null
  }
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
  relayAssetId: string | null

  initialized: boolean
  initialize: () => Promise<void>

  addSegment: (partial?: Partial<TimelineSegment>) => TimelineSegment
  removeSegment: (id: string) => void
  updateSegment: (id: string, patch: Partial<TimelineSegment>) => void
  reorderSegments: (orderedIds: string[]) => void
  undo: () => void
  redo: () => void

  addAudioCue: (cue: Omit<AudioCue, 'id'>) => AudioCue
  removeAudioCue: (id: string) => void
  updateAudioCue: (id: string, patch: Partial<AudioCue>) => void

  setRelayVideo: (url: string | null, segmentIds: string[], assetId?: string | null) => void
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

// Helper: trigger persistence after state changes
function persistNow(state: { segments: TimelineSegment[]; audioCues: AudioCue[]; audioTracks: AudioTrackDef[]; relayVideoUrl: string | null; relaySegmentIds: string[]; relayAssetId: string | null }) {
  persistTimeline(state.segments, state.audioCues, state.audioTracks, state.relayVideoUrl, state.relaySegmentIds, state.relayAssetId)
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
  relayAssetId: null,
  initialized: false,

  initialize: async () => {
    if (get().initialized) return
    const loaded = await loadPersistedTimeline()
    if (loaded) {
      // Restore _nextId to avoid ID collisions
      const allIds = [
        ...loaded.segments.map(s => s.id),
        ...loaded.audioCues.map(c => c.id),
        ...loaded.audioTracks.map(t => t.id),
      ]
      for (const id of allIds) {
        const numMatch = id.match(/^seg_(\d+)_/)
        if (numMatch) {
          const num = parseInt(numMatch[1], 10)
          if (num >= _nextId) _nextId = num + 1
        }
      }

      const totalDuration = loaded.segments.reduce((t, seg) => Math.max(t, seg.start + seg.duration), 0)
      set({
        segments: loaded.segments,
        audioCues: loaded.audioCues,
        audioTracks: loaded.audioTracks,
        relayVideoUrl: loaded.relayVideoUrl,
        relaySegmentIds: loaded.relaySegmentIds,
        relayAssetId: loaded.relayAssetId,
        playback: { isPlaying: false, currentTime: 0, totalDuration },
        initialized: true,
      })
      // Seed undo history with loaded state
      pushHistory(loaded.segments, loaded.audioCues)
    } else {
      set({ initialized: true })
    }
  },

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
      controlVideoUrl: null,
      ...partial,
    }
    set((s) => {
      const segments = [...s.segments, seg]
      const totalDuration = segments.reduce((t, seg) => Math.max(t, seg.start + seg.duration), 0)
      pushHistory(segments, s.audioCues)
      return { segments, playback: { ...s.playback, totalDuration } }
    })
    persistNow(get())
    return seg
  },

  removeSegment: (id) => {
    set((s) => {
      const segments = s.segments.filter((seg) => seg.id !== id).map((seg, i) => ({ ...seg, order: i }))
      pushHistory(segments, s.audioCues)
      // Invalidate relay if removed segment was part of it
      const relayActive = s.relaySegmentIds.includes(id)
      return {
        segments,
        selectedSegmentId: s.selectedSegmentId === id ? null : s.selectedSegmentId,
        ...(relayActive ? { relayVideoUrl: null, relaySegmentIds: [] } : {}),
      }
    })
    persistNow(get())
  },

  updateSegment: (id, patch) => {
    set((s) => {
      const segments = s.segments.map((seg) => seg.id === id ? { ...seg, ...patch } : seg)
      pushHistory(segments, s.audioCues)
      return { segments }
    })
    persistNow(get())
  },

  reorderSegments: (orderedIds) => {
    set((s) => {
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
    })
    persistNow(get())
  },

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
    persistNow(get())
    return track
  },

  removeAudioTrack: (id) => {
    set((s) => ({
      audioTracks: s.audioTracks.filter((t) => t.id !== id),
      audioCues: s.audioCues.filter((c) => c.track !== id),
      selectedAudioCueId: s.selectedAudioCueId && s.audioCues.find(c => c.id === s.selectedAudioCueId)?.track === id ? null : s.selectedAudioCueId,
    }))
    persistNow(get())
  },

  addAudioCue: (cue) => {
    const full: AudioCue = { ...cue, id: uid() }
    set((s) => ({ audioCues: [...s.audioCues, full] }))
    persistNow(get())
    return full
  },

  removeAudioCue: (id) => {
    set((s) => ({
      audioCues: s.audioCues.filter((c) => c.id !== id),
      selectedAudioCueId: s.selectedAudioCueId === id ? null : s.selectedAudioCueId,
    }))
    persistNow(get())
  },

  updateAudioCue: (id, patch) => {
    set((s) => ({
      audioCues: s.audioCues.map((c) => c.id === id ? { ...c, ...patch } : c),
    }))
    persistNow(get())
  },

  setRelayVideo: (url, segmentIds, assetId) => {
    set({ relayVideoUrl: url, relaySegmentIds: segmentIds, relayAssetId: assetId })
    persistNow(get())
  },

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
        negativePrompt: '',
        thumbnailUrl,
        videoUrl,
        firstFrameB64: null,
        lastFrameB64: null,
        params: { ...DEFAULT_SEGMENT_PARAMS, fps, frames },
        trimStart: 0,
        sourceStepId: primaryVideoStep,
        status: 'ready',
        error: null,
        controlVideoUrl: null,
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
        audioB64: null,
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
    persistNow(get())
  },

  reset: () => {
    set({
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
      relayAssetId: null,
    })
    // Clear persisted data
    localStorage.removeItem('tech_noir_timeline')
    tlClearAll().catch(() => {})
  },

  undo: () => {
    if (_historyIdx <= 0) return
    _historyIdx--
    const snap = _history[_historyIdx]
    if (snap) {
      set({ segments: snap.segments, audioCues: snap.audioCues })
      persistNow(get())
    }
  },

  redo: () => {
    if (_historyIdx >= _history.length - 1) return
    _historyIdx++
    const snap = _history[_historyIdx]
    if (snap) {
      set({ segments: snap.segments, audioCues: snap.audioCues })
      persistNow(get())
    }
  },
}))
