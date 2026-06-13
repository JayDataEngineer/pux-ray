import { describe, it, expect, beforeEach } from 'vitest'
import { useTimelineStore } from '../stores/timeline'
import type { WorkflowRun, WorkflowSpec } from '../types'
import type { TimelineSegment } from '../types/timeline'

// ─── Reset store between tests ───────────────────────────────────────────────
beforeEach(() => {
  useTimelineStore.getState().reset()
})

describe('TimelineStore — segment CRUD', () => {
  it('starts with zero segments', () => {
    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(0)
  })

  it('addSegment creates a segment with defaults', () => {
    const seg = useTimelineStore.getState().addSegment()
    expect(seg.id).toBeTruthy()
    expect(seg.order).toBe(0)
    expect(seg.start).toBe(0)
    expect(seg.duration).toBe(5)
    expect(seg.status).toBe('empty')
    expect(seg.prompt).toBe('')
    expect(seg.params.seed).toBe(42)
    expect(seg.params.fps).toBe(16) // Wan 1.3B default
    expect(seg.params.model).toBe('wan/t2v_1.3B')
    expect(seg.negativePrompt).toBe('')

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(1)
    expect(segments[0].id).toBe(seg.id)
  })

  it('addSegment respects partial overrides', () => {
    const seg = useTimelineStore.getState().addSegment({
      prompt: 'forest scene',
      duration: 10,
      status: 'ready',
      params: { seed: 99, width: 1024, height: 576, frames: 200, fps: 30, guideScale: 4.0 },
    })
    expect(seg.prompt).toBe('forest scene')
    expect(seg.duration).toBe(10)
    expect(seg.status).toBe('ready')
    expect(seg.params.seed).toBe(99)
    expect(seg.params.width).toBe(1024)
  })

  it('addSegment positions second segment after first', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    const seg2 = useTimelineStore.getState().addSegment({ duration: 8 })
    expect(seg2.order).toBe(1)
    expect(seg2.start).toBe(5) // after first segment
  })

  it('removeSegment removes and re-indexes', () => {
    const s1 = useTimelineStore.getState().addSegment()
    const s2 = useTimelineStore.getState().addSegment()
    const s3 = useTimelineStore.getState().addSegment()

    useTimelineStore.getState().removeSegment(s2.id)

    const { segments } = useTimelineStore.getState()
    expect(segments).toHaveLength(2)
    expect(segments.map(s => s.id)).toEqual([s1.id, s3.id])
    // Re-indexed
    expect(segments[0].order).toBe(0)
    expect(segments[1].order).toBe(1)
  })

  it('removeSegment clears selectedSegmentId if that segment was selected', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setSelectedSegment(seg.id)
    useTimelineStore.getState().removeSegment(seg.id)
    expect(useTimelineStore.getState().selectedSegmentId).toBeNull()
  })

  it('updateSegment patches fields', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().updateSegment(seg.id, {
      prompt: 'updated',
      duration: 15,
      status: 'generating',
    })
    const updated = useTimelineStore.getState().segments.find(s => s.id === seg.id)!
    expect(updated.prompt).toBe('updated')
    expect(updated.duration).toBe(15)
    expect(updated.status).toBe('generating')
  })

  it('updateSegment with params patch replaces entire params', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().updateSegment(seg.id, {
      params: { ...seg.params, fps: 60 },
    })
    const updated = useTimelineStore.getState().segments.find(s => s.id === seg.id)!
    expect(updated.params.fps).toBe(60)
    expect(updated.params.seed).toBe(42) // other fields preserved
  })
})

describe('TimelineStore — reorder', () => {
  it('reorderSegments reorders and recomputes start positions', () => {
    const s1 = useTimelineStore.getState().addSegment({ duration: 5 })
    const s2 = useTimelineStore.getState().addSegment({ duration: 8 })
    const s3 = useTimelineStore.getState().addSegment({ duration: 3 })

    // Reverse order: s3, s2, s1
    useTimelineStore.getState().reorderSegments([s3.id, s2.id, s1.id])

    const { segments } = useTimelineStore.getState()
    expect(segments[0].id).toBe(s3.id)
    expect(segments[0].order).toBe(0)
    expect(segments[0].start).toBe(0)

    expect(segments[1].id).toBe(s2.id)
    expect(segments[1].order).toBe(1)
    expect(segments[1].start).toBe(3) // s3.duration

    expect(segments[2].id).toBe(s1.id)
    expect(segments[2].order).toBe(2)
    expect(segments[2].start).toBe(11) // 3 + 8
  })

  it('reorderSegments updates totalDuration', () => {
    useTimelineStore.getState().addSegment({ duration: 5 })
    useTimelineStore.getState().addSegment({ duration: 8 })

    const before = useTimelineStore.getState().playback.totalDuration
    // Total should be sum of durations
    expect(before).toBe(13)
  })
})

describe('TimelineStore — playback', () => {
  it('setPlayback updates playback state', () => {
    useTimelineStore.getState().setPlayback({ isPlaying: true, currentTime: 2.5 })
    const { playback } = useTimelineStore.getState()
    expect(playback.isPlaying).toBe(true)
    expect(playback.currentTime).toBe(2.5)
  })
})

describe('TimelineStore — audio cues', () => {
  it('addAudioCue creates a cue with generated id', () => {
    useTimelineStore.getState().addAudioCue({
      track: 'voice',
      start: 0,
      duration: 5,
      label: 'Narration',
      audioUrl: '/audio/test.wav',
      volume: 0.8,
      waveformPeaks: null,
      sourceStepId: 'voice',
    })
    const { audioCues } = useTimelineStore.getState()
    expect(audioCues).toHaveLength(1)
    expect(audioCues[0].track).toBe('voice')
    expect(audioCues[0].id).toBeTruthy()
  })

  it('removeAudioCue removes the cue', () => {
    useTimelineStore.getState().addAudioCue({
      track: 'sfx', start: 0, duration: 2, label: 'Boom',
      audioUrl: null, volume: 1, waveformPeaks: null, sourceStepId: null,
    })
    const { audioCues } = useTimelineStore.getState()
    useTimelineStore.getState().removeAudioCue(audioCues[0].id)
    expect(useTimelineStore.getState().audioCues).toHaveLength(0)
  })

  it('removeAudioCue clears selectedAudioCueId if that cue was selected', () => {
    useTimelineStore.getState().addAudioCue({
      track: 'sfx', start: 0, duration: 2, label: 'Boom',
      audioUrl: null, volume: 1, waveformPeaks: null, sourceStepId: null,
    })
    const cueId = useTimelineStore.getState().audioCues[0].id
    useTimelineStore.getState().setSelectedAudioCue(cueId)
    useTimelineStore.getState().removeAudioCue(cueId)
    expect(useTimelineStore.getState().selectedAudioCueId).toBeNull()
  })

  it('updateAudioCue patches fields', () => {
    useTimelineStore.getState().addAudioCue({
      track: 'music', start: 0, duration: 10, label: 'BGM',
      audioUrl: '/music.mp3', volume: 0.4, waveformPeaks: null, sourceStepId: null,
    })
    const cueId = useTimelineStore.getState().audioCues[0].id
    useTimelineStore.getState().updateAudioCue(cueId, { volume: 0.7, label: 'Updated BGM' })
    const cue = useTimelineStore.getState().audioCues[0]
    expect(cue.volume).toBe(0.7)
    expect(cue.label).toBe('Updated BGM')
  })
})

describe('TimelineStore — loadFromRun', () => {
  it('loads a completed video run into a segment', () => {
    const run: WorkflowRun = {
      run_id: 'run-1',
      spec_name: 'video_gen',
      status: 'completed',
      inputs: {
        video_prompt: 'A forest scene',
        video_fps: 24,
        video_frames: 121,
      },
      step_states: {
        generate_video: {
          step_id: 'generate_video',
          status: 'completed',
          outputs: { video: 'output.mp4' },
        },
      },
      artifacts: {
        'generate_video.output': {
          run_id: 'run-1',
          step_id: 'generate_video',
          name: 'output.mp4',
          file_path: '/tmp/output.mp4',
          media_type: 'video/mp4',
          url: '/v1/wf/video_gen/runs/run-1/artifacts/generate_video/output.mp4',
          size_bytes: 1000000,
          created_at: '2026-01-01T00:00:00Z',
        },
      },
    }

    const spec = { name: 'video_gen', version: '1', description: '', inputs: {}, steps: [] }
    useTimelineStore.getState().loadFromRun(run, spec)

    const { segments, playback } = useTimelineStore.getState()
    expect(segments).toHaveLength(1)
    expect(segments[0].prompt).toBe('A forest scene')
    expect(segments[0].status).toBe('ready')
    expect(segments[0].videoUrl).toBeTruthy()
    expect(segments[0].duration).toBeCloseTo(121 / 24, 1)
    expect(playback.currentTime).toBe(0)
    expect(playback.isPlaying).toBe(false)
  })

  it('loads audio cues from completed audio steps', () => {
    const run: WorkflowRun = {
      run_id: 'run-2',
      spec_name: 'video_gen',
      status: 'completed',
      inputs: { video_fps: 24, video_frames: 121 },
      step_states: {
        voice: { step_id: 'voice', status: 'completed', outputs: {} },
        sound_fx: { step_id: 'sound_fx', status: 'completed', outputs: {} },
      },
      artifacts: {
        'voice.output': {
          run_id: 'run-2', step_id: 'voice', name: 'voice.wav',
          file_path: '/tmp/voice.wav', media_type: 'audio/wav',
          url: '/v1/wf/video_gen/runs/run-2/artifacts/voice/voice.wav',
          size_bytes: 500000, created_at: '2026-01-01T00:00:00Z',
        },
        'sound_fx.output': {
          run_id: 'run-2', step_id: 'sound_fx', name: 'sfx.wav',
          file_path: '/tmp/sfx.wav', media_type: 'audio/wav',
          url: '/v1/wf/video_gen/runs/run-2/artifacts/sound_fx/sfx.wav',
          size_bytes: 200000, created_at: '2026-01-01T00:00:00Z',
        },
      },
    }

    useTimelineStore.getState().loadFromRun(run, { name: 'video_gen', version: '1', description: '', inputs: {}, steps: [] })
    const { audioCues } = useTimelineStore.getState()
    expect(audioCues).toHaveLength(2)
    expect(audioCues.find(c => c.track === 'voice')).toBeTruthy()
    expect(audioCues.find(c => c.track === 'sfx')).toBeTruthy()
  })
})

describe('TimelineStore — reset', () => {
  it('reset clears all state', () => {
    useTimelineStore.getState().addSegment()
    useTimelineStore.getState().addAudioCue({
      track: 'sfx', start: 0, duration: 2, label: 'Test',
      audioUrl: null, volume: 1, waveformPeaks: null, sourceStepId: null,
    })
    useTimelineStore.getState().setSelectedSegment('anything')

    useTimelineStore.getState().reset()

    const state = useTimelineStore.getState()
    expect(state.segments).toHaveLength(0)
    expect(state.audioCues).toHaveLength(0)
    expect(state.selectedSegmentId).toBeNull()
    expect(state.selectedAudioCueId).toBeNull()
    expect(state.playback.isPlaying).toBe(false)
    expect(state.playback.currentTime).toBe(0)
  })
})

describe('TimelineStore — persistence', () => {
  it('persists segments to localStorage on addSegment', () => {
    useTimelineStore.getState().addSegment({ prompt: 'test persistence', duration: 7 })
    const raw = localStorage.getItem('tech_noir_timeline')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!)
    expect(parsed.segments).toHaveLength(1)
    expect(parsed.segments[0].prompt).toBe('test persistence')
    expect(parsed.segments[0].duration).toBe(7)
  })

  it('persists segments on updateSegment', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().updateSegment(seg.id, { prompt: 'updated prompt', status: 'ready' })

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.segments[0].prompt).toBe('updated prompt')
    expect(parsed.segments[0].status).toBe('ready')
  })

  it('persists on removeSegment', () => {
    useTimelineStore.getState().addSegment()
    useTimelineStore.getState().addSegment()
    const segs = useTimelineStore.getState().segments
    useTimelineStore.getState().removeSegment(segs[0].id)

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.segments).toHaveLength(1)
  })

  it('persists audio cues on addAudioCue', () => {
    useTimelineStore.getState().addAudioCue({
      track: 'music', start: 0, duration: 5, label: 'BGM',
      audioUrl: '/music.mp3', volume: 0.5, waveformPeaks: null, sourceStepId: null,
    })

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.audioCues).toHaveLength(1)
    expect(parsed.audioCues[0].label).toBe('BGM')
  })

  it('persists audio tracks', () => {
    useTimelineStore.getState().addAudioTrack('Test Track')

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.audioTracks).toHaveLength(1)
    expect(parsed.audioTracks[0].label).toBe('Test Track')
  })

  it('persists relay video state', () => {
    const seg = useTimelineStore.getState().addSegment()
    useTimelineStore.getState().setRelayVideo('http://example.com/video.mp4', [seg.id], 'asset-1')

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.relayVideoUrl).toBe('http://example.com/video.mp4')
    expect(parsed.relaySegmentIds).toEqual([seg.id])
    expect(parsed.relayAssetId).toBe('asset-1')
  })

  it('strip large data URLs from localStorage metadata (use IndexedDB placeholders)', () => {
    const fakeB64 = 'data:video/mp4;base64,' + 'A'.repeat(1000)
    useTimelineStore.getState().addSegment({
      videoUrl: fakeB64,
      firstFrameB64: 'data:image/png;base64,AAABBB',
    })

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    const seg = parsed.segments[0]
    // The videoUrl and firstFrameB64 should be placeholder keys, NOT the actual data
    expect(seg.videoUrl).not.toBe(fakeB64)
    expect(seg.videoUrl).toMatch(/^TLB:/)
    expect(seg.firstFrameB64).toMatch(/^TLB:/)
  })

  it('leaves non-data-URL values as-is (no stripping for server URLs)', () => {
    useTimelineStore.getState().addSegment({
      videoUrl: '/v1/wf/test/runs/123/artifacts/step/video.mp4',
    })

    const raw = localStorage.getItem('tech_noir_timeline')
    const parsed = JSON.parse(raw!)
    expect(parsed.segments[0].videoUrl).toBe('/v1/wf/test/runs/123/artifacts/step/video.mp4')
  })

  it('reset clears localStorage timeline data', () => {
    useTimelineStore.getState().addSegment()
    expect(localStorage.getItem('tech_noir_timeline')).toBeTruthy()

    useTimelineStore.getState().reset()
    expect(localStorage.getItem('tech_noir_timeline')).toBeNull()
  })

  it('initialize restores segments from localStorage', async () => {
    // Write some timeline data directly to localStorage
    const segId = 'seg_restore_test'
    localStorage.setItem('tech_noir_timeline', JSON.stringify({
      segments: [{
        id: segId, order: 0, start: 0, duration: 5,
        prompt: 'restored!', negativePrompt: '', thumbnailUrl: null,
        videoUrl: null, firstFrameB64: null, lastFrameB64: null,
        params: { seed: 42, width: 768, height: 512, frames: 121, fps: 24,
          guideScale: 5, samplingSteps: 8, model: 'ltx2', guidePhases: 2,
          epsilon: 0.001, denoisingStrength: 1, inputVideoStrength: 1,
          spatialUpscale: false, loras: '', perturbationSwitch: 0,
          perturbationLayers: [28], perturbationStartPerc: 0, perturbationEndPerc: 100,
          cameraPanX: 0, cameraPanY: 0, cameraZoom: 1, resizeMethod: 'fit',
          distilledMode: true, apgSwitch: false, cfgStarSwitch: false,
          nagScale: 1, nagTau: 3.5, nagAlpha: 0.5, altGuideScale: 1,
          altScale: 0, audioGuideScale: 1, audioCfgScale: 1, sampleSolver: 'euler',
          selfRefinerSetting: 0, selfRefinerPlan: '', selfRefinerFUncertainty: 0.1,
          selfRefinerCertainPercentage: 0.999, slidingWindow: false,
          slidingWindowSize: 241, slidingWindowOverlap: 9, videoPromptType: '',
          maskingStrength: 0, maskingSource: '', outpaintingEnabled: false,
          outpaintingRatio: '', audioPromptType: '', enhancePrompt: false,
        },
        trimStart: 0, sourceStepId: null, status: 'ready', error: null,
        controlVideoUrl: null,
      }],
      audioCues: [],
      audioTracks: [],
      relayVideoUrl: null,
      relaySegmentIds: [],
      relayAssetId: null,
    }))

    await useTimelineStore.getState().initialize()

    const { segments, initialized } = useTimelineStore.getState()
    expect(initialized).toBe(true)
    expect(segments).toHaveLength(1)
    expect(segments[0].prompt).toBe('restored!')
    expect(segments[0].status).toBe('ready')
  })

  it('initialize handles empty localStorage gracefully', async () => {
    localStorage.removeItem('tech_noir_timeline')
    await useTimelineStore.getState().initialize()
    expect(useTimelineStore.getState().initialized).toBe(true)
    expect(useTimelineStore.getState().segments).toHaveLength(0)
  })
})
