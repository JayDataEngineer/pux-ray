import { describe, it, expect } from 'vitest'
import { AUDIO_TRACKS, DEFAULT_SEGMENT_PARAMS } from '../types/timeline'

describe('Timeline Types — constants', () => {
  it('AUDIO_TRACKS has expected tracks', () => {
    expect(AUDIO_TRACKS).toHaveLength(3)
    expect(AUDIO_TRACKS.map(t => t.id)).toEqual(['voice', 'sfx', 'music'])
    // Each has a color and label
    for (const t of AUDIO_TRACKS) {
      expect(t.label).toBeTruthy()
      expect(t.color).toBeTruthy()
      expect(t.height).toBeGreaterThan(0)
    }
  })

  it('DEFAULT_SEGMENT_PARAMS has reasonable defaults', () => {
    expect(DEFAULT_SEGMENT_PARAMS.seed).toBeTypeOf('number')
    expect(DEFAULT_SEGMENT_PARAMS.width).toBeGreaterThan(0)
    expect(DEFAULT_SEGMENT_PARAMS.height).toBeGreaterThan(0)
    expect(DEFAULT_SEGMENT_PARAMS.frames).toBeGreaterThan(0)
    expect(DEFAULT_SEGMENT_PARAMS.fps).toBeGreaterThan(0)
    expect(DEFAULT_SEGMENT_PARAMS.guideScale).toBeGreaterThan(0)
    expect(DEFAULT_SEGMENT_PARAMS.width).toBe(768)
    expect(DEFAULT_SEGMENT_PARAMS.height).toBe(512)
    expect(DEFAULT_SEGMENT_PARAMS.fps).toBe(24)
  })
})
