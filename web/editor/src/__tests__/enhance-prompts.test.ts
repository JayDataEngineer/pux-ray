import { describe, it, expect } from 'vitest'
import { getEnhancePrompt, ENHANCEABLE_FIELDS } from '../lib/enhance-prompts'

describe('getEnhancePrompt', () => {
  it('returns Z-Image Turbo for generate with default model', () => {
    const prompt = getEnhancePrompt('generate', { model: 'z_image' })
    expect(prompt).toContain('Z-Image')
    expect(prompt).toContain('Turbo')
  })

  it('returns Z-Image Base for generate with z_image_base', () => {
    const prompt = getEnhancePrompt('generate', { model: 'z_image_base' })
    expect(prompt).toContain('Z-Image')
    expect(prompt).toContain('Base')
    expect(prompt).toContain('negative')
  })

  it('returns correct prompt for generate without formValues', () => {
    const prompt = getEnhancePrompt('generate')
    expect(prompt).toContain('Z-Image')
  })

  it('returns Flux Schnell for flux_schnell model', () => {
    const prompt = getEnhancePrompt('generate', { model: 'flux_schnell' })
    expect(prompt).toContain('Flux Schnell')
  })

  it('returns Flux Dev for flux model', () => {
    const prompt = getEnhancePrompt('generate', { model: 'flux' })
    expect(prompt).toContain('Flux')
  })

  it('returns Flux 2 Dev for flux2_dev model', () => {
    const prompt = getEnhancePrompt('generate', { model: 'flux2_dev' })
    expect(prompt).toContain('Flux 2 Dev')
  })

  it('returns Anima for anima_base', () => {
    const prompt = getEnhancePrompt('generate', { model: 'anima_base' })
    expect(prompt).toContain('Anima')
    expect(prompt).toContain('anime')
  })

  it('returns Qwen Image for qwen_image_20B', () => {
    const prompt = getEnhancePrompt('generate', { model: 'qwen_image_20B' })
    expect(prompt).toContain('Qwen Image')
  })

  it('returns EDIT prompt for edit service', () => {
    const prompt = getEnhancePrompt('edit')
    expect(prompt).toContain('edit')
    expect(prompt).toContain('action verb')
  })

  it('returns POSE_EDIT prompt for pose_edit service', () => {
    const prompt = getEnhancePrompt('pose_edit')
    expect(prompt).toContain('pose')
  })

  it('returns CHAR_SHEET prompt for generate_character_sheet service', () => {
    const prompt = getEnhancePrompt('generate_character_sheet')
    expect(prompt).toContain('character sheet')
  })

  it('returns MUSIC prompt for generate_music', () => {
    const prompt = getEnhancePrompt('generate_music')
    expect(prompt).toContain('music')
  })

  it('returns SOUND EFFECT prompt for generate_sound', () => {
    const prompt = getEnhancePrompt('generate_sound')
    expect(prompt).toContain('acoustic')
  })

  it('returns TTS prompt for tts_speak', () => {
    const prompt = getEnhancePrompt('tts_speak')
    expect(prompt).toContain('speech')
  })

  it('returns FALLBACK for unknown service', () => {
    const prompt = getEnhancePrompt('totally_unknown_service')
    expect(prompt).toContain('AI prompt enhancement')
  })

  it('every prompt tells the model to output only the prompt', () => {
    const services = [
      'generate', 'edit', 'pose_edit', 'generate_character_sheet', 'char_sheet',
      'generate_music', 'ace_step', 'generate_sound', 'moss_soundeffect',
      'tts_speak', 'voice_creator', 'unknown',
    ]
    const models = [
      'z_image', 'z_image_base', 'anima_base', 'flux', 'flux_schnell',
      'flux_chroma', 'flux2_dev', 'flux2_klein_9b', 'qwen_image_20B', 'hidream_o1',
    ]
    for (const s of services) {
      const p = getEnhancePrompt(s)
      expect(p.toLowerCase()).toContain('only')
    }
    for (const m of models) {
      const p = getEnhancePrompt('generate', { model: m })
      expect(p.toLowerCase()).toContain('only')
    }
  })
})

describe('ENHANCEABLE_FIELDS', () => {
  it('contains the expected fields', () => {
    expect(ENHANCEABLE_FIELDS.has('prompt')).toBe(true)
    expect(ENHANCEABLE_FIELDS.has('text')).toBe(true)
    expect(ENHANCEABLE_FIELDS.has('negative_prompt')).toBe(true)
    expect(ENHANCEABLE_FIELDS.has('instruct')).toBe(true)
    expect(ENHANCEABLE_FIELDS.has('lyrics')).toBe(true)
    expect(ENHANCEABLE_FIELDS.has('model')).toBe(false)
  })
})
