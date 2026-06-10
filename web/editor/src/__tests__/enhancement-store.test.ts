import { describe, it, expect, beforeEach } from 'vitest'
import { useEnhanceStore } from '../stores/enhancement'

beforeEach(() => {
  // Clear all models
  const { models, activeId } = useEnhanceStore.getState()
  for (const m of models) {
    useEnhanceStore.getState().removeModel(m.id)
  }
})

describe('EnhancementStore', () => {
  it('starts with no models', () => {
    expect(useEnhanceStore.getState().models).toHaveLength(0)
    expect(useEnhanceStore.getState().activeId).toBeNull()
  })

  it('addModel creates and auto-activates first model', () => {
    const model = useEnhanceStore.getState().addModel({
      name: 'GPT-4o',
      baseUrl: 'https://api.openai.com/v1',
      apiKey: 'sk-test',
      model: 'gpt-4o',
    })
    expect(model.id).toBeTruthy()
    expect(useEnhanceStore.getState().models).toHaveLength(1)
    expect(useEnhanceStore.getState().activeId).toBe(model.id)
  })

  it('addModel does not override existing activeId', () => {
    const m1 = useEnhanceStore.getState().addModel({
      name: 'First', baseUrl: 'https://a.com/v1', apiKey: 'k1', model: 'm1',
    })
    const m2 = useEnhanceStore.getState().addModel({
      name: 'Second', baseUrl: 'https://b.com/v1', apiKey: 'k2', model: 'm2',
    })
    // Active should still be m1
    expect(useEnhanceStore.getState().activeId).toBe(m1.id)
  })

  it('updateModel patches fields', () => {
    const m = useEnhanceStore.getState().addModel({
      name: 'Test', baseUrl: 'https://test.com/v1', apiKey: 'key', model: 'm',
    })
    useEnhanceStore.getState().updateModel(m.id, { name: 'Updated', apiKey: 'new-key' })
    const updated = useEnhanceStore.getState().models.find(x => x.id === m.id)!
    expect(updated.name).toBe('Updated')
    expect(updated.apiKey).toBe('new-key')
    expect(updated.baseUrl).toBe('https://test.com/v1') // unchanged
  })

  it('removeModel falls back activeId to next model', () => {
    const m1 = useEnhanceStore.getState().addModel({
      name: 'A', baseUrl: 'https://a.com/v1', apiKey: 'k1', model: 'm1',
    })
    const m2 = useEnhanceStore.getState().addModel({
      name: 'B', baseUrl: 'https://b.com/v1', apiKey: 'k2', model: 'm2',
    })
    useEnhanceStore.getState().setActive(m1.id)
    useEnhanceStore.getState().removeModel(m1.id)
    // Should fall back to m2
    expect(useEnhanceStore.getState().activeId).toBe(m2.id)
  })

  it('removeModel clears activeId if no models left', () => {
    const m = useEnhanceStore.getState().addModel({
      name: 'Only', baseUrl: 'https://a.com/v1', apiKey: 'k', model: 'm',
    })
    useEnhanceStore.getState().removeModel(m.id)
    expect(useEnhanceStore.getState().activeId).toBeNull()
  })

  it('setActive changes the active model', () => {
    const m1 = useEnhanceStore.getState().addModel({
      name: 'A', baseUrl: 'https://a.com/v1', apiKey: 'k1', model: 'm1',
    })
    const m2 = useEnhanceStore.getState().addModel({
      name: 'B', baseUrl: 'https://b.com/v1', apiKey: 'k2', model: 'm2',
    })
    useEnhanceStore.getState().setActive(m2.id)
    expect(useEnhanceStore.getState().activeId).toBe(m2.id)
  })

  it('activeModel returns the correct model', () => {
    const m = useEnhanceStore.getState().addModel({
      name: 'Active', baseUrl: 'https://a.com/v1', apiKey: 'k', model: 'm',
    })
    const active = useEnhanceStore.getState().activeModel()
    expect(active).toBeTruthy()
    expect(active!.id).toBe(m.id)
    expect(active!.name).toBe('Active')
  })

  it('activeModel returns undefined when nothing is active', () => {
    expect(useEnhanceStore.getState().activeModel()).toBeUndefined()
  })

  it('persists to localStorage', () => {
    useEnhanceStore.getState().addModel({
      name: 'Persist', baseUrl: 'https://a.com/v1', apiKey: 'k', model: 'm',
    })
    const raw = localStorage.getItem('tech_noir_enhance')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!)
    expect(parsed.models).toHaveLength(1)
  })
})
