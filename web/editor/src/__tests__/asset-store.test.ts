import { describe, it, expect, beforeEach } from 'vitest'
import { useAssetStore, nextAssetName, CATEGORY_LABEL, CATEGORY_ORDER } from '../stores/assets'

beforeEach(() => {
  useAssetStore.getState().clear()
})

describe('AssetStore — CRUD', () => {
  it('starts with empty assets', () => {
    expect(useAssetStore.getState().assets).toHaveLength(0)
  })

  it('addAsset creates an asset with id and timestamp', () => {
    const asset = useAssetStore.getState().addAsset({
      name: 'test.png',
      type: 'image',
      category: 'image',
      mediaType: 'image/png',
      url: 'data:image/png;base64,abc',
      sizeBytes: 1024,
      source: 'uploaded',
    })
    expect(asset.id).toBeTruthy()
    expect(asset.createdAt).toBeTruthy()
    expect(asset.name).toBe('test.png')
    expect(asset.source).toBe('uploaded')
  })

  it('addAsset persists to localStorage', () => {
    useAssetStore.getState().addAsset({
      name: 'persist.png',
      type: 'image',
      category: 'image',
      mediaType: 'image/png',
      url: 'data:image/png;base64,abc',
      sizeBytes: 512,
      source: 'generated',
    })
    const raw = localStorage.getItem('tech_noir_assets')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!)
    expect(parsed).toHaveLength(1)
    expect(parsed[0].name).toBe('persist.png')
  })

  it('removeAsset removes the asset', () => {
    const a = useAssetStore.getState().addAsset({
      name: 'to-delete.png', type: 'image', category: 'image',
      mediaType: 'image/png', url: 'x', sizeBytes: 0, source: 'uploaded',
    })
    useAssetStore.getState().removeAsset(a.id)
    expect(useAssetStore.getState().assets).toHaveLength(0)
  })

  it('renameAsset renames the asset', () => {
    const a = useAssetStore.getState().addAsset({
      name: 'old.png', type: 'image', category: 'image',
      mediaType: 'image/png', url: 'x', sizeBytes: 0, source: 'uploaded',
    })
    useAssetStore.getState().renameAsset(a.id, 'new.png')
    expect(useAssetStore.getState().assets[0].name).toBe('new.png')
  })

  it('getByType filters correctly', () => {
    useAssetStore.getState().addAsset({
      name: 'img.png', type: 'image', category: 'image',
      mediaType: 'image/png', url: 'x', sizeBytes: 0, source: 'uploaded',
    })
    useAssetStore.getState().addAsset({
      name: 'song.mp3', type: 'audio', category: 'music',
      mediaType: 'audio/mp3', url: 'y', sizeBytes: 0, source: 'uploaded',
    })
    expect(useAssetStore.getState().getByType('image')).toHaveLength(1)
    expect(useAssetStore.getState().getByType('audio')).toHaveLength(1)
    expect(useAssetStore.getState().getByType('video')).toHaveLength(0)
  })

  it('clear removes all assets and localStorage', () => {
    useAssetStore.getState().addAsset({
      name: 'a.png', type: 'image', category: 'image',
      mediaType: 'image/png', url: 'x', sizeBytes: 0, source: 'uploaded',
    })
    useAssetStore.getState().clear()
    expect(useAssetStore.getState().assets).toHaveLength(0)
    expect(localStorage.getItem('tech_noir_assets')).toBeNull()
  })
})

describe('nextAssetName', () => {
  it('generates sequential names', () => {
    const n1 = nextAssetName('generate', 'png')
    const n2 = nextAssetName('generate', 'png')
    expect(n1).toMatch(/generate_\d+\.png/)
    expect(n2).toMatch(/generate_\d+\.png/)
  })
})

describe('CATEGORY_LABEL / CATEGORY_ORDER', () => {
  it('has labels for all categories', () => {
    for (const cat of CATEGORY_ORDER) {
      expect(CATEGORY_LABEL[cat]).toBeTruthy()
    }
  })
})
