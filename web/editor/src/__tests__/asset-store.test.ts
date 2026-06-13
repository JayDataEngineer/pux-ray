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

  it('addAsset persists metadata to localStorage', () => {
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

  it('addAsset strips data URLs from localStorage (uses IndexedDB placeholder)', () => {
    const bigDataUrl = 'data:image/png;base64,' + 'A'.repeat(5000)
    useAssetStore.getState().addAsset({
      name: 'big.png',
      type: 'image',
      category: 'image',
      mediaType: 'image/png',
      url: bigDataUrl,
      sizeBytes: 3750,
      source: 'generated',
    })
    const raw = localStorage.getItem('tech_noir_assets')
    const parsed = JSON.parse(raw!)
    // The stored url should be the asset ID (placeholder), NOT the data URL
    expect(parsed[0].url).not.toBe(bigDataUrl)
    expect(parsed[0].url).toBeTruthy() // should be the ID
  })

  it('addAsset leaves non-data URLs as-is in localStorage', () => {
    useAssetStore.getState().addAsset({
      name: 'server.png',
      type: 'image',
      category: 'image',
      mediaType: 'image/png',
      url: '/v1/artifacts/123/image.png',
      sizeBytes: 0,
      source: 'uploaded',
    })
    const raw = localStorage.getItem('tech_noir_assets')
    const parsed = JSON.parse(raw!)
    expect(parsed[0].url).toBe('/v1/artifacts/123/image.png')
  })

  it('in-memory asset retains the full data URL', () => {
    const dataUrl = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='
    useAssetStore.getState().addAsset({
      name: 'voice.wav',
      type: 'audio',
      category: 'voice',
      mediaType: 'audio/wav',
      url: dataUrl,
      sizeBytes: 44,
      source: 'generated',
    })
    // The in-memory asset should still have the full URL
    const assets = useAssetStore.getState().assets
    expect(assets[0].url).toBe(dataUrl)
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
