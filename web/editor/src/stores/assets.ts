import { create } from 'zustand'

export type AssetCategory = 'image' | 'music' | 'voice' | 'sfx' | 'video' | 'other'

export interface Asset {
  id: string
  name: string
  type: 'image' | 'audio' | 'video' | 'other'
  category: AssetCategory
  mediaType: string
  url: string
  thumbnailUrl?: string
  sizeBytes: number
  source: 'generated' | 'uploaded'
  sourceRunId?: string
  sourceStepId?: string
  createdAt: string
  prompt?: string
}

export const CATEGORY_LABEL: Record<AssetCategory, string> = {
  image: 'Images', music: 'Music', voice: 'Voices', sfx: 'SFX', video: 'Video', other: 'Other',
}

export const CATEGORY_ORDER: AssetCategory[] = ['image', 'music', 'voice', 'sfx', 'video', 'other']

interface AssetStore {
  assets: Asset[]
  addAsset: (asset: Omit<Asset, 'id' | 'createdAt'>) => Asset
  removeAsset: (id: string) => void
  renameAsset: (id: string, name: string) => void
  getByType: (type: Asset['type']) => Asset[]
  clear: () => void
}

let _nextId = 1
let _nameCounters: Record<string, number> = {}
function uid(): string { return `asset_${_nextId++}_${Date.now().toString(36)}` }
export function nextAssetName(service: string, ext: string): string {
  if (!_nameCounters[service]) _nameCounters[service] = 0
  _nameCounters[service]++
  return `${service}_${_nameCounters[service]}.${ext}`
}

// Load persisted assets on init
function loadPersisted(): Asset[] {
  try {
    const raw = localStorage.getItem('tech_noir_assets')
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function persist(assets: Asset[]) {
  try { localStorage.setItem('tech_noir_assets', JSON.stringify(assets)) } catch {}
}

export const useAssetStore = create<AssetStore>((set, get) => ({
  assets: loadPersisted(),

  addAsset: (partial) => {
    const asset: Asset = {
      ...partial,
      id: uid(),
      createdAt: new Date().toISOString(),
    }
    set((s) => {
      const updated = [asset, ...s.assets]
      persist(updated)
      return { assets: updated }
    })
    return asset
  },

  removeAsset: (id) => set((s) => {
    const updated = s.assets.filter((a) => a.id !== id)
    persist(updated)
    return { assets: updated }
  }),

  renameAsset: (id, name) => set((s) => {
    const updated = s.assets.map((a) => a.id === id ? { ...a, name } : a)
    persist(updated)
    return { assets: updated }
  }),

  getByType: (type) => get().assets.filter((a) => a.type === type),

  clear: () => {
    localStorage.removeItem('tech_noir_assets')
    set({ assets: [] })
  },
}))
