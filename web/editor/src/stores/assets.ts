import { create } from 'zustand'

export interface Asset {
  id: string
  name: string
  type: 'image' | 'audio' | 'video' | 'other'
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

interface AssetStore {
  assets: Asset[]
  addAsset: (asset: Omit<Asset, 'id' | 'createdAt'>) => Asset
  removeAsset: (id: string) => void
  getByType: (type: Asset['type']) => Asset[]
  clear: () => void
}

let _nextId = 1
function uid(): string { return `asset_${_nextId++}_${Date.now().toString(36)}` }

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

  getByType: (type) => get().assets.filter((a) => a.type === type),

  clear: () => {
    localStorage.removeItem('tech_noir_assets')
    set({ assets: [] })
  },
}))
