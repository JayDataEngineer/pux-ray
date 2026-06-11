import { create } from 'zustand'

// ═══════════════════════════════════════════════════════════════════════════
// IndexedDB for large video assets (beyond localStorage ~5MB limit)
// ═══════════════════════════════════════════════════════════════════════════
const DB_NAME = 'TechNoirAssetsDB'
const DB_VERSION = 1
const STORE_NAME = 'videos'

let dbPromise: Promise<IDBDatabase> | null = null

function getDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    }
  })

  return dbPromise
}

async function saveVideoToIndexedDB(assetId: string, dataUrl: string): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  const store = tx.objectStore(STORE_NAME)
  store.put(dataUrl, assetId)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

async function loadVideoFromIndexedDB(assetId: string): Promise<string | null> {
  try {
    const db = await getDB()
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.get(assetId)

    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error)
    })
  } catch {
    return null
  }
}

async function deleteVideoFromIndexedDB(assetId: string): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  const store = tx.objectStore(STORE_NAME)
  store.delete(assetId)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

// ═══════════════════════════════════════════════════════════════════════════

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
  initialized: boolean
  initialize: () => Promise<void>
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
async function loadPersisted(): Promise<Asset[]> {
  try {
    const raw = localStorage.getItem('tech_noir_assets')
    const assets: Asset[] = raw ? JSON.parse(raw) : []

    // Restore video data URLs from IndexedDB
    const videos = assets.filter(a => a.type === 'video')
    for (const video of videos) {
      if (!video.url.startsWith('data:')) {
        // URL is a placeholder (just asset ID), load actual data from IndexedDB
        const dataUrl = await loadVideoFromIndexedDB(video.id)
        if (dataUrl) {
          video.url = dataUrl
        } else {
          // Video not found in IndexedDB, remove from assets
          console.warn('[Assets] Video not found in IndexedDB:', video.name)
          assets.splice(assets.indexOf(video), 1)
        }
      }
    }

    console.log('[Assets] Loaded', assets.length, 'assets from localStorage + IndexedDB')
    return assets
  } catch (e) {
    console.error('[Assets] Failed to load assets:', e)
    return []
  }
}

function persist(assets: Asset[]) {
  try {
    // Separate videos from other assets
    const videos = assets.filter(a => a.type === 'video')
    const others = assets.filter(a => a.type !== 'video')

    // Save non-video assets to localStorage (images, audio - small enough)
    localStorage.setItem('tech_noir_assets', JSON.stringify(others))

    // Save video data URLs to IndexedDB, store placeholder in localStorage
    const videoMetadata = videos.map(({ id, name, type, category, mediaType, sizeBytes, source, createdAt, prompt }) => ({
      id, name, type, category, mediaType, sizeBytes, source, createdAt, prompt,
      url: id  // Placeholder - actual data loaded from IndexedDB
    }))

    // Save video metadata to separate localStorage key
    localStorage.setItem('tech_noir_videos', JSON.stringify(videoMetadata))

    // Save actual video data to IndexedDB
    Promise.all(
      videos.map(v => {
        if (v.url.startsWith('data:')) {
          return saveVideoToIndexedDB(v.id, v.url)
        }
        return Promise.resolve()
      })
    ).catch(err => console.error('[Assets] Failed to save videos to IndexedDB:', err))

    console.log('[Assets] Persisted', others.length, 'small assets to localStorage,', videos.length, 'videos to IndexedDB')
  } catch (e) {
    console.error('[Assets] Failed to persist assets:', e)
  }
}

export const useAssetStore = create<AssetStore>((set, get) => ({
  assets: [],
  initialized: false,

  initialize: async () => {
    if (get().initialized) return
    const assets = await loadPersisted()
    set({ assets, initialized: true })
  },

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

    // Save video data to IndexedDB if it's a video with data URL
    if (asset.type === 'video' && asset.url.startsWith('data:')) {
      saveVideoToIndexedDB(asset.id, asset.url).catch(err =>
        console.error('[Assets] Failed to save video to IndexedDB:', err)
      )
    }

    return asset
  },

  removeAsset: (id) => {
    // Delete from IndexedDB if it's a video
    const existing = get().assets.find(a => a.id === id)
    if (existing?.type === 'video') {
      deleteVideoFromIndexedDB(id).catch(err =>
        console.error('[Assets] Failed to delete video from IndexedDB:', err)
      )
    }

    set((s) => {
      const updated = s.assets.filter((a) => a.id !== id)
      persist(updated)
      return { assets: updated }
    })
  },

  renameAsset: (id, name) => set((s) => {
    const updated = s.assets.map((a) => a.id === id ? { ...a, name } : a)
    persist(updated)
    return { assets: updated }
  }),

  getByType: (type) => get().assets.filter((a) => a.type === type),

  clear: () => {
    localStorage.removeItem('tech_noir_assets')
    localStorage.removeItem('tech_noir_videos')

    // Clear IndexedDB
    getDB().then(db => {
      const tx = db.transaction(STORE_NAME, 'readwrite')
      tx.objectStore(STORE_NAME).clear()
    }).catch(err => console.error('[Assets] Failed to clear IndexedDB:', err))

    set({ assets: [] })
  },
}))
