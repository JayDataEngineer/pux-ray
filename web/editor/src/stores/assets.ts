import { create } from 'zustand'

// ═══════════════════════════════════════════════════════════════════════════
// IndexedDB for all asset blobs (images, audio, video — anything with data: URL)
// localStorage only holds metadata (~few KB). Base64 blobs go to IndexedDB.
// ═══════════════════════════════════════════════════════════════════════════
const DB_NAME = 'TechNoirAssetsDB'
const DB_VERSION = 2  // bumped: renamed store semantics
const STORE_NAME = 'blobs'

let dbPromise: Promise<IDBDatabase> | null = null

function getDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      // If old 'videos' store exists from v1, rename isn't possible — just
      // create 'blobs' and rely on loadPersisted to migrate data.
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
      // Delete the old v1 'videos' store if it exists
      if (db.objectStoreNames.contains('videos')) {
        db.deleteObjectStore('videos')
      }
    }
  })

  return dbPromise
}

async function saveBlob(assetId: string, dataUrl: string): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  tx.objectStore(STORE_NAME).put(dataUrl, assetId)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

async function loadBlob(assetId: string): Promise<string | null> {
  try {
    const db = await getDB()
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).get(assetId)
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result || null)
      request.onerror = () => reject(request.error)
    })
  } catch {
    return null
  }
}

async function deleteBlob(assetId: string): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  tx.objectStore(STORE_NAME).delete(assetId)
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

async function clearBlobs(): Promise<void> {
  const db = await getDB()
  const tx = db.transaction(STORE_NAME, 'readwrite')
  tx.objectStore(STORE_NAME).clear()
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

// Backward-compatible aliases (used by timeline store which imports the old names)
export const _saveVideoToIndexedDB = saveBlob
export const _loadVideoFromIndexedDB = loadBlob
export const _deleteVideoFromIndexedDB = deleteBlob

/** Check if a URL is a base64 data: URL (large — must go to IndexedDB) */
function isDataUrl(url: string | undefined): url is string {
  return typeof url === 'string' && url.startsWith('data:')
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
    // Load metadata from localStorage (single key now)
    const raw = localStorage.getItem('tech_noir_assets')
    if (!raw) {
      console.log('[Assets] No persisted assets found')
      return []
    }

    const assets: Asset[] = JSON.parse(raw)

    // Restore data URLs from IndexedDB for any asset whose url is a blob placeholder
    let restoredCount = 0
    let failedCount = 0
    for (const asset of assets) {
      // Placeholder format: url is just the asset ID (meaning blob is in IndexedDB)
      if (!isDataUrl(asset.url) && asset.url === asset.id) {
        const dataUrl = await loadBlob(asset.id)
        if (dataUrl) {
          asset.url = dataUrl
          restoredCount++
        } else {
          console.warn('[Assets] Blob not found in IndexedDB:', asset.name, asset.id)
          // Don't remove — keep metadata so user knows it existed
          asset.url = ''
          failedCount++
        }
      }
    }

    console.log(`[Assets] Loaded ${assets.length} assets from localStorage + IndexedDB (${restoredCount} restored, ${failedCount} missing)`)
    return assets
  } catch (e) {
    console.error('[Assets] Failed to load assets:', e)
    return []
  }
}

function persist(assets: Asset[]) {
  try {
    // Collect blob saves
    const blobSaves: Promise<void>[] = []

    // Build metadata: strip data: URLs, replace with ID placeholder
    const metadata = assets.map(a => {
      if (isDataUrl(a.url)) {
        // Queue the blob save
        blobSaves.push(saveBlob(a.id, a.url))
        // Return metadata with placeholder
        const { url: _url, thumbnailUrl: _thumb, ...rest } = a
        return { ...rest, url: a.id } as Asset
      }
      // Also handle thumbnailUrl if it's a data URL
      if (isDataUrl(a.thumbnailUrl) && !isDataUrl(a.url)) {
        blobSaves.push(saveBlob(a.id + '_thumb', a.thumbnailUrl!))
        const { thumbnailUrl: _thumb, ...rest } = a
        return { ...rest, thumbnailUrl: a.id + '_thumb' } as Asset
      }
      return a
    })

    // Save only metadata to localStorage (small — no base64 blobs)
    localStorage.setItem('tech_noir_assets', JSON.stringify(metadata))
    console.log('[Assets] Persisted', metadata.length, 'asset metadata to localStorage')

    // Fire-and-forget blob saves to IndexedDB
    const blobCount = blobSaves.length
    if (blobCount > 0) {
      Promise.all(blobSaves).then(() => {
        console.log('[Assets] Saved', blobCount, 'blobs to IndexedDB')
      }).catch(err => console.error('[Assets] Failed to save blobs to IndexedDB:', err))
    }
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

    return asset
  },

  removeAsset: (id) => {
    // Delete blob from IndexedDB
    deleteBlob(id).catch(err =>
      console.error('[Assets] Failed to delete blob from IndexedDB:', err)
    )

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
    // Also clean up old v1 key if present
    localStorage.removeItem('tech_noir_videos')

    // Clear IndexedDB
    clearBlobs().catch(err => console.error('[Assets] Failed to clear IndexedDB:', err))

    set({ assets: [] })
  },
}))
