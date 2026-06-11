# Video Asset Persistence - IndexedDB Implementation

## Problem
- localStorage has ~5-10MB limit
- Generated videos are base64 data URLs (several MB each)
- Users waited 5+ minutes for generation, then lost everything on refresh
- **UNACCEPTABLE UX**

## Solution: IndexedDB
IndexedDB provides **50-100MB+ storage** - enough for multiple generated videos.

## Implementation Details

### 1. IndexedDB Wrapper (`src/stores/assets.ts`)
```typescript
const DB_NAME = 'TechNoirAssetsDB'
const DB_VERSION = 1
const STORE_NAME = 'videos'

// Save video data URL to IndexedDB
await saveVideoToIndexedDB(assetId, dataUrl)

// Load video data URL from IndexedDB
const dataUrl = await loadVideoFromIndexedDB(assetId)

// Delete video from IndexedDB
await deleteVideoFromIndexedDB(assetId)
```

### 2. Persistence Strategy
- **Small assets** (images, audio): localStorage (fast, synchronous)
- **Large videos**: IndexedDB (async, ~50-100MB capacity)
- **Video metadata** (name, size, etc.): localStorage
- **Video data** (base64): IndexedDB

### 3. Loading Flow
```typescript
// On app initialization (App.tsx)
useEffect(() => {
  initializeAssets()  // Loads from localStorage + IndexedDB
}, [])

// Asset store initialization
async function loadPersisted(): Promise<Asset[]> {
  // Load small assets from localStorage
  const assets = JSON.parse(localStorage.getItem('tech_noir_assets') || '[]')
  
  // Load video metadata from localStorage
  const videos = JSON.parse(localStorage.getItem('tech_noir_videos') || '[]')
  
  // Restore video data URLs from IndexedDB
  for (const video of videos) {
    video.url = await loadVideoFromIndexedDB(video.id) || null
  }
  
  return [...assets, ...videos]
}
```

### 4. Saving Flow
```typescript
function persist(assets: Asset[]) {
  const videos = assets.filter(a => a.type === 'video')
  const others = assets.filter(a => a.type !== 'video')
  
  // Save small assets to localStorage
  localStorage.setItem('tech_noir_assets', JSON.stringify(others))
  
  // Save video metadata to localStorage (placeholder URLs)
  localStorage.setItem('tech_noir_videos', JSON.stringify(videos.map(v => ({
    ...v,
    url: v.id  // Placeholder - actual data in IndexedDB
  }))))
  
  // Save actual video data to IndexedDB (parallel async)
  Promise.all(videos.map(v => 
    v.url.startsWith('data:') 
      ? saveVideoToIndexedDB(v.id, v.url)
      : Promise.resolve()
  ))
}
```

## Storage Capacity

| Storage Method | Capacity | Use Case |
|----------------|----------|----------|
| localStorage | ~5-10MB | Images, small audio files |
| IndexedDB | ~50-100MB+ | Generated videos |

## Browser Support
IndexedDB is supported in all modern browsers:
- Chrome 11+
- Firefox 4+
- Safari 7.1+
- Edge (all versions)

## Benefits
✅ Generated videos persist across hard refresh  
✅ No data loss after waiting 5+ minutes  
✅ Supports multiple video assets  
✅ Fallback gracefully if IndexedDB fails  
✅ No backend changes required  

## Console Logging
Debug logs help track persistence:
```
[Assets] Persisted 5 small assets to localStorage, 2 videos to IndexedDB
[Assets] Loaded 7 assets from localStorage + IndexedDB
[Assets] Video not found in IndexedDB: K1_ltx2
```

## Testing
1. Generate a video (wait 5 minutes)
2. Hard refresh browser (Ctrl+Shift+R)
3. Video asset still appears in left sidebar
4. Video thumbnail plays on hover

## Future Improvements
- Add compression before storing (reduce size 50-70%)
- Implement cleanup for old videos (>30 days)
- Add export/import functionality
- Backend file storage as backup
