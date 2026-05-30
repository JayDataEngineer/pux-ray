import { useState } from 'react'
import { FolderOpen, History, Plus, Music, Play, Trash2, ChevronDown, ChevronRight, Image, Video, Mic, Volume2 } from 'lucide-react'
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory } from '../../stores/assets'
import { useTimelineStore } from '../../stores/timeline'
import { useToastStore } from '../../stores/toast'
import { Sidebar, SidebarHeader, SidebarContent, SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuItem, SidebarMenuButton, SidebarTrigger, useSidebar } from '../ui/sidebar'

const CATEGORY_ICONS: Record<AssetCategory, typeof Image> = {
  image: Image, music: Music, voice: Mic, sfx: Volume2, video: Video, other: FolderOpen,
}

function SidebarInner() {
  const { state } = useSidebar()
  const collapsed = state === 'collapsed'
  const assets = useAssetStore((s) => s.assets)
  const removeAsset = useAssetStore((s) => s.removeAsset)
  const addAsset = useAssetStore((s) => s.addAsset)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const toast = useToastStore((s) => s.addToast)
  const [expandedCats, setExpandedCats] = useState<Set<AssetCategory>>(new Set(['image']))
  const [playingId, setPlayingId] = useState<string | null>(null)

  const toggleCat = (cat: AssetCategory) => setExpandedCats((prev) => {
    const next = new Set(prev); next.has(cat) ? next.delete(cat) : next.add(cat); return next
  })

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      const t = file.type.startsWith('image/') ? 'image' as const : file.type.startsWith('audio/') ? 'audio' as const : file.type.startsWith('video/') ? 'video' as const : 'other' as const
      const cat: AssetCategory = t === 'audio' ? 'other' : t
      addAsset({ name: file.name, type: t, category: cat, mediaType: file.type, url: dataUrl, sizeBytes: file.size, source: 'uploaded' })
      if (t === 'audio') addAudioCue({ track: 'sfx', start: 0, duration: 5, label: file.name.replace(/\.[^.]+$/, ''), audioUrl: dataUrl, volume: 0.8, waveformPeaks: null, sourceStepId: null })
      toast('info', `"${file.name}" added`)
    }
    reader.readAsDataURL(file)
  }

  return (
    <>
      <SidebarHeader>
        {!collapsed && <span className="text-[11px] font-medium tracking-wider text-accent uppercase">ASSETS</span>}
        <div className="flex gap-1 ml-auto">
          <label className="flex h-7 w-7 items-center justify-center rounded-sm text-muted-foreground hover:text-accent cursor-pointer">
            <Plus size={16} />
            <input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} />
          </label>
          <SidebarTrigger />
        </div>
      </SidebarHeader>
      <SidebarContent>
        {collapsed ? (
          <SidebarMenu>
            <SidebarMenuItem><SidebarMenuButton isActive><FolderOpen size={16} /></SidebarMenuButton></SidebarMenuItem>
            <SidebarMenuItem><SidebarMenuButton><History size={16} /></SidebarMenuButton></SidebarMenuItem>
          </SidebarMenu>
        ) : (
          <>
            {CATEGORY_ORDER.map((cat) => {
              const items = assets.filter((a) => a.category === cat)
              if (items.length === 0) return null
              const Icon = CATEGORY_ICONS[cat]
              const expanded = expandedCats.has(cat)
              return (
                <SidebarGroup key={cat}>
                  <SidebarGroupLabel className="cursor-pointer flex items-center gap-1" onClick={() => toggleCat(cat)}>
                    {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                    <Icon size={12} />
                    <span>{CATEGORY_LABEL[cat]}</span>
                    <span className="ml-auto text-[10px] bg-card/50 px-1.5 rounded">{items.length}</span>
                  </SidebarGroupLabel>
                  {expanded && (
                    cat === 'image' ? (
                      <div className="grid grid-cols-2 gap-1">
                        {items.map((a) => (
                          <div key={a.id} className="relative group cursor-pointer border border-border hover:border-accent/50 transition-colors"
                            draggable onDragStart={(e) => { e.dataTransfer.setData('application/tech-noir-asset', JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})); e.dataTransfer.setData('text/plain', a.url); e.dataTransfer.effectAllowed = 'copy' }}
                            onDoubleClick={() => {
                              const ov = document.createElement('div'); ov.className = 'fixed inset-0 z-50 bg-black/85 flex items-center justify-center cursor-pointer'; ov.onclick = () => ov.remove()
                              const img = document.createElement('img'); img.src = a.url; img.className = 'max-w-[90vw] max-h-[90vh] object-contain'; ov.appendChild(img); document.body.appendChild(ov)
                            }}>
                            <img src={a.url} alt={a.name} className="w-full aspect-[4/5] object-cover" draggable={false} />
                            <span className="absolute bottom-0 left-0 right-0 bg-black/70 px-1 py-0.5 text-[9px] truncate">{a.name.slice(0,14)}</span>
                            <button className="absolute top-0.5 right-0.5 hidden group-hover:flex bg-black/70 rounded-sm p-0.5 text-red-400" onClick={(e) => { e.stopPropagation(); removeAsset(a.id) }}><Trash2 size={10} /></button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      items.map((a) => (
                        <div key={a.id} className={`flex items-center gap-1.5 px-1 py-0.5 text-xs hover:bg-accent/10 rounded cursor-pointer ${playingId===a.id ? 'bg-accent/10' : ''}`}>
                          <button className="p-0.5 hover:text-accent" onClick={() => setPlayingId((p) => p===a.id ? null : a.id)}>
                            {playingId===a.id ? <span className="text-[10px]">⏸</span> : <Play size={10} />}
                          </button>
                          <Icon size={12} />
                          <span className="flex-1 truncate">{a.name}</span>
                          <button className="p-0.5 hover:text-red-400 opacity-0 group-hover:opacity-100" onClick={() => removeAsset(a.id)}><Trash2 size={10} /></button>
                        </div>
                      ))
                    )
                  )}
                </SidebarGroup>
              )
            })}
            {assets.length === 0 && (
              <div className="px-3 py-4 text-[11px] text-muted-foreground text-center">No assets<br/>Generate or Import</div>
            )}
          </>
        )}
      </SidebarContent>
    </>
  )
}

export function NewSidebar() {
  return (
    <Sidebar>
      <SidebarInner />
    </Sidebar>
  )
}
