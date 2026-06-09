import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory, type Asset } from "@/stores/assets"
import { useToastStore } from "@/stores/toast"
import { FolderOpen, Plus, Music, Image, Mic, Volume2, Trash2, ChevronDown, PanelLeftClose, X, Download, Pencil } from "lucide-react"
import { useState } from "react"

const ICONS: Record<AssetCategory, typeof Image> = {
  image: Image, music: Music, voice: Mic, sfx: Volume2, video: Image, other: FolderOpen,
}

interface AppSidebarProps {
  open: boolean
  onToggle: () => void
  onSelectAsset?: (asset: Asset) => void
}

export function AppSidebar({ open, onToggle, onSelectAsset }: AppSidebarProps) {
  const assets = useAssetStore((s) => s.assets)
  const removeAsset = useAssetStore((s) => s.removeAsset)
  const renameAsset = useAssetStore((s) => s.renameAsset)
  const addAsset = useAssetStore((s) => s.addAsset)
  const toast = useToastStore((s) => s.addToast)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState("")

  const handleDownload = (a: Asset) => {
    const link = document.createElement("a")
    link.href = a.url
    link.download = a.name
    link.click()
  }

  const startRename = (a: Asset) => {
    setRenamingId(a.id)
    setRenameValue(a.name)
  }

  const commitRename = () => {
    if (renamingId && renameValue.trim()) {
      renameAsset(renamingId, renameValue.trim())
    }
    setRenamingId(null)
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      const t = file.type.startsWith("image/") ? "image" : file.type.startsWith("audio/") ? "audio" : "video"
      const cat: AssetCategory = t === "audio" ? "other" : (t as AssetCategory)
      addAsset({ name: file.name, type: t as any, category: cat, mediaType: file.type, url: dataUrl, sizeBytes: file.size, source: "uploaded" })
      toast("info", `Added: ${file.name}`)
    }
    reader.readAsDataURL(file)
  }

  const sidebar = (
    <div className="flex h-full w-64 flex-col bg-sidebar text-sidebar-foreground border-r">
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-semibold text-sm px-1">Assets</span>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onToggle}>
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      </div>
      <div className="p-2 border-b">
        <label className="flex w-full cursor-pointer">
          <input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} />
          <span className="flex w-full items-center justify-center gap-1 rounded-md border border-input bg-background px-3 py-1.5 text-xs shadow-sm hover:bg-accent hover:text-accent-foreground cursor-pointer">
            <Plus className="h-3 w-3" />Import
          </span>
        </label>
      </div>
      <ScrollArea className="flex-1">
        {assets.length === 0 && <p className="px-4 py-8 text-center text-xs text-muted-foreground">No assets yet.</p>}
        {CATEGORY_ORDER.map((cat) => {
          const items = assets.filter((a) => a.category === cat)
          if (items.length === 0) return null
          const Icon = ICONS[cat]
          return (
            <Collapsible key={cat} defaultOpen={cat === "image"}>
              <CollapsibleTrigger className="group/trigger flex w-full items-center gap-1 px-2 py-1 text-xs font-medium text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground">
                <Icon className="h-3.5 w-3.5" />
                <span className="flex-1 text-left">{CATEGORY_LABEL[cat]}</span>
                <Badge variant="secondary" className="h-4 px-1 text-[10px]">{items.length}</Badge>
                <ChevronDown className="h-3 w-3 transition-transform [[data-panel-open]_&]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent>
                {cat === "image" ? (
                  <div className="grid grid-cols-2 gap-1 p-1">
                    {items.map((a) => (
                      <div key={a.id} className="relative group/item cursor-pointer rounded-md border overflow-hidden"
                        onClick={() => onSelectAsset?.(a)}
                        draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}>
                        <img src={a.url} alt={a.name} className="w-full aspect-square object-cover" draggable={false} />
                        {renamingId === a.id ? (
                          <div className="absolute bottom-0 inset-x-0 bg-background/90 px-1 py-0.5">
                            <input className="w-full bg-background border border-border rounded px-1 py-0 text-[9px] outline-none focus:border-primary"
                              value={renameValue}
                              onChange={(e) => setRenameValue(e.target.value)}
                              onBlur={commitRename}
                              onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null) }}
                              onClick={(e) => e.stopPropagation()}
                              autoFocus />
                          </div>
                        ) : (
                          <span className="absolute bottom-0 inset-x-0 bg-background/80 px-1.5 py-0.5 text-[9px] truncate">{a.name}</span>
                        )}
                        <div className="absolute top-0.5 right-0.5 hidden group-hover/item:flex gap-px">
                          <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();handleDownload(a)}}><Download className="h-3 w-3" /></Button>
                          <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();startRename(a)}}><Pencil className="h-3 w-3" /></Button>
                          <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();removeAsset(a.id);toast("info", "Deleted: "+a.name)}}><Trash2 className="text-destructive h-3 w-3" /></Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="px-1 py-0.5 space-y-0.5">
                    {items.map((a) => (
                      <div key={a.id} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs hover:bg-sidebar-accent cursor-pointer group/item"
                        onClick={() => onSelectAsset?.(a)}
                        draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}>
                        <Icon className="h-3 w-3 shrink-0" />
                        {renamingId === a.id ? (
                          <input className="flex-1 bg-background border border-border rounded px-1 py-0 text-xs outline-none focus:border-primary"
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onBlur={commitRename}
                            onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null) }}
                            onClick={(e) => e.stopPropagation()}
                            autoFocus />
                        ) : (
                          <span className="flex-1 truncate">{a.name}</span>
                        )}
                        <div className="hidden group-hover/item:flex gap-px">
                          <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();handleDownload(a)}}><Download className="h-3 w-3" /></Button>
                          <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();startRename(a)}}><Pencil className="h-3 w-3" /></Button>
                          <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();removeAsset(a.id);toast("info", "Deleted: "+a.name)}}><Trash2 className="text-destructive h-3 w-3" /></Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CollapsibleContent>
            </Collapsible>
          )
        })}
      </ScrollArea>
      <div className="p-2 border-t">
        <Button variant="ghost" size="sm" className="w-full justify-start gap-2 text-xs">
          <FolderOpen className="h-3.5 w-3.5" />Library
        </Button>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop sidebar */}
      {open && <div className="hidden md:block">{sidebar}</div>}

      {/* Mobile sidebar — overlay */}
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="fixed inset-0 bg-black/50" onClick={onToggle} />
          <div className="fixed left-0 top-0 bottom-0">
            <div className="flex h-full w-64 flex-col bg-sidebar text-sidebar-foreground border-r">
              <div className="flex items-center justify-between p-2 border-b">
                <span className="font-semibold text-sm px-1">Assets</span>
                <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onToggle}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="p-2 border-b">
                <label className="flex w-full cursor-pointer">
                  <input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} />
                  <span className="flex w-full items-center justify-center gap-1 rounded-md border border-input bg-background px-3 py-1.5 text-xs shadow-sm hover:bg-accent hover:text-accent-foreground cursor-pointer">
                    <Plus className="h-3 w-3" />Import
                  </span>
                </label>
              </div>
              <ScrollArea className="flex-1">
                {assets.length === 0 && <p className="px-4 py-8 text-center text-xs text-muted-foreground">No assets yet.</p>}
                {CATEGORY_ORDER.map((cat) => {
                  const items = assets.filter((a) => a.category === cat)
                  if (items.length === 0) return null
                  const Icon = ICONS[cat]
                  return (
                    <Collapsible key={cat} defaultOpen={cat === "image"} className="group/collapsible">
                      <CollapsibleTrigger className="flex w-full items-center gap-1 px-2 py-1 text-xs font-medium text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground">
                        <Icon className="h-3.5 w-3.5" />
                        <span className="flex-1 text-left">{CATEGORY_LABEL[cat]}</span>
                        <Badge variant="secondary" className="h-4 px-1 text-[10px]">{items.length}</Badge>
                        <ChevronDown className="h-3 w-3 transition-transform group-data-[panel-open]/collapsible:rotate-180" />
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        {cat === "image" ? (
                          <div className="grid grid-cols-2 gap-1 p-1">
                            {items.map((a) => (
                              <div key={a.id} className="relative group/item cursor-pointer rounded-md border overflow-hidden"
                                onClick={() => onSelectAsset?.(a)}
                                draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}>
                                <img src={a.url} alt={a.name} className="w-full aspect-square object-cover" draggable={false} />
                                {renamingId === a.id ? (
                                  <div className="absolute bottom-0 inset-x-0 bg-background/90 px-1 py-0.5">
                                    <input className="w-full bg-background border border-border rounded px-1 py-0 text-[9px] outline-none focus:border-primary"
                                      value={renameValue}
                                      onChange={(e) => setRenameValue(e.target.value)}
                                      onBlur={commitRename}
                                      onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null) }}
                                      onClick={(e) => e.stopPropagation()}
                                      autoFocus />
                                  </div>
                                ) : (
                                  <span className="absolute bottom-0 inset-x-0 bg-background/80 px-1.5 py-0.5 text-[9px] truncate">{a.name}</span>
                                )}
                                <div className="absolute top-0.5 right-0.5 hidden group-hover/item:flex gap-px">
                                  <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();handleDownload(a)}}><Download className="h-3 w-3" /></Button>
                                  <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();startRename(a)}}><Pencil className="h-3 w-3" /></Button>
                                  <Button variant="ghost" size="icon" className="h-5 w-5 bg-background/80" onClick={(e)=>{e.stopPropagation();removeAsset(a.id);toast("info", "Deleted: "+a.name)}}><Trash2 className="text-destructive h-3 w-3" /></Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="px-1 py-0.5 space-y-0.5">
                            {items.map((a) => (
                              <div key={a.id} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs hover:bg-sidebar-accent cursor-pointer group/item"
                                onClick={() => onSelectAsset?.(a)}
                                draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}>
                                <Icon className="h-3 w-3 shrink-0" />
                                {renamingId === a.id ? (
                                  <input className="flex-1 bg-background border border-border rounded px-1 py-0 text-xs outline-none focus:border-primary"
                                    value={renameValue}
                                    onChange={(e) => setRenameValue(e.target.value)}
                                    onBlur={commitRename}
                                    onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingId(null) }}
                                    onClick={(e) => e.stopPropagation()}
                                    autoFocus />
                                ) : (
                                  <span className="flex-1 truncate">{a.name}</span>
                                )}
                                <div className="hidden group-hover/item:flex gap-px">
                                  <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();handleDownload(a)}}><Download className="h-3 w-3" /></Button>
                                  <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();startRename(a)}}><Pencil className="h-3 w-3" /></Button>
                                  <Button variant="ghost" size="icon" className="h-5 w-5 shrink-0" onClick={(e)=>{e.stopPropagation();removeAsset(a.id);toast("info", "Deleted: "+a.name)}}><Trash2 className="text-destructive h-3 w-3" /></Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </CollapsibleContent>
                    </Collapsible>
                  )
                })}
              </ScrollArea>
              <div className="p-2 border-t">
                <Button variant="ghost" size="sm" className="w-full justify-start gap-2 text-xs">
                  <FolderOpen className="h-3.5 w-3.5" />Library
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
