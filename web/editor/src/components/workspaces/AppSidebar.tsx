import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupContent,
  SidebarGroupLabel, SidebarHeader, SidebarMenu, SidebarMenuButton,
  SidebarMenuItem, SidebarTrigger,
} from "@/components/ui/sidebar"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory } from "@/stores/assets"
import { useToastStore } from "@/stores/toast"
import { FolderOpen, Plus, Music, Image, Mic, Volume2, Trash2 } from "lucide-react"

const CATEGORY_ICONS: Record<AssetCategory, typeof Image> = {
  image: Image, music: Music, voice: Mic, sfx: Volume2, video: Image, other: FolderOpen,
}

export function AppSidebar() {
  const assets = useAssetStore((s) => s.assets)
  const removeAsset = useAssetStore((s) => s.removeAsset)
  const addAsset = useAssetStore((s) => s.addAsset)
  const toast = useToastStore((s) => s.addToast)

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

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg">
              <FolderOpen className="h-4 w-4" />
              <span className="font-semibold">Assets</span>
            </SidebarMenuButton>
            <SidebarTrigger className="ml-auto" />
          </SidebarMenuItem>
          <SidebarMenuItem>
            <Label className="flex w-full cursor-pointer">
              <SidebarMenuButton tooltip="Import">
                <Plus className="h-4 w-4" /><span>Import</span>
              </SidebarMenuButton>
              <input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} />
            </Label>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <ScrollArea>
          {assets.length === 0 && <div className="px-4 py-8 text-center text-sm text-muted-foreground">No assets yet.</div>}
          {CATEGORY_ORDER.map((cat) => {
            const items = assets.filter((a) => a.category === cat)
            if (items.length === 0) return null
            const Icon = CATEGORY_ICONS[cat]
            return (
              <Collapsible key={cat} defaultOpen={cat === "image"}>
                <SidebarGroup>
                  <SidebarGroupLabel>
                    <CollapsibleTrigger className="flex w-full items-center gap-1">
                      <Icon className="h-3 w-3" />
                      <span className="flex-1 text-left text-xs">{CATEGORY_LABEL[cat]}</span>
                      <Badge variant="secondary" className="h-4 px-1 text-[10px]">{items.length}</Badge>
                    </CollapsibleTrigger>
                  </SidebarGroupLabel>
                  <CollapsibleContent>
                    <SidebarGroupContent>
                      {cat === "image" ? (
                        <div className="grid grid-cols-2 gap-1 p-1">
                          {items.map((a) => (
                            <div key={a.id} className="relative group/item cursor-pointer rounded-md border overflow-hidden"
                              draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}
                              onDoubleClick={() => { const o=document.createElement("div"); o.className="fixed inset-0 z-50 bg-background/95 flex items-center justify-center cursor-zoom-out"; o.onclick=()=>o.remove(); const i=document.createElement("img"); i.src=a.url; i.className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"; o.appendChild(i); document.body.appendChild(o) }}>
                              <img src={a.url} alt={a.name} className="w-full aspect-square object-cover" draggable={false} />
                              <span className="absolute bottom-0 inset-x-0 bg-background/80 px-1.5 py-0.5 text-[9px] truncate">{a.name.slice(0,14)}</span>
                              <Button variant="ghost" size="icon" className="absolute top-0.5 right-0.5 h-5 w-5 hidden group-hover/item:flex bg-background/80" onClick={(e)=>{e.stopPropagation();removeAsset(a.id)}}><Trash2 className="h-3 w-3 text-destructive" /></Button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <SidebarMenu>
                          {items.map((a) => (
                            <SidebarMenuItem key={a.id}>
                              <SidebarMenuButton className="text-xs h-auto py-1"><Icon className="h-3 w-3 shrink-0" /><span className="flex-1 truncate">{a.name}</span></SidebarMenuButton>
                            </SidebarMenuItem>
                          ))}
                        </SidebarMenu>
                      )}
                    </SidebarGroupContent>
                  </CollapsibleContent>
                </SidebarGroup>
              </Collapsible>
            )
          })}
        </ScrollArea>
      </SidebarContent>
    </Sidebar>
  )
}
