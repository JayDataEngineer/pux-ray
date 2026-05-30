import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup,
  SidebarGroupContent, SidebarGroupLabel, SidebarHeader,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useAssetStore, CATEGORY_LABEL, CATEGORY_ORDER, type AssetCategory } from "@/stores/assets"
import { useToastStore } from "@/stores/toast"
import { FolderOpen, Plus, Music, Image, Mic, Volume2, Trash2, ChevronDown } from "lucide-react"

const ICONS: Record<AssetCategory, typeof Image> = {
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
            <SidebarMenuButton size="lg" asChild>
              <a href="#"><FolderOpen /><span className="font-semibold">Assets</span></a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <label className="flex w-full cursor-pointer">
              <input type="file" className="hidden" accept="image/*,audio/*,video/*" onChange={handleImport} />
              <SidebarMenuButton asChild><span><Plus /><span>Import</span></span></SidebarMenuButton>
            </label>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <ScrollArea>
          {assets.length === 0 && <p className="px-4 py-8 text-center text-sm text-muted-foreground">No assets yet.</p>}
          {CATEGORY_ORDER.map((cat) => {
            const items = assets.filter((a) => a.category === cat)
            if (items.length === 0) return null
            const Icon = ICONS[cat]
            return (
              <Collapsible key={cat} defaultOpen={cat === "image"} className="group/collapsible">
                <SidebarGroup>
                  <SidebarGroupLabel asChild>
                    <CollapsibleTrigger>
                      <Icon /><span className="flex-1 text-left">{CATEGORY_LABEL[cat]}</span>
                      <Badge variant="secondary" className="h-4 px-1 text-[10px]">{items.length}</Badge>
                      <ChevronDown className="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-180" />
                    </CollapsibleTrigger>
                  </SidebarGroupLabel>
                  <CollapsibleContent>
                    <SidebarGroupContent>
                      {cat === "image" ? (
                        <div className="grid grid-cols-2 gap-1 p-1">
                          {items.map((a) => (
                            <div key={a.id} className="relative group/item cursor-pointer rounded-md border overflow-hidden"
                              draggable onDragStart={(e) => { e.dataTransfer.setData("application/tech-noir-asset", JSON.stringify({url:a.url,type:a.type,name:a.name,id:a.id})) }}>
                              <img src={a.url} alt={a.name} className="w-full aspect-square object-cover" draggable={false} />
                              <span className="absolute bottom-0 inset-x-0 bg-background/80 px-1.5 py-0.5 text-[9px] truncate">{a.name.slice(0,12)}</span>
                              <Button variant="ghost" size="icon" className="absolute top-0.5 right-0.5 h-5 w-5 hidden group-hover/item:flex" onClick={(e)=>{e.stopPropagation();removeAsset(a.id)}}><Trash2 className="text-destructive" /></Button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <SidebarMenu>
                          {items.map((a) => (
                            <SidebarMenuItem key={a.id}>
                              <SidebarMenuButton className="text-xs h-auto py-1"><Icon /><span className="flex-1 truncate">{a.name}</span></SidebarMenuButton>
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
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton><FolderOpen /><span>Library</span></SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
