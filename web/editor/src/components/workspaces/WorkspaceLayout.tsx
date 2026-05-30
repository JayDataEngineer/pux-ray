import { useState, useEffect, useCallback } from "react"
import { NavigationMenu, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { forgeStatus } from "@/mcp"
import { Cpu, HardDrive } from "lucide-react"
import { AppSidebar } from "./AppSidebar"

type TabId = "assets" | "video"

// ── Main Layout ─────────────────────────────────────────────────────────────

export function WorkspaceLayout(_props: any = {}) {
  const [tab, setTab] = useState<TabId>("assets")

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="flex flex-col">
        <header className="flex items-center h-11 px-4 border-b gap-4 shrink-0">
          <SidebarTrigger />
          <span className="font-bold text-sm tracking-tight">TECH NOIR</span>
          <NavigationMenu>
            <NavigationMenuList>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} active={tab === "assets"} onClick={() => setTab("assets")}>
                  Assets
                </NavigationMenuLink>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuLink className={navigationMenuTriggerStyle()} active={tab === "video"} onClick={() => setTab("video")}>
                  Video
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
          <div className="flex-1" />
          <GpuStatus />
        </header>
        <div className="flex-1 min-h-0">
          {tab === "assets" ? <AssetsTab /> : <VideoTab />}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

// ── GPU Status Indicator ────────────────────────────────────────────────────

function GpuStatus() {
  const [status, setStatus] = useState<{ loaded: number; vram_free_mb: number; vram_total_mb: number } | null>(null)
  const [error, setError] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const s = await forgeStatus()
      setStatus({
        loaded: Object.keys(s.loaded).length,
        vram_free_mb: s.vram_free_mb,
        vram_total_mb: s.vram_total_mb || 22528,
      })
      setError(false)
    } catch {
      setError(true)
    }
  }, [])

  useEffect(() => { refresh(); const id = setInterval(refresh, 15000); return () => clearInterval(id) }, [refresh])

  if (error) return <Badge variant="outline" className="text-xs gap-1"><Cpu className="h-3 w-3 text-destructive" />Offline</Badge>
  if (!status) return <Skeleton className="h-5 w-20" />

  const used = status.vram_total_mb - status.vram_free_mb
  const pct = Math.round((used / status.vram_total_mb) * 100)
  return (
    <Badge variant="outline" className="text-xs gap-1 cursor-pointer" onClick={refresh} title={`${status.loaded} service(s) loaded, ${used}MB / ${status.vram_total_mb}MB VRAM`}>
      <HardDrive className="h-3 w-3" />
      {pct}% GPU
    </Badge>
  )
}

// ── Assets Tab ─────────────────────────────────────────────────────────────

function AssetsTab() {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="text-center space-y-4">
        <h2 className="text-lg font-semibold">Asset Library</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Browse generated and imported assets in the sidebar. Drag them into the Video tab to build your timeline.
        </p>
        <p className="text-xs text-muted-foreground">
          Use the <strong>Generate</strong> tab to create new images, audio, 3D models, and more.
        </p>
      </div>
    </div>
  )
}

// ── Video Tab (unchanged) ──────────────────────────────────────────────────

function VideoTab() {
  const segments = useTimelineStore((s) => s.segments)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const selectedSegmentId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelectedSegment = useTimelineStore((s) => s.setSelectedSegment)
  const toast = useToastStore((s) => s.addToast)
  const selectedSegment = segments.find((s) => s.id === selectedSegmentId)

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    try {
      const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
      if (d.type === "image") {
        const s = addSegment({ prompt: d.name, firstFrameB64: d.url, thumbnailUrl: d.url, status: "empty" })
        setSelectedSegment(s.id)
        toast("info", `Keyframe: ${d.name}`)
      }
    } catch { /* ignore */ }
  }

  return (
    <div className="flex-1 flex gap-6 p-6">
      <div className="flex-1 flex flex-col gap-4">
        <Card className="flex-1" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
          <CardContent className="flex items-center justify-center h-full min-h-[200px]">
            {selectedSegment?.videoUrl ? (
              <video src={selectedSegment.videoUrl} controls className="max-w-full max-h-full rounded-lg" />
            ) : segments.length === 0 ? (
              <p className="text-muted-foreground text-sm">Drag images from the Assets sidebar</p>
            ) : (
              <p className="text-muted-foreground text-sm">{segments.length} keyframe(s) — select one to generate</p>
            )}
          </CardContent>
        </Card>
        <div className="h-24 flex items-center gap-1 p-2 border rounded-lg overflow-x-auto bg-muted/30">
          {segments.map((seg) => (
            <div
              key={seg.id}
              className={`h-full flex items-center justify-center rounded-md text-[10px] cursor-pointer relative overflow-hidden shrink-0 border-2 ${seg.id === selectedSegmentId ? "border-primary" : "border-border"}`}
              style={{ width: `${seg.duration * 40}px` }}
              onClick={() => setSelectedSegment(seg.id)}
            >
              {seg.thumbnailUrl && <img src={seg.thumbnailUrl} alt="" className="absolute inset-0 w-full h-full object-cover opacity-30" />}
              <span className="relative z-10 font-medium">K_{String(seg.order + 1).padStart(2, "0")}</span>
            </div>
          ))}
          <Button variant="outline" size="icon" className="h-full w-8 shrink-0" onClick={() => { const s = addSegment({ duration: 5, status: "empty" }); setSelectedSegment(s.id) }}>+</Button>
        </div>
      </div>
    </div>
  )
}
