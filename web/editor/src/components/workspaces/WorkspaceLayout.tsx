import { useState, useEffect, useCallback } from "react"
import { NavigationMenu, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { useAssetStore } from "@/stores/assets"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, Image, Music, Mic, Wand2 } from "lucide-react"
import { AppSidebar } from "./AppSidebar"

type TabId = "assets" | "video"

const GENRE_TABS = [
  { id:"image" as const, label:"Image", icon:Image },
  { id:"audio" as const, label:"Audio", icon:Music },
  { id:"voice" as const, label:"Voice", icon:Mic },
]

const TOOL_GENRE: Record<string, string> = {
  generate_sound: "audio",
  generate_music: "audio",
  tts_speak: "voice",
}

function toolGenre(name: string): string {
  return TOOL_GENRE[name] ?? "image"
}

function renderField(
  name: string,
  prop: NonNullable<MCPTool["inputSchema"]["properties"]>[string],
  value: unknown,
  onChange: (v: unknown) => void,
) {
  const label = prop.description || name
  const placeholder = typeof prop.default === "string" ? String(prop.default) : label

  if (prop.enum) {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label>{label}</Label>
        <Select value={String(value ?? prop.default ?? "")} onValueChange={onChange}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            {prop.enum.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
    )
  }

  const schemaType = prop.type ?? "string"

  if (schemaType === "number" || schemaType === "integer") {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label>{label}</Label>
        <Input type="number" value={String(value ?? prop.default ?? "")}
          onChange={(e) => onChange(e.target.value ? Number(e.target.value) : "")}
          placeholder={placeholder} />
      </div>
    )
  }

  if (schemaType === "object") {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label>{label}</Label>
        <Textarea value={typeof value === "object" ? JSON.stringify(value ?? prop.default ?? {}, null, 2) : String(value ?? "")}
          onChange={(e) => {
            try { onChange(JSON.parse(e.target.value)) }
            catch { onChange(e.target.value) }
          }}
          placeholder={placeholder} rows={4} />
      </div>
    )
  }

  const isLong = (prop.description?.length ?? 0) > 80 || name === "lyrics" || name === "instruct"
  if (isLong) {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label>{label}</Label>
        <Textarea value={String(value ?? prop.default ?? "")}
          onChange={(e) => onChange(e.target.value || "")}
          placeholder={placeholder} rows={3} />
      </div>
    )
  }

  return (
    <div key={name} className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <Input value={String(value ?? prop.default ?? "")}
        onChange={(e) => onChange(e.target.value || "")}
        placeholder={placeholder} />
    </div>
  )
}

export function WorkspaceLayout(_props: any = {}) {
  const [tab, setTab] = useState<TabId>("assets")
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen w-full bg-background">
      <AppSidebar open={sidebarOpen} onToggle={() => setSidebarOpen((o) => !o)} />
      <div className="flex flex-1 flex-col min-w-0">
        <header className="flex items-center h-11 px-4 border-b gap-4 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSidebarOpen((o) => !o)}>
            <PanelLeft className="h-4 w-4" />
          </Button>
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
      </div>
    </div>
  )
}

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

function AssetsTab() {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [tools, setTools] = useState<MCPTool[]>([])
  const [loading, setLoading] = useState(true)
  const [genre, setGenre] = useState("image")
  const [selected, setSelected] = useState<string>("")
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    listTools().then((all) => {
      setTools(all)
      setLoading(false)
      if (all.length > 0) setSelected(all[0].name)
    }).catch(() => {
      setLoading(false)
      toast("error", "Failed to load forge tools")
    })
  }, [])

  const currentTool = tools.find((t) => t.name === selected)
  const props_ = currentTool?.inputSchema?.properties ?? {}

  useEffect(() => {
    if (!currentTool) return
    const d: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(props_)) {
      if (v.default !== undefined) d[k] = v.default
    }
    setValues(d)
  }, [selected])

  const handleGenerate = async () => {
    if (!currentTool) return
    setGenerating(true)
    try {
      const args: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(values)) {
        if (v !== null && v !== "") args[k] = v
      }
      const result = await callTool<{ status: string; data?: string; media_type?: string; error?: string; message?: string }>(currentTool.name, args)
      if (result.status === "ok" || result.status === "success") {
        if (result.data) {
          const mt = result.media_type || "image/png"
          const isAud = mt.includes("audio")
          const cat = mt.includes("audio") && currentTool.name === "generate_music" ? "music" as const : isAud ? "sfx" as const : "image" as const
          addAsset({
            name: `${currentTool.name} ${new Date().toLocaleTimeString()}`,
            type: isAud ? "audio" : "image",
            category: cat,
            mediaType: mt,
            url: `data:${mt};base64,${result.data}`,
            sizeBytes: Math.round((result.data as string).length * 0.75),
            source: "generated",
          })
          toast("success", `${currentTool.name} generated`)
        }
      } else {
        toast("error", String(result.error || result.message || "Unknown error"))
      }
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <Skeleton className="h-8 w-48" />
      </div>
    )
  }

  return (
    <div className="flex-1 flex gap-6 p-6">
      <div className="w-56 shrink-0">
        <Tabs value={genre} onValueChange={(v) => { setGenre(v); const t = tools.find((x) => toolGenre(x.name) === v); if (t) setSelected(t.name) }}>
          <TabsList className="w-full">
            {GENRE_TABS.map((g) => (
              <TabsTrigger key={g.id} value={g.id} className="flex-1">{g.label}</TabsTrigger>
            ))}
          </TabsList>
          {GENRE_TABS.map((g) => (
            <TabsContent key={g.id} value={g.id} className="mt-2 space-y-1">
              {tools.filter((t) => {
                if (t.name === "list_models" || t.name === "list_services" || t.name === "get_service" ||
                    t.name === "forge_status" || t.name === "transcribe" || t.name === "chat" ||
                    t.name === "llm_configure" || t.name === "load_service" || t.name === "unload_services" ||
                    t.name === "tts_voices" ||
                    t.name.startsWith("workflow_")) return false
                return toolGenre(t.name) === g.id
              }).map((t) => {
                const Icon = GENRE_TABS.find((x) => x.id === g.id)?.icon ?? Wand2
                return (
                  <Button key={t.name} variant={selected === t.name ? "secondary" : "ghost"}
                    className="w-full justify-start gap-2 h-auto py-2" onClick={() => setSelected(t.name)}>
                    <Icon />{t.description?.split("—")[0]?.trim() || t.name}
                  </Button>
                )
              })}
            </TabsContent>
          ))}
        </Tabs>
      </div>
      <Separator orientation="vertical" />
      <Card className="flex-1 max-w-lg overflow-y-auto">
        {currentTool && (
          <>
            <CardHeader><CardTitle className="text-base">{currentTool.description || currentTool.name}</CardTitle></CardHeader>
            <CardContent className="flex flex-col gap-4">
              {Object.entries(props_).map(([name, prop]) =>
                renderField(name, prop, values[name], (v) => setValues((prev) => ({ ...prev, [name]: v })))
              )}
              <Button className="mt-2" disabled={generating} onClick={handleGenerate}>
                {generating ? "Generating..." : `Generate ${currentTool.name}`}
              </Button>
            </CardContent>
          </>
        )}
      </Card>
    </div>
  )
}

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
