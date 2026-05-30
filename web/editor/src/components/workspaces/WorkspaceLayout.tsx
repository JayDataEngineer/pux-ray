import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { useAssetStore } from "@/stores/assets"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, PanelRightClose, Wand2 } from "lucide-react"
import { AppSidebar } from "./AppSidebar"

type TabId = "assets" | "video"

const TOOL_GENRE: Record<string, string> = {
  generate_sound: "audio",
  generate_music: "audio",
  tts_speak: "voice",
}

function toolGenre(name: string): string {
  return TOOL_GENRE[name] ?? "image"
}

function isAssetTool(t: MCPTool): boolean {
  if (t.name.startsWith("workflow_") || t.name.startsWith("llm_")) return false
  const admin = ["list_models", "list_services", "get_service", "forge_status", "load_service", "unload_services", "tts_voices", "chat", "transcribe"]
  return !admin.includes(t.name)
}

function renderField(
  name: string,
  prop: NonNullable<NonNullable<MCPTool["inputSchema"]["properties"]>[string]>,
  value: unknown,
  onChange: (v: unknown) => void,
) {
  const label = prop.description || name
  const placeholder = typeof prop.default === "string" ? String(prop.default) : label

  if (prop.enum) {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label className="text-xs text-muted-foreground">{label}</Label>
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
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <Input type="number" value={String(value ?? prop.default ?? "")}
          onChange={(e) => onChange(e.target.value ? Number(e.target.value) : "")}
          placeholder={placeholder} />
      </div>
    )
  }

  if (schemaType === "object") {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <Textarea value={typeof value === "object" ? JSON.stringify(value ?? prop.default ?? {}, null, 2) : String(value ?? "")}
          onChange={(e) => {
            try { onChange(JSON.parse(e.target.value)) }
            catch { onChange(e.target.value) }
          }}
          placeholder={placeholder} rows={4} className="font-mono text-xs" />
      </div>
    )
  }

  const isLong = (prop.description?.length ?? 0) > 80 || name === "lyrics" || name === "instruct"
  if (isLong) {
    return (
      <div key={name} className="flex flex-col gap-1.5">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <Textarea value={String(value ?? prop.default ?? "")}
          onChange={(e) => onChange(e.target.value || "")}
          placeholder={placeholder} rows={3} />
      </div>
    )
  }

  return (
    <div key={name} className="flex flex-col gap-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <Input value={String(value ?? prop.default ?? "")}
        onChange={(e) => onChange(e.target.value || "")}
        placeholder={placeholder} />
    </div>
  )
}

export function WorkspaceLayout(_props: any = {}) {
  const [tab, setTab] = useState<TabId>("assets")
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)

  return (
    <div className="flex h-screen w-full bg-background">
      <AppSidebar open={leftOpen} onToggle={() => setLeftOpen((o) => !o)} />
      <div className="flex flex-1 flex-col min-w-0">
        <header className="flex items-center h-11 px-4 border-b gap-2 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setLeftOpen((o) => !o)}>
            <PanelLeft className="h-4 w-4" />
          </Button>
          <span className="font-bold text-sm tracking-tight">TECH NOIR</span>
          <Separator orientation="vertical" className="h-5 mx-1" />
          <Button variant={tab === "assets" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs"
            onClick={() => setTab("assets")}>Assets</Button>
          <Button variant={tab === "video" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs"
            onClick={() => setTab("video")}>Video</Button>
          <div className="flex-1" />
          <GpuStatus />
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setRightOpen((o) => !o)}>
            <PanelRightClose className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 overflow-auto">
            {tab === "assets" ? <AssetsTab /> : <VideoTab />}
          </div>
          {rightOpen && <ServicesSidebar />}
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
    <Badge variant="outline" className="text-xs gap-1 cursor-pointer" onClick={refresh}>
      <HardDrive className="h-3 w-3" />
      {pct}% GPU
    </Badge>
  )
}

function ServicesSidebar() {
  const [tools, setTools] = useState<MCPTool[]>([])
  const [selected, setSelected] = useState("")
  const [genre, setGenre] = useState("image")
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)

  useEffect(() => {
    listTools().then((all) => {
      setTools(all)
      if (all.length > 0) setSelected(all[0].name)
    }).catch(() => toast("error", "Failed to load forge tools"))
  }, [])

  const assetTools = tools.filter(isAssetTool)
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
          const cat = isAud && currentTool.name === "generate_music" ? "music" as const : isAud ? "sfx" as const : "image" as const
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

  const genres = [...new Set(assetTools.map((t) => toolGenre(t.name)))]

  return (
    <div className="w-80 border-l bg-sidebar text-sidebar-foreground flex flex-col shrink-0">
      <div className="flex items-center justify-between p-3 border-b">
        <span className="font-semibold text-sm">Forge Services</span>
      </div>
      <div className="flex border-b">
        {genres.map((g) => (
          <button key={g} onClick={() => setGenre(g)}
            className={`flex-1 py-2 text-xs font-medium text-center transition-colors ${genre === g ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/60 hover:text-sidebar-foreground"}`}>
            {g}
          </button>
        ))}
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {assetTools.filter((t) => toolGenre(t.name) === genre).map((t) => (
            <button key={t.name} onClick={() => setSelected(t.name)}
              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-left transition-colors ${selected === t.name ? "bg-sidebar-accent text-sidebar-accent-foreground" : "hover:bg-sidebar-accent/50"}`}>
              <Wand2 className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{t.description?.split("—")[0]?.trim() || t.name}</span>
            </button>
          ))}
        </div>
      </ScrollArea>
      {currentTool && (
        <div className="border-t p-3 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">{currentTool.description || currentTool.name}</span>
            <Badge variant="outline" className="text-[10px]">{currentTool.name}</Badge>
          </div>
          <div className="space-y-2 max-h-[60vh] overflow-y-auto">
            {Object.entries(props_).map(([name, prop]) =>
              renderField(name, prop, values[name], (v) => setValues((prev) => ({ ...prev, [name]: v })))
            )}
          </div>
          <Button className="w-full h-8 text-xs" disabled={generating} onClick={handleGenerate}>
            {generating ? "Generating..." : `Generate`}
          </Button>
        </div>
      )}
    </div>
  )
}

function AssetsTab() {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="text-center space-y-4">
        <h2 className="text-lg font-semibold">Asset Library</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Select a forge service from the right sidebar to generate assets.
          Drag them from the left sidebar into the Video tab.
        </p>
      </div>
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
