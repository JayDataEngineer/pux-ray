import { useState, useEffect, useCallback, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { useAssetStore } from "@/stores/assets"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, PanelRightClose, Wand2, Loader2, CheckCircle2, XCircle, Clock } from "lucide-react"
import { AppSidebar } from "./AppSidebar"

type TabId = "assets" | "video"

interface JobEntry {
  id: number
  name: string
  status: "pending" | "running" | "completed" | "failed"
  startedAt: number
  endedAt?: number
  error?: string
}

const COMMON_PARAMS = [
  "model", "prompt", "text", "image_b64", "audio_b64",
  "seed", "steps", "guidance", "width", "height", "frames",
  "negative_prompt", "voice", "language",
]

const GENRE_ORDER = ["image", "audio", "voice"]

const SERVICE_GENRE: Record<string, string> = {
  z_image: "image", comfyui: "image", nvidia_upscale: "image", dwpose: "image",
  ace_step: "audio", moss_soundeffect: "audio",
  kokoro: "voice", espeak: "voice", index_tts: "voice", faster_qwen3_tts: "voice",
  generate_sound: "audio", generate_music: "audio", tts_speak: "voice",
  generate_image: "image",
}

type FieldDef = {
  name: string; type: "text" | "number" | "select" | "file" | "textarea" | "json"
  label: string; default?: unknown; options?: string[]; required?: boolean
}

function extractCommonParams(desc: string): FieldDef[] {
  if (!desc.toLowerCase().includes("common:")) return []
  return COMMON_PARAMS.map((n) => ({
    name: n, label: n.replace(/_/g, " "),
    type: (["seed","steps","guidance","width","height","frames"].includes(n) ? "number" : "textarea") as FieldDef["type"],
    default: undefined,
  }))
}

export function WorkspaceLayout(_props: any = {}) {
  const [tab, setTab] = useState<TabId>("assets")
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [selectedService, setSelectedService] = useState("")
  const [jobs, setJobs] = useState<JobEntry[]>([])
  const nextJobId = useRef(1)

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
            {tab === "assets"
              ? <AssetsTab selectedService={selectedService} onStartJob={() => {}} />
              : <VideoTab />
            }
          </div>
          {rightOpen && (
            <ServicesSidebar
              selected={selectedService}
              onSelect={setSelectedService}
              jobs={jobs}
              onJobsChange={setJobs}
              nextJobId={nextJobId}
            />
          )}
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
      setStatus({ loaded: Object.keys(s.loaded).length, vram_free_mb: s.vram_free_mb, vram_total_mb: s.vram_total_mb || 22528 })
      setError(false)
    } catch { setError(true) }
  }, [])
  useEffect(() => { refresh(); const id = setInterval(refresh, 15000); return () => clearInterval(id) }, [refresh])
  if (error) return <Badge variant="outline" className="text-xs gap-1"><Cpu className="h-3 w-3 text-destructive" />Offline</Badge>
  if (!status) return <Skeleton className="h-5 w-20" />
  const used = status.vram_total_mb - status.vram_free_mb
  const pct = Math.round((used / status.vram_total_mb) * 100)
  return (
    <Badge variant="outline" className="text-xs gap-1 cursor-pointer" onClick={refresh}>
      <HardDrive className="h-3 w-3" />{pct}% GPU
    </Badge>
  )
}

function ServicesSidebar({ selected, onSelect, jobs, onJobsChange, nextJobId }: {
  selected: string; onSelect: (n: string) => void
  jobs: JobEntry[]; onJobsChange: (j: JobEntry[] | ((prev: JobEntry[]) => JobEntry[])) => void; nextJobId: React.MutableRefObject<number>
}) {
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])
  const [genre, setGenre] = useState("image")
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const allItems = [
    ...tools.filter((t) => !["run","list_models","list_services","get_service","forge_status","load_service","unload_services","tts_voices","chat","transcribe","llm_configure"].includes(t.name) && !t.name.startsWith("workflow_")),
    ...services.filter((s) => !tools.find((t) => t.name === s.name)),
  ]
  const items = allItems.filter((s) => SERVICE_GENRE[("name" in s ? (s as any).name : (s as any).name)] === genre)

  // Tool widget state
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)
  const [fields, setFields] = useState<FieldDef[]>([])

  const currentTool = tools.find((t) => t.name === selected)
  const currentService = services.find((s) => s.name === selected)

  useEffect(() => {
    if (!currentTool && !currentService) return
    if (currentTool && currentTool.inputSchema?.properties) {
      const props_ = currentTool.inputSchema.properties as Record<string, Record<string, unknown>>
      const extracted: FieldDef[] = []
      for (const [k, v] of Object.entries(props_)) {
        const isFreeform = v.type === "object" || v.additionalProperties || (Array.isArray(v.anyOf) && v.anyOf.some((a: any) => a.type === "object"))
        if (isFreeform) {
          const common = extractCommonParams((v.description as string) || "")
          if (common.length > 0) { extracted.push(...common); continue }
        }
        const isNum = v.type === "number" || v.type === "integer"
        const isLong = ((v.description as string)?.length ?? 0) > 80 || k === "lyrics" || k === "instruct"
        extracted.push({
          name: k, label: (v.description as string) || k,
          type: v.enum ? "select" as const : isNum ? "number" as const : isLong ? "textarea" as const : "text" as const,
          default: v.default, options: v.enum as string[] | undefined,
          required: currentTool.inputSchema.required?.includes(k),
        })
      }
      setFields(extracted)
      const d: Record<string, unknown> = {}
      for (const f of extracted) { if (f.default !== undefined) d[f.name] = f.default }
      setValues(d)
    }
  }, [selected, currentTool])

  const handleGenerate = async () => {
    if (!currentTool && !currentService) return
    const useTool = currentTool || tools.find((t) => t.name === "run")
    if (!useTool) return

    const jobId = nextJobId.current++
    const jobName = currentService?.label || currentTool?.description?.split("—")[0]?.trim() || selected
    onJobsChange([{ id: jobId, name: jobName, status: "running", startedAt: Date.now() }, ...jobs])

    setGenerating(true)
    try {
      const args: Record<string, unknown> = {}
      if (useTool.name === "run") {
        args.service = currentService?.name || selected
        const params: Record<string, unknown> = {}
        for (const [k, v] of Object.entries(values)) { if (v !== null && v !== "") params[k] = v }
        args.params = params
      } else {
        for (const [k, v] of Object.entries(values)) { if (v !== null && v !== "") args[k] = v }
      }
      const result = await callTool<{ status: string; data?: string; media_type?: string; error?: string; message?: string }>(useTool.name, args)
      if (result.status === "ok" || result.status === "success") {
        if (result.data) {
          const mt = result.media_type || "image/png"
          const isAud = mt.includes("audio")
          const cat = isAud && selected.includes("music") ? "music" as const : isAud ? "sfx" as const : "image" as const
          addAsset({
            name: `${jobName} ${new Date().toLocaleTimeString()}`,
            type: isAud ? "audio" : "image", category: cat, mediaType: mt,
            url: `data:${mt};base64,${result.data}`,
            sizeBytes: Math.round((result.data as string).length * 0.75), source: "generated",
          })
          toast("success", `${jobName} generated`)
        }
        onJobsChange((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "completed", endedAt: Date.now() } : j))
      } else {
        onJobsChange((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: result.error || result.message || "Unknown error" } : j))
        toast("error", String(result.error || result.message || "Unknown error"))
      }
    } catch (e) {
      onJobsChange((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: e instanceof Error ? e.message : String(e) } : j))
      toast("error", e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  const running = jobs.filter((j) => j.status === "running")

  return (
    <div className="w-80 border-l bg-sidebar text-sidebar-foreground flex flex-col shrink-0">
      {/* Services list */}
      <div className="p-2 border-b">
        <span className="text-xs font-semibold">Services</span>
      </div>
      <div className="flex border-b text-xs">
        {GENRE_ORDER.map((g) => (
          <button key={g} onClick={() => setGenre(g)}
            className={`flex-1 py-1.5 text-center font-medium transition-colors ${genre === g ? "bg-sidebar-accent text-sidebar-accent-foreground" : "text-sidebar-foreground/60 hover:text-sidebar-foreground"}`}>
            {g}
          </button>
        ))}
      </div>
      <div className="p-1.5 space-y-0.5 border-b max-h-40 overflow-y-auto">
        {items.map((item) => {
          const name = "name" in item ? (item as any).name : (item as any).name
          const label = "label" in item ? (item as any).label : name
          return (
            <button key={name} onClick={() => onSelect(name)}
              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left transition-colors ${selected === name ? "bg-sidebar-accent text-sidebar-accent-foreground" : "hover:bg-sidebar-accent/50"}`}>
              <Wand2 className="h-3 w-3 shrink-0" />
              <span className="truncate">{label}</span>
            </button>
          )
        })}
      </div>

      {/* Tool widget */}
      {currentTool && fields.length > 0 && (
        <div className="p-2 border-b space-y-2 max-h-[40vh] overflow-y-auto">
          {fields.map((f) => (
            <div key={f.name} className="space-y-1">
              <Label className="text-xs text-muted-foreground capitalize">{f.label}</Label>
              {f.type === "select" && f.options ? (
                <Select value={String(values[f.name] ?? f.default ?? "")}
                  onValueChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}>
                  <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {f.options.map((o) => <SelectItem key={o} value={o} className="text-xs">{o}</SelectItem>)}
                  </SelectContent>
                </Select>
              ) : f.type === "number" ? (
                <Input type="number" value={String(values[f.name] ?? f.default ?? "")}
                  onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                  className="h-7 text-xs" placeholder={f.label} />
              ) : f.type === "file" ? (
                <Label className="flex items-center justify-center h-10 border-2 border-dashed rounded cursor-pointer hover:border-primary/50 text-xs text-muted-foreground">
                  {values[f.name] ? "Loaded" : "Upload"}
                  <Input type="file" className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (!file) return
                      const r = new FileReader()
                      r.onload = () => setValues((p) => ({ ...p, [f.name]: (r.result as string).split(",")[1] || "" }))
                      r.readAsDataURL(file)
                    }} />
                </Label>
              ) : f.type === "textarea" ? (
                <Textarea value={String(values[f.name] ?? f.default ?? "")}
                  onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                  className="text-xs" placeholder={f.label} rows={2} />
              ) : (
                <Input value={String(values[f.name] ?? f.default ?? "")}
                  onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                  className="h-7 text-xs" placeholder={f.label} />
              )}
            </div>
          ))}
          <Button className="w-full h-7 text-xs" disabled={generating || running.length > 0} onClick={handleGenerate}>
            {generating ? "Generating..." : "Generate"}
          </Button>
        </div>
      )}

      {/* Job panel */}
      {jobs.length > 0 && (
        <div className="flex-1 overflow-y-auto">
          <div className="px-2 pt-2 pb-1 text-xs font-semibold flex items-center gap-1.5">
            <Clock className="h-3 w-3" /> Jobs
            {running.length > 0 && (
              <Badge variant="secondary" className="text-[10px] px-1 py-0">{running.length} active</Badge>
            )}
          </div>
          <div className="px-1.5 pb-2 space-y-0.5">
            {jobs.map((j) => (
              <div key={j.id} className="flex items-center gap-2 px-2 py-1.5 rounded text-xs hover:bg-sidebar-accent/30">
                {j.status === "running" && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />}
                {j.status === "completed" && <CheckCircle2 className="h-3 w-3 shrink-0 text-green-500" />}
                {j.status === "failed" && <XCircle className="h-3 w-3 shrink-0 text-destructive" />}
                <span className="flex-1 truncate">{j.name}</span>
                {j.status === "running" && <span className="text-[10px] text-muted-foreground">{Math.round((Date.now() - j.startedAt) / 1000)}s</span>}
                {j.status === "completed" && j.endedAt && <span className="text-[10px] text-muted-foreground">{(j.endedAt - j.startedAt) / 1000}s</span>}
                {j.status === "failed" && <span className="text-[10px] text-destructive truncate max-w-20">{j.error}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function AssetsTab(_props: { selectedService: string; onStartJob: () => void }) {
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
              <p className="text-muted-foreground text-sm">Drag images from the sidebar</p>
            ) : (
              <p className="text-muted-foreground text-sm">{segments.length} keyframe(s)</p>
            )}
          </CardContent>
        </Card>
        <div className="h-24 flex items-center gap-1 p-2 border rounded-lg overflow-x-auto bg-muted/30">
          {segments.map((seg) => (
            <div key={seg.id}
              className={`h-full flex items-center justify-center rounded-md text-[10px] cursor-pointer relative overflow-hidden shrink-0 border-2 ${seg.id === selectedSegmentId ? "border-primary" : "border-border"}`}
              style={{ width: `${seg.duration * 40}px` }}
              onClick={() => setSelectedSegment(seg.id)}>
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
