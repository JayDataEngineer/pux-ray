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
import { useAssetStore, type Asset } from "@/stores/assets"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, PanelRightClose, Wand2, Loader2, CheckCircle2, XCircle, Clock, ListTodo, X } from "lucide-react"
import { AppSidebar } from "./AppSidebar"
import { VideoEditor } from "./VideoEditor"

type TabId = "assets" | "video"

interface JobEntry {
  id: number; name: string
  status: "pending" | "running" | "completed" | "failed"
  startedAt: number; endedAt?: number; error?: string
}

interface FieldDef {
  name: string; type: "text" | "number" | "select" | "file" | "textarea" | "json"
  label: string; default?: unknown; options?: string[]; required?: boolean
}

const COMMON_PARAMS = [
  "model", "prompt", "text", "image_b64", "audio_b64",
  "seed", "steps", "guidance", "width", "height", "frames",
  "negative_prompt", "voice", "language",
]

const GENRE_ORDER = ["image", "audio", "voice"]

const SERVICE_GENRE: Record<string, string> = {
  ace_step: "audio", moss_soundeffect: "audio",
  kokoro: "voice", espeak: "voice", index_tts: "voice", faster_qwen3_tts: "voice",
  generate_sound: "audio", generate_music: "audio", tts_speak: "voice",
  generate_image: "image", char_sheet: "image", pose_edit: "image",
}

// Services covered by a dedicated MCP tool — hide from sidebar to avoid duplicates
const COVERED_SERVICES = new Set(["z_image"])

function extractCommonParams(desc: string): FieldDef[] {
  if (!desc.toLowerCase().includes("common:")) return []
  return COMMON_PARAMS.map((n) => ({
    name: n, label: n.replace(/_/g, " "),
    type: (["seed","steps","guidance","width","height","frames"].includes(n) ? "number" : "textarea") as FieldDef["type"],
    default: undefined,
  }))
}

export function WorkspaceLayout() {
  const [tab, setTab] = useState<TabId>("assets")
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [selectedService, setSelectedService] = useState("")
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const [jobs, setJobs] = useState<JobEntry[]>([])
  const nextJobId = useRef(1)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedAsset(null)
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [])

  return (
    <div className="flex h-screen w-full bg-background">
      <AppSidebar open={leftOpen} onToggle={() => setLeftOpen((o) => !o)} onSelectAsset={(a) => { setSelectedService(""); setSelectedAsset(a) }} />
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
          <JobsButton jobs={jobs} />
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setRightOpen((o) => !o)}>
            <PanelRightClose className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 overflow-auto">
            {tab === "assets"
              ? <AssetsTab selectedService={selectedService} selectedAsset={selectedAsset} onCloseAsset={() => setSelectedAsset(null)} jobs={jobs} onAddJob={(j) => setJobs(j)} nextJobId={nextJobId} />
              : <VideoEditor />
            }
          </div>
          {rightOpen && <ServicesSidebar selected={selectedService} onSelect={setSelectedService} />}
        </div>
      </div>
    </div>
  )
}

function JobsButton({ jobs }: { jobs: JobEntry[] }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  const running = jobs.filter((j) => j.status === "running")

  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" size="icon" className="h-7 w-7 relative" onClick={() => setOpen(!open)}
        title={`${jobs.length} job(s)`}>
        {running.length > 0 ? (
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
        ) : (
          <ListTodo className="h-4 w-4" />
        )}
        {jobs.length > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-primary text-[8px] text-primary-foreground">
            {jobs.length > 9 ? "9+" : jobs.length}
          </span>
        )}
      </Button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-72 rounded-md border bg-popover text-popover-foreground shadow-md z-50">
          <div className="p-2 border-b text-xs font-medium flex items-center gap-1.5">
            <Clock className="h-3 w-3" /> Jobs ({jobs.length})
          </div>
          {jobs.length === 0 ? (
            <p className="text-xs text-muted-foreground/50 py-4 text-center">No jobs</p>
          ) : (
            <div className="max-h-72 overflow-y-auto p-1 space-y-0.5">
              {jobs.map((j) => (
                <div key={j.id} className="flex items-center gap-2 px-2 py-1.5 rounded text-xs hover:bg-accent">
                  {j.status === "running" && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />}
                  {j.status === "completed" && <CheckCircle2 className="h-3 w-3 shrink-0 text-green-500" />}
                  {j.status === "failed" && <XCircle className="h-3 w-3 shrink-0 text-destructive" />}
                  <span className="flex-1 truncate">{j.name}</span>
                  {j.status === "running" && <span className="text-[10px] text-muted-foreground">{Math.round((Date.now() - j.startedAt) / 1000)}s</span>}
                  {j.status === "completed" && j.endedAt && <span className="text-[10px] text-muted-foreground">{(j.endedAt - j.startedAt) / 1000}s</span>}
                  {j.status === "failed" && <span className="text-[10px] text-destructive truncate max-w-24">{j.error}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
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

function ServicesSidebar({ selected, onSelect }: { selected: string; onSelect: (n: string) => void }) {
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])
  const [genre, setGenre] = useState("image")

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const allItems = [
    ...tools.filter((t) => !["run","list_models","list_services","get_service","forge_status","load_service","unload_services","tts_voices","chat","transcribe","llm_configure"].includes(t.name) && !t.name.startsWith("workflow_")),
    ...services.filter((s) => !tools.find((t) => t.name === s.name) && !COVERED_SERVICES.has(s.name)),
  ]
  const items = allItems.filter((s) => SERVICE_GENRE[("name" in s ? (s as any).name : (s as any).name)] === genre)

  return (
    <div className="w-56 border-l bg-sidebar text-sidebar-foreground flex flex-col shrink-0">
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
      <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
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
    </div>
  )
}

function AssetsTab({ selectedService, selectedAsset, onCloseAsset, jobs, onAddJob, nextJobId }: {
  selectedService: string; selectedAsset: Asset | null; onCloseAsset: () => void
  jobs: JobEntry[]
  onAddJob: (j: JobEntry[] | ((prev: JobEntry[]) => JobEntry[])) => void
  nextJobId: React.MutableRefObject<number>
}) {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)
  const [fields, setFields] = useState<FieldDef[]>([])

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const currentTool = tools.find((t) => t.name === selectedService)
  const currentService = services.find((s) => s.name === selectedService)

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
  }, [selectedService, currentTool])

  const handleGenerate = async () => {
    if (!currentTool && !currentService) return
    const useTool = currentTool || tools.find((t) => t.name === "run")
    if (!useTool) return

    const jobId = nextJobId.current++
    const jobName = currentService?.label || currentTool?.description?.split("—")[0]?.trim() || selectedService
    onAddJob((prev) => [{ id: jobId, name: jobName, status: "running", startedAt: Date.now() }, ...prev])

    setGenerating(true)
    try {
      const args: Record<string, unknown> = {}
      if (useTool.name === "run") {
        args.service = currentService?.name || selectedService
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
          const cat = isAud && selectedService.includes("music") ? "music" as const : isAud ? "sfx" as const : "image" as const
          addAsset({
            name: `${jobName} ${new Date().toLocaleTimeString()}`,
            type: isAud ? "audio" : "image", category: cat, mediaType: mt,
            url: `data:${mt};base64,${result.data}`,
            sizeBytes: Math.round((result.data as string).length * 0.75), source: "generated",
          })
          toast("success", `${jobName} generated`)
        }
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "completed", endedAt: Date.now() } : j))
      } else {
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: result.error || result.message || "Unknown error" } : j))
        toast("error", String(result.error || result.message || "Unknown error"))
      }
    } catch (e) {
      onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: e instanceof Error ? e.message : String(e) } : j))
      toast("error", e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  const running = jobs.filter((j) => j.status === "running")
  const label = currentService?.label || currentTool?.description?.split("—")[0]?.trim() || selectedService

  if (selectedAsset) {
    const { url, name, mediaType, sizeBytes } = selectedAsset
    const isImage = mediaType.startsWith("image/") || url.startsWith("data:image/")
    return (
      <div className="flex-1 p-6 flex justify-center">
        <div className="w-full max-w-2xl">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold">{name}</h2>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onCloseAsset} title="Close (Esc)">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <Card>
            <CardContent className="p-4 flex items-center justify-center min-h-[300px]">
              {isImage ? (
                <img src={url} alt={name} className="max-w-full max-h-[70vh] object-contain rounded-lg" />
              ) : (
                <audio src={url} controls className="w-full" />
              )}
            </CardContent>
          </Card>
          <div className="mt-2 text-xs text-muted-foreground">
            Size: {sizeBytes ? `${Math.round(sizeBytes / 1024)} KB` : "Unknown"}
            {mediaType ? <> · Type: {mediaType}</> : null}
          </div>
        </div>
      </div>
    )
  }

  if (!selectedService) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">Select a service from the right sidebar, or click an asset in the left sidebar</p>
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 flex justify-center">
      <div className="w-full max-w-xl">
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">{label}</h2>
              {selectedService && <Badge variant="outline" className="text-[10px]">{selectedService}</Badge>}
            </div>
            {fields.map((f) => {
              const handleDrop = (e: React.DragEvent) => {
                e.preventDefault()
                try {
                  const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                  if (d.url && (f.type === "file" || f.name.includes("image") || f.name.includes("b64"))) {
                    setValues((p) => ({ ...p, [f.name]: d.url.split(",")[1] || d.url }))
                  } else if (d.name && f.type === "text") {
                    setValues((p) => ({ ...p, [f.name]: d.name }))
                  }
                } catch {}
              }
              return (
              <div key={f.name} className="space-y-1" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
                <Label className="text-xs text-muted-foreground capitalize">{f.label}</Label>
                {f.type === "select" && f.options ? (
                  <Select value={String(values[f.name] ?? f.default ?? "")}
                    onValueChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {f.options.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                    </SelectContent>
                  </Select>
                ) : f.type === "number" ? (
                  <Input type="number" value={String(values[f.name] ?? f.default ?? "")}
                    onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                    placeholder={f.label} />
                ) : f.type === "file" ? (
                  <Label className="flex items-center justify-center h-16 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/50 text-sm text-muted-foreground"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault()
                      try {
                        const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                        if (d.url) setValues((p) => ({ ...p, [f.name]: d.url.split(",")[1] || d.url }))
                      } catch {}
                    }}>
                    {values[f.name] ? "Loaded from asset" : "Drop asset or click to upload"}
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
                    placeholder={f.label} rows={3} />
                ) : (
                  <Input value={String(values[f.name] ?? f.default ?? "")}
                    onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                    placeholder={f.label} />
                )}
              </div>
              )
            })}
            {fields.length > 0 && (
              <Button className="w-full" disabled={generating || running.length > 0} onClick={handleGenerate}>
                {generating ? "Generating..." : "Generate"}
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}


