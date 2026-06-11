import { useState, useEffect, useCallback, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { LoraPicker } from "@/components/LoraPicker"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useToastStore } from "@/stores/toast"
import { useAssetStore, type Asset, nextAssetName } from "@/stores/assets"
import { useEnhanceStore } from "@/stores/enhancement"
import { enhancePrompt, storeEnhanceKey } from "@/lib/enhance"
import { getEnhancePrompt } from "@/lib/enhance-prompts"
import { EnhanceConfigDialog } from "@/components/EnhanceConfigDialog"
import { LLMChatDialog } from "@/components/LLMChatDialog"
import { kimodoUrl } from "@/mcp"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, ChevronLeft, ChevronRight, Wand2, Loader2, CheckCircle2, XCircle, Clock, ListTodo, Maximize2, Download, Sparkles, AlertTriangle, ChevronDown, Plus, Trash2, HelpCircle } from "lucide-react"
import { AppSidebar } from "./AppSidebar"
import { VideoEditor } from "./VideoEditor"
import { AudioWaveform } from "@/components/AudioWaveform"

type TabId = "assets" | "video"

export interface JobEntry {
  id: number; name: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  startedAt: number; endedAt?: number; error?: string
  abortController?: AbortController
}

interface FieldDef {
  name: string; type: "text" | "number" | "select" | "file" | "textarea" | "json" | "boolean"
  label: string; default?: unknown; options?: string[]; required?: boolean
}

const COMMON_PARAMS = [
  "model", "prompt", "text", "image_b64", "audio_b64",
  "seed", "steps", "guidance", "width", "height", "frames",
  "negative_prompt", "voice", "language",
]

// ── Dynamic TTS: per-engine field visibility ────────────────────────────
// When engine changes, irrelevant fields for other engines are hidden.

const TTS_ENGINE_VISIBLE_FIELDS: Record<string, string[]> = {
  kokoro: ["text", "engine", "voice"],
  espeak: ["text", "engine", "language"],
  moss_tts: [
    "text", "engine", "instruct", "ref_audio_b64", "language", "seed",
    "max_new_tokens",
    "text_temperature", "text_top_p", "text_top_k", "text_repetition_penalty",
    "audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty",
    "n_vq_for_inference",
  ],
  index_tts: ["text", "engine"],
  qwen3_tts: ["text", "engine", "mode", "voice", "instruct", "ref_audio_b64", "language"],
}

function ttsVisibleFields(engine: string, allFields: FieldDef[]): FieldDef[] {
  const visible = TTS_ENGINE_VISIBLE_FIELDS[engine]
  if (!visible) return allFields  // Unknown engine → show all
  return allFields.filter((f) => visible.includes(f.name))
}

// ── Dynamic Voice Creator: per-engine field visibility ──────────────────────
function voiceCreatorVisibleFields(engine: string, mode: string, allFields: FieldDef[]): FieldDef[] {
  if (engine === "qwen3_tts") {
    if (mode === "voice_clone") {
      return allFields.filter((f) =>
        ["text", "engine", "mode", "ref_audio_b64", "ref_audio_b64_list", "language"].includes(f.name)
      )
    } else if (mode === "custom_voice") {
      return allFields.filter((f) =>
        ["text", "engine", "mode", "voice", "language"].includes(f.name)
      )
    } else {
      // voice_design
      return allFields.filter((f) =>
        ["text", "engine", "mode", "instruct", "language"].includes(f.name)
      )
    }
  } else {
    // moss_voicegenerator - show all fields including ref_audio_b64_list
    return allFields
  }
}

const GENRE_ORDER = ["image", "audio", "motion", "3d", "external"]

const GENRE_ICONS: Record<string, string> = { image: "◎", audio: "♪", motion: "↝", "3d": "⟁", external: "⤴" }

// Map backend category → genre for sidebar grouping
const CATEGORY_TO_GENRE: Record<string, string> = {
  audio: "audio", motion: "motion", "3d": "3d",
}

// Helper component for field labels with tooltips
function FieldLabel({ label, tooltip }: { label: string; tooltip?: string }) {
  if (!tooltip) {
    return <Label className="text-xs text-muted-foreground capitalize">{label}</Label>
  }
  return (
    <div className="flex items-center gap-1">
      <Label className="text-xs text-muted-foreground capitalize">{label}</Label>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            {tooltip}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  )
}

const SERVICE_GENRE: Record<string, string> = {
  generate: "image", edit: "image", generate_character_sheet: "image",
  generate_image: "image", pose_edit: "image", char_sheet: "image",
  generate_music: "audio", ace_step: "audio",
  generate_sound: "audio", moss_soundeffect: "audio",
  tts_speak: "audio",
  voice_creator: "audio",
  kimodo: "motion", kimodo_demo: "motion", hy_motion: "motion", gemx: "motion",
  _kimodo_studio: "external",
  _llm_chat: "external",
}

const SERVICE_LABELS: Record<string, string> = {
  generate: "Generate", generate_image: "Generate",
  edit: "Edit", pose_edit: "Edit",
  generate_character_sheet: "Character Sheet", char_sheet: "Character Sheet",
  generate_music: "Music", ace_step: "Music",
  generate_sound: "Sound Effect", moss_soundeffect: "Sound Effect",
  tts_speak: "Text to Speech",
  voice_creator: "Voice Creator",
  kimodo: "Kimodo Motion",
  kimodo_demo: "Kimodo Demo",
  hy_motion: "HY-Motion",
  gemx: "GEM-X Pose",
  body_mesh: "BodyMesh",
  _kimodo_studio: "Kimodo Motion Studio",
  _llm_chat: "AI Chat",
}

// Backend-only services hidden from sidebar
const HIDDEN_SERVICES = new Set([
  "z_image", "wan2gp", "comfyui", "llm", "faster_whisper", "vibevoice_asr",
  "see_through", "nvidia_upscale", "dwpose", "lance", "kohya", "avatar",
  "kokoro", "espeak", "index_tts", "faster_qwen3_tts",
  "moss_voicegenerator", "moss_tts",
  "clone_character", "list_pipelines",
  "kimodo", "kimodo_demo", "hy_motion", "hy_motion_lite", "gemx",
  "trellis", "anigen", "body_mesh", "pixal3d",
])


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
  const [enhanceConfigOpen, setEnhanceConfigOpen] = useState(false)
  const [kimodoOpen, setKimodoOpen] = useState(false)
  const [llmChatOpen, setLLMChatOpen] = useState(false)
  const nextJobId = useRef(1)

  const enhanceActiveModel = useEnhanceStore((s) => s.activeModel)
  const hasEnhanceModel = !!enhanceActiveModel()

  const handleCancelJob = useCallback((jobId: number) => {
    setJobs((prevJobs) => {
      const job = prevJobs.find((j) => j.id === jobId)
      if (job && job.status === "running" && job.abortController) {
        // Abort the fetch request
        job.abortController.abort()
        // Update job status to cancelled
        return prevJobs.map((j) =>
          j.id === jobId
            ? { ...j, status: "cancelled" as const, endedAt: Date.now() }
            : j
        )
      }
      return prevJobs
    })
  }, [])

  return (
    <div className="flex h-screen w-full bg-background">
      <AppSidebar open={leftOpen} onToggle={() => setLeftOpen((o) => !o)} onSelectAsset={(a) => setSelectedAsset(a)} />
      <div className="flex flex-1 flex-col min-w-0">
        <header className="flex items-center h-11 px-4 border-b gap-2 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setLeftOpen((o) => !o)}>
            {leftOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </Button>
          <span className="font-bold text-sm tracking-tight">TECH NOIR</span>
          <Separator orientation="vertical" className="h-5 mx-1" />
          <Button variant={tab === "assets" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs"
            onClick={() => setTab("assets")}>Assets</Button>
          <Button variant={tab === "video" ? "secondary" : "ghost"} size="sm" className="h-7 text-xs"
            onClick={() => setTab("video")}>Video</Button>
          <div className="flex-1" />
          <Button variant="ghost" size="icon" className="h-7 w-7 relative" onClick={() => setEnhanceConfigOpen(true)}
            title="AI Prompt Enhancement">
            <Sparkles className={`h-4 w-4 ${hasEnhanceModel ? "text-primary" : "text-muted-foreground"}`} />
          </Button>
          <GpuStatus />
          <JobsButton jobs={jobs} onCancelJob={handleCancelJob} />
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setRightOpen((o) => !o)}>
            {rightOpen ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </header>
        <div className="flex flex-1 min-h-0">
          {tab === "assets" ? (
            <div className="flex-1 min-w-0 overflow-auto scrollbar-thin">
              <AssetsTab selectedService={selectedService} jobs={jobs} onAddJob={(j) => setJobs(j)} nextJobId={nextJobId} onOpenKimodo={() => setKimodoOpen(true)} onOpenLLM={() => setLLMChatOpen(true)} />
            </div>
          ) : (
            <VideoEditor jobs={jobs} onAddJob={(j) => setJobs(j)} />
          )}
          {rightOpen && tab === "assets" && <ServicesSidebar selected={selectedService} onSelect={setSelectedService} onOpenKimodo={() => setKimodoOpen(true)} onOpenLLM={() => setLLMChatOpen(true)} />}
        </div>
      </div>
      <AssetPreviewDialog asset={selectedAsset} onClose={() => setSelectedAsset(null)} onSelect={(a) => setSelectedAsset(a)} />
      <KimodoDialog open={kimodoOpen} onOpenChange={setKimodoOpen} />
      <LLMChatDialog open={llmChatOpen} onOpenChange={setLLMChatOpen} />
      <EnhanceConfigDialog open={enhanceConfigOpen} onOpenChange={setEnhanceConfigOpen} />
    </div>
  )
}

function JobsButton({ jobs, onCancelJob }: { jobs: JobEntry[]; onCancelJob: (id: number) => void }) {
  const [open, setOpen] = useState(false)
  const [currentTime, setCurrentTime] = useState(Date.now())
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  // Update current time every second when there are running jobs
  useEffect(() => {
    const running = jobs.filter((j) => j.status === "running")
    if (running.length === 0) return

    const interval = setInterval(() => {
      setCurrentTime(Date.now())
    }, 1000)

    return () => clearInterval(interval)
  }, [jobs])

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
                <div key={j.id} className="flex items-center gap-2 px-2 py-1.5 rounded text-xs hover:bg-accent group">
                  {j.status === "running" && <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />}
                  {j.status === "completed" && <CheckCircle2 className="h-3 w-3 shrink-0 text-green-500" />}
                  {j.status === "failed" && <XCircle className="h-3 w-3 shrink-0 text-destructive" />}
                  {j.status === "cancelled" && <XCircle className="h-3 w-3 shrink-0 text-muted-foreground" />}
                  <span className="flex-1 truncate">{j.name}</span>
                  {j.status === "running" && <span className="text-[10px] text-muted-foreground">{Math.round((currentTime - j.startedAt) / 1000)}s</span>}
                  {j.status === "completed" && j.endedAt && <span className="text-[10px] text-muted-foreground">{(j.endedAt - j.startedAt) / 1000}s</span>}
                  {j.status === "failed" && <span className="text-[10px] text-destructive truncate max-w-24">{j.error}</span>}
                  {j.status === "cancelled" && <span className="text-[10px] text-muted-foreground">Cancelled</span>}
                  {j.status === "running" && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => onCancelJob(j.id)}
                      title="Cancel job"
                    >
                      <XCircle className="h-3 w-3 text-destructive" />
                    </Button>
                  )}
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
  const [status, setStatus] = useState<{ loaded: number; vram_free_mb: number; vram_total_mb: number; loaded_models?: Record<string, number> } | null>(null)
  const [error, setError] = useState(false)
  const [hoverOpen, setHoverOpen] = useState(false)
  const [unloading, setUnloading] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const refresh = useCallback(async () => {
    try {
      const s = await forgeStatus()

      // The real GPU info is in gpu.total_mb and gpu.reserved_mb from torch.cuda
      // Forge's vram_free_mb is wrong (returns system RAM, not VRAM)
      const gpu_total = s.gpu?.total_mb
      const gpu_reserved = s.gpu?.reserved_mb

      // Calculate actual free VRAM from torch.cuda data
      const vram_total = gpu_total || 22528
      const vram_used = gpu_reserved || 0  // reserved is what actually matters
      const vram_free = Math.max(0, vram_total - vram_used)

      const loaded_models = s.loaded || {}

      setStatus({
        loaded: Object.keys(loaded_models).length,
        vram_free_mb: vram_free,
        vram_total_mb: vram_total,
        loaded_models: loaded_models
      })
      setError(false)
    } catch (err) {
      console.error('[GpuStatus] Error fetching status:', err)
      setError(true)
    }
  }, [])

  const unloadAll = useCallback(async () => {
    if (!confirm('Are you sure you want to unload all GPU models? This will free all VRAM but any subsequent generation will need to reload models.')) {
      return
    }

    setUnloading(true)
    try {
      const res = await fetch('/forge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'unload' }),
      })
      if (!res.ok) throw new Error(`Failed to unload: ${res.status}`)

      const body = await res.json()
      if (body.status === 'error') throw new Error(body.error || 'Unload failed')

      // Refresh status after unload
      await refresh()
    } catch (err) {
      console.error('[GpuStatus] Unload error:', err)
      setError(true)
    } finally {
      setUnloading(false)
    }
  }, [refresh])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setHoverOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  if (error) return <Badge variant="outline" className="text-xs gap-1"><Cpu className="h-3 w-3 text-destructive" />Offline</Badge>
  if (!status) return <Skeleton className="h-5 w-20" />

  // Use the already-calculated values from refresh()
  const total = Math.max(0, status.vram_total_mb || 0)
  const free = Math.max(0, status.vram_free_mb || 0)
  const used = Math.max(0, total - free)
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0

  // Debug log if values seem wrong
  if (pct < 0 || pct > 100) {
    console.warn('[GpuStatus] Unexpected percentage:', { total, free, used, pct })
  }

  return (
    <div ref={ref} className="relative">
      <Badge
        variant="outline"
        className="text-xs gap-1 cursor-pointer"
        onClick={() => setHoverOpen(!hoverOpen)}
        title={hoverOpen ? "" : "Click to see loaded models"}
      >
        <HardDrive className="h-3 w-3" />{pct}% GPU
      </Badge>
      {hoverOpen && (
        <div className="absolute right-0 top-full mt-1 w-72 rounded-md border bg-popover text-popover-foreground shadow-md z-50">
          <div className="p-2 border-b text-xs font-medium flex items-center gap-1.5">
            <Cpu className="h-3 w-3" /> GPU Status
          </div>
          <div className="p-2 space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">VRAM Usage:</span>
              <span className="font-medium">{used} MB / {total} MB ({pct}%)</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Free:</span>
              <span className="font-medium">{free} MB</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Loaded Models:</span>
              <span className="font-medium">{status.loaded}</span>
            </div>
            {status.loaded_models && Object.keys(status.loaded_models).length > 0 && (
              <div className="pt-1.5 border-t mt-2">
                <div className="text-[10px] text-muted-foreground mb-1">Loaded Services:</div>
                {Object.entries(status.loaded_models).map(([name, vram]) => (
                  <div key={name} className="flex justify-between text-xs py-0.5">
                    <span className="text-muted-foreground">{name}:</span>
                    <span className="font-medium">{vram} MB</span>
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2 mt-2">
              <Button
                variant="ghost"
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={(e) => {
                  e.stopPropagation()
                  refresh()
                }}
              >
                Refresh
              </Button>
              {status.loaded > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  className="flex-1 h-7 text-xs"
                  onClick={(e) => {
                    e.stopPropagation()
                    unloadAll()
                  }}
                  disabled={unloading}
                >
                  {unloading ? (
                    <>
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      Unloading...
                    </>
                  ) : (
                    'Unload All'
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ServicesSidebar({ selected, onSelect, onOpenKimodo, onOpenLLM }: { selected: string; onSelect: (n: string) => void; onOpenKimodo: () => void; onOpenLLM: () => void }) {
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const allItems = [
    ...tools.filter((t) =>
      !["run","list_models","list_services","get_service","forge_status","load_service","unload_services","tts_voices","chat","transcribe","llm_configure"].includes(t.name) &&
      !t.name.startsWith("workflow_") &&
      !HIDDEN_SERVICES.has(t.name)
    ),
    ...services.filter((s) => !tools.find((t) => t.name === s.name) && !HIDDEN_SERVICES.has(s.name)),
    // External links
    { name: "_kimodo_studio", label: "Kimodo Motion Studio", category: "external" } as { name: string; label: string; category: string },
    { name: "_llm_chat", label: "AI Chat", category: "external" } as { name: string; label: string; category: string },
  ]

  const getLabel = (item: MCPTool | { name: string; label: string }) => {
    const name = ("name" in item) ? (item as any).name : ""
    if (SERVICE_LABELS[name]) return SERVICE_LABELS[name]
    if ("label" in item && (item as any).label && (item as any).label !== name) return (item as any).label
    return name.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase())
  }

  // Resolve genre: only items in SERVICE_GENRE or with mapped category get shown
  const getGenre = (item: MCPTool | { name: string; label: string; category?: string }) => {
    const name = ("name" in item) ? (item as any).name : ""
    if (SERVICE_GENRE[name]) return SERVICE_GENRE[name]
    const cat = ("category" in item && (item as any).category) ? (item as any).category : ""
    return CATEGORY_TO_GENRE[cat] || ""
  }

  // Group by genre, deduplicate by label
  const grouped = GENRE_ORDER.map((genre) => {
    const genreItems = allItems.filter((s) => getGenre(s) === genre)
    const seen = new Set<string>()
    const unique = genreItems.filter((item) => {
      const label = getLabel(item)
      if (seen.has(label)) return false
      seen.add(label)
      return true
    })
    return { genre, items: unique }
  }).filter((g) => g.items.length > 0)

  return (
    <div className="w-56 border-l bg-sidebar text-sidebar-foreground flex flex-col shrink-0">
      <div className="p-2 border-b">
        <span className="text-xs font-semibold">Services</span>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin p-1.5">
        {grouped.map(({ genre, items }) => (
          <div key={genre} className="mb-2">
            <div className="flex items-center gap-1.5 px-1.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-sidebar-foreground/40">
              <span>{GENRE_ICONS[genre]}</span>
              <span>{genre}</span>
            </div>
            {items.map((item) => {
              const name = ("name" in item ? (item as any).name : (item as any).name) as string
              const label = getLabel(item)
              const isKimodo = name === "_kimodo_studio"
              const isLLM = name === "_llm_chat"
              return (
                <button key={name} onClick={() => {
                  if (isKimodo) onOpenKimodo()
                  else if (isLLM) onOpenLLM()
                  else onSelect(name)
                }}
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left transition-colors ${selected === name ? "bg-sidebar-accent text-sidebar-accent-foreground" : "hover:bg-sidebar-accent/50"}`}>
                  <Wand2 className="h-3 w-3 shrink-0 opacity-50" />
                  <span className="truncate">{label}</span>
                </button>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Kimodo Picture-in-Picture Dialog ──────────────────────────────────────────

function KimodoDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [loading, setLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Refs to avoid re-creating preload callback on state changes (prevents race)
  const loadingRef = useRef(false)
  const loadedRef = useRef(false)

  // Trigger preload when dialog opens — uses refs so the effect doesn't
  // re-run on state updates from the fetch itself.
  useEffect(() => {
    if (!open) { setError(null); return }
    if (loadedRef.current || loadingRef.current) return

    const controller = new AbortController()
    let cancelled = false
    loadingRef.current = true
    setLoading(true)
    setError(null)

    fetch('/forge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'preload', service: 'kimodo_demo' }),
      signal: controller.signal,
    })
      .then(async (res) => {
        const text = await res.text()
        if (!res.ok) {
          let msg = `HTTP ${res.status}`
          try { const b = JSON.parse(text); msg = b.error || msg } catch { /* use default */ }
          throw new Error(msg)
        }
        let body: any = {}
        try { body = JSON.parse(text) } catch { /* empty */ }
        if (body.status === 'error') throw new Error(body.error || 'Failed to load')
        console.log('[KimodoDialog] preload success:', body.status)
        if (!cancelled) { loadedRef.current = true; setLoaded(true) }
      })
      .catch((e) => {
        if (cancelled) return
        console.error('[KimodoDialog] preload error:', e.message)
        setError(e.message || 'Failed to load Kimodo')
      })
      .finally(() => {
        if (!cancelled) { loadingRef.current = false; setLoading(false) }
      })
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [open]) // intentionally only depends on `open`

  const handleClose = useCallback((willOpen: boolean) => {
    onOpenChange(willOpen)
  }, [onOpenChange])

  // The full Kimodo URL — same origin, proxied through ingress
  const url = kimodoUrl()

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent
        showCloseButton={true}
        className="sm:max-w-[95vw] w-full h-[90vh] max-h-[90vh] p-0 gap-0 flex flex-col overflow-hidden"
      >
        <DialogHeader className="px-4 py-2 border-b shrink-0 flex-row items-center justify-between space-y-0">
          <div className="flex items-center gap-2">
            <DialogTitle className="text-sm">Kimodo Motion Studio</DialogTitle>
            <Badge variant="outline" className="text-[9px]">NVIDIA</Badge>
          </div>
          <DialogDescription className="text-[10px] text-muted-foreground sr-only">
            Pose a character in 3D, then take a screenshot (Exports → Screenshot) and drag the PNG into the pose image field.
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 min-h-0 relative bg-black">
          {loading ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="text-xs">Loading Kimodo on GPU…</span>
              <span className="text-[10px] text-muted-foreground/60">Evicting other models & loading — takes 1–3 min</span>
            </div>
          ) : error ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-3 px-8">
              <AlertTriangle className="h-8 w-8 text-destructive" />
              <span className="text-xs text-center max-w-md">Failed to load Kimodo: {error}</span>
              <span className="text-[10px] text-muted-foreground/60 text-center">
                The forge will auto-evict other models to free VRAM. This may fail if the model files are missing or corrupted.
              </span>
              <Button
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => { setError(null); setLoaded(false); loadedRef.current = false; }}
              >
                Retry
              </Button>
            </div>
          ) : loaded ? (
            <iframe
              src={url}
              className="w-full h-full border-0"
              allow="clipboard-write"
              title="Kimodo Motion Studio"
            />
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function AssetPreviewDialog({ asset, onClose, onSelect }: { asset: Asset | null; onClose: () => void; onSelect: (a: Asset) => void }) {
  const allAssets = useAssetStore((s) => s.assets)
  const [idx, setIdx] = useState(-1)

  useEffect(() => {
    if (asset) {
      const i = allAssets.findIndex((a) => a.id === asset.id)
      setIdx(i)
    }
  }, [asset, allAssets])

  const hasPrev = idx > 0
  const hasNext = idx >= 0 && idx < allAssets.length - 1

  const goPrev = () => { if (hasPrev) onSelect(allAssets[idx - 1]) }
  const goNext = () => { if (hasNext) onSelect(allAssets[idx + 1]) }

  useEffect(() => {
    if (!asset) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") goPrev()
      else if (e.key === "ArrowRight") goNext()
      else if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  })

  const handleDownload = () => {
    if (!asset) return
    const link = document.createElement("a")
    link.href = asset.url
    link.download = asset.name
    link.click()
  }

  if (!asset) return null
  const { url, name, mediaType, sizeBytes } = asset
  const isImage = mediaType.startsWith("image/") || url.startsWith("data:image/")

  return (
    <Dialog open={!!asset} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="sm:max-w-3xl p-0 gap-0 flex flex-col overflow-hidden">
        <DialogHeader className="px-4 py-3 border-b shrink-0">
          <DialogTitle className="text-sm font-medium truncate">{name}</DialogTitle>
          <DialogDescription className="sr-only">Preview asset {name}</DialogDescription>
        </DialogHeader>
        <div className="flex-1 min-h-0 relative flex items-center justify-center bg-black/40 p-4">
          {hasPrev && (
            <Button variant="ghost" size="icon" className="absolute left-2 z-10 h-9 w-9 rounded-full bg-background/80 hover:bg-background"
              onClick={goPrev}>
              <ChevronLeft className="h-5 w-5" />
            </Button>
          )}
          {isImage ? (
            <img src={url} alt={name} className="max-w-full max-h-[70vh] object-contain rounded-lg select-none" />
          ) : (
            <audio src={url} controls className="w-full max-w-lg" />
          )}
          {hasNext && (
            <Button variant="ghost" size="icon" className="absolute right-2 z-10 h-9 w-9 rounded-full bg-background/80 hover:bg-background"
              onClick={goNext}>
              <ChevronRight className="h-5 w-5" />
            </Button>
          )}
        </div>
        <div className="px-4 py-2 border-t flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground">{idx + 1} / {allAssets.length}</span>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-muted-foreground">
              {sizeBytes ? `${Math.round(sizeBytes / 1024)} KB` : ""}{mediaType ? ` · ${mediaType}` : ""}
            </span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleDownload} title="Download">
              <Download className="h-3.5 w-3.5" />
            </Button>
          </div>
          <span className="text-[10px] text-muted-foreground">← → navigate</span>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function AssetsTab({ selectedService, jobs, onAddJob, nextJobId, onOpenKimodo, onOpenLLM }: {
  selectedService: string
  jobs: JobEntry[]
  onAddJob: (j: JobEntry[] | ((prev: JobEntry[]) => JobEntry[])) => void
  nextJobId: React.MutableRefObject<number>
  onOpenKimodo: () => void
  onOpenLLM: () => void
}) {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)
  const [fields, setFields] = useState<FieldDef[]>([])
  const [enhancing, setEnhancing] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [promptingOpen, setPromptingOpen] = useState(false)
  const [voiceExamples, setVoiceExamples] = useState<any[]>([])
  const [voiceExampleCategories, setVoiceExampleCategories] = useState<Record<string, string>>({})
  const [voiceSamplingPresets, setVoiceSamplingPresets] = useState<Record<string, any>>({})
  const [voiceQwen3Voices, setVoiceQwen3Voices] = useState<string[]>([])
  const [voiceQwen3Modes, setVoiceQwen3Modes] = useState<Record<string, string>>({})
  const [loadingVoiceExamples, setLoadingVoiceExamples] = useState(false)
  const [voiceAdvancedOpen, setVoiceAdvancedOpen] = useState(false)
  const [quantity, setQuantity] = useState(1)
  const [voiceComparisonMode, setVoiceComparisonMode] = useState(false)
  const [comparisonResults, setComparisonResults] = useState<any[]>([])
  const [dialogueMode, setDialogueMode] = useState(false)
  const [batchMode, setBatchMode] = useState(false)
  const [dialogueScript, setDialogueScript] = useState("")
  const [pauseControlExamples, setPauseControlExamples] = useState<any[]>([])
  const [dialogueExamples, setDialogueExamples] = useState<any[]>([])
  const [multipleRefAudios, setMultipleRefAudios] = useState<string[]>([])
  const prevServiceRef = useRef("")

  const enhanceActiveModel = useEnhanceStore((s) => s.activeModel)

  // Load voice examples when voice_creator service is selected
  useEffect(() => {
    if (selectedService === "voice_creator" && voiceExamples.length === 0 && !loadingVoiceExamples) {
      setLoadingVoiceExamples(true)
      callTool<{ status: string; examples?: any[]; categories?: Record<string, string>; sampling_presets?: Record<string, any>; qwen3_voices?: string[]; qwen3_modes?: Record<string, string>; pause_control_examples?: any[]; dialogue_examples?: any[] }>(
        "voice_creator_examples",
        {}
      ).then((result) => {
        if (result.status === "ok") {
          setVoiceExamples(result.examples || [])
          setVoiceExampleCategories(result.categories || {})
          setVoiceSamplingPresets(result.sampling_presets || {})
          setVoiceQwen3Voices(result.qwen3_voices || [])
          setVoiceQwen3Modes(result.qwen3_modes || {})
          setPauseControlExamples(result.pause_control_examples || [])
          setDialogueExamples(result.dialogue_examples || [])
        }
      }).catch(() => {
        // Examples failed to load, continue without them
      }).finally(() => {
        setLoadingVoiceExamples(false)
      })
    }
  }, [selectedService, voiceExamples.length, loadingVoiceExamples])

  // Handle voice example selection
  const handleVoiceExampleSelect = useCallback((exampleId: string) => {
    const example = voiceExamples.find((ex) => ex.id === exampleId)
    if (example) {
      setValues((prev) => ({
        ...prev,
        instruct: example.instruction,
        text: example.text,
        language: example.language,
      }))
      toast("info", `Loaded example: ${example.instruction.substring(0, 50)}...`)
    }
  }, [voiceExamples, toast])

  // Handle sampling preset selection
  const handleSamplingPresetSelect = useCallback((presetName: string) => {
    const preset = voiceSamplingPresets[presetName]
    if (preset) {
      setValues((prev) => ({
        ...prev,
        audio_temperature: preset.audio_temperature,
        audio_top_p: preset.audio_top_p,
        audio_top_k: preset.audio_top_k,
        audio_repetition_penalty: preset.audio_repetition_penalty,
      }))
      toast("info", `Applied sampling preset: ${presetName}`)
    }
  }, [voiceSamplingPresets, toast])

  // Handle voice comparison
  const handleVoiceComparison = useCallback(async () => {
    if (!values.text) {
      toast("error", "Enter sample text for voice comparison")
      return
    }

    setVoiceComparisonMode(true)
    setComparisonResults([])

    // Generate variations with different settings
    const variations = [
      { ...values, instruct: (values.instruct || "") + " - Original style" },
      { ...values, audio_temperature: 1.8, instruct: (values.instruct || "") + " - More expressive" },
      { ...values, audio_temperature: 1.2, instruct: (values.instruct || "") + " - More stable" },
    ]

    const results = []
    for (let i = 0; i < variations.length; i++) {
      try {
        const result = await callTool("voice_creator", variations[i])
        results.push({
          index: i,
          label: `Variation ${i + 1}`,
          params: variations[i],
          data: (result as any).data,
          media_type: (result as any).media_type,
          error: (result as any).error,
        })
      } catch (e) {
        results.push({
          index: i,
          label: `Variation ${i + 1}`,
          params: variations[i],
          error: e instanceof Error ? e.message : String(e),
        })
      }
    }

    setComparisonResults(results)
    toast("success", "Voice comparison complete")
  }, [values, toast])

  // Handle dialogue script generation
  const handleDialogueGenerate = useCallback(async () => {
    if (!dialogueScript.trim()) {
      toast("error", "Enter a dialogue script first")
      return
    }

    // Parse dialogue script (simple format: Speaker: Text)
    const lines = dialogueScript.split("\n").filter((line) => line.trim())
    const requests: any[] = []

    lines.forEach((line) => {
      const match = line.match(/^([^:]+):\s*(.+)$/)
      if (match) {
        const speaker = match[1].trim()
        const text = match[2].trim()
        requests.push({
          text,
          instruct: `Voice for ${speaker}`,
          language: values.language || "English",
          engine: values.engine || "moss_voicegenerator",
          mode: values.mode || "voice_design",
        })
      }
    })

    if (requests.length === 0) {
      toast("error", "No valid dialogue lines found. Use format: Speaker: Text")
      return
    }

    const jobId = nextJobId.current++
    const jobName = `Dialogue (${requests.length} lines)`
    const abortController = new AbortController()
    onAddJob((prev) => [{ id: jobId, name: jobName, status: "running", startedAt: Date.now(), abortController }, ...prev])

    try {
      const result = await callTool<{ status: string; results: any[]; total: number; successful: number; failed: number }>("voice_creator_batch", { requests }, abortController.signal)

      if (result.status === "ok" && result.results) {
        result.results.forEach((r: any, idx: number) => {
          if (r.status === "ok" && r.data) {
            addAsset({
              name: `${lines[idx].split(":")[0].trim()}_${nextAssetName("voice", "wav")}`,
              type: "audio",
              category: "voice",
              mediaType: "audio/wav",
              url: `data:audio/wav;base64,${r.data}`,
              sizeBytes: Math.round((r.data as string).length * 0.75),
              source: "generated",
            })
          }
        })
        toast("success", `Dialogue complete: ${result.successful}/${result.total} lines`)
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "completed", endedAt: Date.now() } : j))
      } else {
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now() } : j))
        toast("error", "Dialogue generation failed")
      }
    } catch (e) {
      // Check if the error is due to abort
      if (e instanceof Error && e.name === 'AbortError') {
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "cancelled", endedAt: Date.now() } : j))
        toast("info", `${jobName} cancelled`)
        return
      }
      onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: e instanceof Error ? e.message : String(e) } : j))
      toast("error", e instanceof Error ? e.message : String(e))
    }
  }, [dialogueScript, values, nextJobId, onAddJob, addAsset, toast])

  const handleEnhance = useCallback(async () => {
    const model = enhanceActiveModel()
    if (!model) {
      toast("error", "No AI model configured. Click the ✨ icon in the header to set one up.")
      return
    }

    // Find prompt and negative_prompt fields
    const promptField = fields.find((f) => f.name === "prompt" || f.name === "text")
    const negField = fields.find((f) => f.name === "negative_prompt")
    const promptVal = String(promptField ? (values[promptField.name] ?? "") : "").trim()

    if (!promptVal) {
      toast("error", "Enter a prompt first, then enhance it.")
      return
    }

    setEnhancing(true)
    try {
      // Ensure the API key is stored securely on the backend
      let keyId = model.keyId

      if (!keyId) {
        // Legacy model without keyId - store it now
        if (!model.apiKey) {
          toast("error", "API key missing. Please re-add your LLM endpoint.")
          return
        }
        toast("info", "Storing API key securely on server...")
        keyId = await storeEnhanceKey(model)

        // Update the model in the store with the keyId
        // This way we don't need to store the API key locally anymore
        const updateModel = useEnhanceStore.getState().updateModel
        updateModel(model.id, { keyId, apiKey: '' }) // Clear apiKey from local storage
      }

      // Enhance positive prompt using the secure backend
      const systemPrompt = getEnhancePrompt(selectedService, values)
      const enhanced = await enhancePrompt(keyId, systemPrompt, promptVal)
      const updates: Record<string, unknown> = {}
      if (promptField) updates[promptField.name] = enhanced

      // Enhance negative prompt if the field exists and the model supports negatives
      if (negField && values[negField.name] !== undefined) {
        const negVal = String(values[negField.name] ?? "").trim()
        if (negVal) {
          try {
            const negSystemPrompt = getEnhancePrompt(selectedService, { ...values, _field: "negative_prompt" })
            const enhancedNeg = await enhancePrompt(keyId, negSystemPrompt, negVal)
            updates[negField.name] = enhancedNeg
          } catch {
            // Negative prompt enhancement is best-effort — don't fail the whole thing
          }
        }
      }

      setValues((p) => ({ ...p, ...updates }))
      toast("success", "Prompt enhanced")
    } catch (e) {
      toast("error", e instanceof Error ? e.message : "Enhancement failed")
    } finally {
      setEnhancing(false)
    }
  }, [enhanceActiveModel, values, fields, toast, selectedService])

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const currentTool = tools.find((t) => t.name === selectedService)
  const currentService = services.find((s) => s.name === selectedService)

  // Reset fields and values ONLY when selectedService actually changes
  // (not when currentTool reference changes from tools list refresh)
  useEffect(() => {
    if (prevServiceRef.current === selectedService) return
    prevServiceRef.current = selectedService

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
        const isBool = v.type === "boolean"
        const isNum = v.type === "number" || v.type === "integer"
        const isLong = ((v.description as string)?.length ?? 0) > 80 || k === "lyrics" || k === "instruct"
        const isImgField = k.includes("image") || k.includes("b64") || k.includes("img") || k.includes("photo")
        extracted.push({
          name: k, label: (v.description as string) || k,
          type: isBool ? "boolean" as const : isImgField ? "file" as const : v.enum ? "select" as const : isNum ? "number" as const : isLong ? "textarea" as const : "text" as const,
          default: v.default, options: v.enum as string[] | undefined,
          required: currentTool.inputSchema.required?.includes(k),
        })
      }
      setFields(extracted)
      // ── Dynamic TTS: inject engine options for tts_speak ─────────────────
      if (selectedService === "tts_speak") {
        const engineField = extracted.find((f) => f.name === "engine")
        if (engineField) {
          engineField.type = "select"
          engineField.options = ["kokoro", "qwen3_tts", "moss_tts", "espeak", "index_tts"]
          engineField.default = "kokoro"
        }
        const modeField = extracted.find((f) => f.name === "mode")
        if (modeField) {
          modeField.type = "select"
          modeField.options = ["custom_voice", "voice_design", "voice_clone"]
          modeField.default = "custom_voice"
        }
      }
      const d: Record<string, unknown> = {}
      for (const f of extracted) { if (f.default !== undefined) d[f.name] = f.default }
      setValues(d)
    }
  }, [selectedService, currentTool, currentService])

  // ── Load model presets and pre-fill defaults ─────────────────────────────────
  useEffect(() => {
    const modelName = String(values.model || "")
    if (!modelName || (selectedService !== "generate" && selectedService !== "generate_image")) return

    callTool<{ sampling_steps?: number; guide_scale?: number; width?: number; height?: number; description?: string }>(
      "get_model_preset",
      { model: modelName }
    ).then((preset) => {
      if (preset && typeof preset === "object") {
        setValues((prev) => ({
          ...prev,
          sampling_steps: preset.sampling_steps ?? prev.sampling_steps,
          guide_scale: preset.guide_scale ?? prev.guide_scale,
          width: preset.width ?? prev.width,
          height: preset.height ?? prev.height,
        }))
      }
    }).catch(() => {
      // Preset fetch failed, continue with defaults
    })
  }, [values.model, selectedService])

  const handleGenerate = async () => {
    if (!currentTool && !currentService) return
    const useTool = currentTool || tools.find((t) => t.name === "run")
    if (!useTool) return

    // Generate multiple items based on quantity
    const totalItems = quantity
    const jobName = SERVICE_LABELS[selectedService] || currentService?.label || selectedService
    const jobId = nextJobId.current++
    const abortController = new AbortController()

    onAddJob((prev) => [{ id: jobId, name: totalItems > 1 ? `${jobName} x${totalItems}` : jobName, status: "running", startedAt: Date.now(), abortController }, ...prev])

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

      let successCount = 0
      let failCount = 0

      // Generate quantity items
      for (let i = 0; i < totalItems; i++) {
        try {
          const result = await callTool<{ status: string; data?: string; media_type?: string; error?: string; message?: string }>(useTool.name, args, abortController.signal)
          if (result.status === "ok" || result.status === "success") {
            if (result.data) {
              const mt = result.media_type || "image/png"
              const isAud = mt.includes("audio")
              const cat = isAud && selectedService.includes("music") ? "music" as const : isAud ? "sfx" as const : "image" as const
              const ext = mt.includes("png") ? "png" : mt.includes("jpeg") || mt.includes("jpg") ? "jpg" : mt.includes("webp") ? "webp" : mt.includes("wav") ? "wav" : mt.includes("mp3") ? "mp3" : mt.split("/")[1] || "bin"

              // Add index suffix if generating multiple
              const itemName = totalItems > 1
                ? `${nextAssetName(selectedService, ext).replace(/\.[^.]+$/, '')}_${i + 1}.${ext}`
                : nextAssetName(selectedService, ext)

              addAsset({
                name: itemName,
                type: isAud ? "audio" : "image", category: cat, mediaType: mt,
                url: `data:${mt};base64,${result.data}`,
                sizeBytes: Math.round((result.data as string).length * 0.75), source: "generated",
              })
              successCount++
            }
          } else {
            failCount++
          }
        } catch (e) {
          // Check if the error is due to abort
          if (e instanceof Error && e.name === 'AbortError') {
            onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "cancelled", endedAt: Date.now() } : j))
            toast("info", `${jobName} cancelled`)
            return
          }
          failCount++
        }
      }

      if (successCount === totalItems) {
        toast("success", `${jobName} generated (${successCount}/${totalItems} successful)`)
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "completed", endedAt: Date.now() } : j))
      } else if (successCount > 0) {
        toast("warning", `${jobName} partially complete (${successCount}/${totalItems} successful, ${failCount} failed)`)
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "completed", endedAt: Date.now() } : j))
      } else {
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: "All items failed" } : j))
        toast("error", `${jobName} failed`)
      }
    } catch (e) {
      // Check if the error is due to abort
      if (e instanceof Error && e.name === 'AbortError') {
        onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "cancelled", endedAt: Date.now() } : j))
        toast("info", `${jobName} cancelled`)
        return
      }
      onAddJob((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "failed", endedAt: Date.now(), error: e instanceof Error ? e.message : String(e) } : j))
      toast("error", e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  const running = jobs.filter((j) => j.status === "running")
  const label = currentService?.label || currentTool?.description?.split("—")[0]?.trim() || selectedService

  // Show loading state while tools/services are being fetched
  const isLoading = tools.length === 0 && services.length === 0

  if (!selectedService) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">Select a service from the right sidebar, or click an asset in the left sidebar</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading services...</p>
        </div>
      </div>
    )
  }

  if (!currentTool && !currentService) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">Service "{selectedService}" not found. Select a different service from the right sidebar.</p>
      </div>
    )
  }

  if (fields.length === 0 && currentTool && currentTool.inputSchema?.properties) {
    // Tool exists but no fields were extracted - shouldn't happen but handle gracefully
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">No form fields available for this service.</p>
      </div>
    )
  }

  return (
    <>
    <div className="flex-1 p-6 flex justify-center">
      <div className="w-full max-w-xl">
        <Card>
          <CardContent className="pt-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">{label}</h2>
            </div>
            {/* Voice Creator Example Selection */}
            {selectedService === "voice_creator" && !loadingVoiceExamples && voiceExamples.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-xs font-medium">Voice Examples</Label>
                  <span className="text-[10px] text-muted-foreground">From vendor demos</span>
                </div>
                <Select
                  value=""
                  onValueChange={(value) => value && handleVoiceExampleSelect(value)}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="Select an example..." />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(voiceExampleCategories).map(([cat, label]) => (
                      <div key={cat} className="space-y-1">
                        <div className="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase">
                          {label}
                        </div>
                        {voiceExamples
                          .filter((ex) => ex.category === cat)
                          .map((ex) => (
                            <SelectItem key={ex.id} value={ex.id} className="text-xs">
                              {ex.instruction.substring(0, 60)}...
                            </SelectItem>
                          ))}
                      </div>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* MOSS Sampling Presets */}
            {selectedService === "voice_creator" && String(values.engine || "moss_voicegenerator") === "moss_voicegenerator" && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-xs font-medium">MOSS Sampling Presets</Label>
                  <span className="text-[10px] text-muted-foreground">Vendor-recommended settings</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {Object.keys(voiceSamplingPresets).map((preset) => (
                    <Button
                      key={preset}
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => handleSamplingPresetSelect(preset)}
                    >
                      {preset}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Voice Advanced Settings Toggle */}
            {selectedService === "voice_creator" && (
              <div className="space-y-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full h-8 text-xs gap-2 justify-start px-2"
                  onClick={() => setVoiceAdvancedOpen(!voiceAdvancedOpen)}
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${voiceAdvancedOpen ? "rotate-180" : ""}`} />
                  Advanced Voice Settings
                </Button>

                {/* Pause Control Help */}
                {String(values.model_variant || "default") === "v1.5" && (
                  <div className="rounded-md border bg-blue-50/50 dark:bg-blue-950/20 p-2 space-y-1">
                    <div className="text-xs font-medium text-blue-700 dark:text-blue-300">⏸️ Pause Control (MOSS v1.5)</div>
                    <div className="text-[10px] text-muted-foreground">
                      Use <code className="bg-background px-1 rounded">[pause X.Xs]</code> in your text for timing control:
                    </div>
                    <div className="space-y-0.5">
                      {pauseControlExamples.slice(0, 2).map((ex, idx) => (
                        <div key={idx} className="text-[9px] text-muted-foreground bg-background px-2 py-0.5 rounded">
                          <span className="font-medium">{ex.name}:</span> {ex.text}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Dialogue Example Loading */}
                {dialogueMode && dialogueExamples.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] text-muted-foreground">Load dialogue example:</div>
                    <div className="flex flex-wrap gap-1">
                      {dialogueExamples.map((ex) => (
                        <Button
                          key={ex.name}
                          variant="outline"
                          size="sm"
                          className="h-6 text-xs"
                          onClick={() => setDialogueScript(ex.script)}
                        >
                          {ex.name}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Voice Special Features */}
                <div className="flex flex-wrap gap-1">
                  <Button
                    variant={batchMode ? "default" : "outline"}
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setBatchMode(!batchMode)}
                  >
                    {batchMode ? "📦 Batch Mode ON" : "📦 Batch Mode"}
                  </Button>
                  <Button
                    variant={voiceComparisonMode ? "default" : "outline"}
                    size="sm"
                    className="h-7 text-xs"
                    onClick={handleVoiceComparison}
                    disabled={!values.text || generating}
                  >
                    🔄 Compare Voices
                  </Button>
                  <Button
                    variant={dialogueMode ? "default" : "outline"}
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setDialogueMode(!dialogueMode)}
                  >
                    💬 Dialogue Mode
                  </Button>
                </div>
              </div>
            )}

            {/* Dialogue Mode UI */}
            {dialogueMode && (
              <div className="space-y-2 rounded-md border bg-muted/50 p-3">
                <div className="text-xs font-medium">Dialogue Script</div>
                <div className="text-[10px] text-muted-foreground">
                  Format: Speaker: Text (one line per speaker)
                </div>
                <Textarea
                  value={dialogueScript}
                  onChange={(e) => setDialogueScript(e.target.value)}
                  placeholder={`Mother: How was your day today?\nChild: It was great! We played games.\nMother: That sounds wonderful!`}
                  rows={6}
                  className="text-xs"
                />
                <Button
                  variant="default"
                  size="sm"
                  className="w-full h-8 text-xs"
                  onClick={handleDialogueGenerate}
                  disabled={!dialogueScript.trim() || generating || running.length > 0}
                >
                  {generating ? "Generating Dialogue..." : "Generate Dialogue"}
                </Button>
              </div>
            )}

            {/* Voice Comparison Results */}
            {voiceComparisonMode && comparisonResults.length > 0 && (
              <div className="space-y-2 rounded-md border bg-muted/50 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">Voice Comparison Results</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs"
                    onClick={() => { setVoiceComparisonMode(false); setComparisonResults([]) }}
                  >
                    Close
                  </Button>
                </div>
                <div className="space-y-2">
                  {comparisonResults.map((result) => (
                    <div key={result.index} className="space-y-1 bg-background p-2 rounded">
                      <div className="text-xs font-medium">{result.label}</div>
                      {result.error ? (
                        <div className="text-[10px] text-destructive">{result.error}</div>
                      ) : (
                        <>
                          {true && (
                            <AudioWaveform
                              audioUrl={`data:${result.media_type};base64,${result.data}`}
                              height={40}
                              color="#3b82f6"
                            />
                          )}
                          <audio
                            src={`data:${result.media_type};base64,${result.data}`}
                            controls
                            className="w-full h-8 text-xs"
                          />
                          <div className="text-[9px] text-muted-foreground">
                            Temp: {result.params.audio_temperature}, Top-P: {result.params.audio_top_p}
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 text-xs"
                            onClick={() => {
                              addAsset({
                                name: `voice_comparison_${result.index + 1}.wav`,
                                type: "audio",
                                category: "voice",
                                mediaType: "audio/wav",
                                url: `data:audio/wav;base64,${result.data}`,
                                sizeBytes: Math.round((result.data as string).length * 0.75),
                                source: "generated",
                              })
                              toast("success", "Saved to assets")
                            }}
                          >
                            Save to Assets
                          </Button>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Compact Form Layout */}
            <div className="space-y-4">
              {/* ── Prompt/Text Section (Always Top Priority) ──────────────────────────── */}
              {fields.filter(f => f.name === "prompt" || f.name === "text").map(f => (
                <div key={f.name} className="space-y-2">
                  <FieldLabel
                    label={f.label}
                    tooltip={f.name === "prompt" ? "60-200 words works best for detailed results" : "Enter text content"}
                  />
                  <Textarea
                    value={String(values[f.name] ?? f.default ?? "")}
                    onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                    placeholder={f.label}
                    rows={3}
                    className="text-sm"
                  />
                </div>
              ))}

              {/* ── Basic Configuration Grid ────────────────────────────────────────────── */}
              {(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) && (
                <div className="grid grid-cols-2 gap-4">
                  {/* Model Selection */}
                  {fields.filter(f => f.name === "model").map(f => (
                    <div key={f.name} className="space-y-2">
                      <FieldLabel label={f.label} tooltip="Select the AI model to use" />
                      <Select value={String(values[f.name] ?? f.default ?? "")}
                        onValueChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}>
                        <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {f.options?.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    </div>
                  ))}

                  {/* Seed */}
                  {fields.filter(f => f.name === "seed").map(f => (
                    <div key={f.name} className="space-y-2">
                      <FieldLabel label={f.label} tooltip="Random seed for reproducibility (-1 for random)" />
                      <Input
                        type="number"
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                        placeholder={f.label}
                        className="h-9"
                      />
                    </div>
                  ))}
                </div>
              )}

              {/* ── Dimensions Grid ─────────────────────────────────────────────────────── */}
              {(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) && (
                <div className="grid grid-cols-2 gap-4">
                  {/* Width */}
                  {fields.filter(f => f.name === "width").map(f => (
                    <div key={f.name} className="space-y-2">
                      <FieldLabel label={f.label} tooltip="Must be divisible by 16" />
                      <Input
                        type="number"
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                        placeholder={f.label}
                        className="h-9"
                      />
                    </div>
                  ))}

                  {/* Height */}
                  {fields.filter(f => f.name === "height").map(f => (
                    <div key={f.name} className="space-y-2">
                      <FieldLabel label={f.label} tooltip="Must be divisible by 16" />
                      <Input
                        type="number"
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                        placeholder={f.label}
                        className="h-9"
                      />
                    </div>
                  ))}
                </div>
              )}

              {/* ── Advanced Settings Accordion ─────────────────────────────────────────── */}
              {(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) && (
                <Accordion type="single" collapsible className="border rounded-lg px-3">
                  <AccordionItem value="advanced-settings" className="border-b-0">
                    <AccordionTrigger className="py-3 text-sm font-medium hover:no-underline">
                      Advanced Settings
                    </AccordionTrigger>
                    <AccordionContent className="space-y-4 pb-4">
                      <div className="grid grid-cols-2 gap-4">
                        {/* Sampling Steps */}
                        {fields.filter(f => f.name === "sampling_steps" || f.name === "steps").map(f => (
                          <div key={f.name} className="space-y-2">
                            <FieldLabel label={f.label} tooltip="Number of denoising steps (higher = better quality)" />
                            <Input
                              type="number"
                              value={String(values[f.name] ?? f.default ?? "")}
                              onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                              placeholder={f.label}
                              className="h-9"
                            />
                          </div>
                        ))}

                        {/* Guidance Scale */}
                        {fields.filter(f => f.name === "guide_scale" || f.name === "guidance").map(f => (
                          <div key={f.name} className="space-y-2">
                            <FieldLabel label={f.label} tooltip="Guidance scale for prompt adherence (typically 4-8)" />
                            <Input
                              type="number"
                              step="0.1"
                              value={String(values[f.name] ?? f.default ?? "")}
                              onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                              placeholder={f.label}
                              className="h-9"
                            />
                          </div>
                        ))}

                        {/* Sampler (for image generation) */}
                        {(selectedService === "generate" || selectedService === "generate_image") && (
                          <div className="space-y-2">
                            <FieldLabel label="Sampler" tooltip="Sampler algorithm for denoising" />
                            <Select
                              value={String(values.sampler ?? "")}
                              onValueChange={(v) => setValues((p) => ({ ...p, sampler: v }))}
                            >
                              <SelectTrigger className="h-9">
                                <SelectValue placeholder="Default sampler" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="">Default</SelectItem>
                                <SelectItem value="er_sde">ER SDE (default, neutral)</SelectItem>
                                <SelectItem value="euler_a">Euler A (softer)</SelectItem>
                                <SelectItem value="dpmpp_2m_sde_gpu">DPM++ 2M SDE GPU (creative)</SelectItem>
                                <SelectItem value="dpmpp_2m">DPM++ 2M (fast)</SelectItem>
                                <SelectItem value="dpmpp_sde">DPM++ SDE (balanced)</SelectItem>
                                <SelectItem value="ddim">DDIM (classic)</SelectItem>
                                <SelectItem value="uni_pc">UniPC (unified)</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        )}
                      </div>

                      {/* Negative Prompt */}
                      {fields.filter(f => f.name === "negative_prompt").map(f => (
                        <div key={f.name} className="space-y-2">
                          <FieldLabel label={f.label} tooltip="Describe what to avoid in the output" />
                          <Textarea
                            value={String(values[f.name] ?? f.default ?? "")}
                            onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                            placeholder={f.label}
                            rows={2}
                            className="text-sm"
                          />
                        </div>
                      ))}

                      {/* Additional Advanced Fields */}
                      {(selectedService === "tts_speak"
                        ? ttsVisibleFields(String(values.engine || "kokoro"), fields)
                        : selectedService === "voice_creator"
                        ? voiceCreatorVisibleFields(
                            String(values.engine || "moss_voicegenerator"),
                            String(values.mode || "voice_design"),
                            fields
                          )
                        : fields
                      ).filter(f =>
                        !["prompt", "text", "model", "seed", "width", "height", "sampling_steps", "steps", "guide_scale", "guidance", "negative_prompt", "loras_selected"].includes(f.name) &&
                        f.type !== "file" &&
                        f.type !== "boolean"
                      ).map(f => (
                        <div key={f.name} className="space-y-2">
                          <FieldLabel label={f.label} />
                          {f.type === "select" && f.options ? (
                            <Select value={String(values[f.name] ?? f.default ?? "")}
                              onValueChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}>
                              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                              <SelectContent>
                                {f.options.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                              </SelectContent>
                            </Select>
                          ) : f.type === "number" ? (
                            <Input
                              type="number"
                              value={String(values[f.name] ?? f.default ?? "")}
                              onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                              placeholder={f.label}
                              className="h-9"
                            />
                          ) : f.type === "textarea" ? (
                            <Textarea
                              value={String(values[f.name] ?? f.default ?? "")}
                              onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                              placeholder={f.label}
                              rows={2}
                              className="text-sm"
                            />
                          ) : (
                            <Input
                              value={String(values[f.name] ?? f.default ?? "")}
                              onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                              placeholder={f.label}
                              className="h-9"
                            />
                          )}
                        </div>
                      ))}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              )}

              {/* ── File Upload Fields (Outside Accordion) ──────────────────────────────── */}
              {fields.filter(f => f.type === "file" || f.name.includes("image") || f.name.includes("b64")).map(f => {
                const handleDrop = (e: React.DragEvent) => {
                  e.preventDefault()
                  try {
                    const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                    if (d.url) setValues((p) => ({ ...p, [f.name]: d.url.split(",")[1] || d.url }))
                  } catch {}
                }

                return (
                  <div key={f.name} className="space-y-2" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
                    <div className="flex items-center gap-1">
                      <FieldLabel label={f.label} />
                      {selectedService === "edit" && f.name === "pose_image_b64" && (
                        <button type="button"
                          onClick={onOpenKimodo}
                          className="inline-flex items-center gap-1 text-[10px] text-primary hover:underline ml-auto">
                          <Maximize2 className="h-3 w-3" /> Open Kimodo Studio
                        </button>
                      )}
                    </div>

                    {f.name === "ref_audio_b64_list" ? (
                      <div className="space-y-2">
                        <Label className="flex items-center justify-center h-16 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/50 text-sm text-muted-foreground"
                          onDragOver={(e) => e.preventDefault()}
                          onDrop={(e) => {
                            e.preventDefault()
                            try {
                              const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                              if (d.url) {
                                setMultipleRefAudios((prev) => [...prev, d.url.split(",")[1] || d.url])
                                setValues((p) => ({ ...p, [f.name]: [...(p[f.name] as string[] || []), d.url.split(",")[1] || d.url] }))
                              }
                            } catch {}
                          }}>
                          {multipleRefAudios.length > 0 ? `Loaded ${multipleRefAudios.length} audio files` : "Drop multiple audio files or click to upload"}
                          <input type="file" className="hidden" multiple accept="audio/*"
                            onChange={(e) => {
                              const files = Array.from(e.target.files || [])
                              files.forEach((file) => {
                                const r = new FileReader()
                                r.onload = () => {
                                  const base64 = (r.result as string).split(",")[1] || ""
                                  setMultipleRefAudios((prev) => [...prev, base64])
                                  setValues((p) => ({ ...p, [f.name]: [...(p[f.name] as string[] || []), base64] }))
                                }
                                r.readAsDataURL(file)
                              })
                            }} />
                        </Label>
                        {multipleRefAudios.length > 0 && (
                          <div className="max-h-24 overflow-y-auto space-y-1">
                            {multipleRefAudios.map((audio, idx) => (
                              <div key={idx} className="flex items-center gap-2 text-xs bg-background px-2 py-1 rounded">
                                <span className="text-muted-foreground">{idx + 1}.</span>
                                <audio src={`data:audio/wav;base64,${audio}`} controls className="flex-1 h-6" />
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-5 w-5"
                                  onClick={() => {
                                    setMultipleRefAudios((prev) => prev.filter((_, i) => i !== idx))
                                    setValues((p) => ({ ...p, [f.name]: (p[f.name] as string[] || []).filter((_, i) => i !== idx) }))
                                  }}
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <Label className="flex items-center justify-center h-16 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/50 text-sm text-muted-foreground"
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}>
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
                    )}
                  </div>
                )
              })}

              {/* ── Boolean Fields (Switch instead of checkbox) ──────────────────────────── */}
              {fields.filter(f => f.type === "boolean").map(f => (
                <div key={f.name} className="flex items-center justify-between">
                  <FieldLabel label={f.label} />
                  <Switch
                    checked={!!values[f.name]}
                    onCheckedChange={(checked) => setValues((p) => ({ ...p, [f.name]: checked }))}
                  />
                </div>
              ))}

              {/* ── Remaining Text Fields (for non-image services) ──────────────────────────── */}
              {!(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) &&
                (selectedService === "tts_speak"
                  ? ttsVisibleFields(String(values.engine || "kokoro"), fields)
                  : selectedService === "voice_creator"
                  ? voiceCreatorVisibleFields(
                      String(values.engine || "moss_voicegenerator"),
                      String(values.mode || "voice_design"),
                      fields
                    ).filter((f) => voiceAdvancedOpen || ![
                      "max_new_tokens", "audio_temperature", "audio_top_p",
                      "audio_top_k", "audio_repetition_penalty", "seed"
                    ].includes(f.name))
                  : fields
                ).filter(f =>
                  !["prompt", "text", "model", "seed", "width", "height", "sampling_steps", "steps", "guide_scale", "guidance", "negative_prompt"].includes(f.name) &&
                  f.name !== "loras_selected" &&
                  f.type !== "file" &&
                  f.type !== "boolean"
                ).map(f => (
                  <div key={f.name} className="space-y-2">
                    <FieldLabel label={f.label} />
                    {f.type === "select" && f.options ? (
                      <Select value={String(values[f.name] ?? f.default ?? "")}
                        onValueChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}>
                        <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {f.options.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    ) : f.type === "number" ? (
                      <Input
                        type="number"
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value ? Number(e.target.value) : "" }))}
                        placeholder={f.label}
                        className="h-9"
                      />
                    ) : f.type === "textarea" ? (
                      <Textarea
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                        placeholder={f.label}
                        rows={2}
                        className="text-sm"
                      />
                    ) : (
                      <Input
                        value={String(values[f.name] ?? f.default ?? "")}
                        onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.value }))}
                        placeholder={f.label}
                        className="h-9"
                      />
                    )}
                  </div>
                ))
              }
            </div>

            {/* ── Advanced Settings Guide ──────────────────────────────────────── */}
            {(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) && (
              <div className="space-y-3 pt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full h-8 text-xs gap-2 justify-start px-2"
                  onClick={() => setAdvancedOpen(!advancedOpen)}
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
                  Advanced Settings Guide
                </Button>
                {advancedOpen && (
                  <div className="rounded-lg bg-muted/50 p-3 space-y-2 text-[11px]">
                    {String(values.model || "") === "anima_base" ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Anima Base Recommended Settings</div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          <span className="text-muted-foreground">Steps:</span>
                          <span>30-50 (default: {values.sampling_steps ?? "30"})</span>
                          <span className="text-muted-foreground">CFG Scale:</span>
                          <span>4-5 (default: {values.guide_scale ?? "4.0"})</span>
                          <span className="text-muted-foreground">Resolution:</span>
                          <span>512² - 1536² px</span>
                        </div>
                        <div className="pt-1">
                          <span className="text-muted-foreground">Sampler options:</span>
                          <ul className="mt-1 space-y-0.5 text-muted-foreground">
                            <li>• <span className="text-foreground">er_sde</span> — neutral, flat colors, sharp lines (default)</li>
                            <li>• <span className="text-foreground">euler_a</span> — softer, thinner lines</li>
                            <li>• <span className="text-foreground">dpmpp_2m_sde_gpu</span> — more creative variety</li>
                          </ul>
                        </div>
                      </>
                    ) : String(values.model || "") === "z_image" ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Z-Image Turbo Recommended Settings</div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          <span className="text-muted-foreground">Steps:</span>
                          <span>8 (distilled, no CFG)</span>
                          <span className="text-muted-foreground">CFG Scale:</span>
                          <span>0.0 (disabled for turbo)</span>
                          <span className="text-muted-foreground">Quality:</span>
                          <span>Turbo (fastest)</span>
                        </div>
                        <div className="pt-1 text-[10px] text-muted-foreground">
                          Best for photorealism and speed. Use Z-Image Base for creative work with full control.
                        </div>
                      </>
                    ) : String(values.model || "") === "z_image_base" ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Z-Image Base Recommended Settings</div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          <span className="text-muted-foreground">Steps:</span>
                          <span>50 (default: {values.sampling_steps ?? "50"})</span>
                          <span className="text-muted-foreground">CFG Scale:</span>
                          <span>4.0 (default: {values.guide_scale ?? "4.0"})</span>
                          <span className="text-muted-foreground">Negative Prompt:</span>
                          <span>Active</span>
                        </div>
                        <div className="pt-1 text-[10px] text-muted-foreground">
                          Full model with maximum control. Best for fine-tuning and creative diversity.
                        </div>
                      </>
                    ) : String(values.model || "")?.startsWith("flux") ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Flux Recommended Settings</div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          <span className="text-muted-foreground">Resolution:</span>
                          <span>{values.width ?? "1280"}×{values.height ?? "720"}</span>
                          <span className="text-muted-foreground">Steps:</span>
                          <span>{String(values.model || "")?.includes("schnell") || String(values.model || "")?.includes("klein") ? "4" : String(values.model || "")?.startsWith("flux2") ? "30" : "20"}</span>
                        </div>
                        <div className="pt-1 text-[10px] text-muted-foreground">
                          {String(values.model || "")?.includes("schnell") || String(values.model || "")?.includes("klein")
                            ? "Distilled for speed. Natural language prompts work best."
                            : "Full model with high quality. Natural language prompts work best."}
                        </div>
                      </>
                    ) : (
                      <div className="text-muted-foreground text-[10px]">
                        Model-specific settings will appear here when you select a model.
                      </div>
                    )}

                    {/* ── LoRA Picker ──────────────────────────────────────────────────────── */}
                    {fields.filter(f => f.name === "loras_selected").map(f => (
                      <div key={f.name} className="pt-2 border-t">
                        <div className="space-y-2">
                          <div className="font-semibold text-xs">LoRA Enhancement</div>
                          <div className="text-[10px] text-muted-foreground mb-2">
                            Select LoRA models to enhance generation quality
                          </div>
                          <LoraPicker
                            model={String(values.model || "")}
                            value={String(values[f.name] ?? "")}
                            onChange={(v) => setValues((p) => ({ ...p, [f.name]: v }))}
                            variant="light"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── Prompting Guide ─────────────────────────────────────────────────────── */}
            {(selectedService === "generate" || selectedService === "generate_image" || selectedService?.includes("generate")) && (
              <div className="space-y-3 pt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full h-8 text-xs gap-2 justify-start px-2"
                  onClick={() => setPromptingOpen(!promptingOpen)}
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${promptingOpen ? "rotate-180" : ""}`} />
                  Prompting Guide
                </Button>
                {promptingOpen && (
                  <div className="rounded-lg bg-muted/50 p-3 space-y-3 text-[11px]">
                    {String(values.model || "") === "anima_base" ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Anima Prompting Guide</div>
                        <div className="space-y-2">
                          <div>
                            <span className="text-muted-foreground">Format:</span>
                            <span className="ml-2">Danbooru-style tags (NOT prose)</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Prefix:</span>
                            <span className="ml-2 font-mono text-[10px]">masterpiece, best quality, score_7, safe,</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Structure:</span>
                            <span className="ml-2">[quality] [subject] [character] [series] [artist] [tags]</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">What to include:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Use tags with spaces (not underscores): <code className="bg-background px-1 rounded">long hair</code> not <code className="bg-background px-1 rounded">long_hair</code></li>
                              <li>• Artist tags with <code className="bg-background px-1 rounded">@</code>: <code className="bg-background px-1 rounded">@wlop</code> for style</li>
                              <li>• Quality tags: <code className="bg-background px-1 rounded">masterpiece, best quality, score_9</code></li>
                              <li>• Time period: <code className="bg-background px-1 rounded">year 2025, newest</code> for latest styles</li>
                            </ul>
                          </div>
                          <div className="pt-1 border-t">
                            <span className="text-muted-foreground">Example:</span>
                            <div className="mt-1 text-[10px] font-mono bg-background p-2 rounded">
                              masterpiece, best quality, score_7, safe, year 2025, highres, 1girl, brown hair, smile, looking at viewer, white background, @artist_name
                            </div>
                          </div>
                        </div>
                      </>
                    ) : String(values.model || "").startsWith("z_image") ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Z-Image Prompting Guide</div>
                        <div className="space-y-2">
                          <div>
                            <span className="text-muted-foreground">Format:</span>
                            <span className="ml-2">Continuous descriptive prose (NOT tag lists)</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Length:</span>
                            <span className="ml-2">{String(values.model || "") === "z_image_base" ? "80-200 words" : "80-200 words"}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Structure (order matters):</span>
                            <ol className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>1. Subject identity and action</li>
                              <li>2. Physical description (face, hair, clothing, textures)</li>
                              <li>3. Hand/object interactions and spatial details</li>
                              <li>4. Background/environment with depth layers</li>
                              <li>5. Lighting, atmosphere, color palette</li>
                              <li>6. Camera/framing (for photorealism)</li>
                            </ol>
                          </div>
                          <div>
                            <span className="text-muted-foreground">What to include:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Specific demographics: <code className="bg-background px-1 rounded">young Chinese woman</code> not <code className="bg-background px-1 rounded">woman</code></li>
                              <li>• Specific garments: <code className="bg-background px-1 rounded">cream-colored wool turtleneck</code></li>
                              <li>• Fabric behavior: <code className="bg-background px-1 rounded">slightly wrinkled linen</code></li>
                              <li>• Spatial positions: <code className="bg-background px-1 rounded">on the left</code>, <code className="bg-background px-1 rounded">in the foreground</code></li>
                              <li>• Lighting: <code className="bg-background px-1 rounded">warm golden-hour sunlight from camera left</code></li>
                            </ul>
                          </div>
                          <div className="pt-1 border-t">
                            <span className="text-muted-foreground">What to avoid:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Meta-tags: <code className="bg-background px-1 rounded">8K, masterpiece, trending on artstation</code></li>
                              <li>• Other-model syntax: <code className="bg-background px-1 rounded">score_9, plms, euler a</code></li>
                              <li>• Negation: <code className="bg-background px-1 rounded">no hat</code> → use <code className="bg-background px-1 rounded">bareheaded</code></li>
                            </ul>
                          </div>
                          {String(values.model || "") === "z_image_base" && (
                            <div className="pt-1 border-t">
                              <span className="text-muted-foreground">Negative prompts:</span>
                              <span className="ml-2">Active - use for quality issues and unwanted features</span>
                            </div>
                          )}
                        </div>
                      </>
                    ) : String(values.model || "").startsWith("flux") ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Flux Prompting Guide</div>
                        <div className="space-y-2">
                          <div>
                            <span className="text-muted-foreground">Format:</span>
                            <span className="ml-2">Natural language descriptive prose</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Length:</span>
                            <span className="ml-2">{String(values.model || "").includes("schnell") || String(values.model || "").includes("klein") ? "40-100 words" : "60-150 words"}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">What to describe:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Subject details (age, appearance, expression)</li>
                              <li>• Environment and setting</li>
                              <li>• Lighting and mood</li>
                              <li>• Composition and camera angle</li>
                              <li>• Color palette and style</li>
                            </ul>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Tips:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Use flowing prose with vivid descriptions</li>
                              <li>• Include one clear style signal: <code className="bg-background px-1 rounded">photograph</code>, <code className="bg-background px-1 rounded">digital illustration</code>, <code className="bg-background px-1 rounded">oil painting</code></li>
                              <li>• For photorealism: <code className="bg-background px-1 rounded">85mm lens, shallow DOF</code></li>
                            </ul>
                          </div>
                          {!String(values.model || "").includes("schnell") && !String(values.model || "").includes("klein") && (
                            <div className="pt-1 border-t">
                              <span className="text-muted-foreground">Negative prompts:</span>
                              <span className="ml-2">Effective - use <code className="bg-background px-1 rounded">blurry, low quality, deformed, bad anatomy</code></span>
                            </div>
                          )}
                        </div>
                      </>
                    ) : String(values.model || "").startsWith("qwen") ? (
                      <>
                        <div className="font-semibold text-xs mb-2">Qwen Image Prompting Guide</div>
                        <div className="space-y-2">
                          <div>
                            <span className="text-muted-foreground">Strength:</span>
                            <span className="ml-2">Excellent at text rendering in images</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Length:</span>
                            <span className="ml-2">60-150 words</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Text in images:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Wrap text in double quotes: <code className="bg-background px-1 rounded">poster reading "SALE 50% OFF"</code></li>
                              <li>• Chinese text rendering is especially strong</li>
                              <li>• For signs/posters: describe content, font style, and layout</li>
                            </ul>
                          </div>
                          <div>
                            <span className="text-muted-foreground">What to describe:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Subject and scene</li>
                              <li>• Any visible text content and layout</li>
                              <li>• Style and atmosphere</li>
                            </ul>
                          </div>
                        </div>
                      </>
                    ) : String(values.model || "").startsWith("hidream") ? (
                      <>
                        <div className="font-semibold text-xs mb-2">HiDream O1 Prompting Guide</div>
                        <div className="space-y-2">
                          <div>
                            <span className="text-muted-foreground">Architecture:</span>
                            <span className="ml-2">Unified text+pixel token space - very responsive to detail</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Length:</span>
                            <span className="ml-2">60-150 words</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Focus on:</span>
                            <ul className="mt-1 space-y-0.5 text-muted-foreground">
                              <li>• Visual qualities: textures, materials, reflections, transparency</li>
                              <li>• Subject, scene, lighting, atmosphere</li>
                              <li>• Color palette and composition</li>
                              <li>• One style signal: <code className="bg-background px-1 rounded">photograph</code>, <code className="bg-background px-1 rounded">digital painting</code></li>
                            </ul>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-muted-foreground text-[10px]">
                        Model-specific prompting guide will appear here when you select a model.
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {fields.length > 0 && (
              <div className="flex items-center gap-2">
                {/* Enhance prompt button */}
                {fields.some((f) => f.name === "prompt" || f.name === "text") && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 text-xs gap-2"
                    disabled={enhancing || generating || !String(values[fields.find((f) => f.name === "prompt" || f.name === "text")?.name ?? ""] ?? "").trim()}
                    onClick={handleEnhance}
                  >
                    {enhancing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    {enhancing ? "Enhancing…" : "Enhance"}
                  </Button>
                )}

                {/* Generate button with quantity */}
                <div className="flex items-center gap-0 flex-1">
                  <Button
                    className="rounded-r-none flex-1"
                    disabled={generating || running.length > 0}
                    onClick={handleGenerate}
                  >
                    {generating ? "Generating..." : `Generate${quantity > 1 ? ` ${quantity}` : ""}`}
                  </Button>

                  {/* Quantity selector */}
                  <div className="flex items-center border rounded-l-none rounded-r-md">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-none"
                      onClick={() => setQuantity(Math.max(1, quantity - 1))}
                      disabled={quantity <= 1 || generating}
                    >
                      -
                    </Button>
                    <Input
                      type="number"
                      value={quantity}
                      onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                      className="h-8 w-12 text-center text-sm border-0 rounded-none focus-visible:z-10 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none [-moz-appearance:textfield]"
                      disabled={generating}
                      min={1}
                      max={10}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-none"
                      onClick={() => setQuantity(Math.min(10, quantity + 1))}
                      disabled={quantity >= 10 || generating}
                    >
                      +
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
    </>
  )
}


