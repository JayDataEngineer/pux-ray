import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { useToastStore } from "@/stores/toast"
import { useAssetStore, type Asset, nextAssetName } from "@/stores/assets"
import { useEnhanceStore } from "@/stores/enhancement"
import { enhancePrompt } from "@/lib/enhance"
import { getEnhancePrompt } from "@/lib/enhance-prompts"
import { EnhanceConfigDialog } from "@/components/EnhanceConfigDialog"
import { kimodoUrl } from "@/mcp"
import { callTool, forgeStatus, listTools, type MCPTool } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, PanelRightClose, Wand2, Loader2, CheckCircle2, XCircle, Clock, ListTodo, X, Maximize2, ChevronLeft, ChevronRight, Download, Sparkles } from "lucide-react"
import { AppSidebar } from "./AppSidebar"
import { VideoEditor } from "./VideoEditor"

type TabId = "assets" | "video"

interface JobEntry {
  id: number; name: string
  status: "pending" | "running" | "completed" | "failed"
  startedAt: number; endedAt?: number; error?: string
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

const GENRE_ORDER = ["image", "audio"]

const GENRE_ICONS: Record<string, string> = { image: "◎", audio: "♪" }

// Map backend category → genre for sidebar grouping
const CATEGORY_TO_GENRE: Record<string, string> = {
  audio: "audio", motion: "motion", "3d": "3d",
}

const SERVICE_GENRE: Record<string, string> = {
  generate: "image", edit: "image", generate_character_sheet: "image",
  generate_image: "image", pose_edit: "image", char_sheet: "image",
  generate_music: "audio", ace_step: "audio",
  generate_sound: "audio", moss_soundeffect: "audio",
  tts_speak: "audio",
  voice_creator: "audio",
  kimodo: "motion", kimodo_demo: "motion", hy_motion: "motion", gemx: "motion",
  trellis: "3d", anigen: "3d", body_mesh: "3d",
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
  trellis: "TRELLIS 3D",
  anigen: "AniGen 3D",
  body_mesh: "BodyMesh",
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
  const nextJobId = useRef(1)

  const enhanceActiveModel = useEnhanceStore((s) => s.activeModel)
  const hasEnhanceModel = !!enhanceActiveModel()

  return (
    <div className="flex h-screen w-full bg-background">
      <AppSidebar open={leftOpen} onToggle={() => setLeftOpen((o) => !o)} onSelectAsset={(a) => setSelectedAsset(a)} />
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
          <Button variant="ghost" size="icon" className="h-7 w-7 relative" onClick={() => setEnhanceConfigOpen(true)}
            title="AI Prompt Enhancement">
            <Sparkles className={`h-4 w-4 ${hasEnhanceModel ? "text-primary" : "text-muted-foreground"}`} />
          </Button>
          <GpuStatus />
          <JobsButton jobs={jobs} />
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setRightOpen((o) => !o)}>
            <PanelRightClose className="h-4 w-4" />
          </Button>
        </header>
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 overflow-auto scrollbar-thin">
            {tab === "assets"
              ? <AssetsTab selectedService={selectedService} jobs={jobs} onAddJob={(j) => setJobs(j)} nextJobId={nextJobId} onOpenKimodo={() => setKimodoOpen(true)} />
              : <VideoEditor />
            }
          </div>
          {rightOpen && <ServicesSidebar selected={selectedService} onSelect={setSelectedService} onOpenKimodo={() => setKimodoOpen(true)} />}
        </div>
      </div>
      <AssetPreviewDialog asset={selectedAsset} onClose={() => setSelectedAsset(null)} onSelect={(a) => setSelectedAsset(a)} />
      <KimodoDialog open={kimodoOpen} onOpenChange={setKimodoOpen} />
      <EnhanceConfigDialog open={enhanceConfigOpen} onOpenChange={setEnhanceConfigOpen} />
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

function ServicesSidebar({ selected, onSelect, onOpenKimodo }: { selected: string; onSelect: (n: string) => void; onOpenKimodo: () => void }) {
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])

  useEffect(() => {
    listTools().then(setTools).catch(() => {})
    fetch("/v1/services").then((r) => r.json()).then(setServices).catch(() => {})
  }, [])

  const allItems = [
    ...tools.filter((t) => !["run","list_models","list_services","get_service","forge_status","load_service","unload_services","tts_voices","chat","transcribe","llm_configure"].includes(t.name) && !t.name.startsWith("workflow_")),
    ...services.filter((s) => !tools.find((t) => t.name === s.name) && !HIDDEN_SERVICES.has(s.name)),
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
              return (
                <button key={name} onClick={() => onSelect(name)}
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left transition-colors ${selected === name ? "bg-sidebar-accent text-sidebar-accent-foreground" : "hover:bg-sidebar-accent/50"}`}>
                  <Wand2 className="h-3 w-3 shrink-0 opacity-50" />
                  <span className="truncate">{label}</span>
                </button>
              )
            })}
          </div>
        ))}
      </div>
      <div className="p-2 border-t">
        <button
          onClick={onOpenKimodo}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left transition-colors hover:bg-sidebar-accent/50 text-sidebar-foreground/60"
        >
          <span className="text-sm">↝</span>
          <span className="truncate">Kimodo Motion Studio</span>
        </button>
      </div>
    </div>
  )
}

// ── Kimodo Picture-in-Picture Dialog ──────────────────────────────────────────

function KimodoDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [loading, setLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)

  // Trigger preload when dialog opens programmatically (from sidebar link)
  useEffect(() => {
    if (!open || loaded || loading) return
    let cancelled = false
    setLoading(true)
    fetch('/forge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'preload', service: 'kimodo_demo' }),
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load Kimodo')
        if (!cancelled) setLoaded(true)
      })
      .catch((e) => {
        console.error('Kimodo load failed:', e)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [open, loaded, loading])

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
              <span className="text-[10px] text-muted-foreground/60">This takes ~60s on first load</span>
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

function AssetsTab({ selectedService, jobs, onAddJob, nextJobId, onOpenKimodo }: {
  selectedService: string
  jobs: JobEntry[]
  onAddJob: (j: JobEntry[] | ((prev: JobEntry[]) => JobEntry[])) => void
  nextJobId: React.MutableRefObject<number>
  onOpenKimodo: () => void
}) {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [tools, setTools] = useState<MCPTool[]>([])
  const [services, setServices] = useState<{ name: string; label: string; category: string }[]>([])
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [generating, setGenerating] = useState(false)
  const [fields, setFields] = useState<FieldDef[]>([])
  const [enhancing, setEnhancing] = useState(false)
  const prevServiceRef = useRef("")

  const enhanceActiveModel = useEnhanceStore((s) => s.activeModel)

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
      // Enhance positive prompt
      const systemPrompt = getEnhancePrompt(selectedService, values)
      const enhanced = await enhancePrompt(model, systemPrompt, promptVal)
      const updates: Record<string, unknown> = {}
      if (promptField) updates[promptField.name] = enhanced

      // Enhance negative prompt if the field exists and the model supports negatives
      if (negField && values[negField.name] !== undefined) {
        const negVal = String(values[negField.name] ?? "").trim()
        if (negVal) {
          try {
            const negSystemPrompt = getEnhancePrompt(selectedService, { ...values, _field: "negative_prompt" })
            const enhancedNeg = await enhancePrompt(model, negSystemPrompt, negVal)
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
  }, [selectedService])

  const handleGenerate = async () => {
    if (!currentTool && !currentService) return
    const useTool = currentTool || tools.find((t) => t.name === "run")
    if (!useTool) return

    const jobId = nextJobId.current++
    const jobName = SERVICE_LABELS[selectedService] || currentService?.label || selectedService
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
          const ext = mt.includes("png") ? "png" : mt.includes("jpeg") || mt.includes("jpg") ? "jpg" : mt.includes("webp") ? "webp" : mt.includes("wav") ? "wav" : mt.includes("mp3") ? "mp3" : mt.split("/")[1] || "bin"
          addAsset({
            name: nextAssetName(selectedService, ext),
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

  if (!selectedService) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">Select a service from the right sidebar, or click an asset in the left sidebar</p>
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
              {selectedService && <Badge variant="outline" className="text-[10px]">{selectedService}</Badge>}
            </div>
            {fields.some((f) => f.name === "prompt" || f.name === "text") && (
              <Button variant="outline" size="sm" className="w-full h-8 text-xs gap-2"
                disabled={enhancing || generating || !String(values[fields.find((f) => f.name === "prompt" || f.name === "text")?.name ?? ""] ?? "").trim()}
                onClick={handleEnhance}>
                {enhancing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                {enhancing ? "Enhancing…" : "Enhance Prompt"}
              </Button>
            )}
            {(selectedService === "tts_speak"
              ? ttsVisibleFields(String(values.engine || "kokoro"), fields)
              : fields
            ).map((f) => {
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
                <div className="flex items-center gap-1">
                  <Label className="text-xs text-muted-foreground capitalize">{f.label}</Label>
                  {selectedService === "edit" && f.name === "pose_image_b64" && (
                    <button type="button"
                      onClick={onOpenKimodo}
                      className="inline-flex items-center gap-1 text-[10px] text-primary hover:underline ml-auto">
                      <Maximize2 className="h-3 w-3" /> Open Kimodo Studio
                    </button>
                  )}
                </div>
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
                ) : f.type === "boolean" ? (
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={!!values[f.name]}
                      onChange={(e) => setValues((p) => ({ ...p, [f.name]: e.target.checked }))}
                      className="h-4 w-4 rounded border-border text-primary focus:ring-primary" />
                    <span className="text-xs text-muted-foreground">{values[f.name] ? "Yes" : "No"}</span>
                  </label>
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
    </>
  )
}


