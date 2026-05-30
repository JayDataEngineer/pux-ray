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
import { useAssetStore, type AssetCategory } from "@/stores/assets"
import { invokeService } from "@/mcp"
import { forgeStatus } from "@/mcp"
import { Cpu, HardDrive, PanelLeft, Image, Music, Mic, Volume2, Wand2 } from "lucide-react"
import { AppSidebar } from "./AppSidebar"

type TabId = "assets" | "video"

type ParamDef = { n: string; t: string; l: string; p: string; d?: string | number; opts?: string[] }
const GENRES: Record<string, { id: string; label: string; icon: typeof Image; service: string; model: string; params: ParamDef[] }[]> = {
  image: [
    { id:"z_image", label:"Z-Image Turbo", icon:Image, service:"z_image", model:"", params:[{ n:"prompt", t:"text", l:"Prompt", p:"A cyberpunk samurai..." }] },
    { id:"z_image_base", label:"Z-Image Base", icon:Image, service:"z_image", model:"", params:[{ n:"prompt", t:"text", l:"Prompt", p:"A cyberpunk samurai..." }, { n:"quality", t:"select", l:"Quality", opts:["turbo","standard"], p:"", d:"standard" }] },
  ],
  audio: [
    { id:"ace_step", label:"ACE-Step Music", icon:Music, service:"ace_step", model:"", params:[{ n:"input_prompt", t:"text", l:"Prompt", p:"epic cinematic orchestral" }, { n:"duration_seconds", t:"number", l:"Duration (s)", p:"30", d:30 }] },
    { id:"moss_sfx", label:"MOSS Sound Effect", icon:Volume2, service:"moss_soundeffect", model:"", params:[{ n:"input_prompt", t:"text", l:"Description", p:"rain and thunder" }, { n:"duration_seconds", t:"number", l:"Duration (s)", p:"5", d:5 }] },
    { id:"kokoro", label:"Kokoro TTS", icon:Mic, service:"kokoro", model:"", params:[{ n:"text", t:"text", l:"Text", p:"Hello world" }] },
    { id:"moss_voice_clone", label:"Voice Clone", icon:Mic, service:"wan2gp", model:"moss/moss-tts", params:[{ n:"text", t:"text", l:"Text", p:"Hello world" }, { n:"reference_audio_b64", t:"file", l:"Reference Audio", p:"Upload voice sample" }] },
    { id:"moss_voice_gen", label:"Voice Generator", icon:Wand2, service:"wan2gp", model:"moss/moss-voicegenerator", params:[{ n:"input_prompt", t:"text", l:"Voice Description", p:"deep British male voice" }] },
  ],
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

// ── Assets Tab (Forge generators) ─────────────────────────────────────────

function AssetsTab() {
  const toast = useToastStore((s) => s.addToast)
  const addAsset = useAssetStore((s) => s.addAsset)
  const [genre, setGenre] = useState<"image" | "audio">("image")
  const [selected, setSelected] = useState(GENRES.image[0])
  const [paramVals, setParamVals] = useState<Record<string, string|number>>(()=>{const d:Record<string,string|number>={};GENRES.image[0].params.forEach((p)=>{if(p.d!==undefined)d[p.n]=p.d});return d})
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    const d: Record<string, string|number> = {}
    selected.params.forEach((p) => { if (p.d !== undefined) d[p.n] = p.d })
    setParamVals(d)
  }, [selected])

  const handleGenerate = async () => {
    setGenerating(true)
    try {
      const result = selected.model
        ? await invokeService(selected.service, paramVals, selected.model)
        : await invokeService(selected.service, paramVals)
      if (result.status === "ok" || result.status === "success") {
        if (result.data) {
          const mt = result.media_type || "image/png"
          const isAud = mt.includes("audio")
          const cat: AssetCategory = selected.id.includes("ace") ? "music" : selected.id.includes("sfx") ? "sfx" : selected.id.includes("voice") || selected.id === "kokoro" ? "voice" : "image"
          addAsset({
            name: `${selected.label} ${new Date().toLocaleTimeString()}`,
            type: isAud ? "audio" : "image",
            category: cat,
            mediaType: mt,
            url: `data:${mt};base64,${result.data}`,
            sizeBytes: Math.round((result.data as string).length * 0.75),
            source: "generated",
          })
          toast("success", `${selected.label} generated`)
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

  return (
    <div className="flex-1 flex gap-6 p-6">
      <div className="w-56 shrink-0">
        <Tabs value={genre} onValueChange={(v) => { setGenre(v as "image" | "audio"); setSelected(GENRES[v as "image" | "audio"][0]) }}>
          <TabsList className="w-full">
            <TabsTrigger value="image" className="flex-1">Image</TabsTrigger>
            <TabsTrigger value="audio" className="flex-1">Audio</TabsTrigger>
          </TabsList>
          <TabsContent value="image" className="mt-2 space-y-1">
            {GENRES.image.map((t) => {
              const Icon = t.icon
              return (
                <Button key={t.id} variant={selected.id === t.id ? "secondary" : "ghost"}
                  className="w-full justify-start gap-2 h-auto py-2" onClick={() => setSelected(t)}>
                  <Icon />{t.label}
                </Button>
              )
            })}
          </TabsContent>
          <TabsContent value="audio" className="mt-2 space-y-1">
            {GENRES.audio.map((t) => {
              const Icon = t.icon
              return (
                <Button key={t.id} variant={selected.id === t.id ? "secondary" : "ghost"}
                  className="w-full justify-start gap-2 h-auto py-2" onClick={() => setSelected(t)}>
                  <Icon />{t.label}
                </Button>
              )
            })}
          </TabsContent>
        </Tabs>
      </div>
      <Separator orientation="vertical" />
      <Card className="flex-1 max-w-lg">
        <CardHeader><CardTitle className="text-base">{selected.label}</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-4">
          {selected.params.map((p) => (
            <div key={p.n} className="flex flex-col gap-1.5">
              <Label htmlFor={p.n}>{p.l}</Label>
              {p.t === "select" && p.opts ? (
                <Select value={String(paramVals[p.n] ?? p.d ?? "")}
                  onValueChange={(v) => setParamVals((prev) => ({ ...prev, [p.n]: v } as Record<string, string | number>))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {p.opts.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                  </SelectContent>
                </Select>
              ) : p.t === "number" ? (
                <Input id={p.n} type="number" value={paramVals[p.n] ?? p.d ?? ""}
                  onChange={(e) => setParamVals((prev) => ({ ...prev, [p.n]: parseInt(e.target.value) || 0 }))}
                  placeholder={p.p} />
              ) : p.t === "file" ? (
                <Label htmlFor={p.n}
                  className="flex items-center justify-center h-20 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/50 text-sm text-muted-foreground">
                  {paramVals[p.n] ? "File loaded" : p.p}
                  <Input id={p.n} type="file" className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (!f) return
                      const r = new FileReader()
                      r.onload = () => {
                        const d = r.result as string
                        setParamVals((prev) => ({ ...prev, [p.n]: d.includes(",") ? (d.split(",")[1] || "") : d }))
                      }
                      r.readAsDataURL(f)
                    }} />
                </Label>
              ) : p.t === "text" ? (
                <Textarea id={p.n} value={String(paramVals[p.n] ?? "")}
                  onChange={(e) => setParamVals((prev) => ({ ...prev, [p.n]: e.target.value }))}
                  placeholder={p.p} rows={3} />
              ) : (
                <Input id={p.n} value={String(paramVals[p.n] ?? p.d ?? "")}
                  onChange={(e) => setParamVals((prev) => ({ ...prev, [p.n]: e.target.value }))}
                  placeholder={p.p} />
              )}
            </div>
          ))}
          <Button className="mt-2" disabled={generating} onClick={handleGenerate}>
            {generating ? "Generating..." : `Generate ${selected.label}`}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

// ── Video Tab ──────────────────────────────────────────────────

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
