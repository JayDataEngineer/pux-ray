import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { callTool } from "@/mcp"
import {
  Play, Pause, SkipBack, Plus, Trash2,
  Music, Mic, Volume2, Film, Settings, Wand2, Loader2,
} from "lucide-react"

type TrackId = "video" | "voice" | "sfx" | "music"

interface TrackDef {
  id: TrackId; label: string; color: string; icon: typeof Film
}

const TRACKS: TrackDef[] = [
  { id: "video", label: "Video", color: "#6366f1", icon: Film },
  { id: "voice", label: "Voice", color: "#4ade80", icon: Mic },
  { id: "sfx", label: "SFX", color: "#facc15", icon: Volume2 },
  { id: "music", label: "Music", color: "#fb923c", icon: Music },
]

const PPS = 80 // pixels per second
const ROW_H = 44
const RULER_H = 24

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  const ms = Math.floor((s % 1) * 100)
  return `${m}:${sec.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`
}

export function VideoEditor() {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const removeSegment = useTimelineStore((s) => s.removeSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const selectedId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelected = useTimelineStore((s) => s.setSelectedSegment)
  const playback = useTimelineStore((s) => s.playback)
  const setPlayback = useTimelineStore((s) => s.setPlayback)
  const toast = useToastStore((s) => s.addToast)

  const [generating, setGenerating] = useState(false)
  const raf = useRef(0)

  const sel = segments.find((s) => s.id === selectedId)
  const total = Math.max(segments.reduce((t, s) => Math.max(t, s.start + s.duration), 0), 5)

  // Playback loop
  useEffect(() => {
    if (!playback.isPlaying) return
    const t0 = performance.now() - playback.currentTime * 1000
    const tick = (now: number) => {
      const elapsed = (now - t0) / 1000
      if (elapsed >= total) { setPlayback({ isPlaying: false, currentTime: 0 }); return }
      setPlayback({ currentTime: elapsed })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playback.isPlaying, total])

  const togglePlay = () => {
    if (playback.isPlaying) { setPlayback({ isPlaying: false }); return }
    if (playback.currentTime >= total) setPlayback({ currentTime: 0, isPlaying: true })
    else setPlayback({ isPlaying: true })
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    try {
      const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
      if (d.type === "image") {
        const s = addSegment({ prompt: d.name, firstFrameB64: d.url, thumbnailUrl: d.url, status: "empty" })
        setSelected(s.id)
        toast("info", `Added: ${d.name}`)
      }
    } catch {}
  }

  const genAll = async () => {
    if (segments.length === 0) return
    setGenerating(true)
    for (const seg of segments) {
      if (seg.status !== "empty" || !seg.firstFrameB64) continue
      updateSegment(seg.id, { status: "generating" })
      try {
        const r = await callTool<{ status: string; data?: string; media_type?: string; error?: string }>("run",
          { service: "wan2gp", model: "wan/t2v_1.3B", image_b64: seg.firstFrameB64, prompt: seg.prompt || "animate", seed: seg.params.seed, frames: seg.params.frames })
        if (r.status === "ok" && r.data) {
          updateSegment(seg.id, { videoUrl: `data:video/mp4;base64,${r.data}`, status: "ready" })
        } else {
          updateSegment(seg.id, { status: "failed" })
          toast("error", `Seg ${seg.order + 1}: ${r.error || "failed"}`)
        }
      } catch (e) { updateSegment(seg.id, { status: "failed" }); toast("error", String(e)) }
    }
    setGenerating(false)
    toast("success", "Generation done")
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex-1 flex min-h-0">
        {/* Preview */}
        <div className="flex-1 p-4 flex flex-col gap-3 min-w-0">
          <div className="flex-1 relative rounded-xl overflow-hidden bg-black/90 flex items-center justify-center"
            onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
            {segments.length > 0 ? (
              <CurrentPreview segments={segments} time={playback.currentTime} />
            ) : (
              <p className="text-sm text-muted-foreground">Drop images from the asset sidebar</p>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setPlayback({ currentTime: 0 })}>
              <SkipBack className="h-3 w-3" />
            </Button>
            <Button variant="default" size="icon" className="h-8 w-8 rounded-full" onClick={togglePlay}>
              {playback.isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <span className="text-xs tabular-nums w-22">{fmt(playback.currentTime)} / {fmt(total)}</span>
            <Separator orientation="vertical" className="h-5 mx-1" />
            <Button variant="outline" size="sm" className="h-7 text-xs gap-1" disabled={generating || segments.length === 0} onClick={genAll}>
              {generating ? <><Loader2 className="h-3 w-3 animate-spin" /> Generating...</> : <><Wand2 className="h-3 w-3" /> I2V</>}
            </Button>
            <div className="flex-1" />
            <Button variant="ghost" size="sm" className="h-7 text-xs gap-1"><Settings className="h-3 w-3" /> Export</Button>
          </div>
        </div>

        {/* Inspector */}
        {sel && (
          <div className="w-64 shrink-0 border-l p-3 space-y-3 overflow-y-auto">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold">K_{String(sel.order + 1).padStart(2, "0")}</span>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => { removeSegment(sel.id); setSelected(null) }}>
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
            <div className="space-y-1">
              <Label className="text-[10px] text-muted-foreground">Prompt</Label>
              <Textarea value={sel.prompt} onChange={(e) => updateSegment(sel.id, { prompt: e.target.value })}
                className="text-xs min-h-[60px]" placeholder="Describe this segment..." />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">Duration (s)</Label>
                <Input type="number" value={sel.duration} onChange={(e) => updateSegment(sel.id, { duration: Math.max(1, Number(e.target.value)) })} className="h-7 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">FPS</Label>
                <Select value={String(sel.params.fps)} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, fps: Number(v) } })}>
                  <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>{[12, 16, 24, 30, 60].map((f) => <SelectItem key={f} value={String(f)}>{f}fps</SelectItem>)}</SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">Width</Label>
                <Input type="number" value={sel.params.width} onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, width: Number(e.target.value) } })} className="h-7 text-xs" />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">Height</Label>
                <Input type="number" value={sel.params.height} onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, height: Number(e.target.value) } })} className="h-7 text-xs" />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={sel.status === "ready" ? "default" : sel.status === "failed" ? "destructive" : "secondary"} className="text-[10px]">{sel.status}</Badge>
              {sel.status === "ready" && sel.videoUrl && <span className="text-[10px] text-green-500">Done</span>}
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="border-t bg-background shrink-0">
        <div className="flex" style={{ height: RULER_H }}>
          <div className="w-24 shrink-0 border-r px-2 flex items-center text-[10px] font-medium text-muted-foreground">Track</div>
          <div className="flex-1 relative overflow-hidden">
            <div className="absolute inset-0 flex">
              {Array.from({ length: Math.ceil(total) + 1 }).map((_, i) => (
                <div key={i} className="flex items-end border-l border-border/20 shrink-0" style={{ width: PPS }}>
                  <span className="text-[9px] text-muted-foreground/40 pl-0.5 leading-none">{i}s</span>
                </div>
              ))}
            </div>
            <div className="absolute top-0 bottom-0 w-px bg-red-500 z-10 pointer-events-none"
              style={{ left: playback.currentTime * PPS, transition: playback.isPlaying ? "none" : "left 0.1s" }} />
          </div>
        </div>

        {TRACKS.map((tr) => (
          <div key={tr.id} className="flex border-t" style={{ height: ROW_H }}>
            <div className="w-24 shrink-0 border-r px-2 flex items-center gap-1.5 bg-muted/10">
              <tr.icon className="h-3 w-3" style={{ color: tr.color }} />
              <span className="text-[10px] font-medium truncate">{tr.label}</span>
            </div>
            <div className="flex-1 relative overflow-hidden" style={tr.id !== "video" ? { background: `${tr.color}08` } : {}}>
              {tr.id === "video" ? segments.map((seg) => (
                <div key={seg.id} onClick={() => setSelected(seg.id)}
                  className={`absolute top-0.5 bottom-0.5 rounded cursor-pointer overflow-hidden border transition-colors ${seg.id === selectedId ? "border-primary ring-1 ring-primary/30" : "border-border/40 hover:border-primary/40"}`}
                  style={{ left: seg.start * PPS, width: Math.max(seg.duration * PPS, 4), background: `${tr.color}22` }}>
                  {seg.thumbnailUrl && <img src={seg.thumbnailUrl} alt="" className="absolute inset-0 w-full h-full object-cover opacity-25" />}
                  <div className="relative z-10 flex items-center gap-1 px-1.5 h-full">
                    <span className="text-[9px] font-medium truncate">K_{String(seg.order + 1).padStart(2, "0")}</span>
                    {seg.status === "ready" && <div className="h-1.5 w-1.5 rounded-full bg-green-500 shrink-0" />}
                    {seg.status === "generating" && <Loader2 className="h-2.5 w-2.5 animate-spin shrink-0" />}
                    {seg.status === "failed" && <div className="h-1.5 w-1.5 rounded-full bg-red-500 shrink-0" />}
                  </div>
                </div>
              )) : (
                audioCues.filter((c) => c.track === tr.id).map((cue) => (
                  <div key={cue.id} className="absolute top-1 bottom-1 rounded border border-border/20"
                    style={{ left: cue.start * PPS, width: Math.max(cue.duration * PPS, 4), background: `${tr.color}33` }}>
                    <span className="text-[9px] truncate px-1.5 leading-[44px]">{cue.label}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        ))}

        <div className="flex border-t">
          <div className="w-24 shrink-0 border-r" />
          <div className="flex-1 p-1">
            <Button variant="ghost" size="sm" className="h-6 w-full text-xs gap-1" onClick={() => { const s = addSegment({ duration: 5, status: "empty" }); setSelected(s.id) }}>
              <Plus className="h-3 w-3" /> Add Keyframe
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Preview component - finds the segment at current time
function CurrentPreview({ segments, time }: { segments: ReturnType<typeof useTimelineStore.getState>["segments"]; time: number }) {
  const seg = segments.find((s) => time >= s.start && time < s.start + s.duration)
  if (!seg) return <p className="text-sm text-muted-foreground">No segment at this time</p>

  if (seg.videoUrl) {
    return <video key={seg.id} src={seg.videoUrl} className="max-w-full max-h-full rounded-lg" controls />
  }
  if (seg.firstFrameB64) {
    return (
      <div className="flex flex-col items-center gap-2">
        <img src={seg.firstFrameB64} alt="" className="max-w-full max-h-[70vh] rounded-lg object-contain" />
        <span className="text-xs text-muted-foreground">Keyframe {seg.order + 1} — {seg.prompt?.slice(0, 60)}</span>
      </div>
    )
  }
  return <p className="text-sm text-muted-foreground">Empty segment</p>
}
