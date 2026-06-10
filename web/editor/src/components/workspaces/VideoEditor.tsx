import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { useAssetStore } from "@/stores/assets"
import { callTool } from "@/mcp"
import { VIDEO_MODELS } from "@/types/timeline"
import type { TimelineSegment } from "@/types/timeline"
import {
  Play, Pause, SkipBack, Plus, Trash2,
  Music, Mic, Volume2, Film, Settings, Wand2, Loader2,
  ZoomIn, ZoomOut, ChevronDown, ChevronRight, Dice5,
  Sparkles, ImagePlus, FilmIcon, ToggleLeft, ToggleRight,
  Headphones, Layers, Sliders,
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

const ROW_H = 52
const RULER_H = 26
const HANDLE_W = 8

function fmt(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  const ms = Math.floor((s % 1) * 100)
  return `${m}:${sec.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`
}

function segLabel(order: number) {
  return `K${String(order + 1).padStart(2, "0")}`
}

// ── Drag/Resize State ─────────────────────────────────────────────────────────
type DragMode = "move" | "resize-left" | "resize-right" | null

interface DragInfo {
  mode: DragMode
  segmentId: string
  startMouseX: number
  startSegLeft: number
  startSegWidth: number
}

// ── Compile timeline into backend payload ─────────────────────────────────────
function buildPayload(seg: TimelineSegment, _allSegments?: TimelineSegment[]) {
  return {
    service: "wan2gp",
    model: seg.params.model,
    image_b64: seg.firstFrameB64 || undefined,
    image_end_b64: seg.lastFrameB64 || undefined,
    input_prompt: seg.prompt || "animate",  // backend expects input_prompt, not prompt
    n_prompt: seg.negativePrompt || undefined,
    seed: seg.params.seed,
    frame_num: seg.params.frames,   // backend expects frame_num, not frames
    fps: seg.params.fps,
    width: seg.params.width,
    height: seg.params.height,
    guide_scale: seg.params.guideScale,
    sampling_steps: seg.params.samplingSteps,
    guide_phases: seg.params.guidePhases || undefined,
    epsilon: seg.params.epsilon !== 0.001 ? seg.params.epsilon : undefined,
    denoising_strength: seg.params.denoisingStrength !== 1.0 ? seg.params.denoisingStrength : undefined,
    spatial_upscale: seg.params.spatialUpscale ? "true" : undefined,
    loras_selected: seg.params.loras || undefined,
    perturbation_switch: seg.params.perturbationSwitch || undefined,
  }
}

export function VideoEditor() {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const removeSegment = useTimelineStore((s) => s.removeSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const selectedId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelected = useTimelineStore((s) => s.setSelectedSegment)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const playback = useTimelineStore((s) => s.playback)
  const setPlayback = useTimelineStore((s) => s.setPlayback)
  const toast = useToastStore((s) => s.addToast)
  const assets = useAssetStore((s) => s.assets)

  const [generating, setGenerating] = useState(false)
  const [pps, setPps] = useState(80)
  const raf = useRef(0)

  const sel = segments.find((s) => s.id === selectedId)
  const total = Math.max(segments.reduce((t, s) => Math.max(t, s.start + s.duration), 0), 5)

  // ── Playback loop ──────────────────────────────────────────────────────────
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

  const seekFromMouseEvent = (e: React.MouseEvent<HTMLElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const relX = e.clientX - rect.left
    const t = Math.max(0, Math.min(total, relX / pps))
    setPlayback({ currentTime: t })
  }

  // ── Drag / Resize ──────────────────────────────────────────────────────────
  const dragRef = useRef<DragInfo | null>(null)

  const onSegmentPointerDown = useCallback((
    e: React.PointerEvent<HTMLDivElement>,
    segId: string,
    mode: DragMode,
  ) => {
    if (e.button !== 0) return
    e.stopPropagation()
    e.preventDefault()

    const seg = useTimelineStore.getState().segments.find(s => s.id === segId)
    if (!seg) return

    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)

    dragRef.current = {
      mode,
      segmentId: segId,
      startMouseX: e.clientX,
      startSegLeft: seg.start * pps,
      startSegWidth: seg.duration * pps,
    }

    setSelected(segId)
  }, [pps, setSelected])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      const dx = e.clientX - drag.startMouseX
      const minDurPx = 0.5 * pps

      const seg = useTimelineStore.getState().segments.find(s => s.id === drag.segmentId)
      if (!seg) return

      if (drag.mode === "move") {
        let newStart = drag.startSegLeft / pps + dx / pps
        newStart = Math.max(0, newStart)
        updateSegment(drag.segmentId, { start: newStart })
      } else if (drag.mode === "resize-left") {
        let newLeft = drag.startSegLeft + dx
        let newWidth = drag.startSegWidth - dx
        if (newWidth < minDurPx) { newWidth = minDurPx; newLeft = drag.startSegLeft + drag.startSegWidth - minDurPx }
        if (newLeft < 0) { newLeft = 0; newWidth = drag.startSegLeft + drag.startSegWidth }
        updateSegment(drag.segmentId, { start: newLeft / pps, duration: newWidth / pps })
      } else if (drag.mode === "resize-right") {
        let newWidth = drag.startSegWidth + dx
        if (newWidth < minDurPx) newWidth = minDurPx
        updateSegment(drag.segmentId, { duration: newWidth / pps })
      }
    }
    const onUp = () => { dragRef.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp) }
  }, [pps, updateSegment])

  // ── Drop from asset sidebar ────────────────────────────────────────────────
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    try {
      const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
      if (d.type === "image") {
        const s = addSegment({ prompt: d.name, firstFrameB64: d.url, thumbnailUrl: d.url, status: "empty" })
        setSelected(s.id)
        toast("info", `Added: ${d.name}`)
      } else if (d.type === "audio") {
        const cue = addAudioCue({ track: "music", start: 0, duration: 5, label: d.name, audioUrl: d.url, audioB64: d.url, volume: 0.5, waveformPeaks: null, sourceStepId: null })
        decodeWaveform(d.url, cue.id)
        toast("info", `Added audio: ${d.name}`)
      }
    } catch {}
  }

  // ── Decode real waveform peaks from audio ────────────────────────────────
  const decodeWaveform = async (url: string, cueId: string) => {
    try {
      const ctx = new AudioContext()
      const res = await fetch(url)
      const buf = await res.arrayBuffer()
      const audio = await ctx.decodeAudioData(buf)
      const raw = audio.getChannelData(0)
      const peaks: number[] = []
      const samplesPerPeak = Math.floor(raw.length / 80)
      for (let i = 0; i < 80; i++) {
        let max = 0
        for (let j = 0; j < samplesPerPeak; j++) {
          const val = Math.abs(raw[i * samplesPerPeak + j] || 0)
          if (val > max) max = val
        }
        peaks.push(max)
      }
      ctx.close()
      useTimelineStore.getState().updateAudioCue(cueId, {
        waveformPeaks: peaks,
        duration: audio.duration,
      })
    } catch { /* non-critical — will show sine placeholder */ }
  }

  // ── Generate single segment ────────────────────────────────────────────────
  const generateSegment = async (seg: TimelineSegment) => {
    updateSegment(seg.id, { status: "generating" })
    setGenerating(true)
    try {
      // If using LTX model, route through ltx_director spec
      const isLtx = seg.params.model.startsWith("ltx")
      const payload = buildPayload(seg)

      // Find audio cues that overlap this segment for audio conditioning
      const overlappingAudio = audioCues.filter(c =>
        c.start < seg.start + seg.duration && c.start + c.duration > seg.start
      )
      const firstAudio = overlappingAudio[0]
      if (firstAudio?.audioB64) {
        payload.audio_b64 = firstAudio.audioB64
        payload.audio_scale = String(firstAudio.volume)
        payload.audio_prompt_type = "A"
      }

      // For LTX models with a single segment, still use prompt relay format
      if (isLtx) {
        payload.local_prompts = seg.prompt || "animate"
        payload.segment_lengths = String(seg.params.frames)
      }

      const r = await callTool<{ status: string; data?: string; media_type?: string; error?: string }>(
        isLtx ? "run_ltx_director" : "run",
        isLtx ? {
          global_prompt: seg.prompt || "animate",
          start_image: seg.firstFrameB64 || "",
          end_image: seg.lastFrameB64 || undefined,
          negative_prompt: seg.negativePrompt || undefined,
          seed: seg.params.seed,
          fps: seg.params.fps,
          frames: seg.params.frames,
          steps: seg.params.samplingSteps,
          guidance: String(seg.params.guideScale),
          width: seg.params.width,
          height: seg.params.height,
          audio_b64: payload.audio_b64,
          audio_scale: payload.audio_scale,
          audio_prompt_type: payload.audio_prompt_type,
          loras: seg.params.loras || undefined,
          guide_phases: seg.params.guidePhases,
          epsilon: String(seg.params.epsilon),
          spatial_upscale: seg.params.spatialUpscale ? "true" : undefined,
        } : payload
      )
      if (r.status === "ok" && r.data) {
        updateSegment(seg.id, { videoUrl: `data:video/mp4;base64,${r.data}`, status: "ready" })
        toast("success", `${segLabel(seg.order)} generated!`)
      } else {
        updateSegment(seg.id, { status: "failed" })
        toast("error", r.error || "Generation failed")
      }
    } catch (e) {
      updateSegment(seg.id, { status: "failed" })
      toast("error", String(e))
    } finally {
      setGenerating(false)
    }
  }

  // ── Generate all — uses Director prompt relay for multi-segment LTX ──────
  const genAll = async () => {
    if (segments.length === 0) return
    setGenerating(true)

    const eligible = segments.filter(s => s.status === "empty" || s.status === "failed")
    if (eligible.length === 0) { setGenerating(false); return }

    // Check if we should use Director relay (multiple LTX segments)
    const ltxSegs = eligible.filter(s => s.params.model.startsWith("ltx"))
    if (ltxSegs.length > 1) {
      // Director prompt relay — single call for all LTX segments
      const fps = ltxSegs[0].params.fps
      const totalFrames = ltxSegs.reduce((t, s) => t + s.params.frames, 0)
      const firstSeg = ltxSegs[0]

      ltxSegs.forEach(s => updateSegment(s.id, { status: "generating" }))

      try {
        // Find audio for conditioning
        const firstAudio = audioCues[0]
        const r = await callTool<{ status: string; data?: string; media_type?: string; error?: string }>("run_ltx_director", {
          global_prompt: firstSeg.prompt || "animate",
          segment_prompts: ltxSegs.map(s => s.prompt || "animate").join("|"),
          segment_lengths: ltxSegs.map(s => String(s.params.frames)).join(","),
          start_image: firstSeg.firstFrameB64 || "",
          end_image: ltxSegs[ltxSegs.length - 1].lastFrameB64 || undefined,
          negative_prompt: firstSeg.negativePrompt || undefined,
          seed: firstSeg.params.seed,
          fps,
          frames: totalFrames,
          steps: firstSeg.params.samplingSteps,
          guidance: String(firstSeg.params.guideScale),
          width: firstSeg.params.width,
          height: firstSeg.params.height,
          audio_b64: firstAudio?.audioB64 || undefined,
          audio_scale: firstAudio ? String(firstAudio.volume) : undefined,
          audio_prompt_type: firstAudio ? "A" : undefined,
          loras: firstSeg.params.loras || undefined,
          guide_phases: firstSeg.params.guidePhases,
          epsilon: String(firstSeg.params.epsilon),
        })

        if (r.status === "ok" && r.data) {
          // Assign the video to the first segment (full Director output)
          updateSegment(ltxSegs[0].id, { videoUrl: `data:video/mp4;base64,${r.data}`, status: "ready" })
          ltxSegs.slice(1).forEach(s => updateSegment(s.id, { status: "ready" }))
          toast("success", "Director relay generated!")
        } else {
          ltxSegs.forEach(s => updateSegment(s.id, { status: "failed" }))
          toast("error", r.error || "Director relay failed")
        }
      } catch (e) {
        ltxSegs.forEach(s => updateSegment(s.id, { status: "failed" }))
        toast("error", String(e))
      }

      // Generate any non-LTX segments individually
      const nonLtx = eligible.filter(s => !s.params.model.startsWith("ltx"))
      for (const seg of nonLtx) {
        await generateSegment(seg)
      }
    } else {
      // Generate each segment individually
      let anyFailed = false
      for (const seg of eligible) {
        if (!seg.firstFrameB64 && !seg.params.model.startsWith("ltx")) { anyFailed = true; continue }
        await generateSegment(seg)
        if (useTimelineStore.getState().segments.find(s => s.id === seg.id)?.status === "failed") anyFailed = true
      }
      if (!anyFailed) toast("success", "All segments generated!")
    }

    setGenerating(false)
  }

  // ── Export ──────────────────────────────────────────────────────────────────
  const doExport = () => {
    if (segments.length === 0) return
    const fps = segments[0]?.params.fps ?? 24
    const data = JSON.stringify({
      fps,
      totalDuration: total,
      segments: segments.map(s => ({
        order: s.order, start: s.start, duration: s.duration,
        prompt: s.prompt, negativePrompt: s.negativePrompt,
        status: s.status,
        params: s.params,
        // LTX Director prompt relay format
        startFrame: Math.round(s.start * fps),
        endFrame: Math.round((s.start + s.duration) * fps),
      })),
      audioCues: audioCues.map(c => ({
        track: c.track, start: c.start, duration: c.duration,
        label: c.label, volume: c.volume,
      })),
      // LTX prompt relay: compile all segments into pipe-separated prompts
      ltxDirector: {
        local_prompts: segments.map(s => s.prompt).filter(Boolean).join("|"),
        segment_lengths: segments.map(s => Math.round(s.duration * fps)).join(","),
      },
    }, null, 2)
    const blob = new Blob([data], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url; a.download = "timeline.json"; a.click()
    URL.revokeObjectURL(url)
    toast("success", "Timeline exported")
  }

  // Zoom
  const zoomIn = () => setPps(p => Math.min(p * 1.25, 400))
  const zoomOut = () => setPps(p => Math.max(p / 1.25, 20))

  // ── Keyboard shortcuts ───────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return
      if (e.key === " ") { e.preventDefault(); togglePlay() }
      else if (e.key === "ArrowLeft") { e.preventDefault(); setPlayback({ currentTime: Math.max(0, playback.currentTime - (e.shiftKey ? 1 : 0.1)) }) }
      else if (e.key === "ArrowRight") { e.preventDefault(); setPlayback({ currentTime: Math.min(total, playback.currentTime + (e.shiftKey ? 1 : 0.1)) }) }
      else if (e.key === "Home" || e.key === "k ") { e.preventDefault(); setPlayback({ currentTime: 0 }) }
      else if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
        e.preventDefault(); removeSegment(selectedId); setSelected(null)
      }
      else if (e.key === "=" || e.key === "+") { e.preventDefault(); zoomIn() }
      else if (e.key === "-") { e.preventDefault(); zoomOut() }
      else if (e.key === "z" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); useTimelineStore.getState().undo() }
      else if (e.key === "y" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); useTimelineStore.getState().redo() }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [playback.currentTime, playback.isPlaying, total, selectedId, togglePlay, setPlayback, removeSegment, setSelected, zoomIn, zoomOut])

  // ── Model change handler (updates default params) ──────────────────────────
  const handleModelChange = (segId: string, modelId: string) => {
    const modelDef = VIDEO_MODELS.find(m => m.id === modelId)
    if (modelDef && sel) {
      updateSegment(segId, {
        params: {
          ...sel.params,
          model: modelId,
          frames: modelDef.defaultFrames,
          fps: modelDef.defaultFps,
          width: modelDef.defaultWidth,
          height: modelDef.defaultHeight,
        },
      })
    } else {
      updateSegment(segId, { params: { ...sel!.params, model: modelId } })
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-[#0a0a0c]">
      {/* ═══ MAIN SPLIT: Preview + Inspector ═══ */}
      <div className="flex-1 flex min-h-0">
        {/* Preview + Transport */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Video Canvas */}
          <div className="flex-1 relative overflow-hidden flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #0c0c10 0%, #111118 50%, #0c0c10 100%)" }}
            onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
            {segments.length > 0 ? (
              <CurrentPreview segments={segments} time={playback.currentTime} />
            ) : (
              <div className="text-center space-y-3 opacity-60">
                <div className="mx-auto w-16 h-16 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-center">
                  <FilmIcon className="h-8 w-8 text-white/30" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white/50">Drop images or audio from the asset sidebar</p>
                  <p className="text-xs text-white/25 mt-1">or click + Add Keyframe on the timeline</p>
                </div>
              </div>
            )}
          </div>

          {/* Transport Bar */}
          <div className="h-10 shrink-0 flex items-center px-3 gap-1.5 border-t border-white/[0.06] bg-[#111114]">
            <Button variant="ghost" size="icon" className="h-7 w-7 text-white/60 hover:text-white hover:bg-white/10"
              onClick={() => setPlayback({ currentTime: 0 })} title="Skip to start (Home)">
              <SkipBack className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="icon"
              className="h-8 w-8 rounded-full bg-white/10 text-white hover:bg-white/20 hover:text-white border border-white/10"
              onClick={togglePlay} title="Play / Pause (Space)">
              {playback.isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
            </Button>
            <span className="text-[11px] tabular-nums text-white/50 font-mono w-28 pl-1">{fmt(playback.currentTime)} / {fmt(total)}</span>

            <div className="w-px h-5 bg-white/10 mx-1" />

            <Button variant="ghost" size="icon" className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/5"
              onClick={() => useTimelineStore.getState().undo()} title="Undo (Ctrl+Z)">
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6.69 3L3 13"/></svg>
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/5"
              onClick={() => useTimelineStore.getState().redo()} title="Redo (Ctrl+Y)">
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6.69 3L21 13"/></svg>
            </Button>

            <div className="w-px h-5 bg-white/10 mx-1" />

            <Button variant="ghost" size="sm"
              className="h-7 text-[11px] gap-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded-md"
              disabled={generating || segments.length === 0} onClick={genAll}>
              {generating ? <><Loader2 className="h-3 w-3 animate-spin" /> Generating...</> : <><Sparkles className="h-3 w-3" /> Generate All</>}
            </Button>

            <div className="flex-1" />

            <Button variant="ghost" size="icon" className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/5"
              onClick={zoomOut}><ZoomOut className="h-3 w-3" /></Button>
            <span className="text-[10px] text-white/30 w-10 text-center font-mono">{Math.round(pps / 80 * 100)}%</span>
            <Button variant="ghost" size="icon" className="h-7 w-7 text-white/40 hover:text-white/70 hover:bg-white/5"
              onClick={zoomIn}><ZoomIn className="h-3 w-3" /></Button>

            <div className="w-px h-5 bg-white/10 mx-1" />

            <Button variant="ghost" size="sm"
              className="h-7 text-[11px] gap-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded-md"
              disabled={segments.length === 0} onClick={doExport}>
              <Settings className="h-3 w-3" /> Export
            </Button>
          </div>
        </div>

        {/* ═══ INSPECTOR PANEL ═══ */}
        {sel && (
          <div className="w-[280px] shrink-0 border-l border-white/[0.06] overflow-y-auto bg-[#111114] scrollbar-thin">
            {/* Inspector Header */}
            <div className="sticky top-0 z-10 bg-[#111114] border-b border-white/[0.06] px-4 py-2.5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold tracking-wide text-white/90">{segLabel(sel.order)}</span>
                <Badge variant={sel.status === "ready" ? "default" : sel.status === "failed" ? "destructive" : "secondary"}
                  className="text-[9px] px-1.5 py-0 font-mono">{sel.status}</Badge>
              </div>
              <Button variant="ghost" size="icon" className="h-6 w-6 text-white/30 hover:text-red-400 hover:bg-red-500/10"
                onClick={() => { removeSegment(sel.id); setSelected(null) }}>
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>

            <div className="p-4 space-y-4">
              {/* Thumbnail / Video Preview */}
              {sel.videoUrl ? (
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black shadow-lg">
                  <video src={sel.videoUrl} className="w-full" controls />
                </div>
              ) : sel.thumbnailUrl ? (
                <div className="rounded-lg overflow-hidden border border-white/10 bg-black shadow-lg">
                  <img src={sel.thumbnailUrl} alt="" className="w-full aspect-video object-cover" />
                </div>
              ) : null}

              {/* ── Model Section ── */}
              <InspectorSection title="Model" icon={<FilmIcon className="h-3 w-3" />}>
                <Select value={sel.params.model} onValueChange={(v) => handleModelChange(sel.id, v)}>
                  <SelectTrigger className="h-8 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VIDEO_MODELS.map(m => <SelectItem key={m.id} value={m.id}>{m.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </InspectorSection>

              {/* ── Prompts Section ── */}
              <InspectorSection title="Prompts" icon={<Sparkles className="h-3 w-3" />}>
                <div className="space-y-2.5">
                  <div className="space-y-1">
                    <Label className="text-[10px] font-medium text-white/40 uppercase tracking-wider">Positive</Label>
                    <Textarea value={sel.prompt} onChange={(e) => updateSegment(sel.id, { prompt: e.target.value })}
                      className="text-xs min-h-[72px] bg-white/5 border-white/10 text-white/80 placeholder:text-white/20 resize-none rounded-md focus:border-[#6366f1]/50"
                      placeholder="Describe this segment..." />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px] font-medium text-white/40 uppercase tracking-wider">Negative</Label>
                    <Textarea value={sel.negativePrompt} onChange={(e) => updateSegment(sel.id, { negativePrompt: e.target.value })}
                      className="text-xs min-h-[40px] bg-white/5 border-white/10 text-white/80 placeholder:text-white/20 resize-none rounded-md focus:border-[#6366f1]/50"
                      placeholder="What to avoid..." />
                  </div>
                </div>
              </InspectorSection>

              {/* ── Timing Section ── */}
              <InspectorSection title="Timing" icon={<Film className="h-3 w-3" />}>
                <div className="grid grid-cols-2 gap-2">
                  <InspectorField label="Start (s)">
                    <Input type="number" value={Number(sel.start.toFixed(1))} step={0.5} min={0}
                      onChange={(e) => updateSegment(sel.id, { start: Math.max(0, Number(e.target.value)) })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Duration (s)">
                    <Input type="number" value={sel.duration} step={0.5} min={0.5}
                      onChange={(e) => updateSegment(sel.id, { duration: Math.max(0.5, Number(e.target.value)) })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
              </InspectorSection>

              {/* ── Resolution & Frame Rate ── */}
              <InspectorSection title="Resolution & Frames" icon={<ImagePlus className="h-3 w-3" />}>
                <div className="grid grid-cols-2 gap-2">
                  <InspectorField label="Width">
                    <Input type="number" value={sel.params.width} step={64} min={256} max={2048}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, width: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Height">
                    <Input type="number" value={sel.params.height} step={64} min={256} max={2048}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, height: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="FPS">
                    <Input type="number" value={sel.params.fps} min={8} max={60}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, fps: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Frames">
                    <Input type="number" value={sel.params.frames} min={9} max={201}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, frames: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
                <div className="text-[10px] text-white/25 mt-1 font-mono">
                  {(sel.params.frames / sel.params.fps).toFixed(1)}s · {sel.params.width}×{sel.params.height} · {sel.params.frames}f @ {sel.params.fps}fps
                </div>
              </InspectorSection>

              {/* ── Generation Parameters ── */}
              <InspectorSection title="Generation" icon={<Wand2 className="h-3 w-3" />}>
                <div className="grid grid-cols-2 gap-2">
                  <InspectorField label="Steps">
                    <Input type="number" value={sel.params.samplingSteps} min={4} max={50}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, samplingSteps: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="CFG Scale">
                    <Input type="number" value={sel.params.guideScale} step={0.5} min={1} max={20}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, guideScale: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
                <div className="mt-2">
                  <InspectorField label="Seed">
                    <div className="flex gap-1">
                      <Input type="number" value={sel.params.seed}
                        onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, seed: Number(e.target.value) } })}
                        className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md flex-1" />
                      <Button variant="ghost" size="icon"
                        className="h-7 w-7 shrink-0 text-white/30 hover:text-white/70 hover:bg-white/10 rounded-md"
                        onClick={() => updateSegment(sel.id, { params: { ...sel.params, seed: Math.floor(Math.random() * 2147483647) } })}>
                        <Dice5 className="h-3 w-3" />
                      </Button>
                    </div>
                  </InspectorField>
                </div>
              </InspectorSection>

              {/* ── First / Last Frame ── */}
              <InspectorSection title="Frames" icon={<ImagePlus className="h-3 w-3" />}>
                <div className="space-y-2">
                  <div className="space-y-1">
                    <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">First Frame (start image)</Label>
                    {sel.firstFrameB64 ? (
                      <div className="relative group rounded-md overflow-hidden border border-white/10">
                        <img src={sel.firstFrameB64} alt="" className="w-full aspect-video object-cover" />
                        <button className="absolute top-1 right-1 h-5 w-5 rounded bg-black/60 text-white/60 hover:text-red-400 flex items-center justify-center"
                          onClick={() => updateSegment(sel.id, { firstFrameB64: null, thumbnailUrl: null })}>
                          <Trash2 className="h-2.5 w-2.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="border border-dashed border-white/10 rounded-md py-3 text-center text-[10px] text-white/20"
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault(); e.stopPropagation();
                          try {
                            const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                            if (d.type === "image") {
                              updateSegment(sel.id, { firstFrameB64: d.url, thumbnailUrl: d.url })
                              toast("info", `Set first frame: ${d.name}`)
                            }
                          } catch {}
                        }}>
                        Drop image here
                      </div>
                    )}
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Last Frame (FFLF end image)</Label>
                    {sel.lastFrameB64 ? (
                      <div className="relative group rounded-md overflow-hidden border border-white/10">
                        <img src={sel.lastFrameB64} alt="" className="w-full aspect-video object-cover" />
                        <button className="absolute top-1 right-1 h-5 w-5 rounded bg-black/60 text-white/60 hover:text-red-400 flex items-center justify-center"
                          onClick={() => updateSegment(sel.id, { lastFrameB64: null })}>
                          <Trash2 className="h-2.5 w-2.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="border border-dashed border-white/10 rounded-md py-3 text-center text-[10px] text-white/20"
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault(); e.stopPropagation();
                          try {
                            const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                            if (d.type === "image") {
                              updateSegment(sel.id, { lastFrameB64: d.url })
                              toast("info", `Set last frame: ${d.name}`)
                            }
                          } catch {}
                        }}>
                        Drop image for FFLF conditioning
                      </div>
                    )}
                  </div>
                </div>
              </InspectorSection>

              {/* ── LoRA ── */}
              <InspectorSection title="LoRA" icon={<Layers className="h-3 w-3" />} defaultOpen={false}>
                <InspectorField label="LoRA files (comma-separated)">
                  <Input type="text" value={sel.params.loras}
                    onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, loras: e.target.value } })}
                    className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md font-mono"
                    placeholder="lora1.safetensors, lora2.safetensors" />
                </InspectorField>
              </InspectorSection>

              {/* ── Advanced Director Controls ── */}
              <InspectorSection title="Director Controls" icon={<Sliders className="h-3 w-3" />} defaultOpen={false}>
                <div className="grid grid-cols-2 gap-2">
                  <InspectorField label="Guide Phases">
                    <Input type="number" value={sel.params.guidePhases} min={1} max={2}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, guidePhases: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Epsilon">
                    <Input type="number" value={sel.params.epsilon} step={0.001} min={0} max={1}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, epsilon: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Denoise Strength">
                    <Input type="number" value={sel.params.denoisingStrength} step={0.1} min={0} max={1}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, denoisingStrength: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Perturbation">
                    <Select value={String(sel.params.perturbationSwitch)}
                      onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, perturbationSwitch: Number(v) } })}>
                      <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">Off</SelectItem>
                        <SelectItem value="1">Skip Layer</SelectItem>
                        <SelectItem value="2">Skip Self-Attn</SelectItem>
                      </SelectContent>
                    </Select>
                  </InspectorField>
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Spatial Upscale 2×</Label>
                  <button className="text-white/40 hover:text-white/70"
                    onClick={() => updateSegment(sel.id, { params: { ...sel.params, spatialUpscale: !sel.params.spatialUpscale } })}>
                    {sel.params.spatialUpscale
                      ? <ToggleRight className="h-5 w-5 text-[#6366f1]" />
                      : <ToggleLeft className="h-5 w-5" />}
                  </button>
                </div>
              </InspectorSection>

              {/* ── Generate Action ── */}
              {(sel.status === "empty" || sel.status === "failed") && sel.firstFrameB64 && (
                <Button size="sm"
                  className="w-full h-9 text-xs gap-2 bg-[#6366f1] hover:bg-[#5558e6] text-white rounded-lg font-medium"
                  disabled={generating} onClick={() => generateSegment(sel)}>
                  {generating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                  {generating ? "Generating..." : "Generate This Segment"}
                </Button>
              )}
              {sel.status === "empty" && !sel.firstFrameB64 && (
                <div className="text-[11px] text-white/25 text-center py-3 border border-dashed border-white/10 rounded-lg bg-white/[0.02]">
                  <ImagePlus className="h-4 w-4 mx-auto mb-1 opacity-40" />
                  Drop an image to set the first frame, then generate
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ═══ TIMELINE ═══ */}
      <div className="border-t border-white/[0.06] shrink-0 select-none bg-[#0d0d10]">
        {/* Timeline Header */}
        <div className="flex items-center h-6 px-3 border-b border-white/[0.06] bg-[#111114]">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-white/30">Timeline</span>
          <div className="flex-1" />
          <span className="text-[10px] font-mono text-white/25">
            {segments.length} seg{segments.length !== 1 ? "s" : ""} · {fmt(total)}
            {audioCues.length > 0 && <span className="ml-2">{audioCues.length} audio</span>}
          </span>
        </div>

        {/* Ruler */}
        <div className="flex border-b border-white/[0.04]" style={{ height: RULER_H }}>
          <div className="w-[88px] shrink-0 border-r border-white/[0.06] px-2 flex items-center text-[9px] font-semibold uppercase tracking-wider text-white/20">
            Track
          </div>
          <div className="flex-1 relative overflow-hidden cursor-pointer" onClick={seekFromMouseEvent}>
            <div className="absolute inset-0 flex">
              {Array.from({ length: Math.ceil(total) + 1 }).map((_, i) => (
                <div key={i} className="flex items-end border-l border-white/[0.06] shrink-0" style={{ width: pps }}>
                  <span className="text-[8px] text-white/15 pl-1 leading-none font-mono">{i}s</span>
                </div>
              ))}
            </div>
            {/* Playhead */}
            <div className="absolute top-0 bottom-0 w-[2px] bg-red-500 z-10 pointer-events-none shadow-[0_0_6px_rgba(239,68,68,0.5)]"
              style={{ left: playback.currentTime * pps, transition: playback.isPlaying ? "none" : "left 0.1s" }}>
              <div className="absolute -top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-red-500 rounded-full" />
            </div>
          </div>
        </div>

        {/* Track Rows */}
        {TRACKS.map((tr) => (
          <div key={tr.id} className="flex border-t border-white/[0.04]" style={{ height: ROW_H }}>
            {/* Track Label */}
            <div className="w-[88px] shrink-0 border-r border-white/[0.06] px-2.5 flex items-center gap-1.5 bg-white/[0.02]">
              <tr.icon className="h-3 w-3" style={{ color: tr.color }} />
              <span className="text-[10px] font-medium text-white/40 truncate">{tr.label}</span>
            </div>

            {/* Track Lane */}
            <div className="flex-1 relative overflow-hidden"
              style={tr.id !== "video" ? { background: `linear-gradient(90deg, ${tr.color}04, ${tr.color}08)` } : { background: "#0a0a0e" }}
              onClick={(e) => { if (!(e.target as HTMLElement).closest("[data-seg]")) seekFromMouseEvent(e) }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                if (tr.id === "video") return
                e.preventDefault()
                try {
                  const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                  if (d.type === "audio") {
                    const trackMap: Record<string, string> = { voice: "voice", sfx: "sfx", music: "music" }
                    const trackId = trackMap[tr.id] || "music"
                    const cue = addAudioCue({ track: trackId, start: 0, duration: 5, label: d.name, audioUrl: d.url, audioB64: d.url, volume: tr.id === "music" ? 0.4 : 1.0, waveformPeaks: null, sourceStepId: null })
                    decodeWaveform(d.url, cue.id)
                    toast("info", `Added ${d.name} to ${tr.label}`)
                  }
                } catch {}
              }}>
              {tr.id === "video" ? segments.map((seg) => {
                const left = seg.start * pps
                const width = Math.max(seg.duration * pps, 8)
                const isSelected = seg.id === selectedId
                const segColor = TRACKS[0].color
                return (
                  <div key={seg.id} data-seg={seg.id}
                    className={`absolute top-[3px] bottom-[3px] rounded-md cursor-pointer overflow-hidden transition-all duration-100 ${isSelected
                      ? "ring-2 ring-[#6366f1]/60 shadow-lg shadow-[#6366f1]/20 border border-[#6366f1]/80"
                      : "border border-white/[0.08] hover:border-white/[0.2]"
                    }`}
                    style={{
                      left, width,
                      background: isSelected
                        ? `linear-gradient(135deg, ${segColor}30, ${segColor}18)`
                        : `linear-gradient(135deg, ${segColor}18, ${segColor}0a)`,
                    }}
                    onClick={(e) => { e.stopPropagation(); setSelected(seg.id) }}>

                    {/* Thumbnail background */}
                    {seg.thumbnailUrl && (
                      <img src={seg.thumbnailUrl} alt="" className="absolute inset-0 w-full h-full object-cover opacity-[0.12] pointer-events-none" />
                    )}

                    {/* Left resize handle */}
                    <div className="absolute top-0 left-0 bottom-0 z-20 cursor-ew-resize group" style={{ width: HANDLE_W }}
                      onPointerDown={(e) => onSegmentPointerDown(e, seg.id, "resize-left")}>
                      <div className="absolute left-[3px] top-1/2 -translate-y-1/2 w-[2px] h-5 rounded-full bg-transparent group-hover:bg-white/40 transition-colors" />
                    </div>

                    {/* Right resize handle */}
                    <div className="absolute top-0 right-0 bottom-0 z-20 cursor-ew-resize group" style={{ width: HANDLE_W }}
                      onPointerDown={(e) => onSegmentPointerDown(e, seg.id, "resize-right")}>
                      <div className="absolute right-[3px] top-1/2 -translate-y-1/2 w-[2px] h-5 rounded-full bg-transparent group-hover:bg-white/40 transition-colors" />
                    </div>

                    {/* Segment body (drag area) */}
                    <div className="absolute inset-0 z-10 cursor-grab active:cursor-grabbing flex items-center px-3 gap-1.5"
                      onPointerDown={(e) => onSegmentPointerDown(e, seg.id, "move")}>
                      <span className="relative z-10 text-[10px] font-bold tracking-wide text-white/70 truncate">
                        {segLabel(seg.order)}
                      </span>
                      {seg.status === "ready" && <div className="relative z-10 h-[6px] w-[6px] rounded-full bg-emerald-400 shrink-0 shadow-sm shadow-emerald-400/50" />}
                      {seg.status === "generating" && <Loader2 className="relative z-10 h-3 w-3 animate-spin shrink-0 text-white/50" />}
                      {seg.status === "failed" && <div className="relative z-10 h-[6px] w-[6px] rounded-full bg-red-400 shrink-0 shadow-sm shadow-red-400/50" />}
                      {seg.prompt && width > 100 && (
                        <span className="relative z-10 text-[9px] text-white/30 truncate ml-1 hidden sm:block">{seg.prompt.slice(0, 30)}</span>
                      )}
                    </div>

                    {/* Subtle gradient overlay for depth */}
                    <div className="absolute inset-0 pointer-events-none rounded-md"
                      style={{ background: "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 40%, rgba(0,0,0,0.1) 100%)" }} />
                  </div>
                )
              }) : (
                /* Audio cues with real waveform */
                audioCues.filter((c) => c.track === tr.id).map((cue) => {
                  const cueWidth = Math.max(cue.duration * pps, 4)
                  return (
                    <div key={cue.id} className="absolute top-[4px] bottom-[4px] rounded-md border overflow-hidden"
                      style={{
                        left: cue.start * pps,
                        width: cueWidth,
                        borderColor: `${tr.color}30`,
                        background: `linear-gradient(135deg, ${tr.color}15, ${tr.color}08)`,
                      }}>
                      {/* Waveform */}
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" preserveAspectRatio="none" viewBox={`0 0 ${cue.waveformPeaks?.length || 50} 100`}>
                        {cue.waveformPeaks && cue.waveformPeaks.length > 0
                          ? cue.waveformPeaks.map((peak, i) => (
                              <rect key={i} x={i} y={50 - peak * 45} width={1} height={peak * 90} rx={0.5}
                                fill={tr.color} opacity={0.5} />
                            ))
                          : Array.from({ length: Math.min(50, Math.max(5, Math.floor(cueWidth / 4))) }).map((_, i) => (
                              <rect key={i} x={i} y={50 - (0.15 + Math.sin(i * 0.7) * 0.25) * 45} width={1}
                                height={(0.15 + Math.sin(i * 0.7) * 0.25) * 90} rx={0.5}
                                fill={tr.color} opacity={0.35} />
                            ))
                        }
                      </svg>
                      <span className="relative text-[9px] truncate px-2 leading-[44px] font-medium" style={{ color: `${tr.color}cc` }}>{cue.label}</span>
                    </div>
                  )
                })
              )}
            </div>
          </div>
        ))}

        {/* Add Keyframe Bar */}
        <div className="flex border-t border-white/[0.06]">
          <div className="w-[88px] shrink-0 border-r border-white/[0.06]" />
          <div className="flex-1 px-1.5 py-1">
            <Button variant="ghost" size="sm"
              className="h-7 w-full text-[11px] gap-1.5 text-white/30 hover:text-white/60 hover:bg-white/[0.04] rounded-md border border-dashed border-white/[0.08] hover:border-white/[0.15]"
              onClick={() => { const s = addSegment({ duration: 5, status: "empty" }); setSelected(s.id) }}>
              <Plus className="h-3 w-3" /> Add Keyframe
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function CurrentPreview({ segments, time }: { segments: ReturnType<typeof useTimelineStore.getState>["segments"]; time: number }) {
  const seg = segments.find((s) => time >= s.start && time < s.start + s.duration)
  if (!seg) {
    if (segments.length > 0) {
      const closest = segments.reduce((prev, curr) =>
        Math.abs(curr.start - time) < Math.abs(prev.start - time) ? curr : prev
      )
      return (
        <div className="flex flex-col items-center gap-3">
          {closest.videoUrl ? (
            <video key={closest.id} src={closest.videoUrl} className="max-w-full max-h-[60vh] rounded-lg shadow-2xl" autoPlay loop muted />
          ) : closest.firstFrameB64 ? (
            <img src={closest.firstFrameB64} alt="" className="max-w-full max-h-[60vh] rounded-lg object-contain shadow-2xl" />
          ) : (
            <div className="w-64 h-36 rounded-xl bg-white/5 flex items-center justify-center text-xs text-white/20 border border-white/[0.06]">Empty</div>
          )}
          <span className="text-[11px] text-white/30 font-mono">{segLabel(closest.order)} — {closest.prompt?.slice(0, 80) || "no prompt"}</span>
        </div>
      )
    }
    return <p className="text-sm text-white/30">No segment at this time</p>
  }

  if (seg.videoUrl) {
    return (
      <div className="flex flex-col items-center gap-3">
        <video key={seg.id} src={seg.videoUrl} className="max-w-full max-h-[70vh] rounded-lg shadow-2xl" autoPlay loop muted />
        <span className="text-[11px] text-white/30 font-mono">{segLabel(seg.order)} — {seg.prompt?.slice(0, 80) || "no prompt"}</span>
      </div>
    )
  }
  if (seg.firstFrameB64) {
    return (
      <div className="flex flex-col items-center gap-3">
        <img src={seg.firstFrameB64} alt="" className="max-w-full max-h-[70vh] rounded-lg object-contain shadow-2xl" />
        <span className="text-[11px] text-white/30 font-mono">{segLabel(seg.order)} — {seg.prompt?.slice(0, 80)}</span>
      </div>
    )
  }
  return (
    <div className="flex flex-col items-center gap-2 opacity-50">
      <FilmIcon className="h-8 w-8 text-white/20" />
      <p className="text-sm text-white/30">Empty segment — add a prompt and generate</p>
    </div>
  )
}

/** Collapsible inspector section with icon header */
function InspectorSection({ title, icon, children, defaultOpen = true }: { title: string; icon: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-white/40 hover:text-white/60 hover:bg-white/[0.02] transition-colors"
        onClick={() => setOpen(!open)}>
        {icon}
        <span className="flex-1 text-left">{title}</span>
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>
      {open && <div className="px-3 pb-3 pt-1">{children}</div>}
    </div>
  )
}

/** Field wrapper for inspector fields */
function InspectorField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">{label}</Label>
      {children}
    </div>
  )
}
