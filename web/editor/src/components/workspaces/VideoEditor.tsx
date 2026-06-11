import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { LoraPicker } from "@/components/LoraPicker"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { useToastStore } from "@/stores/toast"
import { useTimelineStore } from "@/stores/timeline"
import { useAssetStore } from "@/stores/assets"
import { callTool } from "@/mcp"
import { VIDEO_MODELS, CONTROL_VIDEO_MODES, AUDIO_PROMPT_MODES, MASK_MODES } from "@/types/timeline"
import type { TimelineSegment } from "@/types/timeline"
import {
  Play, Pause, SkipBack, Plus, Trash2,
  Music, Mic, Volume2, Film, Settings, Wand2, Loader2,
  ZoomIn, ZoomOut, ChevronDown, ChevronRight, Dice5,
  Sparkles, ImagePlus, FilmIcon, ToggleLeft, ToggleRight,
  Headphones, Layers, Sliders,
} from "lucide-react"

// Dynamic track list: video + user-added audio tracks

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

interface CueDragInfo {
  mode: DragMode
  cueId: string
  startMouseX: number
  startCueLeft: number
  startCueWidth: number
}

// ── Compile timeline into backend payload ─────────────────────────────────────
function buildPayload(seg: TimelineSegment, _allSegments?: TimelineSegment[]) {
  const p = seg.params
  return {
    service: "wan2gp",
    params: {
      model: p.model,
      image_b64: seg.firstFrameB64 || undefined,
      image_end_b64: seg.lastFrameB64 || undefined,
      input_prompt: seg.prompt || "animate",
      n_prompt: seg.negativePrompt || undefined,
      seed: p.seed,
      frame_num: p.frames,
      fps: p.fps,
      width: p.width,
      height: p.height,
      guide_scale: p.guideScale,
      sampling_steps: p.samplingSteps,
      guide_phases: p.guidePhases || undefined,
      input_video_strength: p.inputVideoStrength !== 1.0 ? p.inputVideoStrength : undefined,
      perturbation_switch: p.perturbationSwitch || undefined,
      perturbation_layers: p.perturbationSwitch ? p.perturbationLayers : undefined,
      perturbation_start: p.perturbationSwitch ? p.perturbationStartPerc / 100 : undefined,
      perturbation_end: p.perturbationSwitch ? p.perturbationEndPerc / 100 : undefined,
      apg_switch: p.apgSwitch ? 1 : undefined,
      cfg_star_switch: p.cfgStarSwitch ? 1 : undefined,
      alt_guide_scale: p.altGuideScale !== 1.0 ? p.altGuideScale : undefined,
      alt_scale: p.altScale !== 0.0 ? p.altScale : undefined,
      audio_guidance_scale: p.audioGuideScale !== 1.0 ? p.audioGuideScale : undefined,
      audio_cfg_scale: p.audioCfgScale !== 1.0 ? p.audioCfgScale : undefined,
      NAG_scale: p.nagScale !== 1.0 ? p.nagScale : undefined,
      NAG_tau: p.nagTau !== 3.5 ? p.nagTau : undefined,
      NAG_alpha: p.nagAlpha !== 0.5 ? p.nagAlpha : undefined,
      sample_solver: p.sampleSolver !== 'euler' ? p.sampleSolver : undefined,
      self_refiner_setting: p.selfRefinerSetting || undefined,
      self_refiner_plan: p.selfRefinerPlan || undefined,
      self_refiner_f_uncertainty: p.selfRefinerSetting ? p.selfRefinerFUncertainty : undefined,
      self_refiner_certain_percentage: p.selfRefinerSetting ? p.selfRefinerCertainPercentage : undefined,
      video_prompt_type: p.videoPromptType || undefined,
      denoising_strength: p.videoPromptType ? p.denoisingStrength : (p.denoisingStrength !== 1.0 ? p.denoisingStrength : undefined),
      masking_strength: p.maskingStrength !== 0.0 ? p.maskingStrength : undefined,
      masking_source: p.maskingSource || undefined,
      audio_prompt_type: p.audioPromptType || undefined,
      loras_selected: p.loras || undefined,
      enhance_prompt: p.enhancePrompt ? "true" : undefined,
      ...(p.slidingWindow ? {
        sliding_window_size: p.slidingWindowSize,
        sliding_window_overlap: p.slidingWindowOverlap,
      } : {}),
    }
  }
}

export function VideoEditor() {
  const segments = useTimelineStore((s) => s.segments)
  const audioCues = useTimelineStore((s) => s.audioCues)
  const audioTracks = useTimelineStore((s) => s.audioTracks)
  const addAudioTrack = useTimelineStore((s) => s.addAudioTrack)
  const removeAudioTrack = useTimelineStore((s) => s.removeAudioTrack)
  const addSegment = useTimelineStore((s) => s.addSegment)
  const removeSegment = useTimelineStore((s) => s.removeSegment)
  const updateSegment = useTimelineStore((s) => s.updateSegment)
  const selectedId = useTimelineStore((s) => s.selectedSegmentId)
  const setSelected = useTimelineStore((s) => s.setSelectedSegment)
  const selectedAudioCueId = useTimelineStore((s) => s.selectedAudioCueId)
  const setSelectedAudioCue = useTimelineStore((s) => s.setSelectedAudioCue)
  const removeAudioCue = useTimelineStore((s) => s.removeAudioCue)
  const updateAudioCue = useTimelineStore((s) => s.updateAudioCue)
  const addAudioCue = useTimelineStore((s) => s.addAudioCue)
  const playback = useTimelineStore((s) => s.playback)
  const setPlayback = useTimelineStore((s) => s.setPlayback)
  const relayVideoUrl = useTimelineStore((s) => s.relayVideoUrl)
  const relaySegmentIds = useTimelineStore((s) => s.relaySegmentIds)
  const setRelayVideo = useTimelineStore((s) => s.setRelayVideo)
  const toast = useToastStore((s) => s.addToast)
  const assets = useAssetStore((s) => s.assets)

  const [generating, setGenerating] = useState(false)
  const [generatingSegId, setGeneratingSegId] = useState<string | null>(null)
  const [pps, setPps] = useState(80)
  const [sidebarW, setSidebarW] = useState(300)
  const sidebarDragRef = useRef<{ startX: number; startW: number } | null>(null)
  const raf = useRef(0)
  const videoRef = useRef<HTMLVideoElement>(null)

  const sel = segments.find((s) => s.id === selectedId)
  const selCue = audioCues.find((c) => c.id === selectedAudioCueId)
  const total = Math.max(segments.reduce((t, s) => Math.max(t, s.start + s.duration), 0), 5)
  const isLtx = !!sel?.params.model && (sel.params.model.startsWith("ltx") || sel.params.model === "ltx2" || sel.params.model === "ltx2_19B" || sel.params.model === "ltxv_098_13b")
  const isWan = sel?.params.model.startsWith("wan") ?? false

  // ── Auto-create a segment if none exist so inspector is always visible ──
  useEffect(() => {
    if (segments.length === 0) {
      const s = addSegment({ duration: 5, status: "empty" })
      setSelected(s.id)
    }
  }, [segments.length])

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

  const seekFromClientX = useCallback((clientX: number, containerEl: HTMLElement) => {
    const rect = containerEl.getBoundingClientRect()
    const relX = clientX - rect.left
    const t = Math.max(0, Math.min(total, relX / pps))
    setPlayback({ currentTime: t })
  }, [pps, total, setPlayback])

  const seekDragRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = seekDragRef.current
      if (!el) return
      seekFromClientX(e.clientX, el)
    }
    const onUp = () => { seekDragRef.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp) }
  }, [seekFromClientX])

  // ── Drag / Resize ──────────────────────────────────────────────────────────
  const dragRef = useRef<DragInfo | null>(null)
  const cueDragRef = useRef<CueDragInfo | null>(null)

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

  const onCuePointerDown = useCallback((
    e: React.PointerEvent<HTMLDivElement>,
    cueId: string,
    mode: DragMode,
  ) => {
    if (e.button !== 0) return
    e.stopPropagation()
    e.preventDefault()

    const cue = useTimelineStore.getState().audioCues.find(c => c.id === cueId)
    if (!cue) return

    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)

    cueDragRef.current = {
      mode,
      cueId,
      startMouseX: e.clientX,
      startCueLeft: cue.start * pps,
      startCueWidth: cue.duration * pps,
    }

    setSelectedAudioCue(cueId)
  }, [pps, setSelectedAudioCue])

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
    const onCueMove = (e: PointerEvent) => {
      const drag = cueDragRef.current
      if (!drag) return
      const dx = e.clientX - drag.startMouseX
      const minDurPx = 0.5 * pps

      const cue = useTimelineStore.getState().audioCues.find(c => c.id === drag.cueId)
      if (!cue) return

      if (drag.mode === "move") {
        let newStart = drag.startCueLeft / pps + dx / pps
        newStart = Math.max(0, newStart)
        updateAudioCue(drag.cueId, { start: newStart })
      } else if (drag.mode === "resize-left") {
        let newLeft = drag.startCueLeft + dx
        let newWidth = drag.startCueWidth - dx
        if (newWidth < minDurPx) { newWidth = minDurPx; newLeft = drag.startCueLeft + drag.startCueWidth - minDurPx }
        if (newLeft < 0) { newLeft = 0; newWidth = drag.startCueLeft + drag.startCueWidth }
        updateAudioCue(drag.cueId, { start: newLeft / pps, duration: newWidth / pps })
      } else if (drag.mode === "resize-right") {
        let newWidth = drag.startCueWidth + dx
        if (newWidth < minDurPx) newWidth = minDurPx
        updateAudioCue(drag.cueId, { duration: newWidth / pps })
      }
    }
    const onUp = () => { dragRef.current = null; cueDragRef.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointermove", onCueMove)
    window.addEventListener("pointerup", onUp)
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointermove", onCueMove); window.removeEventListener("pointerup", onUp) }
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
        // Auto-create an audio track if none exist, add cue at start
        let targetTrack = audioTracks[0]
        if (!targetTrack) targetTrack = addAudioTrack("Audio 1")
        addAudioCue({ track: targetTrack.id, start: 0, duration: 5, label: d.name, audioUrl: d.url, audioB64: d.url, volume: 0.8, waveformPeaks: null, sourceStepId: null })
        toast("info", `Added audio: ${d.name} — drag onto a track for precise placement`)
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
    setGeneratingSegId(seg.id)
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
        payload.params.audio_b64 = firstAudio.audioB64
        payload.params.audio_scale = String(firstAudio.volume)
        payload.params.audio_prompt_type = "A"
      }

      const r = await callTool<{ status: string; data?: string; media_type?: string; error?: string }>(
        "run",
        payload
      )
      if (r.status === "ok" && r.data) {
        updateSegment(seg.id, { videoUrl: `data:video/mp4;base64,${r.data}`, status: "ready", error: null })
        toast("success", `${segLabel(seg.order)} generated!`)
      } else {
        const errMsg = r.error || "Generation failed"
        updateSegment(seg.id, { status: "failed", error: errMsg })
        toast("error", errMsg)
      }
    } catch (e) {
      const errMsg = String(e)
      updateSegment(seg.id, { status: "failed", error: errMsg })
      toast("error", errMsg)
    } finally {
      setGenerating(false)
      setGeneratingSegId(null)
    }
  }

  // ── Generate all — uses Director prompt relay for multi-segment LTX ──────
  const genAll = async () => {
    if (segments.length === 0) return
    setGenerating(true)

    const eligible = segments.filter(s => s.status === "empty" || s.status === "failed" || s.status === "generating")
    if (eligible.length === 0) { setGenerating(false); return }

    // ── LTX relay path: all LTX segments → single Director relay call ──
    const ltxSegs = eligible.filter(s => s.params.model.startsWith("ltx"))
    if (ltxSegs.length >= 1) {
      const fps = ltxSegs[0].params.fps
      const totalFrames = ltxSegs.reduce((t, s) => t + s.params.frames, 0)
      const firstSeg = ltxSegs[0]
      const lastSeg = ltxSegs[ltxSegs.length - 1]

      ltxSegs.forEach(s => updateSegment(s.id, { status: "generating" }))

      try {
        // Find audio cues overlapping the relay span for audio conditioning
        const relayStart = Math.min(...ltxSegs.map(s => s.start))
        const relayEnd = Math.max(...ltxSegs.map(s => s.start + s.duration))
        const firstAudio = audioCues.find(c =>
          c.audioB64 && c.start < relayEnd && c.start + c.duration > relayStart
        )

        // Build _relay_config — arrays, not strings
        const relayConfig = {
          global_prompt: firstSeg.prompt || "animate",
          local_prompts: ltxSegs.map(s => s.prompt || "animate"),
          segment_lengths: ltxSegs.map(s => s.params.frames),
          epsilon: firstSeg.params.epsilon,
        }

        const r = await callTool<{ status: string; data?: string; media_type?: string; error?: string }>("run", {
          service: "wan2gp",
          params: {
            model: firstSeg.params.model,
            input_prompt: firstSeg.prompt || "animate",
            _relay_config: JSON.stringify(relayConfig),
            image_b64: firstSeg.firstFrameB64 || undefined,
            image_end_b64: lastSeg.lastFrameB64 || undefined,
            n_prompt: firstSeg.negativePrompt || undefined,
            seed: firstSeg.params.seed,
            fps,
            frame_num: totalFrames,
            sampling_steps: firstSeg.params.samplingSteps,
            guide_scale: firstSeg.params.guideScale,
            width: firstSeg.params.width,
            height: firstSeg.params.height,
            audio_b64: firstAudio?.audioB64 || undefined,
            audio_scale: firstAudio ? String(firstAudio.volume) : undefined,
            audio_prompt_type: firstAudio ? "A" : undefined,
            loras_selected: firstSeg.params.loras || undefined,
            guide_phases: firstSeg.params.guidePhases,
          }
        })

        if (r.status === "ok" && r.data) {
          const videoUrl = `data:video/mp4;base64,${r.data}`
          // Store the single relay video at timeline level
          const segIds = ltxSegs.map(s => s.id)
          setRelayVideo(videoUrl, segIds)
          // Mark all segments ready
          ltxSegs.forEach(s => updateSegment(s.id, { status: "ready", error: null }))
          toast("success", `Director relay: ${ltxSegs.length} segments, ${totalFrames} frames`)
        } else {
          const errMsg = r.error || "Director relay failed"
          ltxSegs.forEach(s => updateSegment(s.id, { status: "failed", error: errMsg }))
          toast("error", errMsg)
        }
      } catch (e) {
        const errMsg = String(e)
        ltxSegs.forEach(s => updateSegment(s.id, { status: "failed", error: errMsg }))
        toast("error", errMsg)
      }
    }

    // ── Non-LTX segments: generate individually ──
    const nonLtx = eligible.filter(s => !s.params.model.startsWith("ltx"))
    for (const seg of nonLtx) {
      await generateSegment(seg)
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

  // ── Sidebar resize ──────────────────────────────────────────────────────────
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const drag = sidebarDragRef.current
      if (!drag) return
      const dx = drag.startX - e.clientX
      const newW = Math.max(220, Math.min(600, drag.startW + dx))
      setSidebarW(newW)
    }
    const onUp = () => { sidebarDragRef.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp) }
  }, [])

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
              <CurrentPreview segments={segments} time={playback.currentTime} relayVideoUrl={relayVideoUrl} relaySegmentIds={relaySegmentIds} />
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
        {(
          <div className="shrink-0 border-l border-white/[0.06] overflow-y-auto bg-[#111114] scrollbar-thin relative"
            style={{ width: sidebarW }}>
            {/* Sidebar resize handle */}
            <div className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-30 group"
              onPointerDown={(e) => {
                if (e.button !== 0) return
                e.preventDefault()
                ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
                sidebarDragRef.current = { startX: e.clientX, startW: sidebarW }
              }}>
              <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-transparent group-hover:bg-white/20 group-active:bg-[#6366f1]/50 transition-colors"
                style={{ marginLeft: "-1px" }} />
            </div>
            {!sel ? (
              <div className="p-6 text-center space-y-3">
                <div className="mx-auto w-12 h-12 rounded-xl border border-white/10 bg-white/5 flex items-center justify-center">
                  <Plus className="h-5 w-5 text-white/30" />
                </div>
                <p className="text-xs text-white/40">Click a segment on the timeline</p>
                <Button variant="outline" size="sm" className="text-xs"
                  onClick={() => { const s = addSegment({ duration: 5, status: "empty" }); setSelected(s.id) }}>
                  <Plus className="h-3 w-3 mr-1" /> Add Keyframe
                </Button>
              </div>
            ) : (
            <>
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
              {/* Error display */}
              {sel.status === "failed" && sel.error && (
                <div className="rounded-md bg-red-500/10 border border-red-500/30 px-3 py-2 text-[11px] text-red-300 font-mono break-all">
                  {sel.error}
                </div>
              )}
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
                  {isLtx && (
                    <div className="flex items-center justify-between">
                      <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Auto-Enhance Prompt</Label>
                      <button className="text-white/40 hover:text-white/70"
                        onClick={() => updateSegment(sel.id, { params: { ...sel.params, enhancePrompt: !sel.params.enhancePrompt } })}>
                        {sel.params.enhancePrompt ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                      </button>
                    </div>
                  )}
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
                    <Input type="number" value={Number(sel.duration.toFixed(2))} step={0.5} min={0.5}
                      onChange={(e) => {
                        const secs = Math.max(0.5, Number(e.target.value))
                        const newFrames = Math.round(secs * sel.params.fps)
                        updateSegment(sel.id, { duration: secs, params: { ...sel.params, frames: newFrames } })
                      }}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
              </InspectorSection>

              {/* ── Resolution & Frame Rate ── */}
              <InspectorSection title="Resolution & Frames" icon={<ImagePlus className="h-3 w-3" />}>
                <InspectorField label="Video Length (seconds)">
                  <Input type="number" value={Number((sel.params.frames / sel.params.fps).toFixed(2))} step={0.5} min={0.5} max={30}
                    onChange={(e) => {
                      const secs = Math.max(0.5, Number(e.target.value))
                      const newFrames = Math.round(secs * sel.params.fps)
                      updateSegment(sel.id, {
                        duration: secs,
                        params: { ...sel.params, frames: newFrames },
                      })
                    }}
                    className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md font-mono" />
                </InspectorField>
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
                      onChange={(e) => {
                        const newFps = Math.max(8, Number(e.target.value))
                        // Keep video length constant: recompute frames from current duration
                        const currentSecs = sel.params.frames / sel.params.fps
                        const newFrames = Math.round(currentSecs * newFps)
                        updateSegment(sel.id, { params: { ...sel.params, fps: newFps, frames: newFrames } })
                      }}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Frames">
                    <Input type="number" value={sel.params.frames} min={9} max={201}
                      onChange={(e) => {
                        const newFrames = Math.max(9, Number(e.target.value))
                        // Keep FPS constant: recompute duration from frames
                        const newSecs = newFrames / sel.params.fps
                        updateSegment(sel.id, {
                          duration: newSecs,
                          params: { ...sel.params, frames: newFrames },
                        })
                      }}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
                <InspectorField label="Resize Method">
                  <Select value={sel.params.resizeMethod} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, resizeMethod: v as 'stretch' | 'fit' | 'crop' | 'pad' } })}>
                    <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fit">Fit (letterbox)</SelectItem>
                      <SelectItem value="crop">Crop (fill)</SelectItem>
                      <SelectItem value="stretch">Stretch</SelectItem>
                      <SelectItem value="pad">Pad (maintain AR)</SelectItem>
                    </SelectContent>
                  </Select>
                </InspectorField>
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

              {/* ── Start / End Images ── */}
              <InspectorSection title="Start & End Images" icon={<ImagePlus className="h-3 w-3" />}>
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
                  {/* Start image strength */}
                  {isLtx && sel.firstFrameB64 && (
                    <InspectorField label="Start Image Strength">
                      <div className="flex items-center gap-2">
                        <input type="range" min={0} max={1} step={0.05} value={sel.params.inputVideoStrength}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, inputVideoStrength: Number(e.target.value) } })}
                          className="flex-1 h-1 appearance-none bg-white/10 rounded-full accent-[#6366f1] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white/70" />
                        <span className="text-[10px] font-mono text-white/40 w-8">{sel.params.inputVideoStrength.toFixed(2)}</span>
                      </div>
                      <div className="text-[9px] text-white/15 mt-0.5">Lower = more motion freedom</div>
                    </InspectorField>
                  )}
                </div>
              </InspectorSection>

              {/* ── LoRA ── */}
              <InspectorSection title="LoRA" icon={<Layers className="h-3 w-3" />}>
                {isLtx && (
                  <div className="mb-2 flex items-center justify-between">
                    <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Distilled Mode (8 steps)</Label>
                    <button className="text-white/40 hover:text-white/70"
                      onClick={() => updateSegment(sel.id, {
                        params: {
                          ...sel.params,
                          distilledMode: !sel.params.distilledMode,
                          samplingSteps: !sel.params.distilledMode ? 8 : (sel.params.model === 'ltx2' ? 30 : 40),
                        }
                      })}>
                      {sel.params.distilledMode
                        ? <ToggleRight className="h-5 w-5 text-[#6366f1]" />
                        : <ToggleLeft className="h-5 w-5" />}
                    </button>
                  </div>
                )}
                <LoraPicker model={sel.params.model} value={sel.params.loras}
                  onChange={(loras) => updateSegment(sel.id, { params: { ...sel.params, loras } })} />
              </InspectorSection>

              {/* ── Advanced Director Controls (LTX only) ── */}
              {isLtx && (
              <InspectorSection title="Director Controls" icon={<Sliders className="h-3 w-3" />}>
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
              )}

              {/* ── Camera Motion (LTX only) ── */}
              {isLtx && (
              <InspectorSection title="Camera Motion" icon={<Sliders className="h-3 w-3" />}>
                <div className="grid grid-cols-3 gap-2">
                  <InspectorField label="Pan X">
                    <Input type="number" value={sel.params.cameraPanX} step={0.1} min={-1} max={1}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, cameraPanX: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Pan Y">
                    <Input type="number" value={sel.params.cameraPanY} step={0.1} min={-1} max={1}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, cameraPanY: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <InspectorField label="Zoom">
                    <Input type="number" value={sel.params.cameraZoom} step={0.1} min={0.5} max={3}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, cameraZoom: Number(e.target.value) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                </div>
              </InspectorSection>
              )}

              {/* ── Guidance (LTX only) ── */}
              {isLtx && (
              <InspectorSection title="Guidance" icon={<Sliders className="h-3 w-3" />} defaultOpen={false}>
                {sel.params.distilledMode ? (
                  // NAG — distilled only
                  <div className="space-y-2">
                    <div className="text-[9px] text-white/25">Negative Attention Guidance (NAG)</div>
                    <div className="grid grid-cols-3 gap-2">
                      <InspectorField label="NAG Scale">
                        <Input type="number" value={sel.params.nagScale} step={0.1} min={0} max={10}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, nagScale: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                      <InspectorField label="NAG Tau">
                        <Input type="number" value={sel.params.nagTau} step={0.5} min={0} max={10}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, nagTau: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                      <InspectorField label="NAG Alpha">
                        <Input type="number" value={sel.params.nagAlpha} step={0.1} min={0} max={1}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, nagAlpha: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                    </div>
                  </div>
                ) : (
                  // Dev guidance modes
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">APG</Label>
                        <button className="text-white/40 hover:text-white/70"
                          onClick={() => updateSegment(sel.id, { params: { ...sel.params, apgSwitch: !sel.params.apgSwitch, cfgStarSwitch: sel.params.apgSwitch ? sel.params.cfgStarSwitch : false } })}>
                          {sel.params.apgSwitch ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                        </button>
                      </div>
                      <div className="flex items-center justify-between">
                        <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">CFG Star</Label>
                        <button className="text-white/40 hover:text-white/70"
                          onClick={() => updateSegment(sel.id, { params: { ...sel.params, cfgStarSwitch: !sel.params.cfgStarSwitch, apgSwitch: sel.params.cfgStarSwitch ? sel.params.apgSwitch : false } })}>
                          {sel.params.cfgStarSwitch ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <InspectorField label="Alt Guide Scale">
                        <Input type="number" value={sel.params.altGuideScale} step={0.5} min={0} max={20}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, altGuideScale: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                      <InspectorField label="Alt Rescale">
                        <Input type="number" value={sel.params.altScale} step={0.1} min={0} max={1}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, altScale: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <InspectorField label="Audio Guide">
                        <Input type="number" value={sel.params.audioGuideScale} step={0.5} min={0} max={20}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, audioGuideScale: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                      <InspectorField label="Audio CFG">
                        <Input type="number" value={sel.params.audioCfgScale} step={0.5} min={0} max={20}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, audioCfgScale: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                    </div>
                    {sel.params.model === 'ltx2' && (
                      <InspectorField label="Sample Solver">
                        <Select value={sel.params.sampleSolver} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, sampleSolver: v } })}>
                          <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="euler">Euler (standard)</SelectItem>
                            <SelectItem value="res2s">HQ Res2s (slower, higher quality)</SelectItem>
                          </SelectContent>
                        </Select>
                      </InspectorField>
                    )}
                  </div>
                )}
              </InspectorSection>
              )}

              {/* ── Perturbation Detail (LTX dev only, when enabled) ── */}
              {isLtx && !sel.params.distilledMode && sel.params.perturbationSwitch > 0 && (
              <InspectorSection title="Perturbation Detail" icon={<Sliders className="h-3 w-3" />} defaultOpen={false}>
                <div className="space-y-2">
                  <InspectorField label="Layers (comma-separated)">
                    <Input type="text" value={sel.params.perturbationLayers.join(",")}
                      onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, perturbationLayers: e.target.value.split(",").map(Number).filter(n => !isNaN(n)) } })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md font-mono" />
                  </InspectorField>
                  <div className="grid grid-cols-2 gap-2">
                    <InspectorField label="Start %">
                      <Input type="number" value={sel.params.perturbationStartPerc} min={0} max={100}
                        onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, perturbationStartPerc: Number(e.target.value) } })}
                        className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                    </InspectorField>
                    <InspectorField label="End %">
                      <Input type="number" value={sel.params.perturbationEndPerc} min={0} max={100}
                        onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, perturbationEndPerc: Number(e.target.value) } })}
                        className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                    </InspectorField>
                  </div>
                </div>
              </InspectorSection>
              )}

              {/* ── Self-Refiner (LTX only) ── */}
              {isLtx && (
              <InspectorSection title="Self-Refiner" icon={<Sliders className="h-3 w-3" />} defaultOpen={false}>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Enable Refiner</Label>
                    <button className="text-white/40 hover:text-white/70"
                      onClick={() => updateSegment(sel.id, { params: { ...sel.params, selfRefinerSetting: sel.params.selfRefinerSetting ? 0 : 1 } })}>
                      {sel.params.selfRefinerSetting ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                    </button>
                  </div>
                  {sel.params.selfRefinerSetting > 0 && (
                    <>
                      <InspectorField label="Plan (e.g. 2-8:3)">
                        <Input type="text" value={sel.params.selfRefinerPlan}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, selfRefinerPlan: e.target.value } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md font-mono"
                          placeholder="2-8:3,10-14:2" />
                      </InspectorField>
                      <div className="grid grid-cols-2 gap-2">
                        <InspectorField label="Uncertainty">
                          <Input type="number" value={sel.params.selfRefinerFUncertainty} step={0.01} min={0} max={1}
                            onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, selfRefinerFUncertainty: Number(e.target.value) } })}
                            className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                        </InspectorField>
                        <InspectorField label="Certainty %">
                          <Input type="number" value={sel.params.selfRefinerCertainPercentage} step={0.001} min={0} max={1}
                            onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, selfRefinerCertainPercentage: Number(e.target.value) } })}
                            className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                        </InspectorField>
                      </div>
                    </>
                  )}
                </div>
              </InspectorSection>
              )}

              {/* ── Sliding Window (LTX only) ── */}
              {isLtx && (
              <InspectorSection title="Sliding Window" icon={<Film className="h-3 w-3" />} defaultOpen={false}>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Enable (long videos)</Label>
                    <button className="text-white/40 hover:text-white/70"
                      onClick={() => updateSegment(sel.id, { params: { ...sel.params, slidingWindow: !sel.params.slidingWindow } })}>
                      {sel.params.slidingWindow ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                    </button>
                  </div>
                  {sel.params.slidingWindow && (
                    <div className="grid grid-cols-2 gap-2">
                      <InspectorField label="Window Size">
                        <Input type="number" value={sel.params.slidingWindowSize} step={8} min={5} max={501}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, slidingWindowSize: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                      <InspectorField label="Overlap">
                        <Input type="number" value={sel.params.slidingWindowOverlap} step={8} min={1} max={97}
                          onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, slidingWindowOverlap: Number(e.target.value) } })}
                          className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                      </InspectorField>
                    </div>
                  )}
                </div>
              </InspectorSection>
              )}

              {/* ── Control Video / IC-LoRA (LTX distilled only) ── */}
              {isLtx && sel.params.distilledMode && (
              <InspectorSection title="Control Video (IC-LoRA)" icon={<Film className="h-3 w-3" />} defaultOpen={false}>
                <div className="space-y-2">
                  <InspectorField label="Control Mode">
                    <Select value={sel.params.videoPromptType} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, videoPromptType: v } })}>
                      <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {CONTROL_VIDEO_MODES.map(m => (
                          <SelectItem key={m.id} value={m.id}>{m.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </InspectorField>
                  {sel.params.videoPromptType && (
                    <>
                      <div className="grid grid-cols-2 gap-2">
                        <InspectorField label="Control Strength">
                          <Input type="number" value={sel.params.denoisingStrength} step={0.1} min={0} max={1}
                            onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, denoisingStrength: Number(e.target.value) } })}
                            className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                        </InspectorField>
                        <InspectorField label="Mask Strength">
                          <Input type="number" value={sel.params.maskingStrength} step={0.1} min={0} max={1}
                            onChange={(e) => updateSegment(sel.id, { params: { ...sel.params, maskingStrength: Number(e.target.value) } })}
                            className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                        </InspectorField>
                      </div>
                      <InspectorField label="Mask Mode">
                        <Select value={sel.params.maskingSource} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, maskingSource: v } })}>
                          <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {MASK_MODES.map(m => (
                              <SelectItem key={m.id} value={m.id}>{m.label}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </InspectorField>
                      {/* Outpainting (22B only) */}
                      {sel.params.model === 'ltx2' && (
                        <div className="flex items-center justify-between">
                          <Label className="text-[9px] font-medium text-white/30 uppercase tracking-wider">Spatial Outpainting</Label>
                          <button className="text-white/40 hover:text-white/70"
                            onClick={() => updateSegment(sel.id, { params: { ...sel.params, outpaintingEnabled: !sel.params.outpaintingEnabled } })}>
                            {sel.params.outpaintingEnabled ? <ToggleRight className="h-5 w-5 text-[#6366f1]" /> : <ToggleLeft className="h-5 w-5" />}
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </InspectorSection>
              )}

              {/* ── Audio Conditioning Mode ── */}
              {isLtx && (
              <InspectorSection title="Audio Mode" icon={<Headphones className="h-3 w-3" />} defaultOpen={false}>
                <div className="space-y-2">
                  <InspectorField label="Conditioning Mode">
                    <Select value={sel.params.audioPromptType} onValueChange={(v) => updateSegment(sel.id, { params: { ...sel.params, audioPromptType: v } })}>
                      <SelectTrigger className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {AUDIO_PROMPT_MODES.filter(m => {
                          if (m.id === 'A1OF' && sel.params.distilledMode) return false
                          if ((m.id === 'K' || m.id === '2') && !sel.params.distilledMode) return false
                          return true
                        }).map(m => (
                          <SelectItem key={m.id} value={m.id}>{m.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </InspectorField>
                  {sel.params.audioPromptType === 'A1OF' && (
                    <div className="text-[9px] text-[#6366f1]/60 bg-[#6366f1]/5 rounded px-2 py-1.5">
                      Select an ID-LoRA from the LoRA section above for identity preservation
                    </div>
                  )}
                </div>
              </InspectorSection>
              )}

              {/* ── Audio Conditioning (auto-detected) ── */}
              {(() => {
                const overlapping = audioCues.filter(c =>
                  c.start < sel.start + sel.duration && c.start + c.duration > sel.start
                )
                if (overlapping.length === 0) return null
                return (
                  <div className="rounded-lg border border-[#4ade80]/20 bg-[#4ade80]/5 px-3 py-2">
                    <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#4ade80]/70">
                      <Headphones className="h-3 w-3" />
                      Audio Conditioning
                    </div>
                    <div className="mt-1.5 space-y-1">
                      {overlapping.map(c => (
                        <div key={c.id} className="flex items-center justify-between text-[10px] text-white/50">
                          <span>{c.label}</span>
                          <span className="text-white/30">{c.track} · vol {c.volume.toFixed(1)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })()}

              {/* Generate This Segment button */}
              <div className="mt-3">
                <Button
                  size="sm"
                  className="w-full h-9 text-xs gap-2 bg-[#6366f1] hover:bg-[#5558e6] text-white rounded-lg font-medium"
                  disabled={generating || (sel.status !== 'empty' && sel.status !== 'failed')}
                  onClick={() => generateSegment(sel)}>
                  {generating && sel.id === generatingSegId ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-3.5 w-3.5" />
                      Generate This Segment
                    </>
                  )}
                </Button>
              </div>
            </div>
            </>
            )}
          </div>
        )}

        {/* ═══ AUDIO CUE INSPECTOR ═══ */}
        {selCue && !sel && (
          <div className="shrink-0 border-l border-white/[0.06] overflow-y-auto bg-[#111114] scrollbar-thin"
            style={{ width: sidebarW }}>
            <div className="sticky top-0 z-10 bg-[#111114] border-b border-white/[0.06] px-4 py-2.5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Headphones className="h-3.5 w-3.5 text-white/50" />
                <span className="text-xs font-bold tracking-wide text-white/90">{selCue.label}</span>
                <Badge className="text-[9px] px-1.5 py-0 font-mono">{selCue.track}</Badge>
              </div>
              <Button variant="ghost" size="icon" className="h-6 w-6 text-white/30 hover:text-red-400 hover:bg-red-500/10"
                onClick={() => { removeAudioCue(selCue.id); setSelectedAudioCue(null) }}>
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
            <div className="p-4 space-y-4">
              <InspectorSection title="Properties" icon={<Music className="h-3 w-3" />}>
                <div className="space-y-2">
                  <InspectorField label="Label">
                    <Input type="text" value={selCue.label}
                      onChange={(e) => updateAudioCue(selCue.id, { label: e.target.value })}
                      className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                  </InspectorField>
                  <div className="grid grid-cols-2 gap-2">
                    <InspectorField label="Start (s)">
                      <Input type="number" value={Number(selCue.start.toFixed(1))} step={0.5} min={0}
                        onChange={(e) => updateAudioCue(selCue.id, { start: Math.max(0, Number(e.target.value)) })}
                        className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                    </InspectorField>
                    <InspectorField label="Duration (s)">
                      <Input type="number" value={Number(selCue.duration.toFixed(1))} step={0.5} min={0.5}
                        onChange={(e) => updateAudioCue(selCue.id, { duration: Math.max(0.5, Number(e.target.value)) })}
                        className="h-7 text-xs bg-white/5 border-white/10 text-white/80 rounded-md" />
                    </InspectorField>
                  </div>
                </div>
              </InspectorSection>
              <InspectorSection title="Volume" icon={<Volume2 className="h-3 w-3" />}>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input type="range" min={0} max={1} step={0.05} value={selCue.volume}
                      onChange={(e) => updateAudioCue(selCue.id, { volume: Number(e.target.value) })}
                      className="flex-1 h-1 appearance-none bg-white/10 rounded-full accent-[#6366f1] [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white/70" />
                    <span className="text-[10px] font-mono text-white/40 w-8">{Math.round(selCue.volume * 100)}%</span>
                  </div>
                </div>
              </InspectorSection>
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
          <span className="text-[9px] text-white/15 hidden md:inline">Space=play · ←→=scrub · Del=remove · +/-=zoom · ⌘Z=undo</span>
          <span className="text-[9px] text-white/15 mx-2 hidden md:inline">|</span>
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
          <div className="flex-1 relative overflow-hidden cursor-pointer"
            onPointerDown={(e) => {
              if (e.button !== 0) return
              seekFromMouseEvent(e)
              seekDragRef.current = e.currentTarget
              ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
            }}>
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
        {/* Video track — always first */}
        <div className="flex border-t border-white/[0.04]" style={{ height: ROW_H }}>
          <div className="w-[88px] shrink-0 border-r border-white/[0.06] px-2.5 flex items-center gap-1.5 bg-white/[0.02]">
            <Film className="h-3 w-3" style={{ color: "#6366f1" }} />
            <span className="text-[10px] font-medium text-white/40 truncate">Video</span>
          </div>
          <div className="flex-1 relative overflow-hidden"
            style={{ background: "#0a0a0e" }}
            onClick={(e) => { if (!(e.target as HTMLElement).closest("[data-seg]")) seekFromMouseEvent(e) }}>
            {segments.map((seg) => {
                const left = seg.start * pps
                const width = Math.max(seg.duration * pps, 8)
                const isSelected = seg.id === selectedId
                const segColor = "#6366f1"
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

                    {/* Thumbnail filmstrip background */}
                    {seg.thumbnailUrl && (
                      <div className="absolute inset-0 overflow-hidden pointer-events-none">
                        <img src={seg.thumbnailUrl} alt="" className="w-full h-full object-cover opacity-25" />
                        {/* Filmstrip perforations */}
                        <div className="absolute left-0 top-0 bottom-0 w-[6px] flex flex-col justify-around py-1">
                          {Array.from({ length: 4 }).map((_, i) => (
                            <div key={i} className="w-[3px] h-[4px] rounded-[1px] bg-white/10 mx-auto" />
                          ))}
                        </div>
                        <div className="absolute right-0 top-0 bottom-0 w-[6px] flex flex-col justify-around py-1">
                          {Array.from({ length: 4 }).map((_, i) => (
                            <div key={i} className="w-[3px] h-[4px] rounded-[1px] bg-white/10 mx-auto" />
                          ))}
                        </div>
                      </div>
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
                      {audioCues.some(c => c.start < seg.start + seg.duration && c.start + c.duration > seg.start) && (
                        <Headphones className="relative z-10 h-3 w-3 shrink-0 text-[#4ade80]/60" />
                      )}
                      {seg.prompt && width > 100 && (
                        <span className="relative z-10 text-[9px] text-white/30 truncate ml-1 hidden sm:block">{seg.prompt.slice(0, 30)}</span>
                      )}
                    </div>

                    {/* Subtle gradient overlay for depth */}
                    <div className="absolute inset-0 pointer-events-none rounded-md"
                      style={{ background: "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 40%, rgba(0,0,0,0.1) 100%)" }} />
                  </div>
                )
              })}
            </div>
          </div>

        {/* Audio tracks — dynamic */}
        {audioTracks.map((track) => {
          const trackCues = audioCues.filter(c => c.track === track.id)
          return (
            <div key={track.id} className="flex border-t border-white/[0.04]" style={{ height: ROW_H }}>
              <div className="w-[88px] shrink-0 border-r border-white/[0.06] px-2.5 flex items-center gap-1.5 bg-white/[0.02]">
                <Music className="h-3 w-3" style={{ color: track.color }} />
                <span className="text-[10px] font-medium text-white/40 truncate">{track.label}</span>
                <button className="ml-auto text-white/20 hover:text-red-400 shrink-0"
                  onClick={() => removeAudioTrack(track.id)}>
                  <Trash2 className="h-2.5 w-2.5" />
                </button>
              </div>
              <div className="flex-1 relative overflow-hidden"
                style={{ background: "#0a0a0e" }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  try {
                    const d = JSON.parse(e.dataTransfer.getData("application/tech-noir-asset"))
                    if (d.type === "audio") {
                      const rect = e.currentTarget.getBoundingClientRect()
                      const relX = e.clientX - rect.left
                      const startT = Math.max(0, relX / pps)
                      const cue = addAudioCue({
                        track: track.id, start: startT, duration: 5, label: d.name,
                        audioUrl: d.url, audioB64: d.url, volume: 0.8,
                        waveformPeaks: null, sourceStepId: null,
                      })
                      decodeWaveform(d.url, cue.id)
                      toast("info", `Added audio: ${d.name}`)
                    }
                  } catch {}
                }}
                onClick={(e) => { if (!(e.target as HTMLElement).closest("[data-cue]")) seekFromMouseEvent(e) }}>
                {trackCues.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-0 hover:opacity-100 transition-opacity"
                    style={{ opacity: undefined }}>
                    <span className="text-[9px] text-white/10">Drop audio here</span>
                  </div>
                )}
                {trackCues.map((cue) => {
                  const left = cue.start * pps
                  const width = Math.max(cue.duration * pps, 8)
                  const isSelected = cue.id === selectedAudioCueId
                  return (
                    <div key={cue.id} data-cue={cue.id}
                      className={`absolute top-[3px] bottom-[3px] rounded-md cursor-pointer overflow-hidden transition-all duration-100 ${isSelected
                        ? "ring-2 ring-emerald-400/60 shadow-lg shadow-emerald-400/20 border border-emerald-400/80"
                        : "border border-white/[0.08] hover:border-white/[0.2]"
                      }`}
                      style={{
                        left, width,
                        background: isSelected
                          ? `linear-gradient(135deg, ${track.color}30, ${track.color}18)`
                          : `linear-gradient(135deg, ${track.color}18, ${track.color}0a)`,
                      }}
                      onClick={(e) => { e.stopPropagation(); setSelectedAudioCue(cue.id) }}>

                      {/* Left resize handle */}
                      <div className="absolute top-0 left-0 bottom-0 z-20 cursor-ew-resize group" style={{ width: HANDLE_W }}
                        onPointerDown={(e) => onCuePointerDown(e, cue.id, "resize-left")}>
                        <div className="absolute left-[3px] top-1/2 -translate-y-1/2 w-[2px] h-5 rounded-full bg-transparent group-hover:bg-white/40 transition-colors" />
                      </div>

                      {/* Right resize handle */}
                      <div className="absolute top-0 right-0 bottom-0 z-20 cursor-ew-resize group" style={{ width: HANDLE_W }}
                        onPointerDown={(e) => onCuePointerDown(e, cue.id, "resize-right")}>
                        <div className="absolute right-[3px] top-1/2 -translate-y-1/2 w-[2px] h-5 rounded-full bg-transparent group-hover:bg-white/40 transition-colors" />
                      </div>

                      {/* Cue body — drag to move */}
                      <div className="absolute inset-0 z-10 cursor-grab active:cursor-grabbing flex items-center px-3 gap-1.5"
                        onPointerDown={(e) => onCuePointerDown(e, cue.id, "move")}>
                        {/* Waveform preview */}
                        <div className="absolute inset-0 flex items-center px-1 pointer-events-none">
                          {cue.waveformPeaks ? (
                            <div className="flex items-center w-full h-full gap-px">
                              {cue.waveformPeaks.map((peak, i) => (
                                <div key={i} className="flex-1 bg-white/10 rounded-sm min-w-[1px]"
                                  style={{ height: `${Math.max(8, peak * 80)}%` }} />
                              ))}
                            </div>
                          ) : (
                            <div className="flex items-center w-full h-full gap-px">
                              {Array.from({ length: 30 }).map((_, i) => (
                                <div key={i} className="flex-1 rounded-sm min-w-[1px]"
                                  style={{
                                    height: `${12 + Math.sin(i * 0.5) * 20 + Math.random() * 15}%`,
                                    background: `${track.color}15`,
                                  }} />
                              ))}
                            </div>
                          )}
                        </div>
                        <span className="relative z-10 text-[9px] text-white/50 truncate">{cue.label}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}

        {/* Add Keyframe + Add Audio Track Bar */}
        <div className="flex border-t border-white/[0.06]">
          <div className="w-[88px] shrink-0 border-r border-white/[0.06]" />
          <div className="flex-1 px-1.5 py-1 flex gap-1.5">
            <Button variant="ghost" size="sm"
              className="h-7 flex-1 text-[11px] gap-1.5 text-white/30 hover:text-white/60 hover:bg-white/[0.04] rounded-md border border-dashed border-white/[0.08] hover:border-white/[0.15]"
              onClick={() => { const s = addSegment({ duration: 5, status: "empty" }); setSelected(s.id) }}>
              <Plus className="h-3 w-3" /> Add Keyframe
            </Button>
            <Button variant="ghost" size="sm"
              className="h-7 text-[11px] gap-1.5 text-white/30 hover:text-white/60 hover:bg-white/[0.04] rounded-md border border-dashed border-white/[0.08] hover:border-white/[0.15]"
              onClick={() => { addAudioTrack(); toast("info", "Audio track added") }}>
              <Music className="h-3 w-3" /> Add Track
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Video scrubber for a standalone segment video (non-relay) */
function VideoScrubber({ seg, time }: { seg: TimelineSegment; time: number }) {
  const ref = useRef<HTMLVideoElement>(null)
  const segTime = Math.max(0, Math.min(seg.duration, time - seg.start))
  useEffect(() => {
    const el = ref.current
    if (!el || !seg.videoUrl) return
    if (Math.abs(el.currentTime - segTime) > 0.15) {
      el.currentTime = segTime
    }
  }, [segTime, seg.videoUrl])
  return <video ref={ref} key={seg.id} src={seg.videoUrl} className="max-w-full max-h-[70vh] rounded-lg shadow-2xl" autoPlay loop muted />
}

/** Relay video scrubber — scrubs to absolute time in the continuous relay output */
function RelayScrubber({ videoUrl, time }: { videoUrl: string; time: number }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (Math.abs(el.currentTime - time) > 0.15) {
      el.currentTime = time
    }
  }, [time])
  return <video ref={ref} src={videoUrl} className="max-w-full max-h-[70vh] rounded-lg shadow-2xl" autoPlay loop muted />
}

function CurrentPreview({ segments, time, relayVideoUrl, relaySegmentIds }: {
  segments: ReturnType<typeof useTimelineStore.getState>["segments"]
  time: number
  relayVideoUrl: string | null
  relaySegmentIds: string[]
}) {
  const seg = segments.find((s) => time >= s.start && time < s.start + s.duration)

  // If this segment is part of a relay, show the relay video scrubbed to absolute time
  if (relayVideoUrl && seg && relaySegmentIds.includes(seg.id)) {
    return (
      <div className="flex flex-col items-center gap-3">
        <RelayScrubber videoUrl={relayVideoUrl} time={time} />
        <span className="text-[11px] text-white/30 font-mono">{segLabel(seg.order)} — {seg.prompt?.slice(0, 80) || "no prompt"}</span>
        <span className="text-[9px] text-[#6366f1]/50 font-mono uppercase tracking-wider">Director Relay</span>
      </div>
    )
  }

  if (!seg) {
    if (segments.length > 0) {
      const closest = segments.reduce((prev, curr) =>
        Math.abs(curr.start - time) < Math.abs(prev.start - time) ? curr : prev
      )
      return (
        <div className="flex flex-col items-center gap-3">
          {closest.videoUrl ? (
            <VideoScrubber seg={closest} time={time} />
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
        <VideoScrubber seg={seg} time={time} />
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
