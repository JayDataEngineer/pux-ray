"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ChatPanel from "@/components/chat/ChatPanel";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";

type GeneratedContent = {
  mediaType: string;
  data: string;
  model: string;
  prompt: string;
  timestamp: number;
};

type ModelInfo = {
  id: string;
  category: string;
  label: string;
  output_type: string;
  needs_gpu: boolean;
  description: string;
};

type TTSEngine = {
  id: string;
  label: string;
  modes: string[];
  cpu: boolean;
};

type TVoiceCatalog = {
  engines: TTSEngine[];
  voices: Record<string, string[]>;
};

type KimodoStatus = "idle" | "loading" | "ready" | "error";

const API = "/studio/api";
const KIMODO_URL = process.env.NEXT_PUBLIC_KIMODO_URL || "/kimodo/";

export default function Home() {
  const [history, setHistory] = useState<GeneratedContent[]>([]);
  const [generating, setGenerating] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Form state
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [steps, setSteps] = useState(30);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);

  // Chat
  // TTS state
  const [ttsEngine, setTtsEngine] = useState("kokoro");
  const [ttsMode, setTtsMode] = useState("custom_voice");
  const [ttsVoice, setTtsVoice] = useState("af_bella");
  const [ttsText, setTtsText] = useState("Hello! How are you today?");
  const [ttsInstruct, setTtsInstruct] = useState("");
  const [ttsRefAudio, setTtsRefAudio] = useState<string | null>(null);
  const [ttsGenerating, setTtsGenerating] = useState(false);
  const [ttsResult, setTtsResult] = useState<GeneratedContent | null>(null);
  const [voiceCatalog, setVoiceCatalog] = useState<TVoiceCatalog | null>(null);

  // Kimodo state
  const [kimodoStatus, setKimodoStatus] = useState<KimodoStatus>("idle");
  const [kimodoError, setKimodoError] = useState<string | null>(null);

  // 3D Mesh (TRELLIS) state
  const [meshImage, setMeshImage] = useState<File | null>(null);
  const [meshSteps, setMeshSteps] = useState(30);
  const [meshGuidance, setMeshGuidance] = useState(7.5);
  const [meshGenerating, setMeshGenerating] = useState(false);
  const [meshResult, setMeshResult] = useState<GeneratedContent | null>(null);

  // Audio tab state
  const [audioMode, setAudioMode] = useState<"transcribe" | "sound" | "music">("sound");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [asrEngine, setAsrEngine] = useState("whisper");
  const [asrResult, setAsrResult] = useState<string | null>(null);
  const [soundPrompt, setSoundPrompt] = useState("");
  const [soundDuration, setSoundDuration] = useState(5.0);
  const [musicPrompt, setMusicPrompt] = useState("");
  const [musicLyrics, setMusicLyrics] = useState("");
  const [musicDuration, setMusicDuration] = useState(30.0);
  const [audioGenerating, setAudioGenerating] = useState(false);
  const [audioResult, setAudioResult] = useState<GeneratedContent | null>(null);

  // Admin state
  const [adminStatus, setAdminStatus] = useState<Record<string, unknown> | null>(null);
  const [adminLoading, setAdminLoading] = useState(false);

  // Workflow state
  const [wfSpecs, setWfSpecs] = useState<Array<{ name: string; description: string; steps: number }>>([]);
  const [wfSpecsLoading, setWfSpecsLoading] = useState(false);
  const [wfSelectedSpec, setWfSelectedSpec] = useState("");
  const [wfSpecDetail, setWfSpecDetail] = useState<Record<string, unknown> | null>(null);
  const [wfRunId, setWfRunId] = useState<string | null>(null);
  const [wfRunStatus, setWfRunStatus] = useState<Record<string, unknown> | null>(null);
  const [wfExecuting, setWfExecuting] = useState(false);
  const [wfInputs, setWfInputs] = useState<Record<string, string>>({});

  // Fetch model catalog from /v1/models on mount
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API}/models`);
        if (resp.ok) {
          const { data } = await resp.json();
          setModels(data);
          if (data.length > 0) {
            setSelectedCategory(data[0].category);
          }
        }
      } catch {
        // Will show empty catalog
      } finally {
        setModelsLoading(false);
      }
    })();
  }, []);

  // Fetch voice catalog
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API}/tts/voices`);
        if (resp.ok) {
          const catalog = await resp.json();
          setVoiceCatalog(catalog);
        }
      } catch {
        // Will use defaults
      }
    })();
  }, []);

  // Derive categories and filtered models from catalog
  const categories = [...new Set(models.map((m) => m.category))];
  const filteredModels = models.filter((m) => m.category === selectedCategory);

  // Detect if selected model needs image input
  const needsImage = selectedCategory === "3d" ||
    selectedModel?.includes("i2v") ||
    selectedModel?.includes("edit");
  const needsDimensions = ["video", "image"].includes(selectedCategory);

  const handleGenerate = useCallback(async () => {
    if (!selectedModel) return;
    setGenerating(true);

    try {
      const params: Record<string, unknown> = {
        model: selectedModel,
        steps,
      };

      if (prompt) params.prompt = prompt;
      if (needsDimensions) {
        params.width = width;
        params.height = height;
      }

      // Image upload — read as base64
      if (imageFile && needsImage) {
        const b64 = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => {
            const result = reader.result as string;
            resolve(result.split(",")[1]);
          };
          reader.readAsDataURL(imageFile);
        });
        params.image_b64 = b64;
      }

      const resp = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ service: "wan2gp", params }),
      });

      const result = await resp.json();

      if (result.content) {
        const text = result.content[0]?.text || JSON.stringify(result);
        try {
          const parsed = JSON.parse(text);
          if (parsed.status === "ok" && parsed.data) {
            setHistory((prev) => [
              {
                mediaType: parsed.media_type || "image/png",
                data: parsed.data,
                model: parsed.model || selectedModel,
                prompt: prompt || "",
                timestamp: Date.now(),
              },
              ...prev,
            ]);
          }
        } catch {
          // Not JSON
        }
      }
    } catch (e) {
      console.error("Generation failed:", e);
    } finally {
      setGenerating(false);
    }
  }, [selectedModel, prompt, steps, width, height, imageFile, needsImage, needsDimensions]);

  const handleTTS = useCallback(async () => {
    if (!ttsText) return;
    setTtsGenerating(true);
    try {
      const resp = await fetch(`${API}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: ttsText,
          engine: ttsEngine,
          mode: ttsMode,
          voice: ttsMode === "custom_voice" ? ttsVoice : undefined,
          instruct: ttsMode === "voice_design" ? ttsInstruct : undefined,
          ref_audio_b64: ttsMode === "voice_clone" ? ttsRefAudio : undefined,
        }),
      });
      const result = await resp.json();
      const parsed = result.content?.[0]?.text
        ? JSON.parse(result.content[0].text)
        : result;
      if (parsed.status === "success" && parsed.data) {
        setTtsResult({
          mediaType: parsed.media_type || "audio/wav",
          data: parsed.data,
          model: parsed.model || ttsEngine,
          prompt: ttsText,
          timestamp: Date.now(),
        });
      } else {
        console.error("TTS error:", parsed.error || "Unknown error");
      }
    } catch (e) {
      console.error("TTS failed:", e);
    } finally {
      setTtsGenerating(false);
    }
  }, [ttsText, ttsEngine, ttsMode, ttsVoice, ttsInstruct, ttsRefAudio]);

  // Available modes for the selected engine
  const ttsEngineInfo = voiceCatalog?.engines?.find((e) => e.id === ttsEngine);
  const ttsModes = ttsEngineInfo?.modes || ["custom_voice"];
  const ttsVoices = voiceCatalog?.voices?.[ttsEngine] || [];

  const handleKimodoStart = useCallback(async () => {
    setKimodoStatus("loading");
    setKimodoError(null);
    try {
      const resp = await fetch(`${API}/kimodo`, { method: "POST" });
      const result = await resp.json();
      const parsed = result.content?.[0]?.text
        ? JSON.parse(result.content[0].text)
        : result;
      if (parsed.status === "success" || parsed.status === "already_loaded") {
        // Wait for Viser to start serving
        await new Promise((r) => setTimeout(r, 3000));
        setKimodoStatus("ready");
      } else {
        setKimodoError(parsed.error || "Failed to start Kimodo");
        setKimodoStatus("error");
      }
    } catch (e) {
      setKimodoError(e instanceof Error ? e.message : "Failed to start Kimodo");
      setKimodoStatus("error");
    }
  }, []);

  const handleMeshGenerate = useCallback(async () => {
    if (!meshImage) return;
    setMeshGenerating(true);
    try {
      const b64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve((reader.result as string).split(",")[1]);
        reader.readAsDataURL(meshImage);
      });
      const resp = await fetch(`${API}/3d`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, steps: meshSteps, guidance: meshGuidance }),
      });
      const result = await resp.json();
      const parsed = result.content?.[0]?.text
        ? JSON.parse(result.content[0].text)
        : result;
      if (parsed.status === "success" && parsed.data) {
        setMeshResult({
          mediaType: parsed.media_type || "model/gltf-binary",
          data: parsed.data,
          model: "trellis",
          prompt: "3D mesh",
          timestamp: Date.now(),
        });
        setHistory((prev) => [meshResult!, ...prev]);
      } else if (parsed.error) {
        console.error("3D error:", parsed.error);
      }
    } catch (e) {
      console.error("3D failed:", e);
    } finally {
      setMeshGenerating(false);
    }
  }, [meshImage, meshSteps, meshGuidance, meshResult]);

  const handleAudio = useCallback(async () => {
    setAudioGenerating(true);
    try {
      if (audioMode === "transcribe") {
        if (!audioFile) return;
        const b64 = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve((reader.result as string).split(",")[1]);
          reader.readAsDataURL(audioFile);
        });
        const resp = await fetch(`${API}/transcribe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_b64: b64, engine: asrEngine }),
        });
        const result = await resp.json();
        const parsed = result.content?.[0]?.text ? JSON.parse(result.content[0].text) : result;
        setAsrResult(parsed.text || parsed.error || "No transcription");
      } else if (audioMode === "sound") {
        if (!soundPrompt) return;
        const resp = await fetch(`${API}/sound`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: soundPrompt, duration_seconds: soundDuration }),
        });
        const result = await resp.json();
        const parsed = result.content?.[0]?.text ? JSON.parse(result.content[0].text) : result;
        if (parsed.status === "success" && parsed.data) {
          const item: GeneratedContent = {
            mediaType: parsed.media_type || "audio/wav",
            data: parsed.data,
            model: "moss-soundeffect",
            prompt: soundPrompt,
            timestamp: Date.now(),
          };
          setAudioResult(item);
          setHistory((prev) => [item, ...prev]);
        }
      } else if (audioMode === "music") {
        if (!musicPrompt) return;
        const resp = await fetch(`${API}/music`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: musicPrompt,
            lyrics: musicLyrics || undefined,
            duration_seconds: musicDuration,
          }),
        });
        const result = await resp.json();
        const parsed = result.content?.[0]?.text ? JSON.parse(result.content[0].text) : result;
        if (parsed.status === "success" && parsed.data) {
          const item: GeneratedContent = {
            mediaType: parsed.media_type || "audio/wav",
            data: parsed.data,
            model: "ace-step",
            prompt: musicPrompt,
            timestamp: Date.now(),
          };
          setAudioResult(item);
          setHistory((prev) => [item, ...prev]);
        }
      }
    } catch (e) {
      console.error("Audio failed:", e);
    } finally {
      setAudioGenerating(false);
    }
  }, [audioMode, audioFile, asrEngine, soundPrompt, soundDuration, musicPrompt, musicLyrics, musicDuration]);

  const fetchAdminStatus = useCallback(async () => {
    try {
      const resp = await fetch(`${API}/models`);
      if (resp.ok) {
        const data = await resp.json();
        setAdminStatus(data.gpu_status || null);
      }
    } catch { /* ignore */ }
  }, []);

  const handleAdminLoad = useCallback(async (service: string) => {
    setAdminLoading(true);
    try {
      await fetch(`${API}/admin/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ service }),
      });
      await fetchAdminStatus();
    } catch (e) {
      console.error("Load failed:", e);
    } finally {
      setAdminLoading(false);
    }
  }, [fetchAdminStatus]);

  const handleAdminUnload = useCallback(async () => {
    setAdminLoading(true);
    try {
      await fetch(`${API}/admin/unload`, { method: "POST" });
      await fetchAdminStatus();
    } catch (e) {
      console.error("Unload failed:", e);
    } finally {
      setAdminLoading(false);
    }
  }, [fetchAdminStatus]);

  // ── Workflow helpers ──
  async function wfApi(action: string, extra: Record<string, unknown> = {}) {
    const resp = await fetch(`${API}/workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    const raw = await resp.json();
    // Unwrap MCP content wrapper
    if (raw?.content?.[0]?.type === "text") {
      try { return JSON.parse(raw.content[0].text); } catch { return raw; }
    }
    return raw;
  }

  const loadWfSpecs = useCallback(async () => {
    setWfSpecsLoading(true);
    try {
      const data = await wfApi("list_specs");
      const list = Array.isArray(data) ? data : data?.data ?? [];
      setWfSpecs(list);
      if (list.length > 0 && !wfSelectedSpec) setWfSelectedSpec(list[0].name);
    } catch { /* empty */ }
    finally { setWfSpecsLoading(false); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadWfSpec = useCallback(async (name: string) => {
    try {
      const data = await wfApi("get_spec", { spec_name: name });
      setWfSpecDetail(data);
    } catch { /* empty */ }
  }, []);

  const handleWfStartRun = useCallback(async () => {
    if (!wfSelectedSpec) return;
    setWfExecuting(true);
    try {
      const parsed: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(wfInputs)) {
        if (v) { try { parsed[k] = JSON.parse(v); } catch { parsed[k] = v; } }
      }
      const result = await wfApi("start_run", { spec_name: wfSelectedSpec, inputs: parsed, manual: true });
      const rid = result.run_id;
      if (rid) {
        setWfRunId(rid);
        const runData = await wfApi("get_run", { spec_name: wfSelectedSpec, run_id: rid });
        setWfRunStatus(runData);
      }
    } catch (e) { console.error("Workflow start failed:", e); }
    finally { setWfExecuting(false); }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wfSelectedSpec, wfInputs]);

  const handleWfExecuteStep = useCallback(async (stepId: string) => {
    if (!wfSelectedSpec || !wfRunId) return;
    setWfExecuting(true);
    try {
      await wfApi("execute_step", { spec_name: wfSelectedSpec, run_id: wfRunId, step_id: stepId });
      const runData = await wfApi("get_run", { spec_name: wfSelectedSpec, run_id: wfRunId });
      setWfRunStatus(runData);
    } catch (e) { console.error("Step execute failed:", e); }
    finally { setWfExecuting(false); }
  }, [wfSelectedSpec, wfRunId]);

  const handleWfRerunStep = useCallback(async (stepId: string) => {
    if (!wfSelectedSpec || !wfRunId) return;
    setWfExecuting(true);
    try {
      await wfApi("rerun_step", { spec_name: wfSelectedSpec, run_id: wfRunId, step_id: stepId });
      const runData = await wfApi("get_run", { spec_name: wfSelectedSpec, run_id: wfRunId });
      setWfRunStatus(runData);
    } catch (e) { console.error("Step rerun failed:", e); }
    finally { setWfExecuting(false); }
  }, [wfSelectedSpec, wfRunId]);

  const handleWfRefreshRun = useCallback(async () => {
    if (!wfSelectedSpec || !wfRunId) return;
    try {
      const runData = await wfApi("get_run", { spec_name: wfSelectedSpec, run_id: wfRunId });
      setWfRunStatus(runData);
    } catch { /* empty */ }
  }, [wfSelectedSpec, wfRunId]);

  function renderMedia(item: GeneratedContent) {
    if (!item.data) {
      return (
        <div className="w-full h-32 bg-zinc-900 rounded-lg flex items-center justify-center text-zinc-500">
          No preview available
        </div>
      );
    }
    const src = `data:${item.mediaType};base64,${item.data}`;

    if (item.mediaType.startsWith("video")) {
      return <video src={src} controls className="w-full rounded-lg" />;
    }
    if (item.mediaType.startsWith("audio")) {
      return <audio src={src} controls className="w-full" />;
    }
    if (item.mediaType.includes("gltf") || item.mediaType.includes("model")) {
      return (
        <div className="w-full h-48 bg-zinc-900 rounded-lg flex items-center justify-center text-zinc-500">
          3D model — download to view
        </div>
      );
    }
    if (item.mediaType.includes("npz") || item.mediaType === "application/x-motion") {
      return (
        <div className="w-full h-32 bg-zinc-900 rounded-lg flex items-center justify-center text-zinc-500">
          Motion data — preview not available
        </div>
      );
    }
    return <img src={src} alt={item.prompt} className="w-full rounded-lg" />;
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-80 border-r border-zinc-800 flex flex-col">
        <div className="p-4 border-b border-zinc-800">
          <h1 className="text-lg font-semibold">Tech Noir Studio</h1>
          <p className="text-xs text-zinc-500">GPU-powered generation</p>
        </div>

        <Tabs defaultValue="generate" className="flex-1 flex flex-col">
          <TabsList className="w-full rounded-none border-b border-zinc-800 flex flex-wrap h-auto p-0">
            <TabsTrigger value="generate" className="flex-1 text-xs py-2">Generate</TabsTrigger>
            <TabsTrigger value="speech" className="flex-1 text-xs py-2">Speech</TabsTrigger>
            <TabsTrigger value="audio" className="flex-1 text-xs py-2">Audio</TabsTrigger>
            <TabsTrigger value="kimodo" className="flex-1 text-xs py-2">Kimodo</TabsTrigger>
            <TabsTrigger value="3d" className="flex-1 text-xs py-2">3D Mesh</TabsTrigger>
            <TabsTrigger value="admin" className="flex-1 text-xs py-2">Admin</TabsTrigger>
            <TabsTrigger value="workflow" className="flex-1 text-xs py-2">Workflows</TabsTrigger>
            <TabsTrigger value="history" className="flex-1 text-xs py-2">History</TabsTrigger>
          </TabsList>

          <TabsContent value="generate" className="flex-1 overflow-y-auto p-4 space-y-4">
            {modelsLoading && (
              <p className="text-sm text-zinc-500">Loading models...</p>
            )}

            {!modelsLoading && categories.length === 0 && (
              <p className="text-sm text-zinc-500">No models available</p>
            )}

            {categories.length > 0 && (
              <>
                {/* Category */}
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Category</label>
                  <Select value={selectedCategory} onValueChange={(v) => { setSelectedCategory(v ?? ""); setSelectedModel(""); }}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {categories.map((c) => (
                        <SelectItem key={c} value={c}>{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Model */}
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Model</label>
                  <Select
                    value={selectedModel || (filteredModels[0]?.id ?? "")}
                    onValueChange={(v) => setSelectedModel(v ?? "")}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {filteredModels.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Prompt / Text input */}
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Prompt</label>
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe what to generate..."
                    rows={3}
                  />
                </div>

                {/* Image upload for i2v / 3d / edit models */}
                {needsImage && (
                  <div className="space-y-2">
                    <label className="text-xs text-zinc-400">Input image</label>
                    <Input
                      type="file"
                      accept="image/*"
                      onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
                    />
                  </div>
                )}

                {/* Dimensions for video/image */}
                {needsDimensions && (
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <label className="text-xs text-zinc-400">Width</label>
                      <Input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs text-zinc-400">Height</label>
                      <Input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} />
                    </div>
                  </div>
                )}

                {/* Steps */}
                <div className="space-y-1">
                  <label className="text-xs text-zinc-400">Steps: {steps}</label>
                  <input
                    type="range"
                    min={1}
                    max={100}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                <Button
                  className="w-full"
                  onClick={handleGenerate}
                  disabled={generating || !selectedModel}
                >
                  {generating ? "Generating..." : "Generate"}
                </Button>
              </>
            )}
          </TabsContent>

          <TabsContent value="speech" className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Engine */}
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Engine</label>
              <Select value={ttsEngine} onValueChange={(v) => { setTtsEngine(v ?? "kokoro"); setTtsMode("custom_voice"); }}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {(voiceCatalog?.engines || []).map((e) => (
                    <SelectItem key={e.id} value={e.id}>{e.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Mode (only for engines with multiple modes) */}
            {ttsModes.length > 1 && (
              <div className="space-y-2">
                <label className="text-xs text-zinc-400">Mode</label>
                <Select value={ttsMode} onValueChange={(v) => setTtsMode(v ?? "custom_voice")}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ttsModes.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m === "custom_voice" ? "Preset Voice" : m === "voice_design" ? "Voice Design" : "Voice Clone"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Voice picker (custom_voice mode) */}
            {ttsMode === "custom_voice" && ttsVoices.length > 0 && (
              <div className="space-y-2">
                <label className="text-xs text-zinc-400">Voice</label>
                <Select value={ttsVoice} onValueChange={(v) => setTtsVoice(v ?? "af_bella")}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ttsVoices.map((v: string) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Voice description (voice_design mode) */}
            {ttsMode === "voice_design" && (
              <div className="space-y-2">
                <label className="text-xs text-zinc-400">Voice Description</label>
                <Textarea
                  value={ttsInstruct}
                  onChange={(e) => setTtsInstruct(e.target.value)}
                  placeholder="A cute young anime girl with a high-pitched, cheerful voice..."
                  rows={2}
                />
              </div>
            )}

            {/* Reference audio (voice_clone mode) */}
            {ttsMode === "voice_clone" && (
              <div className="space-y-2">
                <label className="text-xs text-zinc-400">Reference Audio</label>
                <Input
                  type="file"
                  accept="audio/*"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const b64 = await new Promise<string>((resolve) => {
                      const reader = new FileReader();
                      reader.onload = () => {
                        const result = reader.result as string;
                        resolve(result.split(",")[1]);
                      };
                      reader.readAsDataURL(file);
                    });
                    setTtsRefAudio(b64);
                  }}
                />
                {ttsResult && !ttsRefAudio && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full text-xs"
                    onClick={() => setTtsRefAudio(ttsResult.data)}
                  >
                    Use last generated audio as reference
                  </Button>
                )}
              </div>
            )}

            {/* Text input */}
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Text</label>
              <Textarea
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                placeholder="Type what you want to hear..."
                rows={3}
              />
            </div>

            <Button
              className="w-full"
              onClick={handleTTS}
              disabled={ttsGenerating || !ttsText}
            >
              {ttsGenerating ? "Generating..." : "Speak"}
            </Button>

            {/* Audio result */}
            {ttsResult && (
              <Card className="bg-zinc-900 border-zinc-800">
                <CardContent className="p-3 space-y-2">
                  <audio
                    src={`data:${ttsResult.mediaType};base64,${ttsResult.data}`}
                    controls
                    className="w-full"
                  />
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{ttsResult.mediaType.split("/")[1]}</Badge>
                    <span className="text-xs text-zinc-500 truncate">{ttsResult.model}</span>
                  </div>
                  {ttsMode === "voice_design" && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full text-xs"
                      onClick={() => {
                        setTtsRefAudio(ttsResult.data);
                        setTtsMode("voice_clone");
                      }}
                    >
                      Save as voice clone reference
                    </Button>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* ── Audio Tab (transcribe / sound effects / music) ── */}
          <TabsContent value="audio" className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Mode selector */}
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Mode</label>
              <Select value={audioMode} onValueChange={(v) => setAudioMode(v as "transcribe" | "sound" | "music")}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="sound">Sound Effect</SelectItem>
                  <SelectItem value="music">Music</SelectItem>
                  <SelectItem value="transcribe">Transcribe</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {audioMode === "transcribe" && (
              <>
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Engine</label>
                  <Select value={asrEngine} onValueChange={(v) => { if (v) setAsrEngine(v); }}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="whisper">Whisper (CPU, fast)</SelectItem>
                      <SelectItem value="vibevoice">VibeVoice (GPU, diarization)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Audio File</label>
                  <Input type="file" accept="audio/*" onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)} />
                </div>
                <Button className="w-full" onClick={handleAudio} disabled={audioGenerating || !audioFile}>
                  {audioGenerating ? "Transcribing..." : "Transcribe"}
                </Button>
                {asrResult && (
                  <Card className="bg-zinc-900 border-zinc-800">
                    <CardContent className="p-3">
                      <p className="text-sm text-zinc-200 whitespace-pre-wrap">{asrResult}</p>
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {audioMode === "sound" && (
              <>
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Sound Description</label>
                  <Textarea
                    value={soundPrompt}
                    onChange={(e) => setSoundPrompt(e.target.value)}
                    placeholder="Thunder rumbling in the distance, sword unsheathing..."
                    rows={3}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-zinc-400">Duration: {soundDuration}s</label>
                  <input type="range" min={1} max={30} step={0.5} value={soundDuration}
                    onChange={(e) => setSoundDuration(Number(e.target.value))} className="w-full" />
                </div>
                <Button className="w-full" onClick={handleAudio} disabled={audioGenerating || !soundPrompt}>
                  {audioGenerating ? "Generating..." : "Generate Sound Effect"}
                </Button>
              </>
            )}

            {audioMode === "music" && (
              <>
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Music Description</label>
                  <Textarea
                    value={musicPrompt}
                    onChange={(e) => setMusicPrompt(e.target.value)}
                    placeholder="Upbeat electronic dance music with heavy bass..."
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-zinc-400">Lyrics (optional)</label>
                  <Textarea
                    value={musicLyrics}
                    onChange={(e) => setMusicLyrics(e.target.value)}
                    placeholder="Optional lyrics for vocal generation..."
                    rows={2}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-zinc-400">Duration: {musicDuration}s</label>
                  <input type="range" min={5} max={60} step={1} value={musicDuration}
                    onChange={(e) => setMusicDuration(Number(e.target.value))} className="w-full" />
                </div>
                <Button className="w-full" onClick={handleAudio} disabled={audioGenerating || !musicPrompt}>
                  {audioGenerating ? "Generating..." : "Generate Music"}
                </Button>
              </>
            )}

            {/* Audio result (shared for sound + music) */}
            {audioResult && audioMode !== "transcribe" && (
              <Card className="bg-zinc-900 border-zinc-800">
                <CardContent className="p-3 space-y-2">
                  <audio
                    src={`data:${audioResult.mediaType};base64,${audioResult.data}`}
                    controls
                    className="w-full"
                  />
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{audioResult.mediaType.split("/")[1]}</Badge>
                    <span className="text-xs text-zinc-500">{audioResult.model}</span>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* ── Kimodo Tab ── */}
          <TabsContent value="kimodo" className="flex-1 flex flex-col">
            {kimodoStatus === "idle" && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 p-4">
                <p className="text-sm text-zinc-400 text-center">Kimodo — NVIDIA interactive 3D mesh motion suite</p>
                <p className="text-xs text-zinc-500 text-center">Loads the Viser 3D posing tool on the GPU (~30s startup)</p>
                <Button onClick={handleKimodoStart}>Start Kimodo</Button>
              </div>
            )}
            {kimodoStatus === "loading" && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 p-4">
                <div className="animate-spin w-8 h-8 border-2 border-zinc-600 border-t-zinc-300 rounded-full" />
                <p className="text-sm text-zinc-400">Starting Kimodo Viser server...</p>
                <p className="text-xs text-zinc-500">Loading 3D posing tool on GPU</p>
              </div>
            )}
            {kimodoStatus === "error" && (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 p-4">
                <p className="text-sm text-red-400">{kimodoError}</p>
                {(kimodoError?.includes("VRAM") || kimodoError?.includes("Cannot free")) && (
                  <p className="text-xs text-zinc-500">Free GPU memory by stopping other services, then retry.</p>
                )}
                <Button onClick={handleKimodoStart}>Retry</Button>
              </div>
            )}
            {kimodoStatus === "ready" && (
              <iframe
                src={KIMODO_URL}
                className="flex-1 w-full border-0"
                title="Kimodo Director"
                allow="clipboard-write"
              />
            )}
          </TabsContent>

          {/* ── 3D Mesh Tab (TRELLIS) ── */}
          <TabsContent value="3d" className="flex-1 overflow-y-auto p-4 space-y-4">
            <p className="text-xs text-zinc-400">TRELLIS — image to 3D mesh (GLB)</p>
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Input Image</label>
              <Input type="file" accept="image/*" onChange={(e) => setMeshImage(e.target.files?.[0] ?? null)} />
              {meshImage && (
                <p className="text-xs text-zinc-500">{meshImage.name} ({(meshImage.size / 1024).toFixed(1)} KB)</p>
              )}
            </div>
            <div className="space-y-1">
              <label className="text-xs text-zinc-400">Steps: {meshSteps}</label>
              <input type="range" min={1} max={100} value={meshSteps} onChange={(e) => setMeshSteps(Number(e.target.value))} className="w-full" />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-zinc-400">Guidance: {meshGuidance}</label>
              <input type="range" min={1} max={20} step={0.5} value={meshGuidance} onChange={(e) => setMeshGuidance(Number(e.target.value))} className="w-full" />
            </div>
            <Button className="w-full" onClick={handleMeshGenerate} disabled={meshGenerating || !meshImage}>
              {meshGenerating ? "Generating mesh..." : "Generate 3D Mesh"}
            </Button>
            {meshResult && (
              <Card className="bg-zinc-900 border-zinc-800">
                <CardContent className="p-3 space-y-2">
                  <div className="w-full h-48 bg-zinc-800 rounded-lg flex flex-col items-center justify-center gap-3">
                    <span className="text-zinc-400 text-sm">3D mesh ready</span>
                    <a
                      href={`data:${meshResult.mediaType};base64,${meshResult.data}`}
                      download="model.glb"
                      className="text-blue-400 text-sm underline"
                    >
                      Download GLB
                    </a>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">glb</Badge>
                    <span className="text-xs text-zinc-500">trellis</span>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* ── Admin Tab ── */}
          <TabsContent value="admin" className="flex-1 overflow-y-auto p-4 space-y-4">
            <p className="text-xs text-zinc-400">GPU service management — load/unload models</p>

            {/* GPU status */}
            <Card className="bg-zinc-900 border-zinc-800">
              <CardContent className="p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">GPU Status</span>
                  <Button variant="outline" size="sm" onClick={fetchAdminStatus}>Refresh</Button>
                </div>
                {adminStatus ? (
                  <pre className="text-xs text-zinc-400 overflow-auto max-h-40">
                    {JSON.stringify(adminStatus, null, 2)}
                  </pre>
                ) : (
                  <p className="text-xs text-zinc-500">Click Refresh to check status</p>
                )}
              </CardContent>
            </Card>

            {/* Load service */}
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Load Service on GPU</label>
              <div className="grid grid-cols-2 gap-2">
                {["wan2gp", "llm", "trellis", "comfyui"].map((svc) => (
                  <Button
                    key={svc}
                    variant="outline"
                    size="sm"
                    disabled={adminLoading}
                    onClick={() => handleAdminLoad(svc)}
                  >
                    {svc}
                  </Button>
                ))}
              </div>
            </div>

            {/* Unload all */}
            <Button
              variant="destructive"
              className="w-full"
              disabled={adminLoading}
              onClick={handleAdminUnload}
            >
              {adminLoading ? "Working..." : "Unload All (Free GPU)"}
            </Button>
          </TabsContent>

          {/* ── Workflow / DAG Tab ── */}
          <TabsContent value="workflow" className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Spec picker */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-zinc-400">Pipeline</label>
                <Button variant="outline" size="sm" onClick={loadWfSpecs} disabled={wfSpecsLoading}>
                  {wfSpecsLoading ? "Loading..." : "Refresh"}
                </Button>
              </div>
              <Select value={wfSelectedSpec} onValueChange={(v) => { setWfSelectedSpec(v ?? ""); setWfRunId(null); setWfRunStatus(null); loadWfSpec(v ?? ""); }}>
                <SelectTrigger><SelectValue placeholder="Select pipeline..." /></SelectTrigger>
                <SelectContent>
                  {wfSpecs.map((s) => (
                    <SelectItem key={s.name} value={s.name}>
                      {s.name} ({s.steps} steps)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Spec detail — steps DAG */}
            {wfSpecDetail && (
              <Card className="bg-zinc-900 border-zinc-800">
                <CardContent className="p-3 space-y-2">
                  <p className="text-sm font-medium">{(wfSpecDetail as { name?: string }).name}</p>
                  <p className="text-xs text-zinc-400">{(wfSpecDetail as { description?: string }).description}</p>

                  {/* Input fields */}
                  {Object.entries((wfSpecDetail as { inputs?: Record<string, unknown> }).inputs || {}).length > 0 && (
                    <div className="space-y-2 mt-2">
                      <p className="text-xs text-zinc-500 font-medium">Inputs</p>
                      {Object.entries((wfSpecDetail as { inputs?: Record<string, unknown> }).inputs || {}).map(([key, _schema]) => (
                        <div key={key} className="space-y-1">
                          <label className="text-xs text-zinc-500">{key}</label>
                          <Input
                            value={wfInputs[key] || ""}
                            onChange={(e) => setWfInputs((prev) => ({ ...prev, [key]: e.target.value }))}
                            placeholder={key}
                            className="h-8 text-xs"
                          />
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Steps list */}
                  <div className="space-y-1 mt-2">
                    <p className="text-xs text-zinc-500 font-medium">Steps</p>
                    {((wfSpecDetail as { steps?: Array<{ id: string; description?: string; depends_on?: string[] }> }).steps || []).map((step) => {
                      const stepState = (wfRunStatus?.steps as Array<{ id: string; status: string }> | undefined)?.find((s) => s.id === step.id);
                      const stateLabel = stepState?.status ? (
                        <Badge variant={stepState.status === "completed" ? "default" : stepState.status === "failed" ? "destructive" : "secondary"}>
                          {stepState.status}
                        </Badge>
                      ) : (
                        <Badge variant="outline">pending</Badge>
                      );

                      return (
                        <div key={step.id} className="flex items-center justify-between gap-2 py-1 border-b border-zinc-800 last:border-0">
                          <div className="flex items-center gap-2 min-w-0">
                            {stateLabel}
                            <span className="text-xs truncate">{step.id}</span>
                            {step.depends_on && step.depends_on.length > 0 && (
                              <span className="text-[10px] text-zinc-600">← {step.depends_on.join(", ")}</span>
                            )}
                          </div>
                          <div className="flex gap-1 shrink-0">
                            {wfRunId && (
                              <>
                                <Button
                                  variant="outline" size="sm" className="h-6 text-[10px] px-2"
                                  disabled={wfExecuting}
                                  onClick={() => handleWfExecuteStep(step.id)}
                                >
                                  Run
                                </Button>
                                {stepState?.status === "completed" && (
                                  <Button
                                    variant="outline" size="sm" className="h-6 text-[10px] px-2"
                                    disabled={wfExecuting}
                                    onClick={() => handleWfRerunStep(step.id)}
                                  >
                                    Rerun
                                  </Button>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Start run / Refresh */}
            <div className="flex gap-2">
              {wfSelectedSpec && !wfRunId && (
                <Button className="flex-1" onClick={handleWfStartRun} disabled={wfExecuting}>
                  {wfExecuting ? "Starting..." : "Start Run (Manual)"}
                </Button>
              )}
              {wfRunId && (
                <>
                  <Button className="flex-1" variant="outline" onClick={handleWfRefreshRun} disabled={wfExecuting}>
                    Refresh Status
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => { setWfRunId(null); setWfRunStatus(null); }}>
                    New Run
                  </Button>
                </>
              )}
            </div>

            {/* Run info */}
            {wfRunId && (
              <div className="text-xs text-zinc-500 space-y-1">
                <p>Run: <code className="text-zinc-300">{wfRunId}</code></p>
                {wfRunStatus && (
                  <pre className="text-[10px] text-zinc-600 bg-zinc-900 rounded p-2 overflow-auto max-h-32">
                    {JSON.stringify(wfRunStatus, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="history" className="flex-1 overflow-y-auto p-4 space-y-3">
            {history.length === 0 && (
              <p className="text-sm text-zinc-500 text-center py-8">
                No generations yet
              </p>
            )}
            {history.map((item, i) => (
              <Card key={i} className="bg-zinc-900 border-zinc-800">
                <CardContent className="p-2 space-y-2">
                  {renderMedia(item)}
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{item.mediaType.split("/")[1]}</Badge>
                    <span className="text-xs text-zinc-500 truncate">{item.model}</span>
                  </div>
                  {item.prompt && (
                    <p className="text-xs text-zinc-400 line-clamp-2">{item.prompt}</p>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>
      </div>

      {/* Main — Chat (assistant-ui powered) */}
      <div className="flex-1 flex flex-col min-h-0">
        <ChatPanel />
      </div>
    </div>
  );
}
