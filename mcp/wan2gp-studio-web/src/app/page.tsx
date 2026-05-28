"use client";

import { useChat } from "@ai-sdk/react";
import { useState, useEffect, useCallback, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  const chat = useChat();
  const [chatInput, setChatInput] = useState("");

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

  // Fetch model catalog from /v1/models on mount
  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch("/api/models");
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
        const resp = await fetch("/api/tts/voices");
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

  const isStreaming = chat.status === "submitted" || chat.status === "streaming";

  function handleChatSubmit(e: FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || isStreaming) return;
    chat.sendMessage({ text: chatInput });
    setChatInput("");
  }

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

      const resp = await fetch("/api/generate", {
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
      const resp = await fetch("/api/tts", {
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
      const resp = await fetch("/api/kimodo", { method: "POST" });
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
      const resp = await fetch("/api/3d", {
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
            <TabsTrigger value="kimodo" className="flex-1 text-xs py-2">Kimodo</TabsTrigger>
            <TabsTrigger value="3d" className="flex-1 text-xs py-2">3D Mesh</TabsTrigger>
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
                src="/kimodo/"
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

      {/* Main — Chat */}
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="font-medium">Chat</h2>
          <p className="text-xs text-zinc-500">
            Ask the AI to generate anything — it will call the right service
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chat.messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-zinc-500 space-y-2">
                <p className="text-lg">What do you want to create?</p>
                <p className="text-sm">
                  Try: &quot;Generate a cyberpunk cityscape&quot; or
                  &quot;Create a video of ocean waves at sunset&quot;
                </p>
              </div>
            </div>
          )}
          {chat.messages.map((m) => (
            <div
              key={m.id}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-800 text-zinc-100"
                }`}
              >
                {m.parts.map((part, i) => (
                  <p key={i} className="text-sm whitespace-pre-wrap">
                    {part.type === "text" ? part.text : JSON.stringify(part)}
                  </p>
                ))}
              </div>
            </div>
          ))}
          {isStreaming && (
            <div className="flex justify-start">
              <div className="bg-zinc-800 rounded-lg px-4 py-2">
                <p className="text-sm text-zinc-400 animate-pulse">Thinking...</p>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-zinc-800">
          <form onSubmit={handleChatSubmit} className="flex gap-2">
            <Textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Describe what to generate..."
              rows={1}
              className="flex-1 resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleChatSubmit(e);
                }
              }}
            />
            <Button type="submit" disabled={isStreaming}>
              Send
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
