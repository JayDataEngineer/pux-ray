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

  function renderMedia(item: GeneratedContent) {
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
          <TabsList className="w-full rounded-none border-b border-zinc-800">
            <TabsTrigger value="generate" className="flex-1">
              Generate
            </TabsTrigger>
            <TabsTrigger value="history" className="flex-1">
              History
            </TabsTrigger>
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
                  <Select value={selectedCategory} onValueChange={(v) => { setSelectedCategory(v); setSelectedModel(""); }}>
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
                    onValueChange={(v) => setSelectedModel(v)}
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
