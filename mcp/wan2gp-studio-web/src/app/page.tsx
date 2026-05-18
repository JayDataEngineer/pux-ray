"use client";

import { useChat } from "@ai-sdk/react";
import { useState, type FormEvent } from "react";
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
  type: "video" | "image" | "3d" | "audio";
  mediaType: string;
  data: string;
  model: string;
  prompt: string;
  timestamp: number;
};

export default function Home() {
  const [history, setHistory] = useState<GeneratedContent[]>([]);
  const [generating, setGenerating] = useState(false);

  // Direct generation form state
  const [genType, setGenType] = useState<"video" | "image" | "3d" | "audio">("image");
  const [genPrompt, setGenPrompt] = useState("");
  const [genModel, setGenModel] = useState("");
  const [genSteps, setGenSteps] = useState(24);
  const [genWidth, setGenWidth] = useState(1024);
  const [genHeight, setGenHeight] = useState(1024);

  // Chat state — AI SDK v6
  const chat = useChat();
  const [chatInput, setChatInput] = useState("");

  const modelsByType: Record<string, { value: string; label: string }[]> = {
    video: [
      { value: "wan/t2v", label: "WAN Text-to-Video" },
      { value: "wan/i2v", label: "WAN Image-to-Video" },
      { value: "hunyuan/t2v", label: "Hunyuan T2V" },
      { value: "hunyuan/i2v", label: "Hunyuan I2V" },
      { value: "ltx2", label: "LTX-Video" },
    ],
    image: [
      { value: "flux", label: "Flux" },
      { value: "flux_schnell", label: "Flux Schnell (fast)" },
      { value: "flux2_dev", label: "Flux 2 Dev" },
      { value: "flux2_klein_4b", label: "Flux 2 Klein 4B" },
      { value: "qwen-image-edit", label: "QWEN Image Edit" },
    ],
    "3d": [
      { value: "trellis", label: "TRELLIS" },
      { value: "anigen", label: "AniGen" },
    ],
    audio: [
      { value: "moss-soundeffect", label: "MOSS Sound Effect" },
      { value: "kokoro", label: "Kokoro TTS" },
      { value: "espeak", label: "eSpeak TTS" },
    ],
  };

  const isStreaming = chat.status === "submitted" || chat.status === "streaming";

  function handleChatSubmit(e: FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || isStreaming) return;
    chat.sendMessage({ text: chatInput });
    setChatInput("");
  }

  async function handleDirectGenerate() {
    if (!genPrompt.trim()) return;
    setGenerating(true);

    try {
      const toolMap = {
        video: "generate_video",
        image: "generate_image",
        "3d": "generate_3d",
        audio: "generate_audio",
      };

      const args: Record<string, unknown> = {
        prompt: genPrompt,
        model: genModel || modelsByType[genType][0].value,
        steps: genSteps,
      };

      if (genType === "video" || genType === "image") {
        args.width = genWidth;
        args.height = genHeight;
      }

      const resp = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool: toolMap[genType], args }),
      });

      const result = await resp.json();

      if (result.content) {
        const text = result.content[0]?.text || JSON.stringify(result);
        try {
          const parsed = JSON.parse(text);
          if (parsed.status === "ok" && parsed.data) {
            setHistory((prev) => [
              {
                type: genType,
                mediaType: parsed.media_type || "image/png",
                data: parsed.data,
                model: parsed.model || (args.model as string),
                prompt: genPrompt,
                timestamp: Date.now(),
              },
              ...prev,
            ]);
          }
        } catch {
          // Result wasn't JSON
        }
      }
    } catch (e) {
      console.error("Generation failed:", e);
    } finally {
      setGenerating(false);
    }
  }

  function renderMedia(item: GeneratedContent) {
    const src = `data:${item.mediaType};base64,${item.data}`;

    if (item.type === "video" || item.mediaType.startsWith("video")) {
      return <video src={src} controls className="w-full rounded-lg" />;
    }
    if (item.type === "audio" || item.mediaType.startsWith("audio")) {
      return <audio src={src} controls className="w-full" />;
    }
    if (item.type === "3d" || item.mediaType.includes("gltf")) {
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
      {/* Sidebar — generation forms + history */}
      <div className="w-80 border-r border-zinc-800 flex flex-col">
        <div className="p-4 border-b border-zinc-800">
          <h1 className="text-lg font-semibold">Wan2GP Studio</h1>
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
            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Type</label>
              <Select value={genType} onValueChange={(v) => { if (v) { setGenType(v as "video" | "image" | "3d" | "audio"); setGenModel(""); } }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="video">Video</SelectItem>
                  <SelectItem value="image">Image</SelectItem>
                  <SelectItem value="3d">3D Mesh</SelectItem>
                  <SelectItem value="audio">Audio</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Model</label>
              <Select value={genModel || modelsByType[genType][0].value} onValueChange={(v) => { if (v) setGenModel(v); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {modelsByType[genType].map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-zinc-400">Prompt</label>
              <Textarea
                value={genPrompt}
                onChange={(e) => setGenPrompt(e.target.value)}
                placeholder="Describe what to generate..."
                rows={3}
              />
            </div>

            {(genType === "video" || genType === "image") && (
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-xs text-zinc-400">Width</label>
                  <Input
                    type="number"
                    value={genWidth}
                    onChange={(e) => setGenWidth(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-zinc-400">Height</label>
                  <Input
                    type="number"
                    value={genHeight}
                    onChange={(e) => setGenHeight(Number(e.target.value))}
                  />
                </div>
              </div>
            )}

            <div className="space-y-1">
              <label className="text-xs text-zinc-400">Steps: {genSteps}</label>
              <input
                type="range"
                min={1}
                max={100}
                value={genSteps}
                onChange={(e) => setGenSteps(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <Button
              className="w-full"
              onClick={handleDirectGenerate}
              disabled={generating || !genPrompt.trim()}
            >
              {generating ? "Generating..." : "Generate"}
            </Button>
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
                    <Badge variant="secondary">{item.type}</Badge>
                    <span className="text-xs text-zinc-500 truncate">
                      {item.model}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 line-clamp-2">
                    {item.prompt}
                  </p>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
        </Tabs>
      </div>

      {/* Main — Chat interface */}
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="font-medium">Chat</h2>
          <p className="text-xs text-zinc-500">
            Ask the AI to generate anything — it will pick the right tool
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
