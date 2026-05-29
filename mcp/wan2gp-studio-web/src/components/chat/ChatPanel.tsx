"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useExternalStoreRuntime, AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import {
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
} from "@assistant-ui/react";

// ---------- MCP App tool → resourceUri mapping ----------
// Mirrors the Python server's meta.ui.resourceUri declarations.
const TOOL_RESOURCE_URIS: Record<string, string> = {
  run: "ui://apps/generate",
  forge_status: "ui://apps/admin",
  load_service: "ui://apps/admin",
  unload_services: "ui://apps/admin",
  tts_speak: "ui://apps/tts",
  transcribe: "ui://apps/audio",
  generate_sound: "ui://apps/audio",
  generate_music: "ui://apps/audio",
  workflow_list_specs: "ui://apps/workflow",
  workflow_get_spec: "ui://apps/workflow",
  workflow_start_run: "ui://apps/workflow",
  workflow_get_run: "ui://apps/workflow",
  workflow_cancel_run: "ui://apps/workflow",
  workflow_execute_step: "ui://apps/workflow",
  workflow_approve_step: "ui://apps/workflow",
  workflow_rerun_step: "ui://apps/workflow",
};

// ---------- Tool call renderer with MCP app widget support ----------
function McpToolCallRenderer(props: {
  toolName: string;
  argsText: string;
  result?: unknown;
  status?: { type: string };
  addResult: (result: unknown) => void;
}) {
  const { toolName, argsText, result, status } = props;
  const resourceUri = TOOL_RESOURCE_URIS[toolName];
  const [html, setHtml] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(true);

  // Load MCP app HTML widget when the tool has a resourceUri
  useEffect(() => {
    if (!resourceUri || html) return;
    setLoading(true);
    fetch("/studio/api/mcp-host", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method: "mcp-apps/read-resource", params: { uri: resourceUri } }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.html) setHtml(data.html);
        else setError(data.error || "No HTML returned");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [resourceUri, html]);

  const isRunning = status?.type === "running";
  const hasResult = result !== undefined;

  // If we have a loaded HTML widget, render it in a sandboxed iframe
  if (html) {
    return (
      <div className="rounded-lg border border-zinc-700 overflow-hidden my-2">
        <div className="px-3 py-1.5 bg-zinc-800/60 flex items-center gap-2 text-xs">
          <span className="text-zinc-400">Tool:</span>
          <span className="font-mono text-blue-400">{toolName}</span>
          {isRunning && <span className="text-yellow-400 animate-pulse">running…</span>}
        </div>
        <iframe
          srcDoc={html}
          sandbox="allow-scripts allow-same-origin"
          className="w-full border-0"
          style={{ minHeight: 200, maxHeight: 600 }}
          title={`MCP App: ${toolName}`}
        />
      </div>
    );
  }

  // Fallback: collapsible JSON display
  return (
    <div className="rounded-lg border border-zinc-700 my-2">
      <button
        type="button"
        className="w-full px-3 py-1.5 bg-zinc-800/60 flex items-center gap-2 text-xs text-left hover:bg-zinc-700/60 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
      >
        <span className="text-zinc-400">Tool:</span>
        <span className="font-mono text-blue-400">{toolName}</span>
        {isRunning && <span className="text-yellow-400 animate-pulse">running…</span>}
        {hasResult && <span className="text-green-400">✓</span>}
        <span className="ml-auto text-zinc-500">{collapsed ? "▶" : "▼"}</span>
      </button>
      {!collapsed && (
        <div className="px-3 py-2 text-xs">
          {argsText && argsText !== "{}" && (
            <pre className="text-zinc-400 whitespace-pre-wrap mb-2">{argsText}</pre>
          )}
          {hasResult && (
            <pre className="text-zinc-300 whitespace-pre-wrap border-t border-zinc-700 pt-2">
              {typeof result === "string" ? result : JSON.stringify(result, null, 2)}
            </pre>
          )}
          {loading && <p className="text-yellow-400">Loading widget…</p>}
          {error && <p className="text-red-400">{error}</p>}
        </div>
      )}
    </div>
  );
}

function MessageRow() {
  return (
    <MessagePrimitive.Root style={{ display: "flex", marginBottom: 8 }}>
      <MessagePrimitive.If user>
        <div style={{ marginLeft: "auto", maxWidth: "80%" }} className="rounded-lg px-4 py-2 bg-blue-600 text-white">
          <MessagePrimitive.Content />
        </div>
      </MessagePrimitive.If>
      <MessagePrimitive.If assistant>
        <div style={{ marginRight: "auto", maxWidth: "95%", width: "100%" }} className="space-y-2">
          <MessagePrimitive.Content
            components={{
              // Use our MCP-aware tool call renderer for all tool calls
              tools: {
                Override: McpToolCallRenderer as any, // eslint-disable-line @typescript-eslint/no-explicit-any
              },
            }}
          />
        </div>
      </MessagePrimitive.If>
    </MessagePrimitive.Root>
  );
}

function ComposerBar() {
  return (
    <div className="flex gap-2 w-full">
      <ComposerPrimitive.Input
        className="flex-1 resize-none rounded-lg bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-blue-500"
        placeholder="Describe what to generate..."
        rows={1}
      />
      <ComposerPrimitive.Send className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50">
        Send
      </ComposerPrimitive.Send>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center text-zinc-500 space-y-2">
        <p className="text-lg">What do you want to create?</p>
        <p className="text-sm">
          Try: &quot;Generate a cyberpunk cityscape&quot; or
          &quot;Run the character_sheet workflow&quot;
        </p>
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const chat = useChat({
    transport: new DefaultChatTransport({ api: "/studio/api/chat" }),
  });

  const runtime = useExternalStoreRuntime({
    isRunning: chat.status === "submitted" || chat.status === "streaming",
    messages: chat.messages,
    convertMessage: (msg) => {
      // Map useChat UIMessage → assistant-ui ThreadMessageLike
      const parts = msg.parts.map((part) => {
        if (part.type === "text") return { type: "text" as const, text: part.text };
        if ("toolName" in part) {
          const tp = part as Record<string, unknown>;
          return {
            type: "tool-call" as const,
            toolName: String(tp.toolName),
            args: tp.args as Record<string, unknown> | undefined,
            toolCallId: tp.toolCallId as string | undefined,
            result: tp.result,
          };
        }
        return { type: "text" as const, text: JSON.stringify(part) };
      });
      return {
        role: msg.role as "user" | "assistant" | "system",
        content: parts as any, // eslint-disable-line @typescript-eslint/no-explicit-any
        createdAt: ("createdAt" in msg && msg.createdAt) ? new Date(msg.createdAt as string | number | Date) : undefined,
      };
    },
    onNew: async (message) => {
      const textParts = message.content.filter(
        (p): p is { type: "text"; text: string } => p.type === "text",
      );
      chat.sendMessage({ text: textParts.map((p) => p.text).join("\n") });
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex flex-col h-full">
        <div className="p-4 border-b border-zinc-800">
          <h2 className="font-medium">Chat</h2>
          <p className="text-xs text-zinc-500">
            AI assistant with GPU tools — can call any MCP service
          </p>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <ThreadPrimitive.Empty>
            <EmptyState />
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages
            components={{ Message: MessageRow }}
          />
        </div>
        <div className="p-4 border-t border-zinc-800">
          <ComposerBar />
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
