"use client";

import { useExternalStoreRuntime, AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import {
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
} from "@assistant-ui/react";

function MessageRow() {
  return (
    <MessagePrimitive.Root style={{ display: "flex", marginBottom: 8 }}>
      <MessagePrimitive.If user>
        <div style={{ marginLeft: "auto", maxWidth: "80%" }} className="rounded-lg px-4 py-2 bg-blue-600 text-white">
          <MessagePrimitive.Content />
        </div>
      </MessagePrimitive.If>
      <MessagePrimitive.If assistant>
        <div style={{ marginRight: "auto", maxWidth: "80%" }} className="rounded-lg px-4 py-2 bg-zinc-800 text-zinc-100">
          <MessagePrimitive.Content />
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
