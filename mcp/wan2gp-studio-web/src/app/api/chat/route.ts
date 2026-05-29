import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { z } from "zod";
import { callMcpTool } from "@/lib/mcp-client";

const LLM_BASE_URL =
  process.env.LLM_BASE_URL ||
  "http://tech-noir-ray-serve-svc.ai-services:8000/llm/v1";
const LLM_MODEL = process.env.LLM_MODEL || "default";
const LLM_API_KEY = process.env.LLM_API_KEY || "not-needed";

/**
 * Build AI SDK tool definitions wrapping MCP tools.
 *
 * Each tool proxies to the Python MCP server via callMcpTool().
 * The `maxSteps` on streamText enables multi-turn tool calling.
 */
function buildMcpTools(): Record<string, import("ai").Tool> {
  return {
    run: {
      description:
        "Run any registered GPU inference service. Use list_models first to discover available services.",
      inputSchema: z.object({
        service: z.string().describe("Service name (e.g. 'wan2gp', 'kokoro')"),
        params: z.record(z.string(), z.unknown()).optional().describe(
          "Service parameters: model, prompt, text, image_b64, audio_b64, etc.",
        ),
      }),
      execute: async ({ service, params }: { service: string; params?: Record<string, unknown> }) =>
        callMcpTool("run", { service, ...(params ?? {}) }),
    },

    list_models: {
      description: "List available models by category.",
      inputSchema: z.object({
        category: z.string().optional().describe("Filter: tts, image, video, 3d, audio, llm"),
      }),
      execute: async ({ category }: { category?: string }) => callMcpTool("list_models", { category }),
    },

    tts_speak: {
      description: "Generate speech from text (Kokoro CPU, Qwen3-TTS GPU, MOSS GPU).",
      inputSchema: z.object({
        text: z.string().describe("Text to speak"),
        engine: z.string().optional().describe("kokoro | qwen3_tts | moss_voicegenerator"),
        mode: z.string().optional().describe("custom_voice | voice_design | voice_clone"),
        voice: z.string().optional().describe("Voice preset name"),
        language: z.string().optional().describe("Language (default: English)"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("tts_speak", args),
    },

    transcribe: {
      description: "Transcribe audio to text (Whisper CPU or VibeVoice GPU).",
      inputSchema: z.object({
        audio_b64: z.string().describe("Base64-encoded audio"),
        engine: z.string().optional().describe("whisper | vibevoice"),
        language: z.string().optional().describe("Language hint"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("transcribe", args),
    },

    generate_sound: {
      description: "Generate a sound effect from a text prompt.",
      inputSchema: z.object({
        prompt: z.string().describe("Sound description"),
        duration_seconds: z.number().optional().describe("Duration (1-30s)"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("generate_sound", args),
    },

    generate_music: {
      description: "Generate music from a text prompt.",
      inputSchema: z.object({
        prompt: z.string().describe("Music description"),
        lyrics: z.string().optional().describe("Optional lyrics"),
        duration_seconds: z.number().optional().describe("Duration (5-60s)"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("generate_music", args),
    },

    forge_status: {
      description: "Check GPU status and VRAM usage.",
      inputSchema: z.object({
        detailed: z.boolean().optional().describe("Include per-service breakdown"),
      }),
      execute: async ({ detailed }: { detailed?: boolean }) => callMcpTool("forge_status", { detailed }),
    },

    load_service: {
      description: "Preload a model on GPU.",
      inputSchema: z.object({
        service: z.string().describe("Service name"),
        model: z.string().optional().describe("Model variant"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("load_service", args),
    },

    unload_services: {
      description: "Release all GPU memory.",
      inputSchema: z.object({}),
      execute: async () => callMcpTool("unload_services", {}),
    },

    workflow_list_specs: {
      description: "List available workflow pipeline specs.",
      inputSchema: z.object({}),
      execute: async () => callMcpTool("workflow_list_specs", {}),
    },

    workflow_start_run: {
      description: "Start a new workflow run.",
      inputSchema: z.object({
        spec_name: z.string().describe("Workflow spec name"),
        inputs: z.record(z.string(), z.unknown()).optional().describe("Input values"),
        manual: z.boolean().optional().describe("Manual mode (default: true)"),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("workflow_start_run", args),
    },

    workflow_execute_step: {
      description: "Execute a single workflow step.",
      inputSchema: z.object({
        spec_name: z.string(),
        run_id: z.string(),
        step_id: z.string().describe("Step to execute"),
        params: z.record(z.string(), z.unknown()).optional(),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("workflow_execute_step", args),
    },

    workflow_get_run: {
      description: "Get workflow run status.",
      inputSchema: z.object({
        spec_name: z.string(),
        run_id: z.string(),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("workflow_get_run", args),
    },

    workflow_rerun_step: {
      description: "Rerun a workflow from a specific step.",
      inputSchema: z.object({
        spec_name: z.string(),
        run_id: z.string(),
        step_id: z.string(),
        params: z.record(z.string(), z.unknown()).optional(),
      }),
      execute: async (args: Record<string, unknown>) => callMcpTool("workflow_rerun_step", args),
    },
  } as unknown as Record<string, import("ai").Tool>;
}

export const maxDuration = 300;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const llm = createOpenAI({
    baseURL: LLM_BASE_URL,
    apiKey: LLM_API_KEY,
  });

  const result = streamText({
    model: llm(LLM_MODEL),
    messages,
    tools: buildMcpTools(),
  });

  return result.toTextStreamResponse();
}
