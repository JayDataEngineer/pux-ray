import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const LLM_BASE_URL =
  process.env.LLM_BASE_URL ||
  "http://tech-noir-ray-serve-svc.ai-services:8000/llm/v1";
const LLM_MODEL = process.env.LLM_MODEL || "default";
const LLM_API_KEY = process.env.LLM_API_KEY || "not-needed";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const llm = createOpenAI({
    baseURL: LLM_BASE_URL,
    apiKey: LLM_API_KEY,
  });

  const result = streamText({
    model: llm(LLM_MODEL),
    messages,
    system: `You are a creative AI assistant with access to GPU-powered generation tools.

You have a single tool called "run" that calls any registered service.
Pass {service: "<name>", params: {<whatever the service needs>}}.

Use list_models first to discover available services and models.
Use forge_status to check GPU/VRAM state.

Be creative and helpful. Suggest parameters that would produce good results.`,
  });

  return result.toTextStreamResponse();
}
