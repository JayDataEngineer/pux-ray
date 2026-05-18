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

You can generate videos, images, 3D models, and audio. When a user asks you to create something:
1. Choose the right tool (generate_video, generate_image, generate_3d, generate_audio)
2. Extract parameters from their request (prompt, style, size, etc.)
3. Call the tool and describe the result

Available tools:
- generate_video: Creates videos from text or image input (models: wan/t2v, wan/i2v, hunyuan/t2v, hunyuan/i2v, ltx2)
- generate_image: Creates images from text (models: flux, flux_schnell, flux2_dev, qwen-image-edit)
- generate_3d: Converts images to 3D meshes (models: trellis, anigen)
- generate_audio: Generates speech, sound effects, music (models: moss-soundeffect, kokoro, espeak)
- list_models: Shows all available models and their capabilities
- forge_status: Shows GPU/VRAM status

Be creative and helpful. Suggest parameters that would produce good results.`,
  });

  return result.toTextStreamResponse();
}
