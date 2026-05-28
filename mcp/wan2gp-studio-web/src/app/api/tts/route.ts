import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { text, engine, mode, voice, instruct, ref_audio_b64, language } =
      await req.json();

    if (!text) {
      return Response.json(
        { error: "Missing 'text' parameter" },
        { status: 400 },
      );
    }

    const result = await callMcpTool("tts_speak", {
      text,
      engine: engine || "kokoro",
      mode: mode || "custom_voice",
      voice: voice || null,
      instruct: instruct || null,
      ref_audio_b64: ref_audio_b64 || null,
      language: language || "English",
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "TTS generation failed" },
      { status: 500 },
    );
  }
}
