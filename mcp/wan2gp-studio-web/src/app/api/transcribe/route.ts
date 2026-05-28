import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { audio_b64, engine, language } = await req.json();

    if (!audio_b64) {
      return Response.json({ error: "Missing 'audio_b64'" }, { status: 400 });
    }

    const result = await callMcpTool("transcribe", {
      audio_b64,
      engine: engine || "whisper",
      language: language || null,
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json({ error: e.message || "Transcription failed" }, { status: 500 });
  }
}
