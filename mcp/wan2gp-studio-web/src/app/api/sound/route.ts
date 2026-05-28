import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { prompt, duration_seconds, seed } = await req.json();

    if (!prompt) {
      return Response.json({ error: "Missing 'prompt'" }, { status: 400 });
    }

    const result = await callMcpTool("generate_sound", {
      prompt,
      duration_seconds: duration_seconds || 5.0,
      seed: seed || null,
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json({ error: e.message || "Sound generation failed" }, { status: 500 });
  }
}
