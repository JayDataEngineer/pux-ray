import { callMcpTool } from "@/lib/mcp-client";

export async function GET() {
  try {
    const result = await callMcpTool("tts_voices", {}) as {
      content?: { text?: string }[];
      structuredContent?: { engines?: unknown[]; voices?: Record<string, string[]> };
    };
    // Prefer structuredContent (FastMCP), fall back to parsing text content
    if (result.structuredContent?.engines) {
      return Response.json(result.structuredContent);
    }
    const text = result.content?.[0]?.text;
    if (text) {
      const parsed = JSON.parse(text);
      return Response.json(parsed);
    }
    return Response.json({ engines: [], voices: {} });
  } catch (e: any) {
    return Response.json(
      { engines: [], voices: {}, error: e.message || "Failed to list voices" },
      { status: 500 },
    );
  }
}
