import { callMcpTool } from "@/lib/mcp-client";

export async function GET() {
  try {
    const result = await callMcpTool("tts_voices", {});
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "Failed to list voices" },
      { status: 500 },
    );
  }
}
