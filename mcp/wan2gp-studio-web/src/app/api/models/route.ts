import { callMcpTool } from "@/lib/mcp-client";

export async function GET() {
  try {
    const result = await callMcpTool("list_models", {}) as { content?: { text?: string }[] };
    const content = result.content?.[0]?.text;
    if (content) {
      const parsed = JSON.parse(content);
      return Response.json({
        data: parsed.catalog?.data || [],
        gpu_status: parsed.gpu_status || null,
      });
    }
    return Response.json({ data: [], gpu_status: null });
  } catch (e: any) {
    return Response.json({ data: [], gpu_status: null, error: e.message }, { status: 503 });
  }
}
