import { callMcpTool } from "@/lib/mcp-client";

export async function GET() {
  try {
    const result = await callMcpTool("list_models", {});
    const content = result.content?.[0]?.text;
    if (content) {
      const parsed = JSON.parse(content);
      return Response.json(parsed.catalog || { data: [] });
    }
    return Response.json({ data: [] });
  } catch (e: any) {
    return Response.json({ data: [], error: e.message }, { status: 503 });
  }
}
