import { callMcpTool } from "@/lib/mcp-client";

export async function POST() {
  try {
    const result = await callMcpTool("unload_services", {});
    return Response.json(result);
  } catch (e: any) {
    return Response.json({ error: e.message || "Unload failed" }, { status: 500 });
  }
}
