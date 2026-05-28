import { callMcpTool } from "@/lib/mcp-client";

export async function POST() {
  try {
    // Preload the kimodo_demo forge service, which starts the Viser server
    const result = await callMcpTool("run", {
      service: "wan2gp",
      params: { action: "preload", service: "kimodo_demo" },
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "Failed to start Kimodo" },
      { status: 500 },
    );
  }
}
