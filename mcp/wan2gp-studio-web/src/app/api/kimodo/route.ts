import { callMcpTool } from "@/lib/mcp-client";

export async function POST() {
  try {
    // Preload the kimodo_demo service on GPU via the admin/load endpoint.
    // The forge auto-evicts existing services (including pipeline-locked)
    // when an explicit preload is requested.
    const result = await callMcpTool("load_service", {
      service: "kimodo_demo",
    });
    return Response.json(result);
  } catch (e: unknown) {
    return Response.json(
      { error: e instanceof Error ? e.message : "Failed to start Kimodo" },
      { status: 500 },
    );
  }
}
