import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { service, model } = await req.json();

    if (!service) {
      return Response.json({ error: "Missing 'service'" }, { status: 400 });
    }

    const result = await callMcpTool("load_service", {
      service,
      model: model || null,
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json({ error: e.message || "Load failed" }, { status: 500 });
  }
}
