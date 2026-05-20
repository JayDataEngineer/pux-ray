import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { service, params } = await req.json();

    if (!service) {
      return Response.json({ error: "Missing 'service' parameter" }, { status: 400 });
    }

    const result = await callMcpTool("run", { service, params: params || {} });
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "Generation failed" },
      { status: 500 },
    );
  }
}
