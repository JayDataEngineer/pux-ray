import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { tool, args } = await req.json();

    if (!tool) {
      return Response.json({ error: "Missing 'tool' parameter" }, { status: 400 });
    }

    const result = await callMcpTool(tool, args || {});
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "Generation failed" },
      { status: 500 },
    );
  }
}
