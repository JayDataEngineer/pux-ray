import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { messages, model, temperature, max_tokens } = await req.json();

    if (!messages?.length) {
      return Response.json({ error: "Missing 'messages'" }, { status: 400 });
    }

    const result = await callMcpTool("chat", {
      messages,
      model: model || null,
      temperature: temperature ?? null,
      max_tokens: max_tokens ?? null,
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json({ error: e.message || "Chat failed" }, { status: 500 });
  }
}
