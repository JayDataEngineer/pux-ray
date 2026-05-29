import { callMcpTool } from "@/lib/mcp-client";

export const maxDuration = 300;

export async function POST(req: Request) {
  const { messages } = await req.json();

  try {
    // Call the LLM via MCP chat tool (non-streaming, avoids forge SSE buffer issue).
    // The MCP chat tool sends stream:false to the LLM and returns the full result.
    const mcpResult = await callMcpTool("chat", { messages }) as {
      content?: { text?: string }[];
      structuredContent?: Record<string, unknown>;
    };

    let text = "";
    if (mcpResult.structuredContent) {
      const sc = mcpResult.structuredContent as {
        data?: { choices?: { message?: { content?: string } }[] };
      };
      text = sc?.data?.choices?.[0]?.message?.content || "";
    } else if (mcpResult.content?.[0]?.text) {
      try {
        const parsed = JSON.parse(mcpResult.content[0].text);
        text = parsed?.data?.choices?.[0]?.message?.content || "";
      } catch { /* empty */ }
    }

    // Wrap in AI SDK data-stream format for the client's useChat hook.
    const encoder = new TextEncoder();
    const parts: string[] = [];
    if (text) {
      parts.push(`0:${JSON.stringify(text)}\n`);
    }
    parts.push(`d:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n`);

    const stream = new ReadableStream({
      start(controller) {
        for (const part of parts) {
          controller.enqueue(encoder.encode(part));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "X-Vercel-AI-Data-Stream": "v1",
      },
    });
  } catch (e: any) {
    return Response.json({ error: e.message || "Chat failed" }, { status: 500 });
  }
}
