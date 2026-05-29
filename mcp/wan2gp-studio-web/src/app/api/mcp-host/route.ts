import { callMcpTool, getMcpTools } from "@/lib/mcp-client";

const MCP_APP_MIME = "text/html;profile=mcp-app";

/**
 * MCP App Host endpoint — handles { method, params } POST requests
 * from assistant-ui's McpAppsRemoteHost.
 *
 * Methods:
 *  - mcp-apps/read-resource → return MCP app HTML widget
 *  - tools/call → call an MCP tool
 *  - tools/list → list MCP tools with _meta (for widget binding)
 *  - resources/read → read an MCP resource
 *  - resources/list → list MCP resources
 */
export async function POST(req: Request) {
  const body = await req.json();
  const method = body.method ?? "";
  const params = body.params ?? {};

  try {
    switch (method) {
      case "mcp-apps/read-resource": {
        const uri = params.uri ?? "";
        // Read the resource from MCP server
        const result = await callMcpTool("resources/read", { uri }) as {
          content?: Array<{ type: string; text?: string }>;
        };
        const content = result?.content;
        if (content && content.length > 0 && content[0].text) {
          const text = content[0].text;
          return Response.json({
            uri,
            mimeType: MCP_APP_MIME,
            html: text,
            meta: { ui: { prefersBorder: true } },
          });
        }
        return Response.json({ error: "Resource not found" }, { status: 404 });
      }

      case "tools/call": {
        const name = params.name ?? "";
        const args = params.arguments ?? {};
        const result = await callMcpTool(name, args);
        return Response.json(result);
      }

      case "tools/list": {
        const tools = await getMcpTools();
        // Each tool from the MCP SDK already includes _meta if the server set it
        return Response.json({ tools });
      }

      case "resources/read": {
        const uri = params.uri ?? "";
        const result = await callMcpTool("resources/read", { uri });
        return Response.json(result);
      }

      case "resources/list": {
        const result = await callMcpTool("resources/list", {});
        return Response.json(result);
      }

      default:
        return Response.json({ error: `Unknown method: ${method}` }, { status: 400 });
    }
  } catch (e: unknown) {
    return Response.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    );
  }
}
