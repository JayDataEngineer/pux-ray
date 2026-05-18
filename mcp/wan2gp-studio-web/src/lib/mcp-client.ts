/**
 * MCP client for connecting to the wan2gp-studio Python MCP server.
 */
import { Client } from "@modelcontextprotocol/sdk/client";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp";

const MCP_SERVER_URL =
  process.env.MCP_SERVER_URL || "http://wan2gp-studio-mcp.mcp:8002/mcp";

let mcpClient: Client | null = null;

export async function getMcpClient(): Promise<Client> {
  if (mcpClient) return mcpClient;

  const client = new Client({
    name: "wan2gp-studio-web",
    version: "0.1.0",
  });

  const transport = new StreamableHTTPClientTransport(
    new URL(MCP_SERVER_URL),
  );

  await client.connect(transport);
  mcpClient = client;
  return client;
}

export async function getMcpTools() {
  const client = await getMcpClient();
  const { tools } = await client.listTools();
  return tools;
}

export async function callMcpTool(name: string, args: Record<string, unknown>) {
  const client = await getMcpClient();
  const result = await client.callTool({ name, arguments: args });
  return result;
}
