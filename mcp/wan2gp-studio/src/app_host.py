"""MCP App Host endpoint for assistant-ui.

Handles POST requests from the McpAppsRemoteHost client:
  - mcp-apps/read-resource → returns MCP app HTML
  - tools/call → calls an MCP tool
  - tools/list → lists tools with _meta (for assistant-ui widget binding)
  - resources/read → reads an MCP resource
  - resources/list → lists MCP resources

Mounted at /host
"""
from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import JSONResponse

from .apps.registry import get_app_html, list_apps, MCP_APP_MIME


async def handle_app_host(request: Request) -> JSONResponse:
    """Handle MCP app host requests from assistant-ui."""
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "mcp-apps/read-resource":
        uri = params.get("uri", "")
        html = get_app_html(uri)
        if html is None:
            return JSONResponse(
                {"error": f"Resource not found: {uri}"}, status_code=404,
            )
        return JSONResponse({
            "uri": uri,
            "mimeType": MCP_APP_MIME,
            "html": html,
            "meta": {"ui": {"prefersBorder": True}},
        })

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        from .server import mcp
        result = await mcp.call_tool(tool_name, tool_args)
        # ToolResult -> dict (ToolResult is not JSON serializable)
        if hasattr(result, "content"):
            result_dict = {
                "content": [
                    {"type": c.type, "text": c.text}
                    for c in (result.content or [])
                ],
                "isError": getattr(result, "is_error", False),
            }
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"result": str(result)}
        return JSONResponse(result_dict)

    if method == "tools/list":
        from .server import mcp
        tools = await mcp.list_tools()
        # Convert to wire format with _meta
        wire_tools = []
        for t in tools:
            entry: dict = {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.parameters,
            }
            if t.meta:
                entry["_meta"] = t.meta
            wire_tools.append(entry)
        return JSONResponse({"tools": wire_tools})

    if method == "resources/read":
        uri = params.get("uri", "")
        html = get_app_html(uri)
        if html is None:
            return JSONResponse(
                {"error": f"Resource not found: {uri}"}, status_code=404,
            )
        return JSONResponse({
            "contents": [{
                "uri": uri,
                "mimeType": MCP_APP_MIME,
                "text": html,
                "_meta": {"ui": {"prefersBorder": True}},
            }],
        })

    if method == "resources/list":
        return JSONResponse({
            "resources": [
                {
                    "uri": app["resourceUri"],
                    "name": app["name"],
                    "description": app["description"],
                    "mimeType": MCP_APP_MIME,
                    "_meta": {"ui": {"prefersBorder": True}},
                }
                for app in list_apps()
            ],
        })

    return JSONResponse({"error": f"Unknown method: {method}"}, status_code=400)
