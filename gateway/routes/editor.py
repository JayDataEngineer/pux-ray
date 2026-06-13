"""Video editor frontend — serves the built React SPA.

Serves static files from gateway/editor/ (built by web/editor/ via Vite).
The SPA calls /v1/wf API endpoints directly from the browser.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
logger = logging.getLogger(__name__)

_EDITOR_DIR = Path(__file__).resolve().parents[1] / "editor"
_LORAS_DIR = Path(__file__).resolve().parents[2] / "opt" / "wan2gp" / "loras"

# Map frontend model IDs to the lora subdirectory on disk
_MODEL_TO_LORA_DIR: dict[str, str] = {
    "wan/t2v_1.3B": "wan_1.3B",
    "wan/t2v": "wan_5B",
    "wan/i2v": "wan_i2v",
    "ltx2": "ltx2",
    "ltx2_19B": "ltx2",
    "ltxv_098_13b": "ltxv",
    # Z-Image models
    "z_image": "z_image",
    "z_image_base": "z_image",
    "anima_base": "anima",
    "flux": "flux",
    "flux_schnell": "flux",
    "flux_chroma": "flux",
    "flux2_dev": "flux2",
    "flux2_klein_4b": "flux2_klein",
    "flux2_klein_9b": "flux2_klein",
    "qwen_image_20B": "qwen_image",
    "qwen_image_2512_20B": "qwen_image",
    "hidream_o1": "hidream",
    "hidream_o1_dev": "hidream",
}


async def lora_list(request: Request) -> JSONResponse:
    """GET /v1/loras?model=ltx2 — list available LoRA files for a model."""
    model = request.query_params.get("model", "")
    lora_dir_name = _MODEL_TO_LORA_DIR.get(model, model)
    lora_dir = _LORAS_DIR / lora_dir_name

    if not lora_dir.exists() or not lora_dir.is_dir():
        return JSONResponse({"model": model, "loras": []})

    # Prevent directory traversal
    try:
        lora_dir.resolve().relative_to(_LORAS_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "invalid model"}, status_code=400)

    files = sorted(
        f.name for f in lora_dir.iterdir()
        if f.is_file() and f.suffix in (".safetensors", ".pt", ".bin")
    )
    return JSONResponse({"model": model, "loras": files})

_MIME_TYPES = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


async def editor_page(request: Request) -> HTMLResponse:
    """GET /editor — serve the video editor SPA."""
    index = _EDITOR_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h1>Editor not built</h1><p>Run <code>cd web/editor && pnpm build</code></p>",
            status_code=404,
        )
    return HTMLResponse(
        index.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


async def editor_static(request: Request) -> Response:
    """GET /editor/{path} — serve static assets."""
    path = request.path_params.get("path", "")
    file_path = _EDITOR_DIR / path

    # Prevent directory traversal
    try:
        file_path = file_path.resolve()
        file_path.relative_to(_EDITOR_DIR.resolve())
    except ValueError:
        return Response(status_code=403)

    if not file_path.exists() or not file_path.is_file():
        # SPA fallback — serve index.html for client-side routing
        index = _EDITOR_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return Response(status_code=404)

    ext = file_path.suffix.lower()
    media_type = _MIME_TYPES.get(ext, "application/octet-stream")

    # Hashed asset filenames (e.g. index-AbCd123.js) are immutable
    is_hashed = len(file_path.stem.split("-")) > 1 and any(c.isdigit() for c in file_path.stem)
    cache = "public, immutable, max-age=31536000" if is_hashed else "no-cache"

    return Response(
        content=file_path.read_bytes(),
        media_type=media_type,
        headers={"Cache-Control": cache},
    )


# ── Secure LLM Enhancement Endpoints ─────────────────────────────────────────

# Encryption key for storing API keys (derived from environment variable or default)
_ENCRYPTION_KEY_SALT = os.environ.get("LLM_KEY_SALT", "tech-noir-llm-keys-default").encode()
_ENCRYPTION_KEY = hashlib.pbkdf2_hmac(
    "sha256",
    _ENCRYPTION_KEY_SALT,
    b"tech-noir-llm-key-derivation",
    100000,
    dklen=32
)
_CIPHER = Fernet(base64.urlsafe_b64encode(_ENCRYPTION_KEY))

# In-memory storage for encrypted API keys (in production, use a database)
# Structure: { key_id: {"name": str, "encrypted_key": str, "baseUrl": str, "model": str} }
_STORED_KEYS: dict[str, dict[str, Any]] = {}


def _encrypt_key(api_key: str) -> str:
    """Encrypt an API key for storage."""
    return _CIPHER.encrypt(api_key.encode()).decode()


def _decrypt_key(encrypted_key: str) -> str:
    """Decrypt a stored API key."""
    return _CIPHER.decrypt(encrypted_key.encode()).decode()


def _generate_key_id() -> str:
    """Generate a unique ID for a stored API key."""
    import time
    import uuid
    return f"llm_key_{int(time.time())}_{uuid.uuid4().hex[:8]}"


async def llm_key_store(request: Request) -> JSONResponse:
    """POST /v1/llm/keys — Store an encrypted LLM API key."""
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        base_url = data.get("baseUrl", "").strip()
        api_key = data.get("apiKey", "").strip()
        model = data.get("model", "").strip()

        if not all([name, base_url, api_key, model]):
            return JSONResponse(
                {"error": "Missing required fields: name, baseUrl, apiKey, model"},
                status_code=400
            )

        # Encrypt the API key
        encrypted_api_key = _encrypt_key(api_key)
        key_id = _generate_key_id()

        # Store metadata and encrypted key (never store plaintext)
        _STORED_KEYS[key_id] = {
            "name": name,
            "encrypted_key": encrypted_api_key,
            "baseUrl": base_url,
            "model": model,
        }

        logger.info(f"Stored LLM key: {name} (ID: {key_id})")

        # Return only the key ID and safe metadata (NOT the API key)
        return JSONResponse({
            "key_id": key_id,
            "name": name,
            "baseUrl": base_url,
            "model": model,
        })

    except Exception as e:
        logger.error(f"Error storing LLM key: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def llm_key_list(request: Request) -> JSONResponse:
    """GET /v1/llm/keys — List stored LLM keys (metadata only)."""
    try:
        keys = []
        for key_id, data in _STORED_KEYS.items():
            # Only return safe metadata, never the encrypted key
            keys.append({
                "key_id": key_id,
                "name": data["name"],
                "baseUrl": data["baseUrl"],
                "model": data["model"],
            })

        return JSONResponse({"keys": keys})

    except Exception as e:
        logger.error(f"Error listing LLM keys: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def llm_key_delete(request: Request) -> JSONResponse:
    """DELETE /v1/llm/keys/{key_id} — Delete a stored LLM key."""
    key_id = request.path_params.get("key_id")

    if not key_id or key_id not in _STORED_KEYS:
        return JSONResponse({"error": "Key not found"}, status_code=404)

    name = _STORED_KEYS[key_id]["name"]
    del _STORED_KEYS[key_id]
    logger.info(f"Deleted LLM key: {name} (ID: {key_id})")

    return JSONResponse({"message": "Key deleted"})


async def llm_enhance(request: Request) -> JSONResponse:
    """POST /v1/llm/enhance — Enhance a prompt using a stored LLM key.
    
    This endpoint makes the actual LLM API call, so the frontend never
    needs to handle or expose the API key.
    """
    try:
        data = await request.json()
        key_id = data.get("key_id")
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_prompt = data.get("prompt", "").strip()

        if not key_id or not user_prompt:
            return JSONResponse(
                {"error": "Missing required fields: key_id, prompt"},
                status_code=400
            )

        # Retrieve the stored key
        if key_id not in _STORED_KEYS:
            return JSONResponse({"error": "Key not found"}, status_code=404)

        key_data = _STORED_KEYS[key_id]

        # Decrypt the API key in memory only
        api_key = _decrypt_key(key_data["encrypted_key"])
        base_url = key_data["baseUrl"].rstrip("/")
        model = key_data["model"]

        # Make the LLM API call
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.8,
                },
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"LLM API error: {response.status_code} - {error_text}")
            return JSONResponse(
                {"error": f"LLM API error ({response.status_code}): {error_text}"},
                status_code=response.status_code
            )

        result = response.json()
        enhanced_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not enhanced_text:
            return JSONResponse({"error": "Model returned empty response"}, status_code=500)

        logger.info(f"Successfully enhanced prompt using {key_data['name']}")

        return JSONResponse({
            "result": enhanced_text.strip(),
            "model": model,
            "provider": key_data["name"],
        })

    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════════
# AI Chat with web research tools (scrape / search / research)
# ══════════════════════════════════════════════════════════════════════════════

_WEB_RESEARCH_URL = os.environ.get("WEB_RESEARCH_URL", "http://localhost:8000/mcp")

_WEB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scrape",
            "description": "Scrape a URL and extract clean markdown content. Use this to read a specific web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web using multiple search engines. Returns titles, URLs, and short snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "top_k": {"type": "integer", "description": "Max results to return (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research",
            "description": "Search the web AND scrape the top results in one call. Best for researching a topic — you get actual page content, not just snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                    "max_results": {"type": "integer", "description": "Number of results to scrape (1-5, default 3)", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to web research tools. "
    "Use them when the user asks about anything that requires current information or web content. "
    "Be concise. When you use a tool, summarize the key findings from the results."
)


async def _call_web_tool(name: str, args: dict) -> str:
    """Call a tool on the web-research MCP server via HTTP JSON-RPC."""
    import httpx

    # Initialize session (stateless, so we do init + call in sequence)
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Initialize
        init_resp = await client.post(
            _WEB_RESEARCH_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "editor-chat", "version": "1.0"},
                },
            },
        )
        session_id = None
        if init_resp.status_code < 400:
            text = init_resp.text
            data_line = next((l for l in text.split("\n") if l.startswith("data: ")), None)
            if data_line:
                msg = json.loads(data_line[6:])
                session_id = (msg.get("result") or {}).get("sessionId")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        # Call tool
        resp = await client.post(
            _WEB_RESEARCH_URL,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            },
        )

        if resp.status_code >= 400:
            return f"Error: HTTP {resp.status_code} from web-research server"

        text = resp.text
        data_line = next((l for l in text.split("\n") if l.startswith("data: ")), None)
        if not data_line:
            return "Error: no data in MCP response"

        msg = json.loads(data_line[6:])
        if msg.get("error"):
            return f"Tool error: {msg['error'].get('message', str(msg['error']))}"

        content = (msg.get("result") or {}).get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]

        return json.dumps(msg.get("result"), default=str)


async def llm_chat(request: Request) -> JSONResponse:
    """POST /v1/llm/chat — Agentic chat loop with web research tools.

    Sends the conversation to the LLM with tool definitions. If the LLM
    requests tools, executes them and loops. Returns the final response
    along with any tool-call metadata for the frontend to display.
    """
    try:
        data = await request.json()
        key_id = data.get("key_id")
        messages = data.get("messages", [])

        if not key_id:
            return JSONResponse({"error": "Missing key_id"}, status_code=400)

        if key_id not in _STORED_KEYS:
            return JSONResponse({"error": "Key not found"}, status_code=404)

        key_data = _STORED_KEYS[key_id]
        api_key = _decrypt_key(key_data["encrypted_key"])
        base_url = key_data["baseUrl"].rstrip("/")
        model = key_data["model"]

        # Build full message list with system prompt
        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages

        import httpx

        tool_calls_log: list[dict] = []
        max_iterations = 6  # safety limit

        for _ in range(max_iterations):
            async with httpx.AsyncClient(timeout=60.0) as client:
                body: dict[str, Any] = {
                    "model": model,
                    "messages": full_messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "tools": _WEB_TOOLS,
                }
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    json=body,
                )

            if resp.status_code != 200:
                error_text = resp.text[:500]
                return JSONResponse(
                    {"error": f"LLM API error ({resp.status_code}): {error_text}"},
                    status_code=resp.status_code,
                )

            result = resp.json()
            choice = result.get("choices", [{}])[0]
            assistant_msg = choice.get("message", {})
            finish_reason = choice.get("finish_reason")

            # If the LLM wants to call tools
            if finish_reason == "tool_calls" and assistant_msg.get("tool_calls"):
                full_messages.append(assistant_msg)

                for tc in assistant_msg["tool_calls"]:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"Chat tool call: {tool_name}({json.dumps(tool_args)[:200]})")
                    tool_calls_log.append({"name": tool_name, "args": tool_args})

                    tool_result = await _call_web_tool(tool_name, tool_args)

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result[:15000],  # cap to protect context window
                    })
                # Loop back to LLM with tool results
                continue

            # Final text response
            content = assistant_msg.get("content", "")
            return JSONResponse({
                "result": content.strip(),
                "model": model,
                "provider": key_data["name"],
                "tool_calls": tool_calls_log,
            })

        # Exhausted iterations — return whatever we have
        return JSONResponse({
            "result": assistant_msg.get("content", "I wasn't able to complete the research in time.").strip(),
            "model": model,
            "provider": key_data["name"],
            "tool_calls": tool_calls_log,
        })

    except Exception as e:
        logger.error(f"Error in llm_chat: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
