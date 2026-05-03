"""Worker HTTP server for Docker-based creative tools.

Runs inside each worker container. Ray head node calls this via HTTPToolMixin.

Endpoints:
  POST /generate  - tool-specific generation
  GET  /health    - readiness check
  POST /load      - pre-load model into memory

The tool-specific handler is selected via the TOOL_NAME env var.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("worker")

app = FastAPI()

# Handler is set by _load_handler() after startup
_handler = None


def _load_handler():
    """Import and instantiate the tool handler based on TOOL_NAME env var."""
    global _handler
    tool_name = os.environ.get("TOOL_NAME")
    if not tool_name:
        raise RuntimeError("TOOL_NAME env var is required")

    # Map tool names to handler module paths
    handlers = {
        "trellis": "workers.trellis_handler",
        "anigen": "workers.anigen_handler",
        "vibevoice": "workers.vibevoice_handler",
    }

    module_path = handlers.get(tool_name)
    if not module_path:
        raise RuntimeError(f"Unknown tool: {tool_name}. Known: {list(handlers.keys())}")

    # Add repo root to path so handler can import tool code
    repo_root = os.environ.get("REPO_ROOT", "/app/repo")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module = importlib.import_module(module_path)
    _handler = module.Handler()
    logger.info("Loaded handler for %s", tool_name)


@app.on_event("startup")
async def startup():
    """Load the tool handler on startup."""
    try:
        _load_handler()
        logger.info("Worker ready")
    except Exception:
        logger.exception("Failed to load handler")
        # Don't crash — /health will return 503 until resolved


@app.get("/health")
async def health():
    """Readiness check. Returns 200 when handler is loaded and model is ready."""
    if _handler is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return await _handler.health()


@app.post("/load")
async def load(request: Request):
    """Pre-load model into GPU memory."""
    if _handler is None:
        return JSONResponse({"error": "handler not loaded"}, status_code=503)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    return await _handler.load(body)


@app.post("/generate")
async def generate(request: Request):
    """Generate output. Request format depends on the tool."""
    if _handler is None:
        return JSONResponse({"error": "handler not loaded"}, status_code=503)
    return await _handler.generate(request)


def main():
    port = int(os.environ.get("PORT", "18401"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
