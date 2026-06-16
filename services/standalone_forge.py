"""Standalone forge — ASGI app without Ray Serve.

Runs ForgeCore directly behind uvicorn. Same API as the Ray Serve wrapper
but zero Ray overhead. For development and single-GPU deployment.

Endpoints:
  GET  /forge          — status (VRAM, loaded models)
  POST /forge          — invoke/preload/release
  GET  /health         — health check

Usage:
  uvicorn services.standalone_forge:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from services.forge_base import ForgeCore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ─── Global forge instance (lazy init on first request) ─────────────────────
_core: ForgeCore | None = None
_load_lock = asyncio.Lock()


async def get_core() -> ForgeCore:
    global _core
    if _core is None:
        async with _load_lock:
            if _core is None:  # Double-check after lock
                from services.forge_base import SERVICE_MAP
                _core = ForgeCore(service_map=SERVICE_MAP)
    return _core


# ─── Handlers ────────────────────────────────────────────────────────────────


async def handle_health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def handle_forge(request: Request) -> Response:
    core = await get_core()

    if request.method == "GET":
        return JSONResponse(await core.status())

    body = await request.json()
    action = body.get("action", "")

    if action == "release":
        svc = body.get("service")
        result = await core.release(svc)
        return JSONResponse(result)

    if action == "status":
        return JSONResponse(await core.status())

    if action == "preload":
        service = body.get("service")
        model = body.get("model")
        quant = body.get("quant")
        result = await core.preload(service, model, quant)
        return JSONResponse(result)

    service = body.get("service")
    if not service:
        return JSONResponse(
            {"status": "error", "error": "Specify 'service'"},
            status_code=400,
        )

    payload = {k: v for k, v in body.items() if k != "service"}
    model = payload.get("model")
    quant = payload.get("quant", None)

    async with _load_lock:
        result = await core.invoke(service, payload, model, quant)

    return JSONResponse(result)


# ─── App ─────────────────────────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/health", handle_health),
        Route("/forge", handle_forge, methods=["GET", "POST"]),
    ],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
