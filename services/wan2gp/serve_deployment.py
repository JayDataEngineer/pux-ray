"""Wan2GP Ray Serve deployment — standalone GPU deployment with mmgp.

Runs as its own deployment (num_gpus: 1.0) separate from the Forge.
All GPU models route through Wan2GPService directly.
mmgp handles ALL VRAM/CPU/RAM management — no external scheduler needed.
"""
from __future__ import annotations

import asyncio
import logging

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from services.wan2gp.deployment import Wan2GPService, discover_models

logger = logging.getLogger(__name__)


@serve.deployment(
    name="wan2gp",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0},
)
class Wan2GPDeployment:
    """Thin Ray Serve wrapper around Wan2GPService."""

    def __init__(self):
        self._svc = Wan2GPService()
        logger.info("Wan2GP deployment initialized: %d models discovered",
                     len(self._svc.registry))

    async def invoke(self, payload: dict, model: str | None = None) -> dict:
        if model:
            payload["model"] = model
        return await asyncio.to_thread(self._svc.infer, payload)

    async def preload(self, model: str) -> dict:
        await asyncio.to_thread(self._svc.load, model)
        return {"status": "loaded", "model": self._svc._loaded_model}

    async def release(self) -> dict:
        prev = self._svc._loaded_model
        await asyncio.to_thread(self._svc.unload)
        return {"status": "released", "was_loaded": prev}

    async def status(self) -> dict:
        return self._svc.status()

    async def __call__(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse(self._svc.status())

        body = await request.json()
        payload = dict(body)
        model = payload.pop("model", None)
        result = await self.invoke(payload, model)
        return JSONResponse(result)


wan2gp_deployment = Wan2GPDeployment.bind()
