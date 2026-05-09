"""Master Router — exclusive GPU access for heavy services on a single RTX 4090.

Ray's logical GPU scheduler (num_gpus: 0.5, 1.0) is a ledger, not a hardware fence.
Multiple deployments with fractional GPU claims will physically OOM when their models
collide in VRAM. This router claims the entire GPU (num_gpus: 1.0) and performs
explicit _load()/_unload() swaps with torch.cuda.empty_cache() between them.

Lightweight GPU services (faster_qwen3_tts, index_tts, vibevoice_cpp) stay as
separate Ray deployments — they're small enough to coexist. Only the heavy hitters
(trellis, ace_step, comfyui, hy_motion, anigen, see_through, moss_soundeffect, llm)
go through this router.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Dict, Optional

import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

HEAVY_SERVICES = {"trellis", "ace_step", "comfyui", "hy_motion", "moss_soundeffect",
                   "anigen", "see_through", "llm"}

# Default _load() arguments for services that need them.
LOAD_KWARGS = {
    "llm": {"model_name": "qwen3.6-27b-q6_k"},
}


@serve.deployment(
    name="master_router",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1.0},
)
class MasterRouter:
    def __init__(self):
        self.active_service: Optional[str] = None
        self._services: Dict[str, Any] = {}
        self._loaded: Dict[str, bool] = {}

    def _get_service(self, name: str):
        if name in self._services:
            return self._services[name]

        imports = {
            "trellis": ("services.creative.trellis", "TRELLISDeployment"),
            "ace_step": ("services.creative.ace_step", "ACEStepDeployment"),
            "comfyui": ("services.image.comfyui", "ComfyUIDeployment"),
            "hy_motion": ("services.creative.hy_motion", "HYMotionDeployment"),
            "moss_soundeffect": ("services.audio.moss_soundeffect", "MossSoundEffectDeployment"),
            "anigen": ("services.creative.anigen", "AniGenDeployment"),
            "see_through": ("services.creative.see_through", "SeeThroughDeployment"),
            "llm": ("services.llm.deployment", "LLMDeployment"),
        }
        if name not in imports:
            raise ValueError(f"Unknown heavy service: {name}")

        module_path, class_name = imports[name]
        import importlib
        mod = importlib.import_module(module_path)
        deployment_obj = getattr(mod, class_name)
        cls = deployment_obj.func_or_class if hasattr(deployment_obj, 'func_or_class') else deployment_obj
        self._services[name] = cls()
        self._loaded[name] = False
        return self._services[name]

    def _unload_active(self):
        if not self.active_service:
            return
        if self._loaded.get(self.active_service):
            svc = self._services.get(self.active_service)
            if svc and hasattr(svc, "_unload"):
                try:
                    svc._unload()
                    logger.info("Unloaded %s", self.active_service)
                except Exception as e:
                    logger.warning("Failed to unload %s: %s", self.active_service, e)
            self._loaded[self.active_service] = False

        self.active_service = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _load_service(self, name: str):
        if self._loaded.get(name, False) and self.active_service == name:
            return

        self._unload_active()

        svc = self._get_service(name)
        kwargs = LOAD_KWARGS.get(name, {})
        svc._load(**kwargs)
        self._loaded[name] = True
        self.active_service = name
        vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
        logger.info("Loaded %s (VRAM: %.0fMB)", name, vram)

    async def __call__(self, request: Request) -> Response:
        body = await request.json()
        service = body.get("service")
        if not service or service not in HEAVY_SERVICES:
            return JSONResponse(
                {"status": "error", "error": f"Specify 'service' as one of {sorted(HEAVY_SERVICES)}"},
                status_code=400,
            )

        import asyncio
        await asyncio.to_thread(self._load_service, service)

        svc = self._services[service]

        # Build a sub-request with the payload minus the router key
        inner_body = {k: v for k, v in body.items() if k != "service"}

        class _InnerRequest:
            def __init__(self, data):
                self._data = data
                self.method = "POST"
                self.url = type("U", (), {"path": f"/{service}", "query": ""})()
                self.headers = {}

            async def json(self):
                return self._data

            async def body(self):
                return json.dumps(self._data).encode()

        inner_req = _InnerRequest(inner_body)
        return await svc(inner_req)


master_router = MasterRouter.bind()
