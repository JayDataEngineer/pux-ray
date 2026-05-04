"""ComfyUI Ray deployment — runs inside Ray-managed container.

Ray manages the container (tech-noir/comfyui:latest). The actor starts
ComfyUI as a subprocess and proxies HTTP requests to it.

GPU managed by Ray: num_gpus=1.0 ensures only one GPU service runs at a time.
GPUScheduler coordinates load/unload across all GPU services.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from services.base import BaseGPUDeployment, SubprocessMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="comfyui",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class ComfyUIDeployment(BaseGPUDeployment, SubprocessMixin):
    """ComfyUI with flash-attn inside Ray-managed container."""

    COMFYUI_PORT = 18465

    def _load(self, model_name: str = "comfyui") -> None:
        self._setup_model_links()
        self.start_process(
            [
                "python3", "main.py",
                "--port", str(self.COMFYUI_PORT),
                "--listen", "0.0.0.0",
                "--preview-method", "auto",
                "--use-split-cross-attention",
            ],
            cwd="/opt/ComfyUI",
        )
        self.wait_for_health(
            f"http://localhost:{self.COMFYUI_PORT}/",
            timeout=120,
        )
        self.model_name = model_name
        self.model = True
        logger.info("ComfyUI ready (port=%d)", self.COMFYUI_PORT)

    def _unload(self) -> None:
        self.stop_process()

    def is_loaded(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _setup_model_links(self) -> None:
        models_dir = "/opt/ComfyUI/models"
        os.makedirs(models_dir, exist_ok=True)
        for d in ["HY-Motion", "RMBG", "sams", "ultralytics"]:
            src = f"/models/image-gen/comfyui/{d}"
            dst = f"{models_dir}/{d}"
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    async def submit_workflow(self, workflow: dict) -> dict:
        self._ensure_loaded()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://localhost:{self.COMFYUI_PORT}/prompt",
                json={"prompt": workflow},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_history(self, prompt_id: str = "") -> dict:
        self._ensure_loaded()
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"http://localhost:{self.COMFYUI_PORT}/history"
            if prompt_id:
                url += f"/{prompt_id}"
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    def _ensure_loaded(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self._load()

    async def __call__(self, request: Request) -> Response:
        self._ensure_loaded()

        async with httpx.AsyncClient(timeout=300) as client:
            path = request.url.path
            if path.startswith("/comfyui"):
                path = path[len("/comfyui"):] or "/"

            target_url = f"http://localhost:{self.COMFYUI_PORT}{path}"
            if request.url.query:
                target_url += f"?{request.url.query}"

            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=body,
            )

            content_type = resp.headers.get("content-type", "application/json")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=content_type,
            )
