"""ComfyUI Ray deployment — Docker-based, flash-attn enabled.

ComfyUI runs in a Docker container (tech-noir/comfyui:latest) with
all extensions baked in during build. Models mounted from host at /models.

GPU managed by Ray: num_gpus=1.0 ensures only one GPU service runs at a time.
GPUScheduler coordinates load/unload across all GPU services.
"""
from __future__ import annotations

import logging

import httpx
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="comfyui",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={"num_gpus": 1.0, "num_cpus": 0.5},
)
class ComfyUIDeployment(BaseGPUDeployment, HTTPToolMixin):
    """ComfyUI with flash-attn via Docker container. Proxies API and WebUI requests."""

    COMFYUI_PORT = 18465

    def _load(self, model_name: str = "comfyui") -> None:
        self._ensure_healthy(
            port=self.COMFYUI_PORT,
            service_name="comfyui",
            timeout=120,
            container_port=self.COMFYUI_PORT,
            health_path="/",
        )
        self.model_name = model_name
        self.model = True

    def _unload(self) -> None:
        self._stop_container()

    def is_loaded(self) -> bool:
        return self._is_container_alive()

    def _ensure_loaded(self) -> None:
        if not self._is_container_alive():
            self._load()

    async def submit_workflow(self, workflow: dict) -> dict:
        """Submit a workflow JSON to ComfyUI's /prompt endpoint."""
        self._ensure_loaded()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://127.0.0.1:{self.COMFYUI_PORT}/prompt",
                json={"prompt": workflow},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_history(self, prompt_id: str = "") -> dict:
        """Get execution history."""
        self._ensure_loaded()
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"http://127.0.0.1:{self.COMFYUI_PORT}/history"
            if prompt_id:
                url += f"/{prompt_id}"
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def __call__(self, proxy_data: dict) -> Response:
        """Proxy HTTP requests to ComfyUI's Docker container.

        proxy_data dict keys: method, path, query, headers, body
        """
        self._ensure_loaded()

        async with httpx.AsyncClient(timeout=300) as client:
            path = proxy_data["path"]
            if path.startswith("/comfyui"):
                path = path[len("/comfyui"):] or "/"

            target_url = f"http://127.0.0.1:{self.COMFYUI_PORT}{path}"
            if proxy_data.get("query"):
                target_url += f"?{proxy_data['query']}"

            resp = await client.request(
                method=proxy_data["method"],
                url=target_url,
                headers={k: v for k, v in proxy_data.get("headers", {}).items()
                         if k.lower() not in ("host",)},
                content=proxy_data.get("body", b""),
            )

            content_type = resp.headers.get("content-type", "application/json")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=content_type,
            )
