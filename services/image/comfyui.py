"""ComfyUI Ray deployment - runs ComfyUI as a managed subprocess.

Preserves the WebUI for visual workflow development while
making it controllable through Ray Serve.

ComfyUI runs as a subprocess on its own port (8188).
The Ray deployment proxies HTTP requests to it.
Models stay in ComfyUI's existing directory structure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

from registry.config import Config
from services.base import SubprocessMixin

logger = logging.getLogger(__name__)

COMFYUI_DIR = Config().get("services.comfyui.working_dir", "/home/ubuntu/Documents/programs/img/comfyui")
COMFYUI_PORT = Config().get("services.comfyui.port", 8188)


@serve.deployment(
    name="comfyui",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={
        "num_gpus": 0.01,
        "runtime_env": {
            "working_dir": COMFYUI_DIR,
        },
    },
)
class ComfyUIDeployment(SubprocessMixin):
    """Runs ComfyUI server as a subprocess. Proxies API and WebUI requests."""

    def __init__(self):
        self.process = None
        self.port = COMFYUI_PORT
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._running = False

    def start_comfyui(self) -> bool:
        """Start ComfyUI subprocess."""
        if self.process and self.process.poll() is None:
            logger.info("ComfyUI already running on port %d", self.port)
            return True

        cmd = [
            "python", "main.py",
            "--port", str(self.port),
            "--listen", "127.0.0.1",
            "--preview-method", "auto",
            "--cpu" if os.environ.get("COMFYUI_CPU") else "",
        ]
        cmd = [c for c in cmd if c]  # remove empty strings

        logger.info("Starting ComfyUI: %s", " ".join(cmd))
        self.start_process(cmd, cwd=COMFYUI_DIR)

        if not self.wait_for_health(f"{self.base_url}/", timeout=120):
            if self.process and self.process.poll() is not None:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise RuntimeError(f"ComfyUI died during startup: {stderr[:500]}")
            raise TimeoutError("ComfyUI didn't start in 120s")

        self._running = True
        logger.info("ComfyUI running on port %d", self.port)
        return True

    def stop_comfyui(self) -> None:
        """Stop ComfyUI subprocess."""
        self.stop_process()
        self._running = False
        logger.info("ComfyUI stopped")

    async def submit_workflow(self, workflow: dict) -> dict:
        """Submit a workflow JSON to ComfyUI's /prompt endpoint."""
        if not self._running:
            self.start_comfyui()

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_history(self, prompt_id: str = "") -> dict:
        """Get execution history."""
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.base_url}/history"
            if prompt_id:
                url += f"/{prompt_id}"
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_output_path(self, filename: str) -> str:
        """Get the full path to an output file."""
        return os.path.join(COMFYUI_DIR, "output", filename)

    async def __call__(self, request: Request) -> Response:
        """Proxy all HTTP requests to ComfyUI's own server.

        This allows the WebUI to work through Ray Serve's HTTP proxy.
        """
        if not self._running:
            self.start_comfyui()

        async with httpx.AsyncClient(timeout=300) as client:
            # Build the target URL
            path = request.url.path
            # Strip /comfyui prefix if present
            if path.startswith("/comfyui"):
                path = path[len("/comfyui"):] or "/"

            target_url = f"{self.base_url}{path}"
            if request.url.query:
                target_url += f"?{request.url.query}"

            # Forward the request
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=body,
            )

            # Return the response
            content_type = resp.headers.get("content-type", "application/json")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=content_type,
            )
