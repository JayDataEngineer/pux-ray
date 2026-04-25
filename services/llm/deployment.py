"""LLM deployment - wraps llama.cpp server as a Ray Serve deployment.

Manages the llama.cpp subprocess lifecycle:
- Start with specific GGUF model
- Proxy OpenAI-compatible API calls
- Kill and restart for model swaps (with ghost VRAM prevention)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from ray import serve

from registry.config import Config
from services.base import BaseGPUDeployment, SubprocessMixin

logger = logging.getLogger(__name__)

LLAMA_SERVER_PATH = os.environ.get(
    "LLAMA_SERVER_PATH",
    Config().get("binaries.llama_server", "llama-server"),
)
DEFAULT_PORT = 8399
DEFAULT_CTX_SIZE = 8192
DEFAULT_N_GPU_LAYERS = 99


@serve.deployment(
    name="llm",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={
        "num_gpus": 0.01,
    },
)
class LLMDeployment(BaseGPUDeployment, SubprocessMixin):
    """Ray Serve deployment wrapping llama.cpp server."""

    def __init__(self):
        super().__init__()
        self.port = DEFAULT_PORT
        self.base_url = f"http://127.0.0.1:{self.port}"

    def _load(self, model_name: str) -> None:
        """Start llama-server with the specified model."""
        from registry.models import ModelRegistry

        registry = ModelRegistry()
        model_path = registry.get_path("llm", model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        cmd = [
            LLAMA_SERVER_PATH,
            "-m", str(model_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-ngl", str(DEFAULT_N_GPU_LAYERS),
            "-c", str(DEFAULT_CTX_SIZE),
            "--parallel", "4",
        ]

        # Add vision/mmproj if available
        meta = registry.get_metadata("llm", model_name)
        mmproj_path = meta.get("mmproj_path")
        if mmproj_path:
            full_mmproj = Path(mmproj_path)
            if not full_mmproj.is_absolute():
                from registry.config import Config
                full_mmproj = Path(Config().models_root) / mmproj_path
            if full_mmproj.exists():
                cmd.extend(["--mmproj", str(full_mmproj)])

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self.start_process(cmd)
        self.model_name = model_name
        self.model = True  # Mark as loaded

        # Wait for server to be healthy
        if not self.wait_for_health(f"{self.base_url}/health", timeout=180):
            # Check if process is still alive
            if self.process and self.process.poll() is not None:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise RuntimeError(f"llama-server died during startup: {stderr[:500]}")
            raise TimeoutError(f"llama-server didn't become healthy in 180s")

        # Wait for model to be loaded (health check passes before model is ready)
        if not self._wait_for_model_ready(model_path.name, timeout=300):
            raise TimeoutError(f"Model {model_name} didn't become ready in 300s")

        logger.info("LLM %s ready on port %d", model_name, self.port)

    def _unload(self) -> None:
        """Kill llama-server subprocess."""
        self.stop_process()

    def _wait_for_model_ready(self, model_filename: str, timeout: int = 300) -> bool:
        """Poll /v1/models until our model appears."""
        import time
        deadline = time.time() + timeout
        model_id = model_filename.replace(".gguf", "")

        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("data", []):
                        if model_id in m.get("id", ""):
                            return True
                    # Single model mode - if only 1 model listed, it's ours
                    if len(data.get("data", [])) == 1:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(3)
        return False

    async def chat(self, messages: list[dict], model: str = "",
                   stream: bool = False, **kwargs) -> dict:
        """OpenAI-compatible chat completion."""
        if not self.is_loaded():
            raise RuntimeError(f"No model loaded. Current: {self.model_name}")

        payload = {
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if model:
            payload["model"] = model

        async with httpx.AsyncClient(timeout=120) as client:
            if stream:
                return await client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ).__aenter__()
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

    async def __call__(self, request) -> dict:
        """HTTP ingress handler. Auto-loads model if not already loaded."""
        body = await request.json()
        model = body.get("model", "")
        if model and not self.is_loaded():
            self.load_model(model)
        elif model and self.model_name != model:
            self.load_model(model)
        return await self.chat(**body)
