"""LLM deployment — llama.cpp server in Docker via HTTPToolMixin.

Manages llama.cpp lifecycle via Docker container (ghcr.io/ggml-org/llama.cpp:server-cuda).
Models mounted from host at /models. Docker container started with custom CLI args
for model selection, context size, GPU layers, etc.

Model swaps: stop current container, start new one with different model.
Health check: polls /v1/models until the model appears.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
from ray import serve

from services.base import BaseGPUDeployment, HTTPToolMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="llm",
    num_replicas=1,
    max_ongoing_requests=8,
    ray_actor_options={"num_gpus": 1.0},
)
class LLMDeployment(BaseGPUDeployment, HTTPToolMixin):
    """Ray Serve deployment wrapping llama.cpp server (Docker)."""

    PORT = 18399

    def __init__(self):
        super().__init__()
        self.base_url = f"http://127.0.0.1:{self.PORT}"

    def _load(self, model_name: str) -> None:
        """Start llama-server Docker container with the specified model."""
        from registry.config import Config
        from registry.models import ModelRegistry

        config = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("llm", model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        meta = registry.get_metadata("llm", model_name)
        ctx_size = meta.get("ctx_size", config.get("llm.ctx_size", 8192))
        parallel = meta.get("parallel", config.get("llm.parallel", 4))
        n_gpu_layers = meta.get("n_gpu_layers", 99)

        # Resolve model path inside container (/models is host models_root)
        container_model = f"/models/{model_path.relative_to(config.models_root)}"

        # Docker CMD args: override the default entrypoint with our model
        docker_args = [
            "-m", container_model,
            "--port", str(self.PORT),
            "--host", "0.0.0.0",
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
        ]

        # Add vision/mmproj if available
        mmproj_path = meta.get("mmproj_path")
        if mmproj_path:
            full_mmproj = Path(mmproj_path)
            if not full_mmproj.is_absolute():
                full_mmproj = Path(config.models_root) / mmproj_path
            if full_mmproj.exists():
                container_mmproj = f"/models/{full_mmproj.relative_to(config.models_root)}"
                docker_args.extend(["--mmproj", container_mmproj])

        logger.info("Starting llama-server: model=%s (port=%d)...", model_name, self.PORT)

        self._init_http(
            port=self.PORT,
            service_name="llm",
            timeout=300,
            container_port=self.PORT,
            image_name="ghcr.io/ggml-org/llama.cpp:server-cuda",
            health_path="/v1/models",
            docker_args=docker_args,
        )

        self.model_name = model_name
        self.model = True
        logger.info("LLM %s ready on port %d", model_name, self.PORT)

    def _unload(self) -> None:
        """Stop the llama-server Docker container."""
        self._stop_container()
        self.model = None
        self.model_name = None

    async def chat(self, messages: list[dict], model: str = "",
                   stream: bool = False, **kwargs) -> dict:
        """Non-streaming chat completion. Returns parsed JSON dict."""
        if not self.is_loaded():
            raise RuntimeError(f"No model loaded. Current: {self.model_name}")

        payload = {
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        if model:
            payload["model"] = model

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

    async def __call__(self, request) -> dict:
        """HTTP ingress handler — non-streaming only.

        Streaming is handled by the ingress router which proxies
        directly to llama-server's SSE endpoint.
        """
        body = await request.json()
        return await self.chat(**body)
