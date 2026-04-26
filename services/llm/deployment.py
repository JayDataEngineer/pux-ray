"""LLM deployment - wraps llama.cpp server as a Ray Serve deployment.

Manages the llama.cpp subprocess lifecycle:
- Start with specific GGUF model
- Proxy OpenAI-compatible API calls (non-streaming via handle, streaming via ingress)
- Kill and restart for model swaps (with ghost VRAM prevention)

Model loading runs in a thread to avoid blocking the Ray Serve event loop.
The ingress router handles streaming by proxying directly to llama-server.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from ray import serve

from services.base import BaseGPUDeployment, SubprocessMixin

logger = logging.getLogger(__name__)


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
        self.port = 8399
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._llama_server_path: Optional[str] = None

    def _resolve_server_path(self) -> str:
        """Lazily resolve llama-server binary path from config."""
        import os
        from registry.config import Config

        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path:
            return env_path

        config = Config()
        raw = config.get("binaries.llama_server", "llama-server")
        p = Path(raw)
        if not p.is_absolute():
            p = config.project_root / p
        resolved = str(p.resolve())

        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"llama-server not found at {resolved}. "
                f"Set LLAMA_SERVER_PATH or configure binaries.llama_server."
            )
        return resolved

    def _load(self, model_name: str) -> None:
        """Start llama-server with the specified model.

        Runs in a thread via load_model() -> asyncio.to_thread().
        """
        from registry.config import Config
        from registry.models import ModelRegistry

        if not self._llama_server_path:
            self._llama_server_path = self._resolve_server_path()

        config = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("llm", model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        meta = registry.get_metadata("llm", model_name)
        ctx_size = meta.get("ctx_size", config.get("llm.ctx_size", 8192))
        parallel = meta.get("parallel", config.get("llm.parallel", 4))
        n_gpu_layers = meta.get("n_gpu_layers", 99)

        cmd = [
            self._llama_server_path,
            "-m", str(model_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
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
                cmd.extend(["--mmproj", str(full_mmproj)])

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self.start_process(cmd)
        self.model_name = model_name
        self.model = True

        # Single wait loop: poll /v1/models until the model appears
        if not self._wait_for_model_ready(model_path.name, timeout=300):
            if self.process and self.process.poll() is not None:
                stderr = ""
                if hasattr(self, "_stderr_file") and self._stderr_file:
                    try:
                        self._stderr_file.flush()
                        stderr = Path(self._stderr_file.name).read_text()[-500:]
                    except Exception:
                        pass
                raise RuntimeError(f"llama-server died during startup: {stderr}")
            raise TimeoutError(f"Model {model_name} didn't become ready in 300s")

        logger.info("LLM %s ready on port %d", model_name, self.port)

    def _unload(self) -> None:
        """Kill llama-server subprocess."""
        self.stop_process()

    def _wait_for_model_ready(self, model_filename: str, timeout: int = 300) -> bool:
        """Poll /v1/models until our model appears or server is healthy.

        Single loop replaces the old double wait_for_health + _wait_for_model_ready.
        Checks /v1/models first (model ready), falls back to /health (server up).
        """
        deadline = time.time() + timeout
        model_id = model_filename.replace(".gguf", "")

        while time.time() < deadline:
            # Check if subprocess died
            if self.process and self.process.poll() is not None:
                return False

            try:
                resp = httpx.get(f"{self.base_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    # Check if our model is listed
                    for m in models:
                        if model_id in m.get("id", ""):
                            return True
                    # Single model mode — if only 1 model listed, it's ours
                    if len(models) == 1:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(2)
        return False

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
