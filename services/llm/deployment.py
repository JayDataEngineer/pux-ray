"""LLM deployment — llama.cpp server inside Ray-managed container.

Ray manages the container (ghcr.io/ggml-org/llama.cpp:server-cuda).
The actor starts llama-server as a subprocess and proxies requests.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, SubprocessMixin

logger = logging.getLogger(__name__)


@serve.deployment(
    name="llm",
    num_replicas=1,
    max_ongoing_requests=8,
)
class LLMDeployment(BaseGPUDeployment, SubprocessMixin):
    """Ray Serve deployment wrapping llama.cpp server."""

    vram_mb = 20_480
    _service_name = "llm"
    PORT = 18399
    DEFAULT_MODEL = "qwen3.6-27b-q6_k"

    def __init__(self):
        super().__init__()
        self.base_url = f"http://localhost:{self.PORT}"

    def _load(self, model_name: str) -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        config = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("llm", model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        meta = registry.get_metadata("llm", model_name)
        ctx_size = meta.get("ctx_size", config.get("llm.ctx_size", 8192))
        parallel = meta.get("parallel", config.get("llm.parallel", 1))
        n_gpu_layers = meta.get("n_gpu_layers", 99)

        container_model = f"/models/{model_path.relative_to(config.models_root)}"

        cmd = [
            "llama-server",
            "-m", container_model,
            "--port", str(self.PORT),
            "--host", "0.0.0.0",
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
        ]

        # Quantized KV cache — critical for tight VRAM fits
        cache_type_k = meta.get("cache_type_k")
        if cache_type_k:
            cmd.extend(["--cache-type-k", cache_type_k])
        cache_type_v = meta.get("cache_type_v")
        if cache_type_v:
            cmd.extend(["--cache-type-v", cache_type_v])

        # Vision projector (mmproj) — kept on DRAM to save VRAM
        mmproj_path = meta.get("mmproj_path")
        if mmproj_path:
            full_mmproj = Path(mmproj_path)
            if not full_mmproj.is_absolute():
                full_mmproj = Path(config.models_root) / mmproj_path
            if full_mmproj.exists():
                container_mmproj = f"/models/{full_mmproj.relative_to(config.models_root)}"
                cmd.extend(["--mmproj", container_mmproj])
                mmproj_offload = meta.get("mmproj_n_gpu_layers", 0) > 0
                if not mmproj_offload:
                    cmd.append("--no-mmproj-offload")

        # Sampling defaults from model config (overridable per-request)
        for flag, key in [
            ("--temp", "temp"),
            ("--top-p", "top_p"),
            ("--top-k", "top_k"),
            ("--min-p", "min_p"),
            ("--presence-penalty", "presence_penalty"),
            ("--repeat-penalty", "repeat_penalty"),
        ]:
            val = meta.get(key)
            if val is not None:
                cmd.extend([flag, str(val)])

        logger.info("Starting llama-server: model=%s (port=%d)...", model_name, self.PORT)
        self.start_process(cmd)
        self.wait_for_health(
            f"{self.base_url}/v1/models",
            timeout=300,
        )

        self.model_name = model_name
        self.model = True
        logger.info("LLM %s ready on port %d", model_name, self.PORT)

    def _unload(self) -> None:
        self.stop_process()
        self.model = None
        self.model_name = None

    async def _resolve_image_urls(self, messages: list[dict]) -> list[dict]:
        """Download image URLs and convert to base64 data URIs.

        llama.cpp is built without HTTPS — it can't fetch remote images.
        We download them here and pass as data:image/...;base64,... instead.
        """
        import base64

        resolved = []
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    new_parts = []
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith(("http://", "https://")):
                                try:
                                    resp = await client.get(url, headers={"User-Agent": "TechNoir/1.0"})
                                    resp.raise_for_status()
                                    ct = resp.headers.get("content-type", "image/png")
                                    b64 = base64.b64encode(resp.content).decode()
                                    part = {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}
                                except Exception as e:
                                    raise ValueError(f"Failed to download image from {url[:80]}: {e}")
                        new_parts.append(part)
                    resolved.append({**msg, "content": new_parts})
                else:
                    resolved.append(msg)
        return resolved

    async def chat(self, messages: list[dict], model: str = "",
                   stream: bool = False, **kwargs) -> dict:
        """Non-streaming chat completion."""
        if not self.is_loaded():
            raise RuntimeError(f"No model loaded. Current: {self.model_name}")

        messages = await self._resolve_image_urls(messages)

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

    async def __call__(self, request):
        """TNAP endpoint + OpenAI-compatible passthrough."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()

            # OpenAI-compatible: {"messages": [...], "model": "..."}
            if "messages" in body and "action" not in body:
                if not self.is_loaded():
                    import asyncio
                    model_name = body.get("model", self.DEFAULT_MODEL)
                    await asyncio.to_thread(self.load_model, self.DEFAULT_MODEL)

                result = await self.chat(
                    messages=body["messages"],
                    model=body.get("model", ""),
                    stream=body.get("stream", False),
                    **{k: v for k, v in body.items()
                       if k not in ("messages", "model", "stream")},
                )
                return JSONResponse(result)

            # TNAP format: {action, input: {messages, stream}, config}
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                model_name = extracted.get("model", self.DEFAULT_MODEL)
                import asyncio
                await asyncio.to_thread(self.load_model, model_name)

            messages = extracted.get("messages", [])
            stream = extracted.get("stream", False)

            result = await self.chat(messages=messages, stream=stream)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    json.dumps(result).encode("utf-8"),
                    "application/json",
                    latency_ms,
                    extra_metrics={"model": self.model_name, "stream": stream},
                )
            )
        except Exception as e:
            logger.error("llm error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)
