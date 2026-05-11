"""LLM deployment — BeeLlama.cpp / upstream llama-server inside Ray-managed container.

Engine-agnostic wrapper: picks binary based on model config or client override,
builds the full llama-server command from config + overrides, smart-diffs to
skip restarts when nothing changed, and merges session defaults into every
inference request.

Conforms to TNAP: unified request/response protocol.
OpenAI-compatible: standard /v1/chat/completions format.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, SubprocessMixin, _free_cuda_cache

logger = logging.getLogger(__name__)

ENGINE_BINARIES = {
    "beellama": "llama-server",
    "upstream": "llama-server-upstream",
}

# Registry key → llama-server API key
_SESSION_KEY_MAP = {
    "temp": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "repeat_penalty": "repeat_penalty",
    "reasoning": "reasoning",
}


@serve.deployment(
    name="llm",
    num_replicas=1,
    max_ongoing_requests=8,
)
class LLMDeployment(BaseGPUDeployment, SubprocessMixin):
    """Ray Serve deployment wrapping llama-server.

    Two engines:
      - ``beellama`` (default): Anbeeld/beellama.cpp fork with DFlash spec
        decoding, TurboQuant KV cache, adaptive draft-max.
      - ``upstream``: ggml-org/llama.cpp stable.

    Client calls ``/configure`` to set model, engine, startup flags, and
    session defaults.  Subsequent ``/v1/chat/completions`` calls merge session
    defaults (lowest priority) with per-request params.
    """

    vram_mb = 20_480
    _service_name = "llm"
    PORT = 18399
    DEFAULT_MODEL = "qwen3.6-27b-q5_k_s"

    def __init__(self):
        super().__init__()
        self.base_url = f"http://localhost:{self.PORT}"
        self._current_cmd: list[str] | None = None
        self._session_defaults: dict[str, Any] = {}
        self._engine: str = "beellama"

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def _build_cmd(
        self,
        model_name: str,
        engine: str | None = None,
        startup_overrides: dict | None = None,
    ) -> list[str]:
        """Build the full llama-server command from registry config + overrides.

        Raises ``FileNotFoundError`` if the model file is missing.
        Raises ``ValueError`` if the engine is unknown.
        """
        from registry.config import Config
        from registry.models import ModelRegistry

        config = Config()
        registry = ModelRegistry()
        meta = registry.get_metadata("llm", model_name)
        overrides = startup_overrides or {}

        # Resolve engine → binary
        engine = engine or meta.get("engine", "beellama")
        binary = ENGINE_BINARIES.get(engine)
        if not binary:
            raise ValueError(f"Unknown engine: {engine}. Use: {list(ENGINE_BINARIES)}")
        self._engine = engine

        # Resolve model path
        model_path = registry.get_path("llm", model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        container_model = f"/models/{model_path.relative_to(config.models_root)}"

        # Merge: override > registry > code default
        def _val(key, fallback=None):
            return overrides.get(key) or meta.get(key) or fallback

        ctx_size = int(_val("ctx_size", 8192))
        parallel = int(_val("parallel", 1))
        n_gpu_layers = int(_val("n_gpu_layers", 99))

        cmd: list[str] = [
            binary,
            "-m", container_model,
            "--port", str(self.PORT),
            "--host", "0.0.0.0",
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
            "--jinja",
            "--kv-unified",
            "--no-mmap", "--mlock",
            "--cache-ram", "0",
        ]

        # Flash attention
        if _val("flash_attn"):
            cmd.extend(["--flash-attn", "on"])

        # Batch sizes
        batch_size = _val("batch_size")
        if batch_size:
            cmd.extend(["-b", str(batch_size)])
        ubatch_size = _val("ubatch_size")
        if ubatch_size:
            cmd.extend(["-ub", str(ubatch_size)])

        # KV cache compression — TurboQuant types are BeeLlama-only
        beellama_cache = {"turbo4", "turbo3_tcq", "turbo3"}
        cache_type_k = _val("cache_type_k")
        if cache_type_k:
            if engine != "beellama" and cache_type_k in beellama_cache:
                cache_type_k = "f16"
            cmd.extend(["--cache-type-k", str(cache_type_k)])
        cache_type_v = _val("cache_type_v")
        if cache_type_v:
            if engine != "beellama" and cache_type_v in beellama_cache:
                cache_type_v = "f16"
            cmd.extend(["--cache-type-v", str(cache_type_v)])

        # Vision projector (mmproj) — always load when path configured; offload=0 keeps it CPU
        mmproj_path = overrides.get("mmproj_path") or meta.get("mmproj_path")
        use_vision = overrides.get("use_vision")
        if use_vision is None:
            use_vision = mmproj_path is not None
        offload = int(_val("mmproj_n_gpu_layers", 0))
        if use_vision and mmproj_path:
            full_mmproj = Path(mmproj_path)
            if not full_mmproj.is_absolute():
                full_mmproj = Path(config.models_root) / mmproj_path
            if full_mmproj.exists():
                container_mmproj = (
                    f"/models/{full_mmproj.relative_to(config.models_root)}"
                )
                cmd.extend(["--mmproj", container_mmproj])
                if offload == 0:
                    cmd.append("--no-mmproj-offload")

        # Engine-specific flags
        if engine == "beellama":
            self._add_beellama_flags(cmd, meta, overrides)

        # Sampling defaults as startup flags (llama-server native)
        for flag, key in [
            ("--temp", "temp"),
            ("--top-p", "top_p"),
            ("--top-k", "top_k"),
            ("--min-p", "min_p"),
            ("--presence-penalty", "presence_penalty"),
            ("--repeat-penalty", "repeat_penalty"),
        ]:
            val = _val(key)
            if val is not None:
                cmd.extend([flag, str(val)])

        # Reasoning mode
        if _val("reasoning"):
            cmd.extend(["--reasoning", "on"])
            cmd.extend(["--chat-template-kwargs", '{"preserve_thinking":true}'])

        # Threads
        threads = _val("threads")
        if threads:
            cmd.extend(["-t", str(threads)])

        return cmd

    def _add_beellama_flags(
        self, cmd: list[str], meta: dict, overrides: dict
    ) -> None:
        """Append BeeLlama-specific flags to *cmd*."""
        from registry.config import Config

        def _val(key, fallback=None):
            return overrides.get(key) or meta.get(key) or fallback

        # Speculative decoding type
        spec_type = _val("spec_type")
        if spec_type:
            cmd.extend(["--spec-type", spec_type])

        # Draft model
        spec_draft_model = _val("spec_draft_model")
        if spec_draft_model:
            draft_path = Path(spec_draft_model)
            if not draft_path.is_absolute():
                draft_path = Path(Config().models_root) / spec_draft_model
            if draft_path.exists():
                container_draft = (
                    f"/models/{draft_path.relative_to(Config().models_root)}"
                )
                cmd.extend(["--spec-draft-model", container_draft])
            else:
                logger.warning(
                    "DFlash draft model not found: %s — DFlash disabled", draft_path
                )

        spec_draft_ngl = _val("spec_draft_ngl")
        if spec_draft_ngl is not None:
            cmd.extend(["--spec-draft-ngl", str(spec_draft_ngl)])

        spec_dflash_cross_ctx = _val("spec_dflash_cross_ctx")
        if spec_dflash_cross_ctx is not None:
            cmd.extend(["--spec-dflash-cross-ctx", str(spec_dflash_cross_ctx)])

        spec_draft_n_max = _val("spec_draft_n_max")
        if spec_draft_n_max is not None:
            cmd.extend(["--spec-draft-n-max", str(spec_draft_n_max)])

        if _val("no_host"):
            cmd.append("--no-host")

        if _val("no_spec_dm_adaptive"):
            cmd.append("--no-spec-dm-adaptive")

    # ------------------------------------------------------------------
    # Configure — the unified endpoint
    # ------------------------------------------------------------------

    def _configure_sync(self, body: dict) -> dict:
        """Synchronous configure implementation (runs in thread)."""
        model_name = body.get("model", self.DEFAULT_MODEL)
        engine = body.get("engine")
        startup_overrides = body.get("startup_overrides", {})
        session_defaults = body.get("session_defaults", {})

        # Build new command
        try:
            new_cmd = self._build_cmd(model_name, engine, startup_overrides)
        except (FileNotFoundError, ValueError) as e:
            return {"status": "error", "error": str(e)}

        # Smart diff — restart only if command changed OR model/subprocess dead
        process_alive = (
            self.process is not None
            and self.process.poll() is None
        )
        if new_cmd == self._current_cmd and self.is_loaded() and process_alive:
            changed = False
        else:
            changed = True
            # Stop current
            if self.process is not None:
                self.stop_process()
            self.model = None
            self.model_name = None
            _free_cuda_cache()

            # Start new
            logger.info(
                "Starting llama-server (%s): model=%s", self._engine, model_name
            )
            self.start_process(new_cmd)
            healthy = self.wait_for_health(
                f"{self.base_url}/v1/models", timeout=300
            )
            if not healthy:
                return {"status": "error", "error": "llama-server failed to start"}
            self._current_cmd = new_cmd
            self.model_name = model_name
            self.model = True

        # Session defaults — map registry keys to llama-server API keys
        from registry.models import ModelRegistry

        meta = ModelRegistry().get_metadata("llm", model_name)
        raw_defaults = {}
        for key in _SESSION_KEY_MAP:
            val = meta.get(key)
            if val is not None:
                raw_defaults[key] = val
        raw_defaults.update(session_defaults)

        # Map to API keys, handle reasoning → chat_template_kwargs
        mapped = {}
        for key, val in raw_defaults.items():
            if key == "reasoning":
                if val:
                    mapped.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
            elif key in _SESSION_KEY_MAP:
                mapped[_SESSION_KEY_MAP[key]] = val
            else:
                mapped[key] = val
        self._session_defaults = mapped

        return {
            "status": "ok",
            "changed": changed,
            "engine": self._engine,
            "model": model_name,
            "startup_command": new_cmd,
            "session_defaults": self._session_defaults,
        }

    async def configure(self, body: dict) -> dict:
        """Public configure endpoint — call via ``handle.configure.remote(body)``."""
        import asyncio

        return await asyncio.to_thread(self._configure_sync, body)

    # ------------------------------------------------------------------
    # Backward-compatible load (used by Master Router)
    # ------------------------------------------------------------------

    def _load(self, model_name: str) -> None:
        """Backward-compatible: Master Router calls this with just model_name."""
        result = self._configure_sync({"model": model_name})
        if result.get("status") == "error":
            raise RuntimeError(result["error"])

    def _unload(self) -> None:
        self.stop_process()
        self.model = None
        self.model_name = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
                                    resp = await client.get(
                                        url, headers={"User-Agent": "TechNoir/1.0"}
                                    )
                                    resp.raise_for_status()
                                    ct = resp.headers.get("content-type", "image/png")
                                    b64 = base64.b64encode(resp.content).decode()
                                    part = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{ct};base64,{b64}"
                                        },
                                    }
                                except Exception as e:
                                    raise ValueError(
                                        f"Failed to download image from {url[:80]}: {e}"
                                    )
                        new_parts.append(part)
                    resolved.append({**msg, "content": new_parts})
                else:
                    resolved.append(msg)
        return resolved

    async def chat(
        self, messages: list[dict], model: str = "", stream: bool = False, **kwargs
    ) -> dict:
        """Non-streaming chat completion.

        Merges *session_defaults* (lowest priority) with per-request
        *kwargs* before sending to llama-server.
        """
        if not self.is_loaded():
            raise RuntimeError(f"No model loaded. Current: {self.model_name}")

        messages = await self._resolve_image_urls(messages)

        # Priority: explicit params > per-request kwargs > session defaults
        payload = {"messages": messages, "model": model, "stream": stream}
        payload = {**self._session_defaults, **payload}
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # HTTP entry point
    # ------------------------------------------------------------------

    async def __call__(self, request):
        """TNAP endpoint + OpenAI-compatible passthrough."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "engine": self._engine,
            }

        start = time.perf_counter()

        try:
            body = await request.json()

            # Configure action — explicit config without inference
            if body.get("action") == "configure":
                import asyncio
                result = await asyncio.to_thread(self._configure_sync, body)
                return JSONResponse(result)

            # OpenAI-compatible: {"messages": [...], "model": "..."}
            if "messages" in body and "action" not in body:
                if not self.is_loaded():
                    raise RuntimeError(f"No model loaded. Current: {self.model_name}")

                result = await self.chat(
                    messages=body["messages"],
                    model=body.get("model", ""),
                    stream=body.get("stream", False),
                    **{
                        k: v
                        for k, v in body.items()
                        if k not in ("messages", "model", "stream")
                    },
                )
                return JSONResponse(result)

            # TNAP format: {action, input: {messages, stream}, config}
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                raise RuntimeError(f"No model loaded. Current: {self.model_name}")

            messages = extracted.get("messages", [])
            stream = extracted.get("stream", False)
            extra = {
                k: v
                for k, v in extracted.items()
                if k not in ("messages", "stream", "model")
            }

            result = await self.chat(messages=messages, stream=stream, **extra)

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
