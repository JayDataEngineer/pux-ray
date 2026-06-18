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
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, SubprocessMixin, _free_cuda_cache
from services.forge_base import ForgeService
from services.forge_subprocess import ForgeSubprocessMixin

logger = logging.getLogger(__name__)

ENGINE_BINARIES = {
    "beellama": "/opt/vendor/beellama.cpp/build/bin/llama-server",
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
    DEFAULT_MODEL = "qwen3.6-27b-q5_k_s-32k"

    def __init__(self):
        super().__init__()
        self.base_url = f"http://localhost:{self.PORT}"
        self._current_cmd: list[str] | None = None
        self._session_defaults: dict[str, Any] = {}
        self._engine: str = "beellama"

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_downloaded(registry, config, registry_key: str) -> Path:
        """Download GGUF model file from HuggingFace if missing. Raises on failure."""
        meta = registry.get_metadata("llm", registry_key)
        rel_path = meta["path"]
        full = Path(config.models_root) / rel_path
        if full.exists():
            return full
        logger.warning("Model file missing: %s — auto-downloading...", full)
        from huggingface_hub import hf_hub_download
        source = meta.get("source", "")
        if source.startswith("hf://"):
            rest = source.removeprefix("hf://")
            parts = rest.split("/")
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            full.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id, filename=filename,
                local_dir=str(full.parent),
                local_dir_use_symlinks=False,
            )
        if not full.exists():
            raise FileNotFoundError(
                f"Model file could not be downloaded: {full}\n"
                f"Source: {source}. Registry key: {registry_key}"
            )
        return full

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

        # Resolve model path — auto-download if missing
        model_path = registry.get_path("llm", model_name)
        if not model_path.exists():
            try:
                model_path = self._ensure_downloaded(registry, config, model_name)
            except Exception as e:
                raise FileNotFoundError(
                    f"Model not found and auto-download failed: {model_path}\n{e}"
                )
        container_model = f"/models/{model_path.relative_to(config.models_root)}"

        cmd = self._base_cmd(binary, container_model, meta, overrides)

        self._add_cache_flags(cmd, engine, meta, overrides)
        self._add_vision_flags(cmd, config, meta, overrides)
        if engine == "beellama":
            self._add_beellama_flags(cmd, model_name, meta, overrides)
        self._add_sampling_flags(cmd, meta, overrides)

        return cmd

    def _val(self, key, overrides, meta, fallback=None):
        return overrides.get(key) or meta.get(key) or fallback

    def _base_cmd(self, binary: str, container_model: str,
                  meta: dict, overrides: dict) -> list[str]:
        ctx_size = int(self._val("ctx_size", overrides, meta, 8192))
        parallel = int(self._val("parallel", overrides, meta, 1))
        n_gpu = int(self._val("n_gpu_layers", overrides, meta, 99))

        cmd: list[str] = [
            binary,
            "-m", container_model,
            "--port", str(self.PORT),
            "--host", "0.0.0.0",
            "-ngl", str(n_gpu),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
            "--jinja",
            "--kv-unified",
            "--no-mmap", "--mlock",
            "--cache-ram", "0",
        ]

        if self._val("flash_attn", overrides, meta):
            cmd.extend(["--flash-attn", "on"])

        for flag, key in [("-b", "batch_size"), ("-ub", "ubatch_size")]:
            val = self._val(key, overrides, meta)
            if val:
                cmd.extend([flag, str(val)])

        threads = self._val("threads", overrides, meta)
        if threads:
            cmd.extend(["-t", str(threads)])

        return cmd

    def _add_cache_flags(self, cmd: list[str], engine: str,
                         meta: dict, overrides: dict) -> None:
        beellama_only = {"turbo4", "turbo3_tcq", "turbo3"}
        for flag, key in [("--cache-type-k", "cache_type_k"),
                          ("--cache-type-v", "cache_type_v")]:
            val = self._val(key, overrides, meta)
            if val:
                if engine != "beellama" and val in beellama_only:
                    val = "f16"
                cmd.extend([flag, str(val)])

    def _add_vision_flags(self, cmd: list[str], config, meta: dict,
                          overrides: dict) -> None:
        mmproj_path = overrides.get("mmproj_path") or meta.get("mmproj_path")
        use_vision = overrides.get("use_vision")
        if use_vision is None:
            use_vision = mmproj_path is not None
        if not (use_vision and mmproj_path):
            return

        full_mmproj = Path(mmproj_path)
        if not full_mmproj.is_absolute():
            full_mmproj = Path(config.models_root) / mmproj_path
        if not full_mmproj.exists():
            return

        container_mmproj = f"/models/{full_mmproj.relative_to(config.models_root)}"
        cmd.extend(["--mmproj", container_mmproj])
        offload = int(self._val("mmproj_n_gpu_layers", overrides, meta, 0))
        if offload == 0:
            cmd.append("--no-mmproj-offload")

    def _add_sampling_flags(self, cmd: list[str], meta: dict,
                            overrides: dict) -> None:
        for flag, key in [
            ("--temp", "temp"),
            ("--top-p", "top_p"),
            ("--top-k", "top_k"),
            ("--min-p", "min_p"),
            ("--presence-penalty", "presence_penalty"),
            ("--repeat-penalty", "repeat_penalty"),
        ]:
            val = self._val(key, overrides, meta)
            if val is not None:
                cmd.extend([flag, str(val)])

        if self._val("reasoning", overrides, meta):
            cmd.extend(["--reasoning", "on"])
            cmd.extend(["--chat-template-kwargs", '{"preserve_thinking":true}'])

    def _add_beellama_flags(
        self, cmd: list[str], model_name: str, meta: dict, overrides: dict
    ) -> None:
        """Append BeeLlama-specific flags to *cmd*."""
        from registry.config import Config

        def _val(key, fallback=None):
            return overrides.get(key) or meta.get(key) or fallback

        spec_type = _val("spec_type")
        if spec_type:
            cmd.extend(["--spec-type", spec_type])

        spec_draft_model = _val("spec_draft_model")
        if spec_draft_model:
            draft_rel = Path(spec_draft_model)
            draft_full = Path(Config().models_root) / draft_rel
            if not draft_full.exists():
                draft_key = f"{model_name}-dflash-draft-q4km"
                from registry.models import ModelRegistry
                try:
                    draft_registry = ModelRegistry()
                    draft_registry.get_metadata("llm", draft_key)
                    draft_full = self._ensure_downloaded(
                        draft_registry, Config(), draft_key
                    )
                except Exception as e:
                    logger.warning(
                        "Draft model auto-download failed (%s): %s — DFlash disabled",
                        draft_key, e,
                    )
                    draft_full = Path("/dev/null")
            if draft_full.exists() and draft_full != Path("/dev/null"):
                container_draft = (
                    f"/models/{draft_full.relative_to(Config().models_root)}"
                )
                cmd.extend(["--spec-draft-model", container_draft])
            else:
                logger.warning(
                    "DFlash draft model not found: %s — DFlash disabled", draft_rel
                )

        for flag, key in [
            ("--spec-draft-ngl", "spec_draft_ngl"),
            ("--spec-dflash-cross-ctx", "spec_dflash_cross_ctx"),
            ("--spec-draft-n-max", "spec_draft_n_max"),
        ]:
            val = _val(key)
            if val is not None:
                cmd.extend([flag, str(val)])

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
        except (FileNotFoundError, ValueError, KeyError) as e:
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


# ─── Forge Service ────────────────────────────────────────────────────────────

class LLMService(ForgeSubprocessMixin, ForgeService):
    """LLM via llama-server subprocess for the Forge. Takes dict, returns dict."""
    vram_mb = 20_480
    service_name = "llm"
    default_model = "qwen3.6-27b-q5_k_s-32k"

    def __init__(self):
        ForgeService.__init__(self)
        self._current_cmd: list[str] | None = None
        self._session_defaults: dict[str, Any] = {}
        self._engine: str = "beellama"
        self.PORT = 18399

    def load(self, model_name: str, quant: str | None = None) -> None:
        result = self._configure({"model": model_name})
        if result.get("status") == "error":
            raise RuntimeError(result["error"])

    def unload(self) -> None:
        self.stop_subprocess()
        self._loaded = False
        self.model_name = None

    def infer(self, payload: dict) -> dict:
        # Raw proxy mode — used by /llm ingress proxy
        if payload.get("raw"):
            method = payload.get("method", "GET")
            path = payload.get("path", "/")
            params = payload.get("params", {})
            kwargs: dict = {"params": params}
            body = payload.get("body")
            if body is not None:
                kwargs["json"] = body
            return self._call_raw_full(method, path, timeout=600, **kwargs)

        # Configure action
        if payload.get("action") == "configure":
            return self._configure(payload)

        # Chat completion
        messages = payload.get("messages", [])
        if not messages:
            return {"status": "error", "error": "messages required"}

        if not self._loaded:
            return {"status": "error", "error": "No model loaded"}

        # Merge session defaults with request params
        request_params = {k: v for k, v in payload.items()
                         if k not in ("messages", "model", "stream", "action")}
        api_payload = {"messages": messages, "model": payload.get("model", ""),
                       "stream": payload.get("stream", False)}
        api_payload = {**self._session_defaults, **api_payload, **request_params}

        resp = self._call("POST", "/v1/chat/completions", json=api_payload, timeout=120)
        return {"status": "success", "data": resp}

    def _configure(self, body: dict) -> dict:
        model_name = body.get("model", self.default_model)
        engine = body.get("engine")
        startup_overrides = body.get("startup_overrides", {})
        session_defaults = body.get("session_defaults", {})

        try:
            new_cmd = self._build_cmd(model_name, engine, startup_overrides)
        except (FileNotFoundError, ValueError, KeyError) as e:
            return {"status": "error", "error": str(e)}

        # Smart diff — skip restart if nothing changed
        process_alive = self.is_running()
        if new_cmd == self._current_cmd and self._loaded and process_alive:
            changed = False
        else:
            changed = True
            self.stop_subprocess()
            self._loaded = False
            self.model_name = None

            logger.info("Starting llama-server (%s): model=%s", self._engine, model_name)
            # Set LD_LIBRARY_PATH so the vendored llama-server can find its
            # bundled shared libraries (libllama-common.so.0, libggml-cuda.so, etc).
            llm_bin_dir = os.path.dirname(ENGINE_BINARIES.get(self._engine, "llama-server"))
            subprocess_env = {}
            if llm_bin_dir:
                subprocess_env["LD_LIBRARY_PATH"] = llm_bin_dir
            self.start_subprocess(new_cmd, port=self.PORT,
                                  health_path="/v1/models", timeout=600,
                                  env=subprocess_env)
            self._current_cmd = new_cmd
            self.model_name = model_name
            self._loaded = True

        # Session defaults
        from registry.models import ModelRegistry
        meta = ModelRegistry().get_metadata("llm", model_name)
        raw_defaults = {}
        for key in _SESSION_KEY_MAP:
            val = meta.get(key)
            if val is not None:
                raw_defaults[key] = val
        raw_defaults.update(session_defaults)

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
            "status": "ok", "changed": changed,
            "engine": self._engine, "model": model_name,
        }

    def _build_cmd(self, model_name: str, engine: str | None = None,
                   startup_overrides: dict | None = None) -> list[str]:
        """Build llama-server command. Reuses the same logic as LLMDeployment."""
        from registry.config import Config
        from registry.models import ModelRegistry

        config = Config()
        registry = ModelRegistry()
        meta = registry.get_metadata("llm", model_name)
        overrides = startup_overrides or {}

        engine = engine or meta.get("engine", "beellama")
        binary = ENGINE_BINARIES.get(engine)
        if not binary:
            raise ValueError(f"Unknown engine: {engine}")
        self._engine = engine

        model_path = registry.get_path("llm", model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        container_model = f"/models/{model_path.relative_to(config.models_root)}"

        def _val(key, fallback=None):
            return overrides.get(key) or meta.get(key) or fallback

        cmd = [
            binary, "-m", container_model,
            "--port", str(self.PORT), "--host", "0.0.0.0",
            "-ngl", str(int(_val("n_gpu_layers", 99))),
            "-c", str(int(_val("ctx_size", 8192))),
            "--parallel", str(int(_val("parallel", 1))),
            "--jinja", "--kv-unified", "--no-mmap", "--mlock", "--cache-ram", "0",
        ]

        if _val("flash_attn"):
            cmd.extend(["--flash-attn", "on"])

        for flag, key in [("--temp", "temp"), ("--top-p", "top_p"),
                          ("--top-k", "top_k"), ("--min-p", "min_p"),
                          ("--presence-penalty", "presence_penalty"),
                          ("--repeat-penalty", "repeat_penalty")]:
            val = _val(key)
            if val is not None:
                cmd.extend([flag, str(val)])

        if _val("reasoning"):
            cmd.extend(["--reasoning", "on"])
            cmd.extend(["--chat-template-kwargs", '{"preserve_thinking":true}'])

        return cmd
