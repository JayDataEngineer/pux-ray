"""Base deployment classes for GPU and CPU AI services.

Tech-Noir AI Protocol (TNAP):
- All services accept the same POST body structure: {action, input, config}
- All services return the same response structure: {status, output, metrics}
- Frontend sees one language regardless of backend quirks

TNAP Request Schema:
    {
        "action": "generate",           # generate | classify | transcribe | ...
        "input": {
            "prompt": "...",
            "image_b64": "...",
            "audio_b64": "...",
            "text": "...",
        },
        "config": {
            "precision": "bf16",
            "quantization": null,
            "low_resource": false,
        }
    }

TNAP Response Schema:
    {
        "status": "success" | "error",
        "output": {
            "type": "image/png | audio/wav | model/gltf-binary | text/plain | ...",
            "content": "base64_encoded_data...",
            "url": "/shared/models/output/..."       # optional path
        },
        "metrics": {
            "vram_used_mb": 14200,
            "latency_ms": 1205,
            "model_version": "trellis-2.4b"
        }
    }

Ray-Native Design:
- GPU memory tracked via torch.cuda (not nvidia-smi subprocess)
- Ray Serve handles port allocation (no manual port scanning)
- ray.available_resources() for cluster-level GPU status
- SubprocessMixin / CLIToolMixin exist only for tools lacking
  Docker images; will be replaced by runtime_env["container"]
  once images are built.

Ghost VRAM prevention:
- unload_model() calls torch.cuda.empty_cache() + gc.collect()
- torch.cuda.synchronize() ensures CUDA ops complete before cleanup
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import ray
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─── TNAP Models ──────────────────────────────────────────────────────────────

class TNAPInput(BaseModel):
    prompt: Optional[str] = None
    text: Optional[str] = None
    image_b64: Optional[str] = None
    audio_b64: Optional[str] = None
    video_b64: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    messages: Optional[list[dict]] = None
    stream: Optional[bool] = None


class TNAPConfig(BaseModel):
    precision: str = "bf16"
    quantization: Optional[str] = None
    low_resource: bool = False


class TNAPRequest(BaseModel):
    action: str = "generate"
    input: TNAPInput = TNAPInput()
    config: Optional[TNAPConfig] = None


class TNAPMetrics(BaseModel):
    vram_used_mb: int = 0
    latency_ms: int = 0
    model_version: str = ""
    extra: dict[str, Any] = {}


class TNAPOutput(BaseModel):
    type: str = "text/plain"
    content: str = ""
    url: Optional[str] = None
    text: Optional[str] = None


class TNAPResponse(BaseModel):
    status: str = "success"
    output: TNAPOutput = TNAPOutput()
    metrics: TNAPMetrics = TNAPMetrics()
    error: Optional[str] = None


def _b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64_decode(data: str) -> bytes:
    return base64.b64decode(data)


# ─── InferenceConfig — runtime precision/quantization knobs ──────────────────

class InferenceConfig(BaseModel):
    """Standardized knobs for every model in the platform.

    Passed in the request body under the ``config`` key::

        {"config": {"precision": "fp16", "quantization": null}, "text": "..."}

    Services check ``self.config`` in ``_load()`` and apply the dtype/quant
    after model loading, so precision can change per-request without redeploy.
    """

    precision: str = "bf16"
    """torch dtype: ``fp32``, ``fp16``, ``bf16`` (default)"""

    quantization: Optional[str] = None
    """Quantization mode: ``"8bit"``, ``"4bit"``, or ``None`` (disabled)"""

    low_resource: bool = False
    """Dev/test mode: use lite model variant, fp16, skip prompt engineering.
    Loads in seconds instead of minutes. Not for production quality."""

    def dtype(self) -> "torch.dtype":
        """Translate precision string to torch dtype."""
        import torch

        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self.precision, torch.bfloat16)


# ─── Ray-native container config ──────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_ROOT = os.environ.get("TECH_NOIR_MODELS_ROOT", "/home/user/Documents/models")


def container_runtime(image: str, extra_mounts: dict[str, str] | None = None) -> dict:
    """Build runtime_env container config with model + project volume mounts.

    Mounts the project root at /app (read-only) so Ray workers inside the
    container can import from services.* without triggering Ray's auto-packaging
    (which conflicts with runtime_env["container"]).

    Usage in deployment decorator::

        @serve.deployment(
            ray_actor_options={
                "num_gpus": 0,
                "runtime_env": container_runtime("tech-noir/vibevoice:latest"),
            }
        )
    """
    run_options = [
        "--gpus", "all",
        f"--volume={MODELS_ROOT}:/models:ro",
        f"--volume={PROJECT_ROOT}:/app:ro",
        "--workdir=/app",
        "--env=PYTHONPATH=/app",
        "--shm-size=16g",
    ]
    if extra_mounts:
        for host, container in extra_mounts.items():
            run_options.append(f"--volume={host}:{container}")
    return {
        "container": {
            "image": image,
            "run_options": run_options,
        },
        "working_dir": None,
    }


# ─── GPU Memory (Ray-native: torch.cuda, not nvidia-smi) ──────────────────────

def gpu_memory_info() -> dict:
    """Get GPU memory state via torch.cuda. No subprocess needed."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            free = total - reserved
            return {
                "total_mb": int(total),
                "allocated_mb": int(allocated),
                "reserved_mb": int(reserved),
                "free_mb": int(free),
                "device_name": props.name,
            }
    except (ImportError, RuntimeError):
        pass
    return {"total_mb": 0, "free_mb": 0}


def gpu_resources() -> dict:
    """Ray-native GPU resource status (no nvidia-smi)."""
    try:
        resources = ray.available_resources()
        return {
            "gpus_available": resources.get("GPU", 0),
            "cpus_available": resources.get("CPU", 0),
            "memory_available_mb": resources.get("memory", 0),
            "node_id": resources.get("node:__internal_head__", 0),
        }
    except Exception:
        return {}


def free_vram_mb() -> int:
    """Convenience: free GPU memory in MB via torch.cuda."""
    return gpu_memory_info()["free_mb"]


# ─── Process management (subprocess pattern — bridges to Docker later) ────────

def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (SIGKILL)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _free_cuda_cache():
    """Release all PyTorch CUDA cache. Idempotent."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─── BaseGPUDeployment ────────────────────────────────────────────────────────

class BaseGPUDeployment:
    """Base class for GPU model deployments with load/unload lifecycle.

    Subclasses must implement:
        _load(model_name) - load model into VRAM
        _unload() - release model from VRAM

    Governor integration: load_model() requests a VRAM lease before loading.
    Heavy services (trellis, ace_step, etc.) coordinate through the GPUGovernor
    to prevent OOM collisions. Lightweight services set vram_mb=0 to skip leasing.

    Subclasses that need Governor leasing MUST set both:
        _service_name: str  — deployment name matching HEAVY_SERVICES key
        vram_mb: int > 0    — estimated VRAM footprint in MB
    """

    vram_mb: int = 0
    """Estimated VRAM footprint in MB. 0 = lightweight (no Governor lease)."""

    _service_name: str = ""
    """Governor service identifier. Must match a key in HEAVY_SERVICES."""

    def __init__(self):
        self.model: Optional[object] = None
        self.model_name: Optional[str] = None
        self._vram_before_load_mb: int = 0
        self.config = InferenceConfig()

    # ── Governor helpers ────────────────────────────────────────────────────

    @staticmethod
    def _governor():
        try:
            return ray.get_actor("gpu_governor")
        except ValueError:
            return None

    def _request_lease(self) -> None:
        """Request VRAM lease from Governor before loading."""
        if self.vram_mb <= 0 or not self._service_name:
            return
        gov = self._governor()
        if not gov:
            return
        try:
            result = ray.get(gov.acquire.remote(self._service_name))
            if result.get("evicted"):
                logger.info("Governor evicted %s for %s", result["evicted"], self._service_name)
        except Exception as e:
            logger.warning("Governor lease request failed for %s: %s", self._service_name, e)

    def _release_lease(self) -> None:
        """Release VRAM lease to Governor after unloading."""
        if self.vram_mb <= 0 or not self._service_name:
            return
        gov = self._governor()
        if not gov:
            return
        try:
            ray.get(gov.release.remote(self._service_name))
        except Exception as e:
            logger.warning("Governor lease release failed for %s: %s", self._service_name, e)

    # ── Model lifecycle ─────────────────────────────────────────────────────

    def _load(self, model_name: str) -> None:
        raise NotImplementedError

    def _unload(self) -> None:
        pass

    def load_model(self, model_name: str) -> None:
        """Public load with Governor lease + VRAM tracking."""
        if self.model_name == model_name and self.model is not None:
            return

        if self.model is not None:
            self.unload_model()

        self._request_lease()

        self._vram_before_load_mb = free_vram_mb()
        logger.info("Loading %s (free VRAM: %dMB)", model_name, self._vram_before_load_mb)
        self._load(model_name)
        logger.info("Loaded %s", model_name)

    def unload_model(self) -> None:
        """Release GPU memory + Governor lease."""
        if self.model is None and self.model_name is None:
            return

        name = self.model_name
        self._unload()
        self.model = None
        self.model_name = None

        _free_cuda_cache()
        gc.collect()
        logger.info("Unloaded %s", name)

        self._release_lease()

    def is_loaded(self) -> bool:
        return self.model is not None

    def current_model(self) -> Optional[str]:
        return self.model_name

    # ─── TNAP helpers ──────────────────────────────────────────────────────────

    def handle_request(self, request_body: dict) -> tuple[TNAPRequest, dict]:
        """Parse and validate a TNAP request. Returns (tnap_req, raw_input).

        Subclasses override _extract_input() to map TNAP input fields to
        whatever format their model expects (numpy array, PIL Image, torch.Tensor).
        """
        tnap = TNAPRequest.model_validate(request_body)

        if tnap.config:
            self.config.precision = tnap.config.precision or self.config.precision
            self.config.quantization = tnap.config.quantization
            self.config.low_resource = tnap.config.low_resource

        extracted = self._extract_input(tnap.input)
        return tnap, extracted

    def _extract_input(self, inp: TNAPInput) -> dict:
        """Override in subclass to convert TNAPInput to model-specific format."""
        result = {}
        if inp.prompt:
            result["prompt"] = inp.prompt
        if inp.text:
            result["text"] = inp.text
        if inp.image_b64:
            result["image"] = _b64_decode(inp.image_b64)
        if inp.audio_b64:
            result["audio"] = _b64_decode(inp.audio_b64)
        if inp.video_b64:
            result["video"] = _b64_decode(inp.video_b64)
        if inp.model:
            result["model"] = inp.model
        if inp.voice:
            result["voice"] = inp.voice
        if inp.seed is not None:
            result["seed"] = inp.seed
        if inp.steps is not None:
            result["steps"] = inp.steps
        if inp.guidance is not None:
            result["guidance"] = inp.guidance
        if inp.messages is not None:
            result["messages"] = inp.messages
        if inp.stream is not None:
            result["stream"] = inp.stream
        return result

    def handle_response(
        self,
        content: bytes | str,
        output_type: str,
        latency_ms: int,
        extra_metrics: Optional[dict] = None,
        url: Optional[str] = None,
    ) -> dict:
        """Wrap raw output in TNAP response structure."""
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        vram_mb = 0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        except Exception:
            pass

        output = {
            "type": output_type,
            "content": _b64_encode(content_bytes),
        }
        if url:
            output["url"] = url
        if isinstance(content, str):
            output["text"] = content

        metrics = {
            "vram_used_mb": vram_mb,
            "latency_ms": latency_ms,
            "model_version": self.model_name or "",
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        return {
            "status": "success",
            "output": output,
            "metrics": metrics,
        }

    def handle_error(self, error_msg: str, latency_ms: int = 0) -> dict:
        """Return a TNAP error response."""
        return {
            "status": "error",
            "output": {},
            "metrics": {"latency_ms": latency_ms},
            "error": error_msg,
        }


# ─── SubprocessMixin — llama.cpp / ComfyUI (→ Docker later) ──────────────────

class SubprocessMixin:
    """Mixin for services that wrap a subprocess (llama.cpp, ComfyUI).

    These will be replaced by runtime_env["container"] once Docker
    images are built. In the meantime, manages subprocess lifecycle:
    start with os.setsid for process-group isolation, stop via SIGKILL.
    """

    process: Optional[subprocess.Popen] = None
    port: int = 0

    def start_process(self, cmd: list[str], cwd: Optional[str] = None,
                      env: Optional[dict] = None) -> subprocess.Popen:
        proc_env = {**os.environ, **(env or {})}
        log_dir = "/tmp/tech-noir"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "subprocess.log")
        log_file = open(log_path, "a")
        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=proc_env,
            stdout=log_file,
            stderr=log_file,
            preexec_fn=os.setsid,
        )
        logger.info("Subprocess PID %d started, logs at %s", self.process.pid, log_path)
        return self.process

    def stop_process(self) -> None:
        """Kill subprocess tree with SIGKILL, free CUDA cache."""
        if self.process is None:
            return

        pid = self.process.pid
        if self.process.poll() is None:
            kill_process_tree(pid)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %d didn't die after SIGKILL", pid)

        self.process = None
        _free_cuda_cache()
        gc.collect()

    def wait_for_health(self, url: str, timeout: int = 120) -> bool:
        """Poll a health endpoint until it returns 200."""
        import httpx
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(url, timeout=5)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(2)
        return False


# ─── SubprocessProxyMixin — in-pod API server proxy (KubeRay) ─────────────────

class SubprocessProxyMixin(SubprocessMixin):
    """Start an API server as a subprocess within the Ray worker pod.

    Used by ComfyUI, TRELLIS, HY-Motion — each runs its own FastAPI/ComfyUI
    server inside the pod, and the Ray Serve __call__ proxies requests to it.
    Replaces HTTPToolMixin (which used Podman containers).
    """

    _proxy_base_url: str = ""
    _proxy_default_endpoint: str = ""

    def _start_proxy(
        self,
        cmd: list[str],
        port: int,
        health_path: str = "/health",
        timeout: int = 300,
        cwd: str | None = None,
        env: dict | None = None,
        default_endpoint: str = "",
    ) -> None:
        self._proxy_default_endpoint = default_endpoint
        self.start_process(cmd, cwd=cwd, env=env)
        self._proxy_base_url = f"http://127.0.0.1:{port}"
        if not self.wait_for_health(f"{self._proxy_base_url}{health_path}", timeout=timeout):
            self.stop_process()
            raise TimeoutError(f"Subprocess not healthy after {timeout}s")

    def _stop_proxy(self) -> None:
        self.stop_process()
        self._proxy_base_url = ""

    async def _ensure_loaded(self, model_name: str) -> None:
        """Load model in a background thread to avoid blocking the async event loop.

        Subprocess startup + health polling is synchronous (time.sleep, httpx.get).
        Running it in a thread keeps the event loop responsive for Ray Serve health
        checks and other requests.
        """
        if self.is_loaded():
            return
        import asyncio
        await asyncio.to_thread(self.load_model, model_name)

    async def _proxy_request(self, request) -> "Response":
        import httpx
        from starlette.responses import Response

        path = request.url.path
        for prefix in ["/comfyui", "/image/comfyui", "/3d/trellis", "/3d/hy-motion",
                        "/3d/anigen", "/creative/see-through", "/music/ace-step",
                        "/tts/gpt-sovits"]:
            if path.startswith(prefix):
                path = path[len(prefix):]
                if path in ("", "/"):
                    path = self._proxy_default_endpoint or "/"
                break

        target = f"{self._proxy_base_url}{path}"
        if request.url.query:
            target += f"?{request.url.query}"

        body = await request.body()
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host",)},
                content=body,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )


# ─── HTTPToolMixin — Podman-based GPU services (legacy) ───────────────────────

_CONTAINER_RUNTIME = os.environ.get("TECH_NOIR_CONTAINER_RUNTIME", "podman")


class HTTPToolMixin:
    """Mixin for services running inside Podman containers.

    Manages Podman container lifecycle (start, health check, stop)
    and provides async HTTP calls to the container's internal API.

    Used by TRELLIS, ComfyUI, and other GPU workers.
    Each service has an image built from infra/docker/Dockerfile.*.
    The container runs a FastAPI or ComfyUI server internally.
    Health check confirms container readiness before accepting requests.

    Subclass example::

        class MyService(BaseGPUDeployment, HTTPToolMixin):
            def _load(self, model_name):
                self._init_http(port=18401, service_name="my_service")
                self.model = True
                self.model_name = model_name

            async def __call__(self, request):
                data = await self._call_worker("endpoint", json={...})
                return Response(content=data)
    """

    _base_url: str = ""
    _container_name: str = ""
    _health_path: str = "/health"

    def _init_http(
        self,
        port: int,
        service_name: str,
        timeout: int = 600,
        container_port: int = 8000,
        health_path: str = "/health",
        image_name: str = "",
        docker_args: list[str] | None = None,
        mounts: dict[str, str] | None = None,
    ) -> None:
        """Start Podman container and wait for health check.

        Args:
            port: Host port to map.
            service_name: Image name suffix (tech-noir/{service_name}:latest).
            timeout: Seconds to wait for health check.
            container_port: Internal container port (8000 for FastAPI, 18465 for ComfyUI).
            health_path: URL path for health check (e.g. "/health", "/").
            image_name: Override the default image.
            docker_args: Extra args appended after image.
            mounts: Additional volume mounts (host_path: container_path).
        """
        import time as _time

        self._health_path = health_path
        image = image_name or f"docker.io/tech-noir/{service_name}:latest"
        self._container_name = f"tech-noir-{service_name}"

        # Check if container already running
        result = subprocess.run(
            [_CONTAINER_RUNTIME, "ps", "-q", "-f", f"name={self._container_name}"],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            self._base_url = f"http://127.0.0.1:{port}"
            logger.info("%s container already running on port %d", service_name, port)
            return

        # Remove any stopped container with same name
        subprocess.run(
            [_CONTAINER_RUNTIME, "rm", "-f", self._container_name],
            capture_output=True,
        )

        from registry.config import Config
        config = Config()
        # Use absolute host path, not Config().models_root which may resolve
        # to a Ray session temp dir when the actor runs inside Ray's working_dir copy
        models_root = MODELS_ROOT

        # Build podman run command
        cmd = [
            _CONTAINER_RUNTIME, "run", "-d",
            "--name", self._container_name,
            "--security-opt=label=disable",
            "--device=nvidia.com/gpu=all",
            "-p", f"{port}:{container_port}",
            "-v", f"{models_root}:/models:ro",
            "--shm-size", "16g",
        ]

        # Pass HF_TOKEN if available (needed for gated models like DINOv2)
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

        if mounts:
            for host_path, container_path in mounts.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        cmd.append(image)

        if docker_args:
            cmd.extend(docker_args)

        logger.info("Starting %s container (port %d → %d)...", service_name, port, container_port)

        # Check if image exists locally
        inspect = subprocess.run(
            [_CONTAINER_RUNTIME, "image", "inspect", image],
            capture_output=True, text=True,
        )
        if inspect.returncode != 0:
            raise RuntimeError(
                f"Image '{image}' not found. "
                f"Build or pull it first:\n"
                f"  podman build -f infra/docker/Dockerfile.{service_name} -t {image} ."
            )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start {service_name} container: {result.stderr}")

        self._base_url = f"http://127.0.0.1:{port}"

        # Wait for health
        import httpx
        deadline = _time.time() + timeout
        health_url = f"{self._base_url}{health_path}"
        while _time.time() < deadline:
            try:
                resp = httpx.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info("%s healthy (port=%d)", service_name, port)
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            _time.sleep(2)
        raise TimeoutError(f"{service_name} not healthy after {timeout}s (tried {health_url})")

    async def _call_worker(
        self,
        endpoint: str,
        method: str = "POST",
        timeout: int = 300,
        **kwargs,
    ) -> bytes:
        """Call the container's internal API and return raw bytes.

        Args:
            endpoint: API endpoint path (e.g. "generate").
            method: HTTP method.
            timeout: Request timeout in seconds.
            **kwargs: Passed to httpx.AsyncClient.request() (files, json, data, etc.).
        """
        import httpx
        url = f"{self._base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.content

    async def _call_worker_json(
        self,
        endpoint: str,
        method: str = "POST",
        timeout: int = 300,
        **kwargs,
    ) -> dict:
        """Call the container's API and return parsed JSON."""
        import httpx
        url = f"{self._base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()

    def _is_container_alive(self) -> bool:
        """Check if the Podman container is running and healthy."""
        if not self._container_name or not self._base_url:
            return False

        result = subprocess.run(
            [_CONTAINER_RUNTIME, "ps", "-q", "-f", f"name={self._container_name}"],
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            return False

        import httpx
        try:
            resp = httpx.get(f"{self._base_url}{self._health_path}", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _ensure_healthy(
        self,
        port: int,
        service_name: str,
        timeout: int = 120,
        container_port: int = 8000,
        health_path: str = "/health",
        image_name: str = "",
        docker_args: list[str] | None = None,
    ) -> None:
        """Ensure the Podman container is alive and healthy. Re-inits if dead.

        Use this instead of custom _ensure_loaded patterns. Checks both
        that the container process is alive AND the health endpoint responds.
        If either fails, tears down and re-initializes the container.
        """
        if self._is_container_alive():
            return

        if self._container_name:
            self._stop_container()

        self._health_path = health_path
        self._init_http(
            port=port,
            service_name=service_name,
            timeout=timeout,
            container_port=container_port,
            health_path=health_path,
            image_name=image_name,
            docker_args=docker_args,
        )

    def _stop_container(self) -> None:
        """Stop and remove the Podman container."""
        if not self._container_name:
            return
        name = self._container_name
        subprocess.run(
            [_CONTAINER_RUNTIME, "stop", "-t", "10", self._container_name],
            capture_output=True,
        )
        subprocess.run(
            [_CONTAINER_RUNTIME, "rm", "-f", self._container_name],
            capture_output=True,
        )
        self._base_url = ""
        self._container_name = ""
        logger.info("Stopped container: %s", name)


# ─── CLIToolMixin — compiled CUDA tools (→ Docker later) ─────────────────────

class CLIToolMixin:
    """Mixin for tools called via subprocess using their own isolated venv.

    These tools have compiled CUDA extensions (flash-attn, o_voxel,
    pytorch3d, spconv, nvdiffrast) that cannot be pip-installed by
    runtime_env. We call their CLI wrappers as subprocesses.

    When Docker images become available for each tool, replace with:
        ray_actor_options={"runtime_env": {"container": {"image": "..."}}}

    Model lifecycle:
    - load_model(): Verifies venv and script exist.
    - unload_model(): No-op (model lives in subprocess, not in our process).
    """

    config_prefix: str = ""

    def _init_cli(self, config_prefix: str) -> None:
        """Read tool paths from config and verify they exist.

        Relative paths are resolved against the project root so that
        ``local.yaml.example`` can use portable defaults like
        ``infra/repos/TRELLIS.2/.venv/bin/python``.
        """
        from registry.config import Config

        config = Config()
        project_root = config.project_root

        raw_venv = config.get(f"{config_prefix}.venv_python", "")
        raw_script = config.get(f"{config_prefix}.script", "")
        raw_cwd = config.get(f"{config_prefix}.working_dir", "")

        if not raw_venv:
            raise ValueError(f"No venv_python configured for {config_prefix}")

        self._venv_python = str(self._resolve_path(raw_venv, project_root))
        self._script = str(self._resolve_path(raw_script, project_root)) if raw_script else ""
        self._working_dir = str(self._resolve_path(raw_cwd, project_root)) if raw_cwd else ""

        if not Path(self._venv_python).exists():
            raise FileNotFoundError(
                f"Tool venv Python not found: {self._venv_python}. "
                f"Install the tool and create its venv first."
            )
        if self._script and not Path(self._script).exists():
            raise FileNotFoundError(f"Tool script not found: {self._script}")

        logger.info(
            "%s CLI ready (python=%s, script=%s)",
            self.__class__.__name__, self._venv_python, self._script,
        )

    @staticmethod
    def _resolve_path(raw: str, project_root: Path) -> Path:
        """Resolve a path that may be relative to project_root."""
        p = Path(raw)
        if not p.is_absolute():
            p = project_root / p
        return p.resolve()

    def _run_cli(
        self,
        args: list[str],
        timeout: int = 600,
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run the tool's CLI with its own venv Python."""
        cmd = [self._venv_python, self._script, *args]
        logger.info("Running CLI: %s", " ".join(cmd[:6]))
        env = {**os.environ, **extra_env} if extra_env else None
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or self._working_dir,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            logger.error("CLI failed (exit %d): %s", result.returncode, stderr_tail)
            raise RuntimeError(f"CLI failed (exit {result.returncode}): {stderr_tail}")
        return result
