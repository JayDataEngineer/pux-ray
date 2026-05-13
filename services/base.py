"""Base deployment classes for GPU and CPU AI services.

Used by ComfyUI and llama.cpp Forge services (subprocess-based).
"""
from __future__ import annotations

import gc
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import ray

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_ROOT = os.environ.get("TECH_NOIR_MODELS_ROOT", "/home/user/Documents/models")


# ─── GPU Memory ────────────────────────────────────────────────────────────────

def gpu_memory_info() -> dict:
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
    return gpu_memory_info()["free_mb"]


def kill_process_tree(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _free_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ─── BaseGPUDeployment ────────────────────────────────────────────────────────

class BaseGPUDeployment:
    """Base class for Ray Serve deployments with load/unload lifecycle.

    Used by ComfyUI and llama.cpp deployments.
    """

    vram_mb: int = 0
    _service_name: str = ""

    def __init__(self):
        self.model: Optional[object] = None
        self.model_name: Optional[str] = None
        self._vram_before_load_mb: int = 0

    def _load(self, model_name: str) -> None:
        raise NotImplementedError

    def _unload(self) -> None:
        pass

    def load_model(self, model_name: str) -> None:
        if self.model_name == model_name and self.model is not None:
            return
        if self.model is not None:
            self.unload_model()
        self._vram_before_load_mb = free_vram_mb()
        logger.info("Loading %s (free VRAM: %dMB)", model_name, self._vram_before_load_mb)
        self._load(model_name)
        logger.info("Loaded %s", model_name)

    def unload_model(self) -> None:
        if self.model is None and self.model_name is None:
            return
        name = self.model_name
        self._unload()
        self.model = None
        self.model_name = None
        _free_cuda_cache()
        gc.collect()
        logger.info("Unloaded %s", name)

    def is_loaded(self) -> bool:
        return self.model is not None

    def handle_request(self, request_body: dict) -> tuple[dict, dict]:
        """Extract TNAP fields from request body."""
        inp = request_body.get("input", request_body)
        extracted = {}
        for key in ("prompt", "text", "image_b64", "audio_b64", "video_b64",
                     "model", "voice", "seed", "steps", "guidance", "messages", "stream"):
            if key in inp:
                extracted[key] = inp[key]
        return request_body, extracted

    def handle_response(self, content: bytes, output_type: str,
                        latency_ms: int, extra_metrics: dict | None = None) -> dict:
        import base64
        return {
            "status": "success",
            "output": {"type": output_type, "content": base64.b64encode(content).decode()},
            "metrics": {"latency_ms": latency_ms, "model_version": self.model_name or "",
                        **(extra_metrics or {})},
        }

    def handle_error(self, error_msg: str, latency_ms: int = 0) -> dict:
        return {"status": "error", "output": {}, "metrics": {"latency_ms": latency_ms},
                "error": error_msg}


# ─── SubprocessMixin ──────────────────────────────────────────────────────────

class SubprocessMixin:
    """Mixin for services that wrap a subprocess (llama.cpp)."""

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
            cmd, cwd=cwd, env=proc_env,
            stdout=log_file, stderr=log_file,
            preexec_fn=os.setsid,
        )
        logger.info("Subprocess PID %d started, logs at %s", self.process.pid, log_path)
        return self.process

    def stop_process(self) -> None:
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


# ─── SubprocessProxyMixin ─────────────────────────────────────────────────────

class SubprocessProxyMixin(SubprocessMixin):
    """Start an API server as a subprocess within the Ray worker pod.

    Used by ComfyUI — runs its own server inside the pod, Ray Serve
    proxies requests to it.
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
        if self.is_loaded():
            return
        import asyncio
        await asyncio.to_thread(self.load_model, model_name)

    async def _proxy_request(self, request) -> "Response":
        import httpx
        from starlette.responses import Response

        path = request.url.path
        for prefix in ["/comfyui", "/image/comfyui"]:
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
