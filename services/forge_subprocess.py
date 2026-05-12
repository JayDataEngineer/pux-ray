"""Forge subprocess mixin — for services that wrap external processes.

Used by ComfyUI (ComfyUI server) and llama.cpp (llama-server).
Manages subprocess lifecycle with process-group isolation.
"""
from __future__ import annotations

import gc
import logging
import os
import signal
import subprocess
import time
from typing import Optional

import httpx

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


class ForgeSubprocessMixin:
    """Mixin for ForgeService subclasses that start external processes.

    Usage:
        class ComfyUIService(ForgeSubprocessMixin, ForgeService):
            def load(self, model_name):
                self.start_subprocess([...cmd...], port=18465)
            def unload(self):
                self.stop_subprocess()
            def infer(self, payload):
                return self._call("POST", "/api/generate", json=payload)
    """

    _process: Optional[subprocess.Popen] = None
    _base_url: str = ""

    def start_subprocess(
        self,
        cmd: list[str],
        port: int,
        health_path: str = "/health",
        timeout: int = 300,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> None:
        """Start a subprocess and wait for its health endpoint."""
        if self._process and self._process.poll() is None:
            logger.info("Subprocess already running on port %d", port)
            self._base_url = f"http://127.0.0.1:{port}"
            return

        proc_env = {**os.environ, **(env or {})}
        log_dir = "/tmp/tech-noir"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "subprocess.log")
        log_file = open(log_path, "a")

        self._process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=proc_env,
            stdout=log_file,
            stderr=log_file,
            preexec_fn=os.setsid,
        )
        self._base_url = f"http://127.0.0.1:{port}"
        logger.info("Subprocess PID %d started, waiting for health...", self._process.pid)

        # Wait for health
        deadline = time.time() + timeout
        health_url = f"{self._base_url}{health_path}"
        while time.time() < deadline:
            try:
                resp = httpx.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Subprocess healthy on port %d", port)
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(2)

        self.stop_subprocess()
        raise TimeoutError(f"Subprocess not healthy after {timeout}s at {health_url}")

    def stop_subprocess(self) -> None:
        """Kill subprocess tree and free CUDA cache."""
        if self._process is None:
            return

        pid = self._process.pid
        if self._process.poll() is None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %d didn't die after SIGKILL", pid)

        self._process = None
        self._base_url = ""

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        logger.info("Subprocess stopped (PID %d)", pid)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _call(
        self,
        method: str,
        path: str,
        timeout: int = 600,
        **kwargs,
    ) -> dict:
        """Synchronous HTTP call to the subprocess. Returns parsed JSON."""
        url = f"{self._base_url}{path}"
        resp = httpx.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _call_raw(
        self,
        method: str,
        path: str,
        timeout: int = 600,
        **kwargs,
    ) -> bytes:
        """Synchronous HTTP call. Returns raw bytes."""
        url = f"{self._base_url}{path}"
        resp = httpx.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.content

    async def _async_call(
        self,
        method: str,
        path: str,
        timeout: int = 600,
        **kwargs,
    ) -> dict:
        """Async HTTP call to the subprocess. Returns parsed JSON."""
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()

    async def _async_call_raw(
        self,
        method: str,
        path: str,
        timeout: int = 600,
        **kwargs,
    ) -> bytes:
        """Async HTTP call. Returns raw bytes."""
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.content
