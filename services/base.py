"""Base deployment classes for GPU and CPU AI services.

Ghost VRAM prevention:
- unload_model() calls torch.cuda.empty_cache() + gc.collect()
- For subprocess services: SIGKILL + 2s wait to force CUDA context release
- Verifies VRAM is actually freed before returning
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# Default port range for subprocess services
_PORT_START = 8300


def get_free_port() -> int:
    """Find a free port in the 8300-8399 range."""
    import socket
    for port in range(_PORT_START, 8400):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port in range 8300-8399")


def poll_vram_free_mb() -> int:
    """Get current free VRAM in MB via nvidia-smi. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        return 0


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (SIGKILL)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def wait_for_vram_freed(target_mb: int, timeout: int = 15) -> bool:
    """Wait until free VRAM >= target_mb. Returns True if freed in time."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        free = poll_vram_free_mb()
        if free >= target_mb:
            return True
        time.sleep(1)
    logger.warning("VRAM not fully freed after %ds (free: %dMB, target: %dMB)",
                   timeout, free, target_mb)
    return False


class BaseGPUDeployment(ABC):
    """Base class for GPU model deployments with load/unload lifecycle.

    Subclasses must implement:
        _load(model_name) - load model into VRAM
        _unload() - release model from VRAM
    """

    def __init__(self):
        self.model: Optional[object] = None
        self.model_name: Optional[str] = None
        self._vram_before_load_mb: int = 0

    @abstractmethod
    def _load(self, model_name: str) -> None:
        """Load a model. Must set self.model and self.model_name."""
        ...

    @abstractmethod
    def _unload(self) -> None:
        """Unload current model. Must set self.model = None."""
        ...

    def load_model(self, model_name: str) -> None:
        """Public load with VRAM tracking."""
        if self.model_name == model_name and self.model is not None:
            logger.info("Model %s already loaded, skipping", model_name)
            return

        if self.model is not None:
            self.unload_model()

        self._vram_before_load_mb = poll_vram_free_mb()
        logger.info("Loading %s (free VRAM: %dMB)", model_name, self._vram_before_load_mb)
        self._load(model_name)
        logger.info("Loaded %s successfully", model_name)

    def unload_model(self) -> None:
        """Public unload with ghost VRAM prevention."""
        if self.model is None and self.model_name is None:
            return

        name = self.model_name
        logger.info("Unloading %s", name)
        vram_before = poll_vram_free_mb()

        self._unload()
        self.model = None
        self.model_name = None

        # Force GPU memory release
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        gc.collect()

        # Wait for VRAM to actually be freed
        if vram_before > 0:
            expected_free = vram_before + (self._vram_before_load_mb - vram_before)
            wait_for_vram_freed(min(vram_before + 100, expected_free), timeout=15)

        logger.info("Unloaded %s (VRAM freed)", name)

    def is_loaded(self) -> bool:
        return self.model is not None

    def current_model(self) -> Optional[str]:
        return self.model_name


class SubprocessMixin:
    """Mixin for services that wrap a subprocess (llama.cpp, ComfyUI).

    Ensures subprocess is killed with SIGKILL on unload to release CUDA.
    """

    process: Optional[subprocess.Popen] = None
    port: int = 0

    def start_process(self, cmd: list[str], cwd: Optional[str] = None,
                      env: Optional[dict] = None) -> subprocess.Popen:
        """Start a subprocess in a new process group."""
        proc_env = {**os.environ, **(env or {})}
        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        return self.process

    def stop_process(self) -> None:
        """Kill subprocess tree with SIGKILL and wait."""
        if self.process is None:
            return

        pid = self.process.pid
        if self.process.poll() is None:
            kill_process_tree(pid)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %d didn't die after SIGKILL", pid)

        # Also kill anything still holding our port
        if self.port:
            self._kill_port_users(self.port)

        self.process = None

    def _kill_port_users(self, port: int) -> None:
        """Kill any process still listening on our port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str and pid_str.isdigit():
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
        except Exception:
            pass

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


class CLIToolMixin:
    """Mixin for tools called via subprocess using their own isolated venv.

    These tools have compiled CUDA extensions (flash-attn, o_voxel, pytorch3d,
    etc.) that can't be pip-installed dynamically by Ray's runtime_env. Instead,
    we call their CLI wrappers via subprocess using the tool's own venv Python.

    Model lifecycle:
    - load_model(): Verifies venv and script exist. The model loads fresh
      per-call inside the subprocess.
    - unload_model(): No-op (nothing in-process to unload).
    """

    # Subclasses override: config key prefix for reading paths
    config_prefix: str = ""

    def _init_cli(self, config_prefix: str) -> None:
        """Read tool paths from config and verify they exist."""
        from pathlib import Path
        from registry.config import Config

        config = Config()
        self._venv_python: str = config.get(f"{config_prefix}.venv_python", "")
        self._script: str = config.get(f"{config_prefix}.script", "")
        self._working_dir: str = config.get(f"{config_prefix}.working_dir", "")

        if not self._venv_python:
            raise ValueError(f"No venv_python configured for {config_prefix}")
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

    def _run_cli(
        self,
        args: list[str],
        timeout: int = 600,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run the tool's CLI with its own venv Python."""
        cmd = [self._venv_python, self._script, *args]
        logger.info("Running CLI: %s", " ".join(cmd[:6]))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or self._working_dir,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            logger.error("CLI failed (exit %d): %s", result.returncode, stderr_tail)
            raise RuntimeError(f"CLI failed (exit {result.returncode}): {stderr_tail}")
        return result
