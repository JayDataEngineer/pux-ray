"""Base deployment classes for GPU and CPU AI services.

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

import gc
import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import ray

logger = logging.getLogger(__name__)


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
        ...

    @abstractmethod
    def _unload(self) -> None:
        ...

    def load_model(self, model_name: str) -> None:
        """Public load with VRAM tracking (torch.cuda, not nvidia-smi)."""
        if self.model_name == model_name and self.model is not None:
            return

        if self.model is not None:
            self.unload_model()

        self._vram_before_load_mb = free_vram_mb()
        logger.info("Loading %s (free VRAM: %dMB)", model_name, self._vram_before_load_mb)
        self._load(model_name)
        logger.info("Loaded %s", model_name)

    def unload_model(self) -> None:
        """Release GPU memory: empty CUDA cache + garbage collect."""
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

    def current_model(self) -> Optional[str]:
        return self.model_name


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
