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


# ─── HTTPToolMixin — Docker-based GPU services ────────────────────────────────

class HTTPToolMixin:
    """Mixin for services running inside Docker containers.

    Manages Docker container lifecycle (start, health check, stop)
    and provides async HTTP calls to the container's internal API.

    Used by TRELLIS, AniGen, VibeVoice, and ComfyUI (Docker workers).
    Each service has a Docker image built from infra/docker/Dockerfile.*.
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
        """Start Docker container and wait for health check.

        Args:
            port: Host port to map.
            service_name: Docker image name (tech-noir/{service_name}:latest).
            timeout: Seconds to wait for health check.
            container_port: Internal container port (8000 for FastAPI, 18465 for ComfyUI).
            health_path: URL path for health check (e.g. "/health", "/").
            image_name: Override the default image (e.g. "ghcr.io/ggml-org/llama.cpp:server-cuda").
            docker_args: Extra args appended after image (e.g. ["-m", "/models/llm/foo.gguf"]).
            mounts: Additional volume mounts (host_path: container_path).
        """
        import time as _time

        image = image_name or f"tech-noir/{service_name}:latest"
        self._container_name = f"tech-noir-{service_name}"

        # Check if container already running
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            self._base_url = f"http://127.0.0.1:{port}"
            logger.info("%s container already running on port %d", service_name, port)
            return

        # Remove any stopped container with same name
        subprocess.run(
            ["docker", "rm", "-f", self._container_name],
            capture_output=True,
        )

        from registry.config import Config
        config = Config()
        models_root = config.models_root

        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", self._container_name,
            "--gpus", "all",
            "-p", f"{port}:{container_port}",
            "-v", f"{models_root}:/models:ro",
            "--shm-size", "16g",
        ]

        if mounts:
            for host_path, container_path in mounts.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        cmd.append(image)

        if docker_args:
            cmd.extend(docker_args)

        logger.info("Starting %s container (port %d → %d)...", service_name, port, container_port)

        # Check if Docker image exists locally
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True,
        )
        if inspect.returncode != 0:
            raise RuntimeError(
                f"Docker image '{image}' not found on this machine. "
                f"Build or pull it first:\n"
                f"  docker build -f infra/docker/Dockerfile.{service_name} -t {image} ."
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

    def _is_container_alive(self) -> bool:
        """Check if the Docker container is running and healthy."""
        if not self._container_name or not self._base_url:
            return False

        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
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
        """Ensure the Docker container is alive and healthy. Re-inits if dead.
        
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
        """Stop and remove the Docker container."""
        if not self._container_name:
            return
        name = self._container_name
        subprocess.run(
            ["docker", "stop", "-t", "10", self._container_name],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "rm", "-f", self._container_name],
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
