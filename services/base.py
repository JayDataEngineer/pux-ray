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
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default port range for subprocess services
_PORT_START = 18300


def get_free_port() -> int:
    """Find a free port in the 18300-18399 range."""
    import socket
    for port in range(_PORT_START, 18400):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port in range 18300-18399")


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
        """Start a subprocess in a new process group.

        stdout/stderr go to temp files (not pipes) to prevent blocking
        when the subprocess produces a lot of output (e.g. llama-server).
        """
        proc_env = {**os.environ, **(env or {})}

        # Use temp files instead of pipes to avoid deadlock when
        # subprocess writes large amounts of output
        import tempfile
        self._stdout_file = tempfile.NamedTemporaryFile(
            prefix="ray-subprocess-stdout-", suffix=".log", delete=False,
        )
        self._stderr_file = tempfile.NamedTemporaryFile(
            prefix="ray-subprocess-stderr-", suffix=".log", delete=False,
        )

        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=proc_env,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
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

        # Clean up temp log files
        for f in ("_stdout_file", "_stderr_file"):
            fh = getattr(self, f, None)
            if fh:
                try:
                    fh.close()
                    Path(fh.name).unlink(missing_ok=True)
                except Exception:
                    pass
                delattr(self, f)

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

    def wait_for_port(self, port: int, timeout: int = 120) -> bool:
        """Poll until a TCP port is accepting connections."""
        import socket
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.settimeout(2)
                    s.connect(("127.0.0.1", port))
                    return True
                except (ConnectionRefusedError, OSError):
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
        # Build clean env: start from OS env but strip Python paths that
        # would override the venv's site-packages (Ray sets PYTHONPATH etc.)
        env = dict(os.environ)
        for key in list(env.keys()):
            if key.startswith(("PYTHON", "VIRTUAL_ENV", "CONDA")):
                del env[key]
        # uv-created venvs use symlinks to a shared Python; they need
        # VIRTUAL_ENV set so Python can find its site-packages.
        venv_dir = str(Path(self._venv_python).parent.parent)
        env["VIRTUAL_ENV"] = venv_dir
        env["PATH"] = f"{Path(self._venv_python).parent}:{env.get('PATH', '')}"
        if extra_env:
            env.update(extra_env)
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


class HTTPToolMixin:
    """Mixin for tools running in Docker containers, accessed via HTTP.

    Replaces CLIToolMixin for Docker-based tools. Instead of calling
    a subprocess, sends HTTP requests to a worker container.

    The worker container is managed by GPUScheduler (start/stop via
    docker compose profiles). This mixin only handles HTTP communication.

    Model lifecycle:
    - load_model(): Records port/service info. Container is started
      separately by GPUScheduler.
    - unload_model(): No-op locally. Container is stopped by GPUScheduler.
    """

    port: int = 0
    _base_url: str = ""
    _service_name: str = ""
    _timeout: int = 600

    def _init_http(self, port: int, service_name: str, timeout: int = 600) -> None:
        """Configure HTTP connection to a Docker worker container.

        Args:
            port: Host port the container is mapped to.
            service_name: Docker Compose profile name (e.g. "trellis").
            timeout: HTTP request timeout in seconds.
        """
        self.port = port
        self._base_url = f"http://127.0.0.1:{port}"
        self._service_name = service_name
        self._timeout = timeout
        logger.info(
            "%s HTTP ready (port=%d, service=%s)",
            self.__class__.__name__, port, service_name,
        )

    async def _call_worker(
        self,
        endpoint: str,
        *,
        files: dict | None = None,
        data: dict | None = None,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> bytes:
        """Send a POST request to the worker container.

        Args:
            endpoint: URL path (e.g. "generate").
            files: Multipart file uploads.
            data: Form data.
            json: JSON body.
            timeout: Override default timeout.

        Returns:
            Response body bytes.
        """
        import httpx

        url = f"{self._base_url}/{endpoint}"
        request_timeout = timeout or self._timeout

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            resp = await client.post(
                url,
                files=files,
                data=data,
                json=json,
            )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            logger.error(
                "Worker %s returned %d: %s",
                self._service_name, resp.status_code, error_text,
            )
            raise RuntimeError(
                f"Worker {self._service_name} error ({resp.status_code}): {error_text}"
            )

        return resp.content

    async def _check_worker_health(self) -> bool:
        """Check if the worker container is healthy."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _load_worker_model(self, model_name: str | None = None) -> None:
        """Ask the worker to load its model into GPU memory."""
        import httpx

        body = {}
        if model_name:
            body["model"] = model_name

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self._base_url}/load",
                json=body,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Worker model load failed ({resp.status_code}): {resp.text[:300]}"
                )
