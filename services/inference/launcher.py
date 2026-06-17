"""Pool launcher — start/stop/status for inference pool containers.

Wraps `docker run` / `docker rm` with the pool's bind-mounts, env vars,
and per-model launch scripts. NO Ray, NO async — straight subprocess.

Each pool has a launch script (referenced from the YAML via
``model_launchers.<model>.script`` for multi-model pools, or via the pool's
``start_args`` for single-model pools). The launcher invokes the script with
the right port/container args, OR — if no script is declared — falls back to
a direct ``docker run`` synthesized from the pool spec.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from services.inference.config import Pool, ModelLauncher
from services.inference.manager import PoolManager, ResolvedTarget

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Result types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LaunchResult:
    pool: str
    container: str
    port: int
    healthy: bool
    elapsed_s: float
    message: str
    model_loaded: str | None = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _docker(*args: str, check: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a docker CLI command, capturing output. Returns CompletedProcess."""
    cmd = ["docker", *args]
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=check, timeout=timeout)


def container_status(container: str) -> str:
    """Return docker status string for a container, or "" if missing."""
    res = _docker("ps", "-a", "--filter", f"name=^{container}$",
                  "--format", "{{.Status}}")
    return res.stdout.strip()


def is_healthy(pool: Pool, timeout: float = 2.0) -> bool:
    """Quick HTTP health probe. No imports of httpx — use urllib for stdlib-only."""
    if not shutil.which("curl"):
        return False
    res = subprocess.run(
        ["curl", "-sf", "--max-time", str(timeout),
         "-o", "/dev/null", pool.health_url],
        capture_output=True, text=True,
    )
    return res.returncode == 0


# ─── Launcher ────────────────────────────────────────────────────────────────

class PoolLauncher:
    """Start/stop pools. Stateless — takes a PoolManager for resolution."""

    def __init__(self, manager: PoolManager):
        self.manager = manager

    # ── Script-based launch (preferred) ────────────────────────────────────

    def _launch_script(self, pool: Pool, launcher: ModelLauncher | None,
                       extra_args: list[str] | None = None) -> LaunchResult:
        """Invoke a pool's bind-mounted launch script."""
        script_path: Path | None = None
        if launcher is not None and launcher.script_path is not None:
            script_path = launcher.script_path
        if script_path is None:
            return LaunchResult(
                pool=pool.name, container=pool.container, port=pool.port,
                healthy=False, elapsed_s=0.0,
                message=f"No launch script for {pool.name}",
            )
        # Resolve relative to repo root.
        if not script_path.is_absolute():
            script_path = REPO_ROOT / script_path
        if not script_path.exists():
            return LaunchResult(
                pool=pool.name, container=pool.container, port=pool.port,
                healthy=False, elapsed_s=0.0,
                message=f"Launch script not found: {script_path}",
            )

        args = [str(pool.port), *(extra_args or [])]
        import time
        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                ["bash", str(script_path), *args],
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return LaunchResult(
                pool=pool.name, container=pool.container, port=pool.port,
                healthy=False, elapsed_s=300.0,
                message="Launch script timed out after 300s",
                model_loaded=launcher.name if launcher else None,
            )
        elapsed = time.perf_counter() - t0
        healthy = is_healthy(pool)
        return LaunchResult(
            pool=pool.name, container=pool.container, port=pool.port,
            healthy=healthy, elapsed_s=elapsed,
            message=(res.stdout.strip().splitlines()[-1] if res.stdout.strip()
                     else ("healthy" if healthy else f"rc={res.returncode}")),
            model_loaded=launcher.name if launcher else None,
        )

    # ── Direct docker run (synthesized) ────────────────────────────────────

    def _docker_run_direct(self, pool: Pool, cmd_args: list[str]) -> LaunchResult:
        """Synthesize a docker run for pools without a launch script."""
        # Stop existing container if present.
        _docker("rm", "-f", pool.container)
        run_args = [
            "run", "-d",
            "--name", pool.container,
            "--gpus", "all",
            "--ipc=host",
            "-p", f"{pool.port}:8080",
        ]
        for k, v in pool.env.items():
            run_args.extend(["-e", f"{k}={v}"])
        for host, cont in pool.volumes.items():
            run_args.extend(["-v", f"{host}:{cont}:ro" if host else f"{cont}"])
        run_args.append(pool.image)
        run_args.extend(cmd_args)
        import time
        t0 = time.perf_counter()
        res = _docker(*run_args)
        elapsed = time.perf_counter() - t0
        if res.returncode != 0:
            return LaunchResult(
                pool=pool.name, container=pool.container, port=pool.port,
                healthy=False, elapsed_s=elapsed,
                message=f"docker run failed: {res.stderr.strip()[:200]}",
            )
        healthy = is_healthy(pool, timeout=30.0)
        return LaunchResult(
            pool=pool.name, container=pool.container, port=pool.port,
            healthy=healthy, elapsed_s=elapsed,
            message="healthy" if healthy else "started, health check pending",
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, pool_name: str, model: str | None = None,
              extra_args: list[str] | None = None) -> LaunchResult:
        """Start a pool. If model is given, uses that model's launch script."""
        pool = self.manager.pool(pool_name)
        if pool is None:
            return LaunchResult(
                pool=pool_name, container="", port=0,
                healthy=False, elapsed_s=0.0,
                message=f"Unknown pool: {pool_name}",
            )
        launcher = None
        if model is not None:
            launcher = self.manager._launcher_for(model, pool)
        if launcher is not None and launcher.script:
            return self._launch_script(pool, launcher, extra_args)
        # Fallback: direct docker run with image default entrypoint.
        return self._docker_run_direct(pool, extra_args or [])

    def stop(self, pool_name: str) -> bool:
        """Stop and remove a pool's container."""
        pool = self.manager.pool(pool_name)
        if pool is None:
            return False
        res = _docker("rm", "-f", pool.container)
        return res.returncode == 0

    def status(self, pool_name: str) -> dict:
        """Return current state of a pool's container + health."""
        pool = self.manager.pool(pool_name)
        if pool is None:
            return {"error": f"Unknown pool: {pool_name}"}
        cs = container_status(pool.container)
        return {
            "pool": pool.name,
            "tier": pool.tier,
            "container": pool.container,
            "port": pool.port,
            "framework": pool.framework,
            "docker_status": cs or "absent",
            "healthy": is_healthy(pool) if cs.startswith("Up") else False,
            "health_url": pool.health_url,
            "models": pool.models,
            "vram_mb": pool.vram_mb,
        }

    def status_all(self) -> list[dict]:
        return [self.status(p.name) for p in self.manager.pools()]
