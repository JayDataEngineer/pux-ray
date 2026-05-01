"""Health check utilities for service lifecycle management."""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class Status(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class HealthResult:
    status: Status
    detail: str = ""


def check_tcp(port: int, host: str = "127.0.0.1", timeout: float = 3) -> HealthResult:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return HealthResult(Status.HEALTHY, f"port {port} open")
    except (ConnectionRefusedError, OSError, TimeoutError):
        return HealthResult(Status.STOPPED, f"port {port} not listening")


def check_http(url: str, timeout: float = 5) -> HealthResult:
    """Check if an HTTP endpoint returns 2xx."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        if 200 <= resp.status_code < 400:
            return HealthResult(Status.HEALTHY, f"HTTP {resp.status_code}")
        return HealthResult(Status.UNHEALTHY, f"HTTP {resp.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return HealthResult(Status.STOPPED, str(e))


def check_docker(compose_dir: str, compose_file: str | None = None) -> HealthResult:
    """Check Docker Compose project status.

    Returns HEALTHY if all services are running, UNHEALTHY if some are down,
    STOPPED if the project is not running at all.
    """
    cmd = ["docker", "compose"]
    if compose_file:
        cmd += ["-f", compose_file]
    cmd += ["ps", "--format", "json"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, cwd=compose_dir,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return HealthResult(Status.STOPPED, "no containers")

        import json
        lines = result.stdout.strip().split("\n")
        total = 0
        running = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
                total += 1
                health = info.get("Health", info.get("Status", ""))
                state = info.get("State", "")
                if state == "running" or "running" in str(health).lower():
                    running += 1
            except json.JSONDecodeError:
                total += 1
                if "running" in line.lower() or "healthy" in line.lower():
                    running += 1

        if total == 0:
            return HealthResult(Status.STOPPED, "no containers")
        if running == total:
            return HealthResult(Status.HEALTHY, f"{running}/{total} running")
        return HealthResult(Status.UNHEALTHY, f"{running}/{total} running")

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return HealthResult(Status.UNKNOWN, str(e))


def check_ray() -> HealthResult:
    """Check Ray cluster connectivity via TCP to dashboard port."""
    return check_tcp(18265)


def wait_healthy(
    check_fn, timeout: int = 120, interval: int = 3,
) -> bool:
    """Poll a check function until it returns HEALTHY or timeout.

    Args:
        check_fn: Callable that returns a HealthResult.
        timeout: Max seconds to wait.
        interval: Seconds between checks.

    Returns:
        True if healthy within timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = check_fn()
        if result.status == Status.HEALTHY:
            return True
        remaining = int(deadline - time.time())
        if remaining > 0:
            logger.debug("Waiting... %s (%ds remaining)", result.detail, remaining)
        time.sleep(interval)
    return False
