"""Service registry and lifecycle management for Tech Noir.

Each service is a typed dataclass. The registry knows how to start, stop,
and check health for Ray, Docker, and process-based services.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from boot.config import get_programs_dir, get_project_root, resolve_service_dir
from boot.health import (
    HealthResult,
    Status,
    check_docker,
    check_http,
    wait_healthy,
)

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    DOCKER = "docker"
    RAY = "ray"
    PROCESS = "process"


@dataclass
class Service:
    name: str
    type: ServiceType
    label: str = ""
    working_dir: str = ""
    port: int | None = None
    health_url: str | None = None
    compose_file: str | None = None
    compose_profile: str = ""
    depends_on: list[str] = field(default_factory=list)
    start_cmd: list[str] | None = None
    relative_to_root: bool = False

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("-", " ").replace("_", " ").title()

    def get_working_dir(self) -> Path:
        return resolve_service_dir(self.working_dir, self.relative_to_root)

    def get_health_url(self) -> str | None:
        if self.health_url:
            return self.health_url
        if self.port:
            return f"http://127.0.0.1:{self.port}"
        return None


# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------

PROGRAMS = str(get_programs_dir())

SERVICES: dict[str, Service] = {}


def register(svc: Service) -> None:
    SERVICES[svc.name] = svc


def get(name: str) -> Service | None:
    return SERVICES.get(name)


def all_services() -> list[Service]:
    return list(SERVICES.values())


# --- Docker services ---

# MCP services are deployed as K3s pods (see infra/k8s/mcp/ and infra/flux/mcp/).
# K8s livenessProbes handle auto-restart — no Docker lifecycle needed.

register(Service(
    name="redshiftdb",
    type=ServiceType.DOCKER,
    working_dir=f"{PROGRAMS}/redshiftdb/ops/docker",
    compose_file="compose.dev.yaml",
    label="RedshiftDB (infra)",
))

register(Service(
    name="act-scheduler-bot",
    type=ServiceType.DOCKER,
    working_dir=f"{PROGRAMS}/act-scheduler-bot",
    label="ACT Scheduler Bot",
))

register(Service(
    name="jellyfin",
    type=ServiceType.DOCKER,
    working_dir=f"{PROGRAMS}/jellyfin_act",
    label="Jellyfin + Nextcloud",
))


# --- Ray services (managed by KubeRay, NOT bare-metal) ---
# Ray is deployed via `kubectl apply -f infra/k8s/ray-service.yaml`.
# Do NOT register ray-cluster, ray-serve, or ingress here —
# those would start bare-metal Ray processes on the host.
# KubeRay manages Ray head + worker pods inside k3s.
# See: infra/k8s/ray-service.yaml


# ---------------------------------------------------------------------------
# Lifecycle operations
# ---------------------------------------------------------------------------

def start_service(svc: Service) -> bool:
    """Start a service. Returns True if started (or already running)."""
    health = get_status(svc)
    if health.status == Status.HEALTHY:
        logger.info("%s already healthy, skipping", svc.label)
        return True

    # Start dependencies first
    for dep_name in svc.depends_on:
        dep = SERVICES.get(dep_name)
        if dep:
            dep_health = get_status(dep)
            if dep_health.status != Status.HEALTHY:
                logger.info("Starting dependency: %s", dep.label)
                start_service(dep)

    logger.info("Starting %s...", svc.label)

    if svc.type == ServiceType.DOCKER:
        return _start_docker(svc)
    else:
        logger.error("Unknown service type: %s", svc.type)
        return False

    return False


def stop_service(svc: Service) -> bool:
    """Stop a service. Returns True if stopped successfully."""
    logger.info("Stopping %s...", svc.label)

    if svc.type == ServiceType.DOCKER:
        return _stop_docker(svc)
    else:
        logger.error("Unknown service type: %s", svc.type)
        return False

    return False


def get_status(svc: Service) -> HealthResult:
    """Check the health of a single service."""
    if svc.type == ServiceType.DOCKER:
        return check_docker(str(svc.get_working_dir()), svc.compose_file)

    return HealthResult(Status.UNKNOWN, f"unsupported type: {svc.type}")


def get_all_status() -> dict[str, HealthResult]:
    """Get status for all registered services."""
    return {name: get_status(svc) for name, svc in SERVICES.items()}


# ---------------------------------------------------------------------------
# Private: Docker lifecycle
# ---------------------------------------------------------------------------

def _start_docker(svc: Service) -> bool:
    cwd = str(svc.get_working_dir())
    cmd = ["docker", "compose"]
    if svc.compose_file:
        cmd += ["-f", svc.compose_file]
    if svc.compose_profile:
        cmd += ["--profile", svc.compose_profile]
    cmd += ["up", "-d"]

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("docker compose up failed: %s", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("docker compose up timed out for %s", svc.label)
        return False

    # Wait for health
    if svc.port:
        return wait_healthy(lambda: check_tcp(svc.port), timeout=60)
    return wait_healthy(
        lambda: check_docker(str(svc.get_working_dir()), svc.compose_file),
        timeout=120,
    )


def _stop_docker(svc: Service) -> bool:
    cwd = str(svc.get_working_dir())
    cmd = ["docker", "compose"]
    if svc.compose_file:
        cmd += ["-f", svc.compose_file]
    if svc.compose_profile:
        cmd += ["--profile", svc.compose_profile]
    cmd += ["down"]

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False



# ---------------------------------------------------------------------------
# Ray lifecycle — REMOVED. Ray is managed by KubeRay (k3s), not bare-metal.
# To deploy/update Ray: kubectl apply -f infra/k8s/ray-service.yaml
# ---------------------------------------------------------------------------
