"""Service registry and lifecycle management for Tech Noir.

Each service is a typed dataclass. The registry knows how to start, stop,
and check health for Ray, Docker, and process-based services.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
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
    check_ray,
    check_tcp,
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

register(Service(
    name="local-web-mcp",
    type=ServiceType.DOCKER,
    working_dir=f"{PROGRAMS}/local-web-mcp",
    port=18327,
    health_url="http://127.0.0.1:18327/health",
))

register(Service(
    name="media-analysis-mcp",
    type=ServiceType.DOCKER,
    working_dir=f"{PROGRAMS}/media-analysis-mcp",
    port=8001,
))

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


# --- Ray services ---

register(Service(
    name="ray-cluster",
    type=ServiceType.RAY,
    working_dir=str(get_project_root()),
    port=18265,
    health_url="http://127.0.0.1:18265",
    relative_to_root=True,
    label="Ray Cluster (KubeRay)",
))

register(Service(
    name="ray-serve",
    type=ServiceType.RAY,
    working_dir=str(get_project_root()),
    port=18800,
    depends_on=["ray-cluster"],
    relative_to_root=True,
    label="Ray Serve Deployments",
))

register(Service(
    name="ingress",
    type=ServiceType.PROCESS,
    working_dir=str(get_project_root()),
    port=18080,
    health_url="http://127.0.0.1:18080/health",
    depends_on=["ray-serve"],
    relative_to_root=True,
    label="API Ingress",
))


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
    elif svc.type == ServiceType.RAY:
        return _start_ray(svc)
    elif svc.type == ServiceType.PROCESS:
        return _start_process(svc)

    return False


def stop_service(svc: Service) -> bool:
    """Stop a service. Returns True if stopped successfully."""
    logger.info("Stopping %s...", svc.label)

    if svc.type == ServiceType.DOCKER:
        return _stop_docker(svc)
    elif svc.type == ServiceType.RAY:
        return _stop_ray(svc)
    elif svc.type == ServiceType.PROCESS:
        return _stop_process(svc)

    return False


def get_status(svc: Service) -> HealthResult:
    """Check the health of a single service."""
    if svc.type == ServiceType.DOCKER:
        return check_docker(str(svc.get_working_dir()), svc.compose_file)

    if svc.type == ServiceType.RAY:
        if svc.name == "ray-cluster":
            return check_ray()
        # ray-serve depends on ray-cluster
        cluster = check_ray()
        if cluster.status != Status.HEALTHY:
            return cluster
        if svc.port:
            return check_tcp(svc.port)
        return cluster

    if svc.type == ServiceType.PROCESS:
        if svc.port:
            return check_tcp(svc.port)
        url = svc.get_health_url()
        if url:
            return check_http(url)
        return HealthResult(Status.UNKNOWN, "no health check")

    return HealthResult(Status.UNKNOWN)


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
# Private: Ray lifecycle
# ---------------------------------------------------------------------------

def _start_ray(svc: Service) -> bool:
    root = str(svc.get_working_dir())

    if svc.name == "ray-cluster":
        ray_bin = f"{root}/.venv/bin/ray"
        try:
            # Check if Ray is already running
            check = subprocess.run(
                [ray_bin, "status"], capture_output=True, text=True, timeout=10,
            )
            if "node" in (check.stdout or ""):
                logger.info("Ray cluster already running")
                return True

            subprocess.run(
                [
                    ray_bin, "start", "--head",
                    "--num-cpus=16",
                    "--num-gpus=1",
                    "--dashboard-host=0.0.0.0",
                    "--dashboard-port=18265",
                    "--min-worker-port=10002",
                    "--max-worker-port=17999",
                    "--object-store-memory=4000000000",
                    "--temp-dir=/tmp/ray",
                ],
                cwd=root, capture_output=True, text=True, timeout=120,
                env={
                    **os.environ,
                    "RAY_memory_usage_threshold": "0.98",
                    "RAY_prestart_python_workers": "4",
                    "RAY_RUNTIME_ENV_IMAGE_WORKER_PULLER": "docker",
                },
            )
        except subprocess.TimeoutExpired:
            logger.error("Ray cluster start timed out")
            return False

        return wait_healthy(check_ray, timeout=60)

    if svc.name == "ray-serve":
        # Deploy services via Python
        venv = f"{root}/.venv/bin/python"
        try:
            result = subprocess.run(
                [venv, "-m", "scripts.deploy_services"],
                cwd=root, capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                logger.error("deploy_services failed: %s", result.stderr[-500:])
                return False
        except subprocess.TimeoutExpired:
            logger.error("Ray Serve deploy timed out")
            return False
        return True

    return False


def _stop_ray(svc: Service) -> bool:
    if svc.name in ("ray-cluster", "ray-serve", "ingress"):
        root = str(svc.get_working_dir())
        ray_bin = f"{root}/.venv/bin/ray"
        try:
            subprocess.run(
                [ray_bin, "stop"], capture_output=True, text=True, timeout=30,
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    return False


# ---------------------------------------------------------------------------
# Private: Process lifecycle
# ---------------------------------------------------------------------------

def _start_process(svc: Service) -> bool:
    root = str(svc.get_working_dir())

    if svc.name == "ingress":
        log_dir = "/tmp/tech-noir"
        os.makedirs(log_dir, exist_ok=True)

        venv = f"{root}/.venv/bin/python"
        cmd = [
            venv, "-c",
            "import ray; ray.init(address='auto', namespace='tech_noir'); "
            "import uvicorn; from gateway.ingress import create_app; "
            "uvicorn.run(create_app(), host='0.0.0.0', port=18080)",
        ]

        log_file = open(f"{log_dir}/ingress.log", "a")
        subprocess.Popen(
            cmd, cwd=root,
            stdout=log_file, stderr=log_file,
            start_new_session=True,
        )

        url = svc.get_health_url()
        if url:
            return wait_healthy(lambda: check_http(url), timeout=30)
        return wait_healthy(lambda: check_tcp(svc.port), timeout=30)

    return False


def _stop_process(svc: Service) -> bool:
    """Kill any process listening on the service's port."""
    if not svc.port:
        return False
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{svc.port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            pid = pid.strip()
            if pid.isdigit():
                os.kill(int(pid), 9)
                logger.info("Killed PID %s on port %d", pid, svc.port)
        return True
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        return False
