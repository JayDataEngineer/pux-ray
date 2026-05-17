"""Tech Noir Flux Sync — pushes infra/flux/ as OCI artifact to local forge-registry.

Usage:
    python -m infra.flux_sync push        Push artifact + reconcile
    python -m infra.flux_sync watch        Watch dir and push on change
    python -m infra.flux_sync reconcile    Reconcile Flux kustomizations

systemd integration:
    infra/flux-sync.path     — triggers flux-sync.service on file change
    infra/flux-sync.service  — runs `push`
    infra/flux-sync.timer    — periodic fallback every 5min
"""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("flux_sync")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLUX_DIR = PROJECT_DIR / "infra" / "flux"
LOGFILE = PROJECT_DIR / "logs" / "flux-sync.log"
LOCKFILE = Path("/tmp/flux-sync.lock")
KUSTOMIZATIONS = ["namespaces", "networking", "ai-services"]

REGISTRY_SVC = "forge-registry"
REGISTRY_NS = "infra"
REGISTRY_PORT = 5000
ARTIFACT_NAME = "flux-manifests"


def _kubectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sudo", "k3s", "kubectl", *args],
        capture_output=True, text=True, timeout=30,
    )


def _flux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["flux", *args],
        capture_output=True, text=True, timeout=120,
    )


def get_registry_ip() -> str:
    result = _kubectl(
        "get", "svc", REGISTRY_SVC,
        "-n", REGISTRY_NS,
        "-o", "jsonpath={.spec.clusterIP}",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot resolve {REGISTRY_SVC}: {result.stderr}")
    ip = result.stdout.strip()
    if not ip:
        raise RuntimeError(f"Empty ClusterIP for {REGISTRY_SVC}")
    return ip


def push_artifact() -> bool:
    logger.info("Pushing OCI artifact from %s ...", FLUX_DIR)
    try:
        registry_ip = get_registry_ip()
    except RuntimeError as e:
        logger.error("Registry lookup failed: %s", e)
        return False

    url = f"oci://{registry_ip}:{REGISTRY_PORT}/{ARTIFACT_NAME}:latest"

    # Get git info
    source = "local"
    revision = "unknown"
    try:
        source = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "local"
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        pass

    result = _flux(
        "push", "artifact", url,
        "--path", str(FLUX_DIR),
        "--source", source,
        "--revision", revision,
        "--insecure-registry",
    )
    if result.returncode != 0:
        logger.error("Push failed: %s", result.stderr[-500:])
        return False

    logger.info("Artifact pushed successfully")
    return True


def reconcile_kustomizations() -> None:
    logger.info("Reconciling Flux kustomizations...")
    for name in KUSTOMIZATIONS:
        result = _kubectl("get", "kustomization", name, "-n", "flux-system")
        if result.returncode != 0:
            logger.debug("Kustomization %s not found, skipping", name)
            continue
        _flux("reconcile", "kustomization", name, "--with-source")
        logger.info("Reconciled %s", name)


def sync() -> bool:
    """Push artifact + reconcile. Returns True on success."""
    if not push_artifact():
        return False
    reconcile_kustomizations()
    return True


def watch_loop() -> None:
    """Watch infra/flux/ for changes, sync on every modification."""
    logger.info("Watching %s for changes...", FLUX_DIR)

    try:
        import inotify_simple
        _watch_inotify()
    except ImportError:
        _watch_poll()


def _watch_inotify() -> None:
    from inotify_simple import INotify, flags

    inotify = INotify()
    watch_flags = flags.CREATE | flags.DELETE | flags.MODIFY | flags.MOVED_TO | flags.MOVED_FROM

    for root, dirs, files in os.walk(str(FLUX_DIR)):
        inotify.add_watch(root, watch_flags)
        for d in dirs:
            inotify.add_watch(os.path.join(root, d), watch_flags)

    logger.info("inotify watcher ready")
    while True:
        events = inotify.read(timeout=1000)
        if events:
            # Debounce
            time.sleep(1)
            logger.debug("Change detected, syncing...")
            sync()


def _watch_poll() -> None:
    """Polling fallback (no inotify_simple)."""
    known = {p: p.stat().st_mtime for p in FLUX_DIR.rglob("*") if p.is_file()}
    logger.info("Polling watcher ready (no inotify_simple)")
    while True:
        time.sleep(5)
        changed = False
        for p in FLUX_DIR.rglob("*"):
            if not p.is_file():
                continue
            mtime = p.stat().st_mtime
            if known.get(p) != mtime:
                known[p] = mtime
                changed = True
        if changed:
            time.sleep(1)
            logger.debug("Change detected, syncing...")
            sync()


def main() -> None:
    # Acquire lock
    try:
        lock_fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, BlockingIOError):
        logger.info("Sync already running, skipping")
        return

    try:
        cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
        if cmd == "push":
            sync()
        elif cmd == "watch":
            watch_loop()
        elif cmd == "reconcile":
            reconcile_kustomizations()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python -m infra.flux_sync [push|watch|reconcile]")
            sys.exit(1)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    main()
