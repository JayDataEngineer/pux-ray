"""ComfyUI Extension Sidecar - polls git repos for updates and restarts ComfyUI.

A named detached Ray actor that:
1. Scans ComfyUI custom_nodes/ for git repos every POLL_INTERVAL seconds
2. Runs git pull on each repo
3. If any repo's HEAD changed, calls stop_comfyui via Serve handle
   (ComfyUI auto-restarts on next request through the proxy)

Registered in deploy_services.py alongside GPUScheduler and JobManager.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import ray
from ray import serve

logger = logging.getLogger(__name__)

# Poll interval in seconds (5 minutes)
POLL_INTERVAL = 300

# Path to ComfyUI custom_nodes directory
COMFYUI_CUSTOM_NODES = Path(
    "/home/user/Documents/programs/ray/infra/repos/ComfyUI/custom_nodes"
)


@ray.remote
class ComfyUISidecar:
    """Polls ComfyUI custom extension git repos and triggers reload on changes.

    Usage:
        sidecar = ray.get_actor("comfyui_sidecar")
        # It runs its poll loop automatically after init.
    """

    def __init__(self):
        self._repo_heads: dict[str, str] = {}
        self._running = False

    async def start(self) -> None:
        """Start the polling loop (called once after actor creation)."""
        if self._running:
            logger.info("ComfyUI sidecar already running")
            return

        self._running = True
        logger.info(
            "ComfyUI sidecar started (poll interval: %ds, path: %s)",
            POLL_INTERVAL,
            COMFYUI_CUSTOM_NODES,
        )
        await self._poll_loop()

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        logger.info("ComfyUI sidecar stopped")

    async def poll_once(self) -> dict[str, str]:
        """Force a single poll cycle. Returns dict of repo -> commit SHA."""
        return await self._check_and_reload()

    async def status(self) -> dict:
        """Return current tracked repo heads."""
        return dict(self._repo_heads)

    async def _poll_loop(self) -> None:
        """Main polling loop. Runs until stopped."""
        while self._running:
            try:
                await self._check_and_reload()
            except Exception:
                logger.exception("ComfyUI sidecar poll error")
            await asyncio.sleep(POLL_INTERVAL)

    async def _check_and_reload(self) -> dict[str, str]:
        """Check all git repos in custom_nodes for changes. Reload ComfyUI if changed."""
        if not COMFYUI_CUSTOM_NODES.is_dir():
            logger.warning("ComfyUI custom_nodes not found: %s", COMFYUI_CUSTOM_NODES)
            return {}

        changed = False
        current: dict[str, str] = {}

        for repo_dir in sorted(COMFYUI_CUSTOM_NODES.iterdir()):
            if not repo_dir.is_dir():
                continue
            git_dir = repo_dir / ".git"
            if not git_dir.exists():
                continue

            repo_name = repo_dir.name
            try:
                result = subprocess.run(
                    ["git", "pull", "--ff-only", "origin", "master"],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Use rev-parse to get current HEAD (even if pull failed due to no remote)
                sha_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                head = sha_result.stdout.strip()

                if result.returncode == 0 and "Already up to date" not in result.stdout:
                    logger.info(
                        "Git pull %s: %s", repo_name, result.stdout.strip()[:200]
                    )

                current[repo_name] = head

                previous = self._repo_heads.get(repo_name)
                if previous and previous != head:
                    changed = True
                    logger.info(
                        "%s changed: %s... -> %s...",
                        repo_name,
                        previous[:12],
                        head[:12],
                    )

            except subprocess.TimeoutExpired:
                logger.warning("Git pull timed out for %s", repo_name)
                continue
            except Exception:
                logger.exception("Git pull failed for %s", repo_name)
                continue

        self._repo_heads = current

        if changed:
            await self._reload_comfyui()

        return current

    async def _reload_comfyui(self) -> None:
        """Call stop_comfyui on the ComfyUI Serve deployment.
        ComfyUI will auto-restart on the next request through its proxy.
        """
        try:
            handle = serve.get_deployment_handle("comfyui", "comfyui")
            await handle.options(method_name="stop_comfyui").remote()
            logger.info("ComfyUI stopped via sidecar; will restart on next request")
        except Exception:
            logger.exception("Failed to reload ComfyUI via sidecar")
