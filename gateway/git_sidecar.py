"""Git Sidecar — polls all infrastructure git repos and reloads on push.

Replaces systemd timer. A named detached Ray actor that:
1. Watches infra/repos/ (ComfyUI, ACE-Step, llama.cpp, qwen, etc.)
2. Watches ComfyUI custom_nodes/ (extensions)
3. Runs git pull on every repo every POLL_INTERVAL seconds
4. If HEAD changed, triggers reload of the corresponding Ray service
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import ray
from ray import serve

logger = logging.getLogger(__name__)

POLL_INTERVAL = 300

REPOS_DIR = Path("/home/user/Documents/programs/ray/infra/repos")
COMFYUI_CUSTOM_NODES = REPOS_DIR / "ComfyUI" / "custom_nodes"

# What to reload when a given repo changes.
# Key: repo directory name. Value: (deployment, app) tuple for stop_comfyui / load_model.
RELOAD_MAP: dict[str, tuple[str, str]] = {
    "ComfyUI": ("comfyui", "comfyui"),
    "llama.cpp": ("llm", "llm"),
}

# Directories outside REPOS_DIR to also watch (e.g. symlinked extensions).
EXTRA_DIRS: list[Path] = []


def _find_git_repos(root: Path) -> list[Path]:
    """Recursively find git repos under root, max depth 2."""
    repos = []
    for p in root.iterdir():
        if (p / ".git").is_dir():
            repos.append(p)
        elif p.is_dir():
            for sub in p.iterdir():
                if (sub / ".git").is_dir():
                    repos.append(sub)
    return sorted(repos)


@ray.remote
class GitSidecar:
    """Polls all infra git repos and triggers reloads on changes."""

    def __init__(self):
        self._repo_heads: dict[str, str] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("GitSidecar started (interval: %ds, root: %s)",
                     POLL_INTERVAL, REPOS_DIR)
        await self._poll_loop()

    async def stop(self) -> None:
        self._running = False
        logger.info("GitSidecar stopped")

    async def poll_once(self) -> dict[str, str]:
        return await self._check_and_reload()

    async def status(self) -> dict:
        return dict(self._repo_heads)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._check_and_reload()
            except Exception:
                logger.exception("GitSidecar poll error")
            await asyncio.sleep(POLL_INTERVAL)

    async def _check_and_reload(self) -> dict[str, str]:
        changed: set[str] = set()
        current: dict[str, str] = {}
        repos: list[Path] = []

        if REPOS_DIR.is_dir():
            repos.extend(_find_git_repos(REPOS_DIR))
        repos.extend(EXTRA_DIRS)

        for repo_dir in repos:
            if not repo_dir.is_dir() or not (repo_dir / ".git").is_dir():
                continue
            repo_name = repo_dir.name
            try:
                result = subprocess.run(
                    ["git", "pull", "--ff-only", "origin", "master"],
                    cwd=str(repo_dir),
                    capture_output=True, text=True, timeout=60,
                )
                sha = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_dir),
                    capture_output=True, text=True, timeout=10,
                )
                head = sha.stdout.strip()

                if result.returncode == 0 and "Already up to date" not in result.stdout:
                    logger.info("git pull %s: %s", repo_name, result.stdout.strip()[:200])

                current[repo_name] = head
                previous = self._repo_heads.get(repo_name)
                if previous and previous != head:
                    changed.add(repo_name)
                    logger.info("%s changed: %s.. -> %s..",
                                repo_name, previous[:12], head[:12])
            except subprocess.TimeoutExpired:
                logger.warning("git pull timed out for %s", repo_name)
            except Exception:
                logger.exception("git pull failed for %s", repo_name)

        self._repo_heads = current

        if changed:
            await self._reload_services(changed)

        return current

    async def _reload_services(self, changed: set[str]) -> None:
        """Reload services whose repos changed."""
        for repo_name in changed:
            target = RELOAD_MAP.get(repo_name)
            if not target:
                continue
            dep_name, app_name = target
            try:
                handle = serve.get_deployment_handle(dep_name, app_name)
                await handle.options(method_name="stop_comfyui").remote()
                logger.info("%s reloaded via GitSidecar", repo_name)
            except Exception:
                logger.exception("Failed to reload %s via GitSidecar", repo_name)
