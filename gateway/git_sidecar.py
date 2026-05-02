"""Universal Git Sidecar — polls all infra repos and auto-installs deps.

A named detached Ray actor that:
1. Sweeps infra/repos/ (bare-metal tools) and ComfyUI custom_nodes/ every check_interval
2. Runs git pull on each repo
3. If HEAD changed, runs uv pip install (requirements.txt and/or pyproject.toml)
4. Logs changes; service reload is handled by each deployment on next request

Registered in deploy_services.py alongside GPUScheduler and JobManager.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import ray

logger = logging.getLogger(__name__)

# Absolute paths (resolved at module import time, before Ray serialization)
_REPOS_DIR = Path(__file__).resolve().parents[1] / "infra" / "repos"
_COMFY_CUSTOM_NODES = _REPOS_DIR / "ComfyUI" / "custom_nodes"
_VENV_PYTHON = _REPOS_DIR.parent / ".venv" / "bin" / "python"
_VENV_UV = _REPOS_DIR.parent / ".venv" / "bin" / "uv"


@ray.remote
class UniversalGitSidecar:
    """Polls all infra git repos, pulls updates, and installs dependencies."""

    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self._running = False
        self._repo_heads: dict[str, str] = {}
        self.repos_dir = _REPOS_DIR
        self.comfy_custom_nodes = _COMFY_CUSTOM_NODES
        self.venv_python = _VENV_PYTHON
        self.venv_uv = _VENV_UV

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info(
            "UniversalGitSidecar started (interval: %ds, repos: %s)",
            self.check_interval,
            self.repos_dir,
        )
        await self._run()

    async def stop(self) -> None:
        self._running = False
        logger.info("UniversalGitSidecar stopped")

    async def poll_once(self) -> dict[str, str]:
        await self.sync_all_repos()
        return dict(self._repo_heads)

    async def status(self) -> dict:
        return dict(self._repo_heads)

    async def _run(self) -> None:
        while self._running:
            try:
                await self.sync_all_repos()
            except Exception:
                logger.exception("UniversalGitSidecar poll error")
            await asyncio.sleep(self.check_interval)

    async def sync_all_repos(self) -> None:
        repos: list[Path] = []

        # Bare-metal tool repos
        if self.repos_dir.exists():
            for d in self.repos_dir.iterdir():
                if d.is_dir() and (d / ".git").exists():
                    repos.append(d)

        # ComfyUI custom_nodes
        if self.comfy_custom_nodes.exists():
            for d in self.comfy_custom_nodes.iterdir():
                if d.is_dir() and (d / ".git").exists():
                    repos.append(d)

        for repo in sorted(set(repos), key=lambda p: p.name):
            try:
                await self._sync_and_install(repo)
            except Exception:
                logger.exception("Sidecar sync failed for %s", repo.name)

    async def _sync_and_install(self, repo_path: Path) -> None:
        repo_name = repo_path.name

        # Fetch to check if behind
        await _run_async("git", "fetch", cwd=repo_path)

        # Get current HEAD
        sha_before = await _run_async("git", "rev-parse", "HEAD", cwd=repo_path)
        sha_before = sha_before.strip()

        # Pull
        result = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "master"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=60,
        )

        sha_after = await _run_async("git", "rev-parse", "HEAD", cwd=repo_path)
        sha_after = sha_after.strip()

        self._repo_heads[repo_name] = sha_after

        if sha_before != sha_after:
            logger.info(
                "%s changed: %s.. -> %s..",
                repo_name, sha_before[:12], sha_after[:12],
            )
            await self._install_deps(repo_path)
            logger.info("%s update complete; will reload on next request", repo_name)

    async def _install_deps(self, repo_path: Path) -> None:
        """Install python dependencies from the updated repo into the Ray .venv."""
        repo_name = repo_path.name
        uv = str(self.venv_uv)

        if (repo_path / "requirements.txt").exists():
            logger.info("Installing requirements.txt for %s", repo_name)
            result = subprocess.run(
                [uv, "pip", "install", "--system", "-r",
                 str(repo_path / "requirements.txt")],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "requirements.txt install for %s: %s",
                    repo_name, result.stderr[:200],
                )

        if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            logger.info("Installing package for %s as editable", repo_name)
            result = subprocess.run(
                [uv, "pip", "install", "--system", "-e", str(repo_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "editable install for %s: %s",
                    repo_name, result.stderr[:200],
                )


async def _run_async(*args: str, cwd: Path) -> str:
    """Run a command asynchronously and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug("Command %s failed: %s", args, stderr.decode()[:200])
    return stdout.decode().strip()
