"""Boot config — resolves service paths from registry.config.Config."""

from __future__ import annotations

from pathlib import Path

from registry.config import Config


def get_config() -> Config:
    """Get the shared Config singleton."""
    return Config()


def get_project_root() -> Path:
    """Get the ray project root."""
    return get_config().project_root


def get_programs_dir() -> Path:
    """Get the programs directory (parent of ray)."""
    return get_project_root().parent


def resolve_service_dir(path: str, relative_to_root: bool = False) -> Path:
    """Resolve a service working directory.

    If relative_to_root is True, the path is relative to the ray project root.
    Otherwise, it's treated as an absolute path.
    """
    p = Path(path)
    if relative_to_root or not p.is_absolute():
        return get_project_root() / p
    return p
