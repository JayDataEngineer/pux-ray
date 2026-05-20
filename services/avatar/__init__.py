"""Avatar pipeline services — text-to-avatar via Kimodo + FluxRT."""

import os


def models_root() -> str:
    """Resolve the models root directory (reuses registry.config.Config)."""
    from registry.config import Config
    return Config().models_root
