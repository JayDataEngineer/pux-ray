"""Avatar pipeline services — text-to-avatar via GEM + SOMA + FluxRT."""

import os


def models_root() -> str:
    """Resolve the models root directory (reuses registry.config.Config)."""
    from registry.config import Config
    return Config().models_root
