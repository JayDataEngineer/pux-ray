"""Inference Pool System — 4-tier priority fallback for model serving.

Public API:
    from services.inference import PoolManager, ResolvedTarget, Pool, Route

    mgr = PoolManager.from_yaml()       # loads config/inference_pools.yaml
    targets = mgr.resolve("z-image")    # → [primary omni-vllm, fallback sglang]
    mgr.validate()                      # → list of config warnings

The manager is a pure resolver. Container lifecycle and HTTP dispatch live
in launcher.py and the gateway integration.
"""
from services.inference.config import (
    ModelLauncher,
    Optimization,
    Pool,
    PoolSystem,
    Route,
    TIERS,
)
from services.inference.manager import (
    DEFAULT_CONFIG_PATH,
    PoolManager,
    ResolvedTarget,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ModelLauncher",
    "Optimization",
    "Pool",
    "PoolManager",
    "PoolSystem",
    "ResolvedTarget",
    "Route",
    "TIERS",
]
