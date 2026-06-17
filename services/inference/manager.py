"""PoolManager — load config, resolve models to pools with priority fallback.

This module has NO runtime dependencies on Docker, httpx, or Ray. It's
purely a resolver: model name → ordered list of (pool, launcher) candidates.

The actual HTTP dispatch / container lifecycle lives in launcher.py and is
wired into the gateway separately. Keeping this pure means the resolution
logic is fully testable from the YAML alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from services.inference.config import (
    Pool,
    PoolSystem,
    Route,
    ModelLauncher,
    Optimization,
)

# ─── Default config location ──────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "inference_pools.yaml"


# ─── Resolution result ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResolvedTarget:
    """A single candidate for serving a model."""
    model: str
    pool: Pool
    launcher: ModelLauncher | None
    is_primary: bool
    fallback_index: int            # 0 for primary, 1+ for fallback slots

    @property
    def base_url(self) -> str:
        return self.pool.base_url

    def __repr__(self) -> str:
        tag = "primary" if self.is_primary else f"fallback#{self.fallback_index}"
        return (f"ResolvedTarget(model={self.model!r}, "
                f"pool={self.pool.name!r} [{self.pool.tier}], {tag})")


class PoolManager:
    """The resolver. Stateless aside from the loaded PoolSystem."""

    def __init__(self, system: PoolSystem):
        self.system = system

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PoolManager":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Inference pool config not found: {path}")
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PoolManager":
        pools = {name: Pool.from_dict(name, cfg)
                 for name, cfg in raw.get("pools", {}).items()}
        routes = {model: Route.from_dict(model, cfg)
                  for model, cfg in raw.get("routes", {}).items()}
        system = PoolSystem(pools=pools, routes=routes,
                            defaults=dict(raw.get("defaults", {})))
        return cls(system)

    # ── Introspection ───────────────────────────────────────────────────────

    def pools(self) -> list[Pool]:
        return self.system.pools_sorted()

    def pool(self, name: str) -> Pool | None:
        return self.system.pool(name)

    def models(self) -> list[str]:
        return sorted(self.system.routes.keys())

    def models_in_pool(self, pool_name: str) -> list[str]:
        pool = self.system.pool(pool_name)
        return list(pool.models) if pool else []

    # ── Resolution ──────────────────────────────────────────────────────────

    def resolve(self, model: str) -> list[ResolvedTarget]:
        """Resolve a model name to an ordered list of candidate targets.

        Order: primary pool first, then fallback pools in declared order.
        Each ResolvedTarget carries its pool + launcher (if any). Use the
        list to walk candidates at request time, skipping unhealthy pools.
        """
        route = self.system.route(model)
        if route is None:
            return []
        targets: list[ResolvedTarget] = []
        for idx, pool_name in enumerate(route.resolution_order):
            pool = self.system.pool(pool_name)
            if pool is None:
                continue
            # Only include pools that actually declare this model OR have a
            # launcher for it. (Rationale: a fallback entry in routes should
            # not silently route to a pool that can't serve the model.)
            if model not in pool.models and model not in pool.model_launchers:
                # Allow single-launcher pools to serve aliased models.
                if not (len(pool.model_launchers) == 1 and len(pool.models) >= 1):
                    continue
            launcher = self._launcher_for(model, pool)
            targets.append(ResolvedTarget(
                model=model, pool=pool, launcher=launcher,
                is_primary=(idx == 0), fallback_index=idx,
            ))
        return targets

    def resolve_primary(self, model: str) -> ResolvedTarget | None:
        targets = self.resolve(model)
        return targets[0] if targets else None

    def _launcher_for(self, model: str, pool: Pool) -> ModelLauncher | None:
        if model in pool.model_launchers:
            return pool.model_launchers[model]
        # Aliased match: a launcher whose extra["aliases"] lists this model.
        for launcher in pool.model_launchers.values():
            if model in launcher.extra.get("aliases", []):
                return launcher
        # Single-model pool implicit match.
        if len(pool.model_launchers) == 1:
            return next(iter(pool.model_launchers.values()))
        return None

    # ── Validation ──────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of human-readable config warnings (empty = OK)."""
        warnings: list[str] = []
        # Every route's primary pool must exist and serve the model.
        for model, route in self.system.routes.items():
            pool = self.system.pool(route.primary)
            if pool is None:
                warnings.append(
                    f"Route {model!r}: primary pool {route.primary!r} not defined")
                continue
            if model not in pool.models and model not in pool.model_launchers:
                if not (len(pool.model_launchers) == 1 and len(pool.models) >= 1):
                    warnings.append(
                        f"Route {model!r}: pool {pool.name!r} does not list it "
                        f"in models: or model_launchers:")
            for fb in route.fallback:
                fb_pool = self.system.pool(fb)
                if fb_pool is None:
                    warnings.append(
                        f"Route {model!r}: fallback pool {fb!r} not defined")
        # Every model declared in a pool should appear in routes (informational).
        route_keys = set(self.system.routes.keys())
        for pool in self.system.pools.values():
            for m in pool.models:
                if m not in route_keys:
                    warnings.append(
                        f"Pool {pool.name!r} serves {m!r} but no route declares it "
                        f"(will be unreachable via resolve())")
        # Tier sanity.
        for pool in self.system.pools.values():
            if pool.tier not in ("A", "B", "C", "D"):
                warnings.append(
                    f"Pool {pool.name!r}: tier {pool.tier!r} not in A/B/C/D")
        # Port collisions.
        seen_ports: dict[int, str] = {}
        for pool in self.system.pools.values():
            if pool.port in seen_ports:
                warnings.append(
                    f"Port collision: pool {pool.name!r} and {seen_ports[pool.port]!r} "
                    f"both use port {pool.port}")
            else:
                seen_ports[pool.port] = pool.name
        return warnings

    # ─── Optimization summary (for /status display) ──────────────────────────

    def optimization_summary(self, model: str) -> dict[str, Any]:
        """Return a compact summary of optimization for a model."""
        target = self.resolve_primary(model)
        if target is None or target.launcher is None:
            return {"model": model, "served": False}
        opt = target.launcher.optimization
        return {
            "model": model,
            "pool": target.pool.name,
            "tier": target.pool.tier,
            "framework": target.pool.framework,
            "script": target.launcher.script,
            "quant": opt.quant if opt else None,
            "cache_dit": opt.cache_dit if opt else False,
            "taylorseer": opt.taylorseer if opt else False,
            "teacache_thresh": opt.teacache_thresh if opt else None,
            "benchmark": target.launcher.benchmark,
        }
