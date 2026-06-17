"""Dataclasses for the inference pool system.

Mirrors config/inference_pools.yaml. Pure data — no I/O, no Docker, no Ray.
This keeps the config schema testable without spinning up services.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── Tier enum ────────────────────────────────────────────────────────────────

TIERS = ("A", "B", "C", "D")


# ─── Optimization spec ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Optimization:
    """Per-model optimization knobs (FP8, Cache-DiT, TeaCache, etc.)."""
    quant: str | None = None
    cache_dit: bool = False
    taylorseer: bool = False
    teacache_thresh: float | None = None
    vae_tiling: bool = False
    vae_slicing: bool = False
    fn_compute_blocks: int | None = None
    bn_compute_blocks: int | None = None
    max_warmup_steps: int | None = None
    warmup_steps: int | None = None
    rdt: float | None = None
    mc: int | None = None
    cpu_offload_gb: int | None = None
    device_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Optimization | None":
        if not d:
            return None
        known = {f for f in cls.__dataclass_fields__} - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)


# ─── Per-model launcher ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelLauncher:
    """How a specific model is launched within its pool."""
    name: str
    pool: str                       # pool key this launcher belongs to
    script: str | None = None       # path to launch script
    patch: str | None = None        # bind-mounted pipeline patch
    model_dir: str | None = None
    binary: str | None = None
    variant: str | None = None      # e.g. "nf4", "fp8"
    api: dict[str, str] = field(default_factory=dict)  # action → endpoint
    optimization: Optimization | None = None
    benchmark: list[dict[str, Any]] = field(default_factory=list)
    env_requires: list[str] = field(default_factory=list)
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, pool: str, d: dict[str, Any]) -> "ModelLauncher":
        return cls(
            name=name,
            pool=pool,
            script=d.get("script"),
            patch=d.get("patch"),
            model_dir=d.get("model_dir"),
            binary=d.get("binary"),
            variant=d.get("variant"),
            api=dict(d.get("api", {})),
            optimization=Optimization.from_dict(d.get("optimization")),
            benchmark=list(d.get("benchmark", [])),
            env_requires=list(d.get("env_requires", [])),
            note=d.get("note"),
            extra={k: v for k, v in d.items()
                   if k not in {"script", "patch", "model_dir", "binary",
                                "variant", "api", "optimization",
                                "benchmark", "env_requires", "note"}},
        )

    @property
    def script_path(self) -> Path | None:
        return Path(self.script) if self.script else None

    @property
    def patch_path(self) -> Path | None:
        return Path(self.patch) if self.patch else None


# ─── Pool ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pool:
    """A serving pool — one docker container serving one or more models."""
    name: str
    tier: str                       # "A" | "B" | "C" | "D"
    priority: int                   # lower = higher priority
    image: str
    container: str
    port: int
    framework: str                  # moss, vllm-omni, sglang, diffusers, ...
    description: str = ""
    models: list[str] = field(default_factory=list)
    vram_mb: int = 0
    health_path: str = "/health"
    env: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    start_args: dict[str, Any] = field(default_factory=dict)
    model_launchers: dict[str, ModelLauncher] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "Pool":
        launchers = {
            mname: ModelLauncher.from_dict(mname, name, mcfg)
            for mname, mcfg in d.get("model_launchers", {}).items()
        }
        return cls(
            name=name,
            tier=d["tier"],
            priority=int(d["priority"]),
            image=d["image"],
            container=d["container"],
            port=int(d["port"]),
            framework=d["framework"],
            description=d.get("description", ""),
            models=list(d.get("models", [])),
            vram_mb=int(d.get("vram_mb", 0)),
            health_path=d.get("health_path", "/health"),
            env={k: str(v) for k, v in d.get("env", {}).items()},
            volumes={k: str(v) for k, v in d.get("volumes", {}).items()},
            start_args=dict(d.get("start_args", {})),
            model_launchers=launchers,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"


# ─── Route ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Route:
    """Resolution plan for a single model name."""
    model: str
    primary: str                    # pool name
    fallback: list[str] = field(default_factory=list)

    @property
    def resolution_order(self) -> list[str]:
        """Ordered list of pool names to try."""
        return [self.primary, *self.fallback]

    @classmethod
    def from_dict(cls, model: str, d: dict[str, Any]) -> "Route":
        return cls(
            model=model,
            primary=d["primary"],
            fallback=list(d.get("fallback", [])),
        )


# ─── Top-level config ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoolSystem:
    """The whole 4-tier system, parsed from inference_pools.yaml."""
    pools: dict[str, Pool]
    routes: dict[str, Route]
    defaults: dict[str, Any] = field(default_factory=dict)

    def pool(self, name: str) -> Pool | None:
        return self.pools.get(name)

    def route(self, model: str) -> Route | None:
        return self.routes.get(model)

    def pools_by_tier(self, tier: str) -> list[Pool]:
        return sorted((p for p in self.pools.values() if p.tier == tier),
                      key=lambda p: p.priority)

    def pools_sorted(self) -> list[Pool]:
        """All pools, lowest priority number first."""
        return sorted(self.pools.values(), key=lambda p: p.priority)

    def launcher_for(self, model: str) -> ModelLauncher | None:
        """Find the launcher for a model on its primary pool."""
        route = self.route(model)
        if route is None:
            return None
        pool = self.pool(route.primary)
        if pool is None:
            return None
        # Direct match on model name.
        if model in pool.model_launchers:
            return pool.model_launchers[model]
        # Some models are aliased (e.g. "z-image" → pool serves "z-image-sglang").
        for launcher in pool.model_launchers.values():
            if model in launcher.extra.get("aliases", []):
                return launcher
        # Fall back to first launcher if pool serves exactly one model.
        if len(pool.model_launchers) == 1:
            return next(iter(pool.model_launchers.values()))
        return None
