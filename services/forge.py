"""The Forge — VRAM-aware GPU manager for commodity hardware.

One Ray Serve deployment claims the GPU (num_gpus: 1.0).
Tracks VRAM per service. Swaps models when needed.
Allows concurrent services if they fit in VRAM.
Evicts only when there isn't enough room.

Multi-node: GPUNode registry for Tailscale workers (Phase 4).
"""
from __future__ import annotations

import asyncio
import gc
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from services.forge_base import ForgeService
from services.forge_persistence import Persistence

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

TOTAL_VRAM_MB = 24_576
RESERVED_MB = 2_048
AVAILABLE_MB = TOTAL_VRAM_MB - RESERVED_MB
# Minimum free VRAM for self-managed services (mmgp) to coexist with
# other loaded services. Below this, the forge must evict first.
MIN_COFREE_MB = 4_096

# ─── Service Registry ────────────────────────────────────────────────────────

SERVICE_MAP: dict[str, tuple[str, str]] = {
    # Wan2GP — unified model pool (mmgp-managed VRAM, vram_mb=0)
    # All vendor + custom family models: wan, flux, hunyuan, trellis, anigen,
    # kokoro, index_tts, ace_step, moss, etc.
    "wan2gp":    ("services.wan2gp.forge_adapter",    "Wan2GPForgeService"),
    # ComfyUI — subprocess, separate GPU
    "comfyui":   ("services.image.comfyui",          "ComfyUIService"),
    # llama.cpp — subprocess, separate GPU
    "llm":       ("services.llm.deployment",          "LLMService"),
    # Avatar pipeline — GEM + SOMA + FluxRT orchestrator
    "avatar":    ("services.avatar.forge_avatar",     "AvatarForgeService"),
    # Kimodo demo — Viser interactive 3D motion authoring (subprocess)
    "kimodo_demo": ("services.motion.kimodo_demo",    "KimodoDemoService"),
}

# ─── GPU Node (multi-node prep) ──────────────────────────────────────────────

@dataclass
class GPUNode:
    """A GPU-bearing node in the cluster."""
    node_id: str
    address: str
    gpu_name: str = ""
    total_vram_mb: int = AVAILABLE_MB
    forge_handle: Any = None


# ─── ForgeCore — testable without Ray ────────────────────────────────────────

class ForgeCore:
    """VRAM-aware GPU manager. Testable without Ray Serve.

    Scheduling logic:
      - Services declare vram_mb (estimated VRAM footprint).
      - vram_mb == 0 means "I manage my own VRAM" (like Wan2GP with mmgp).
      - Multiple services can coexist if their combined VRAM fits.
      - Eviction only happens when a new service needs more VRAM than free.
      - After loading, actual VRAM is measured and tracked.
    """

    def __init__(self, service_map: dict[str, tuple[str, str]] | None = None):
        self._service_map = service_map or SERVICE_MAP
        self._services: Dict[str, ForgeService] = {}
        self._loaded: Dict[str, bool] = {}
        self._vram_allocations: Dict[str, int] = {}
        self._vram_free_mb: int = AVAILABLE_MB
        self._gpu_nodes: Dict[str, GPUNode] = {}
        self._persistence_overrides: Dict[str, Persistence] = {}
        self._loading: set[str] = set()  # services whose load() is in-flight

    # ── Service Resolution ────────────────────────────────────────────────────

    def _get_service(self, name: str) -> ForgeService:
        if name not in self._service_map:
            raise ValueError(f"Unknown service: {name}. Available: {sorted(self._service_map)}")
        if name in self._services:
            return self._services[name]

        module_path, class_name = self._service_map[name]
        if module_path == "__direct__":
            raise ValueError(f"Service {name} was registered but has no instance")

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        svc = cls()
        self._services[name] = svc
        self._loaded[name] = False
        logger.info("Forge: registered %s (%s, vram=%dMB)", name, cls.__name__, svc.vram_mb)
        return svc

    def _register_service(self, name: str, svc: ForgeService) -> None:
        """Directly register a service instance (for testing)."""
        self._service_map[name] = ("__direct__", svc.__class__.__name__)
        self._services[name] = svc
        self._loaded[name] = False

    # ── VRAM Accounting ───────────────────────────────────────────────────────

    def _total_allocated(self) -> int:
        return sum(self._vram_allocations.values())

    def _estimate_vram(self, name: str) -> int:
        svc = self._get_service(name)
        return svc.vram_mb

    def _can_fit(self, name: str) -> bool:
        estimate = self._estimate_vram(name)
        if estimate == 0:
            # Self-managed service (mmgp) — still needs GPU breathing room.
            # If another service has most VRAM pinned, mmgp will thrash or OOM.
            # Treat self-managed as fitting only if at least MIN_COFREE_MB is
            # available (or nothing else is loaded).
            if self._vram_allocations:
                return self._vram_free_mb >= MIN_COFREE_MB
            return True
        return estimate <= self._vram_free_mb

    def _evict_for(self, name: str) -> list[str]:
        needed = self._estimate_vram(name)
        if needed == 0:
            return []

        evicted = []
        loaded_by_priority = sorted(
            [(n, mb) for n, mb in self._vram_allocations.items() if self._loaded.get(n)],
            key=lambda x: (self._get_persistence(x[0]).value, -x[1]),
        )

        for svc_name, svc_mb in loaded_by_priority:
            if self._vram_free_mb >= needed:
                break
            if self._get_persistence(svc_name) >= Persistence.PIPELINE_LOCKED:
                continue
            logger.info("Forge: evicting %s (%dMB, persistence=%s) for %s (%dMB)",
                        svc_name, svc_mb, self._get_persistence(svc_name).name, name, needed)
            self._do_unload(svc_name)
            evicted.append(svc_name)

        if self._vram_free_mb < needed:
            raise RuntimeError(
                f"Cannot free enough VRAM for {name}: need {needed}MB, "
                f"free {self._vram_free_mb}MB (remaining services are pipeline-locked)"
            )

        return evicted

    def _get_persistence(self, name: str) -> Persistence:
        if name in self._persistence_overrides:
            return self._persistence_overrides[name]
        svc = self._services.get(name)
        if svc:
            return svc.persistence
        return Persistence.TRANSIENT

    # ── Model Lifecycle ───────────────────────────────────────────────────────

    def _cleanup_stale_allocations(self) -> None:
        """Clean VRAM allocations for services not actually loaded.

        Handles the case where a preload() call timed out or was cancelled
        after reserving VRAM but before the service finished loading.
        """
        for name in list(self._vram_allocations):
            if not self._loaded.get(name) and name not in self._loading:
                mb = self._vram_allocations.pop(name, 0)
                self._vram_free_mb += mb
                if mb > 0:
                    logger.warning(
                        "Forge: cleaned stale allocation %s (%dMB)", name, mb,
                    )

    def _do_load(self, name: str, model: str | None = None,
                 quant: str | None = None) -> None:
        svc = self._get_service(name)
        target_model = model or svc.default_model
        estimate = svc.vram_mb

        self._loading.add(name)
        self._reserve_vram(name, estimate)
        try:
            svc.load(target_model, quant=quant)
        except Exception:
            # Load failed — release the reserved VRAM
            self._vram_allocations.pop(name, None)
            self._vram_free_mb += estimate
            raise
        finally:
            self._loading.discard(name)
        self._loaded[name] = True
        self._reconcile_vram(name, estimate)

        logger.info(
            "Forge: loaded %s model=%s (estimated=%dMB, actual=%dMB, free=%dMB)",
            name, target_model, estimate, svc.actual_vram_mb(), self._vram_free_mb,
        )

    def _reserve_vram(self, name: str, estimate: int) -> None:
        if estimate > 0:
            self._vram_allocations[name] = estimate
            self._vram_free_mb -= estimate

    def _reconcile_vram(self, name: str, estimate: int) -> None:
        svc = self._get_service(name)
        actual = svc.actual_vram_mb()
        if actual <= 0:
            return
        diff = actual - estimate
        self._vram_allocations[name] = actual
        self._vram_free_mb -= diff

    def _do_unload(self, name: str) -> None:
        svc = self._services.get(name)
        if svc and self._loaded.get(name):
            svc.unload()
        self._loaded[name] = False

        freed = self._vram_allocations.pop(name, 0)
        self._vram_free_mb += freed
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Forge: unloaded %s (freed %dMB, free=%dMB)", name, freed, self._vram_free_mb)

    # ── Public API ────────────────────────────────────────────────────────────

    async def _load_with_cleanup(self, service: str, model: str | None = None,
                                  quant: str | None = None) -> None:
        """Load service with timeout. Cleans up VRAM on failure or timeout."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._do_load, service, model, quant),
                timeout=600,
            )
        except Exception:
            self._cleanup_stale_allocations()
            raise

    async def invoke(self, service: str, payload: dict,
                     model: str | None = None, quant: str | None = None) -> dict:
        if service not in self._service_map:
            return {"status": "error", "error": f"Unknown service: {service}"}

        if self._loaded.get(service):
            svc = self._services[service]
            return await asyncio.to_thread(svc.infer, payload)

        self._cleanup_stale_allocations()
        if not self._can_fit(service):
            self._evict_for(service)

        await self._load_with_cleanup(service, model, quant)

        if not self._loaded.get(service):
            return {"status": "error", "error": f"Failed to load {service}"}

        svc = self._services[service]
        return await asyncio.to_thread(svc.infer, payload)

    async def preload(self, service: str, model: str | None = None,
                      quant: str | None = None) -> dict:
        if service not in self._service_map:
            return {"status": "error", "error": f"Unknown service: {service}"}

        if self._loaded.get(service):
            return {"status": "already_loaded", "service": service}

        self._cleanup_stale_allocations()
        if not self._can_fit(service):
            self._evict_for(service)

        try:
            await self._load_with_cleanup(service, model, quant)
        except Exception:
            return {"status": "error", "error": f"Failed to load {service}"}

        return {
            "status": "loaded",
            "service": service,
            "vram_used_mb": self._vram_allocations.get(service, 0),
            "vram_free_mb": self._vram_free_mb,
        }

    async def release(self, service: str | None = None) -> dict:
        self._cleanup_stale_allocations()
        if service:
            if self._loaded.get(service):
                await asyncio.to_thread(self._do_unload, service)
            return {"status": "released", "service": service}

        to_unload = [n for n, loaded in self._loaded.items() if loaded]
        for name in to_unload:
            await asyncio.to_thread(self._do_unload, name)
        return {"status": "released", "services": to_unload}

    async def unload_service(self, service: str) -> None:
        self._cleanup_stale_allocations()
        if self._loaded.get(service):
            await asyncio.to_thread(self._do_unload, service)

    def status_sync(self) -> dict:
        """Synchronous status — no torch.cuda dependency."""
        self._cleanup_stale_allocations()
        loaded_services = {n: mb for n, mb in self._vram_allocations.items()
                           if self._loaded.get(n)}
        return {
            "loaded": loaded_services,
            "vram_free_mb": self._vram_free_mb,
            "vram_total_mb": AVAILABLE_MB,
            "vram_allocated_mb": self._total_allocated(),
        }

    async def status(self) -> dict:
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "device": props.name,
                    "total_mb": int(props.total_memory / (1024 * 1024)),
                    "allocated_mb": int(torch.cuda.memory_allocated(0) / (1024 * 1024)),
                    "reserved_mb": int(torch.cuda.memory_reserved(0) / (1024 * 1024)),
                }
        except ImportError:
            pass

        result = self.status_sync()
        result["gpu"] = gpu_info
        result["gpu_nodes"] = {k: {"address": v.address, "vram_mb": v.total_vram_mb}
                               for k, v in self._gpu_nodes.items()}
        return result


# ─── Ray Serve Deployment Wrapper ────────────────────────────────────────────

@serve.deployment(
    name="forge",
    num_replicas=1,
    max_ongoing_requests=4,
    ray_actor_options={"num_gpus": 1.0},
)
class Forge:
    """Thin Ray Serve wrapper around ForgeCore."""

    def __init__(self):
        self._core = ForgeCore()

    async def invoke(self, service: str, payload: dict,
                     model: str | None = None, quant: str | None = None) -> dict:
        return await self._core.invoke(service, payload, model, quant)

    async def preload(self, service: str, model: str | None = None,
                      quant: str | None = None) -> dict:
        return await self._core.preload(service, model, quant)

    async def release(self, service: str | None = None) -> dict:
        return await self._core.release(service)

    async def unload_service(self, service: str) -> None:
        await self._core.unload_service(service)

    async def status(self) -> dict:
        return await self._core.status()

    async def run_pipeline(self, pipeline_id: str, params: dict) -> dict:
        """Execute a registered workflow with Forge-aware VRAM tracking.

        Injects a ForgeProxy (instead of bare Wan2GPService) so all
        load/infer calls go through the Forge's VRAM ledger.
        """
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        from services.forge_persistence import Persistence
        from services.workflows.base import set_forge_core, clear_forge_core

        fn = _WORKFLOW_REGISTRY.get(pipeline_id)
        if fn is None:
            return {"status": "error", "error": f"Unknown pipeline: {pipeline_id}"}

        # Pipeline-lock wan2gp for the duration of execution
        self._core._persistence_overrides["wan2gp"] = Persistence.PIPELINE_LOCKED
        set_forge_core(self._core)

        try:
            result = await asyncio.to_thread(fn, **params)
        except TypeError as e:
            return {"status": "error", "error": f"Invalid parameters: {e}"}
        except Exception as e:
            logger.exception("Pipeline %s failed", pipeline_id)
            return {"status": "error", "error": str(e)}
        finally:
            self._core._persistence_overrides.pop("wan2gp", None)
            clear_forge_core()

        return result

    async def __call__(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse(await self.status())

        body = await request.json()

        # Action routes (don't require a service)
        action = body.get("action", "")
        if action == "release":
            svc = body.get("service")
            return JSONResponse(await self.release(svc))
        if action == "status":
            return JSONResponse(await self.status())
        if action == "preload":
            service = body.get("service")
            model = body.get("model")
            quant = body.get("quant")
            return JSONResponse(await self.preload(service, model, quant))

        service = body.get("service")
        if not service or service not in SERVICE_MAP:
            return JSONResponse(
                {"status": "error",
                 "error": f"Specify 'service' as one of {sorted(SERVICE_MAP)}"},
                status_code=400,
            )

        payload = {k: v for k, v in body.items() if k != "service"}
        model = payload.get("model", None)
        quant = payload.get("quant", None)
        result = await self.invoke(service, payload, model, quant)
        return JSONResponse(result)


forge = Forge.bind()
