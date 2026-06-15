"""The Forge — VRAM-aware GPU manager for commodity hardware.

One Ray Serve deployment claims the GPU (num_gpus: 1.0).
Tracks VRAM per service. Swaps models when needed.
Allows concurrent services if they fit in VRAM.
Evicts only when there isn't enough room.

Multi-node: GPUNode registry for Tailscale workers (Phase 4).
"""
from __future__ import annotations

import asyncio
import os
import time
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


def _real_gpu_free_mb() -> int | None:
    """Return actual free GPU memory in MB, or None if CUDA unavailable.

    Uses torch.cuda.mem_get_info() which reports driver-level free memory,
    accounting for ALL GPU allocations (PyTorch, subprocesses like llama.cpp,
    CUDA contexts, etc.), not just PyTorch-tracked allocations.

    Returns 0 (not None) on CUDA errors — this forces the caller to treat
    the GPU as full and trigger eviction.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info(0)
            return int(free_bytes / (1024 * 1024))
    except Exception as exc:
        logger.warning("GPU memory query failed (%s) — treating as 0 free", exc)
        return 0
    return None

# ─── Constants ────────────────────────────────────────────────────────────────

TOTAL_VRAM_MB = 24_576
RESERVED_MB = 2_048
AVAILABLE_MB = TOTAL_VRAM_MB - RESERVED_MB
# Minimum free VRAM for self-managed services (mmgp) to coexist with
# other loaded services. Below this, the forge must evict first.
MIN_COFREE_MB = 4_096

# ─── Service Registry ────────────────────────────────────────────────────────

SERVICE_MAP: dict[str, tuple[str, str]] = {
    # Native diffusers — ALL models served through adaptive VRAM optimization
    # Supports: Z-Image, Anima, FLUX, Wan, LTX, Qwen-Image + LoRA + multi-format
    "native":    ("services.native.forge_adapter",    "NativeForgeService"),
    # Route specific model families to native service
    "z-image":   ("services.native.forge_adapter",    "NativeForgeService"),
    "anima":     ("services.native.forge_adapter",    "NativeForgeService"),
    # NOTE: wan2gp service removed — replaced by native.
    # If legacy wan2gp is needed, uncomment and install mmgp:
    # "wan2gp":    ("services.wan2gp.forge_adapter",    "Wan2GPForgeService"),
    # MOSS audio — standalone container (TTS + Sound Effects)
    "moss":      ("services.audio.forge_moss",        "MossForgeService"),
    # ComfyUI — subprocess, separate GPU
    "comfyui":   ("services.image.comfyui",          "ComfyUIService"),
    # llama.cpp — subprocess, separate GPU
    "llm":       ("services.llm.deployment",          "LLMService"),
    # Lance — ByteDance unified multimodal (subprocess, self-managed VRAM)
    "lance":     ("services.lance.forge_lance",       "LanceForgeService"),
    # Avatar pipeline — GEM + SOMA + FluxRT orchestrator
    "avatar":    ("services.avatar.forge_avatar",     "AvatarForgeService"),
    # Kimodo demo — Viser interactive 3D motion authoring (subprocess)
    "kimodo_demo": ("services.motion.kimodo_demo",    "KimodoDemoService"),
    # GEM-X — NVIDIA video-to-SOMA mesh pose estimation (subprocess)
    "gemx":       ("services.gemx.forge_gemx",        "GemxForgeService"),
    # kohya_ss — LoRA training (subprocess, flux_train_network.py)
    "kohya":      ("services.kohya.forge_kohya",      "KohyaForgeService"),
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
        svc._forge_core = self
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
            # Check real GPU memory, not just our internal ledger, because
            # subprocess services (LLM, ComfyUI) allocate outside PyTorch
            # and our VRAM tracking may underestimate their usage.
            real_free = _real_gpu_free_mb()
            if real_free is not None and real_free < MIN_COFREE_MB:
                return False
            if self._vram_allocations:
                return self._vram_free_mb >= MIN_COFREE_MB
            return True
        # Ledger-managed service — also check real GPU memory when self-managed
        # services (vram_mb=0) are loaded, since they don't appear in the ledger.
        if estimate <= self._vram_free_mb:
            has_self_managed = any(
                self._loaded.get(n) and self._services.get(n) and self._services[n].vram_mb == 0
                for n in self._loaded
            )
            if has_self_managed:
                real_free = _real_gpu_free_mb()
                if real_free is not None and real_free < estimate + MIN_COFREE_MB:
                    return False
            return True
        return False

    def _evict_for(self, name: str, *, force: bool = False) -> list[str]:
        needed = self._estimate_vram(name)
        if needed == 0:
            # Self-managed: evict anything loaded to free real GPU memory
            real_free = _real_gpu_free_mb()
            if real_free is not None and real_free >= MIN_COFREE_MB:
                return []
            # Fall through to evict loaded services so real GPU memory is freed
            needed = MIN_COFREE_MB

        evicted = []

        # Build eviction candidates from BOTH ledger-tracked AND self-managed
        # (vram_mb=0) services. Self-managed services (legacy mmgp) don't
        # appear in _vram_allocations but hold real GPU memory that must be freed.
        candidates: dict[str, int] = {}
        for n in self._loaded:
            if not self._loaded[n]:
                continue
            mb = self._vram_allocations.get(n, 0)
            svc = self._services.get(n)
            if svc and svc.vram_mb == 0:
                mb = 0  # self-managed: real usage unknown, mark as 0 for sorting
            candidates[n] = mb

        loaded_by_priority = sorted(
            candidates.items(),
            key=lambda x: (self._get_persistence(x[0]).value, -x[1]),
        )

        for svc_name, svc_mb in loaded_by_priority:
            # Check both ledger AND real GPU memory — subprocess services
            # (LLM, ComfyUI) use far more VRAM than the ledger tracks.
            real_free = _real_gpu_free_mb()
            ledger_ok = self._vram_free_mb >= needed
            gpu_ok = real_free is None or real_free >= MIN_COFREE_MB
            if ledger_ok and gpu_ok:
                break
            # PIPELINE_LOCKED services are never evicted — only explicit release() unloads them.
            if self._get_persistence(svc_name) >= Persistence.PIPELINE_LOCKED:
                continue
            logger.info("Forge: evicting %s (%dMB, persistence=%s, force=%s) for %s (%dMB)",
                        svc_name, svc_mb, self._get_persistence(svc_name).name, force, name, needed)
            self._do_unload(svc_name)
            evicted.append(svc_name)

        # Final check: if real GPU memory is still too low, we couldn't evict enough
        real_free = _real_gpu_free_mb()
        if real_free is not None and real_free < MIN_COFREE_MB and not evicted:
            locked = [n for n in self._loaded if self._loaded[n]
                      and self._get_persistence(n) >= Persistence.PIPELINE_LOCKED]
            hint = (f" Pipeline-locked services are holding GPU: {locked}. "
                    f"Release them first.") if locked else ""
            raise RuntimeError(
                f"Cannot free enough VRAM for {name}: need {needed}MB, "
                f"GPU free {real_free}MB.{hint}"
            )
        if self._vram_free_mb < needed and needed > 0:
            locked = [n for n in self._loaded if self._loaded[n]
                      and self._get_persistence(n) >= Persistence.PIPELINE_LOCKED]
            hint = (f" Pipeline-locked services are holding GPU: {locked}. "
                    f"Release them first.") if locked else ""
            raise RuntimeError(
                f"Cannot free enough VRAM for {name}: need {needed}MB, "
                f"ledger free {self._vram_free_mb}MB.{hint}"
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
                 quant: str | None = None, payload: dict | None = None) -> None:
        svc = self._get_service(name)
        # Resolve default_model: check registry.py first, then fall back to service class
        registry_default = None
        try:
            from services.registry import SERVICE_REGISTRY
            entry = SERVICE_REGISTRY.get(name)
            if entry and entry.default_model:
                registry_default = entry.default_model
        except Exception:
            pass
        target_model = model or (payload.get("model") or payload.get("model_type") if payload else None) or registry_default or svc.default_model
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
        """Reconcile ledger with real GPU memory after loading a service.

        Uses real GPU memory (torch.cuda.mem_get_info) as ground truth
        instead of torch.cuda.memory_allocated, which doesn't see
        subprocess (llama.cpp) or mmgp allocations.
        """
        real_free = _real_gpu_free_mb()
        if real_free is not None:
            # Real GPU memory is the ground truth. Calculate actual usage
            # as the delta between what the ledger thinks is free and what
            # the GPU driver reports.
            total_real = 0
            try:
                import torch
                if torch.cuda.is_available():
                    _, total_real = torch.cuda.mem_get_info(0)
                    total_real = int(total_real / (1024 * 1024))
            except Exception:
                pass
            if total_real > 0:
                real_used = total_real - real_free
                # Subtract ledger entries for OTHER services (already tracked)
                other_tracked = sum(
                    mb for n, mb in self._vram_allocations.items()
                    if n != name
                )
                this_actual = max(0, real_used - other_tracked)
                diff = this_actual - estimate
                self._vram_allocations[name] = this_actual
                self._vram_free_mb -= diff
                logger.info("Forge: reconciled %s to %dMB (real GPU: %d used, %d free)",
                            name, this_actual, real_used, real_free)
                return
        # Fallback: use PyTorch tracking (misses subprocess/mmgp)
        svc = self._get_service(name)
        actual = svc.actual_vram_mb()
        if actual <= 0:
            return
        diff = actual - estimate
        self._vram_allocations[name] = actual
        self._vram_free_mb -= diff

    def _do_unload(self, name: str) -> None:
        svc = self._services.get(name)
        before_gpu = _real_gpu_free_mb()
        if svc and self._loaded.get(name):
            svc.unload()
        self._loaded[name] = False

        ledger_freed = self._vram_allocations.pop(name, 0)
        self._vram_free_mb += ledger_freed
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Reconcile with real GPU memory — subprocess services (LLM, ComfyUI)
        # allocate outside PyTorch so our ledger underestimates their usage.
        # Wait up to 10s for real GPU memory to be released after killing subprocesses.
        for _ in range(10):
            after_gpu = _real_gpu_free_mb()
            if after_gpu is not None and before_gpu is not None:
                real_freed = after_gpu - before_gpu
                if real_freed > ledger_freed:
                    diff = real_freed - ledger_freed
                    self._vram_free_mb += diff
                    logger.info("Forge: %s real GPU freed %dMB (ledger had %dMB, corrected +%dMB)",
                                name, real_freed, ledger_freed, diff)
                # Check if we actually freed the expected amount
                if after_gpu >= before_gpu + ledger_freed - 1024:  # tolerate 1GB slack
                    break
            time.sleep(1)
        else:
            after_gpu = _real_gpu_free_mb()
            if before_gpu is not None and after_gpu is not None:
                real_freed = after_gpu - before_gpu
                if real_freed > ledger_freed:
                    diff = real_freed - ledger_freed
                    self._vram_free_mb += diff
            logger.warning("Forge: %s GPU memory may not be fully released after 10s", name)

        logger.info("Forge: unloaded %s (freed %dMB, free=%dMB)", name, ledger_freed, self._vram_free_mb)

    # ── Public API ────────────────────────────────────────────────────────────

    async def _wait_gpu_ready(self, service: str) -> None:
        """Wait for real GPU memory to be free after eviction.

        After evicting services (especially large ones like TRELLIS via mmgp),
        GPU memory is freed asynchronously by the driver and CUDA — this waits
        up to 120s for it to settle before the new service tries to allocate.
        """
        estimate = self._estimate_vram(service)
        needed = max(estimate, MIN_COFREE_MB)

        for i in range(120):
            real_free = _real_gpu_free_mb()
            if real_free is None or real_free >= needed:
                return
            if i % 10 == 0:
                logger.info("Forge: waiting for GPU memory (free=%dMB, need=%dMB) for %s",
                            real_free, needed, service)
            await asyncio.sleep(1)

        logger.warning("Forge: GPU still not ready after 120s for %s (proceeding anyway)", service)

    async def _load_with_cleanup(self, service: str, model: str | None = None,
                                  quant: str | None = None, payload: dict | None = None) -> None:
        """Load service with timeout. Cleans up VRAM on failure or timeout."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._do_load, service, model, quant, payload),
                timeout=1200,
            )
        except Exception:
            self._cleanup_stale_allocations()
            raise

    async def invoke(self, service: str, payload: dict,
                     model: str | None = None, quant: str | None = None) -> dict:
        if service not in self._service_map:
            return {"status": "error", "error": f"Unknown service: {service}"}

        # Inject model into payload so service.infer() can find it
        if model and "model" not in payload:
            payload = {**payload, "model": model}

        if self._loaded.get(service):
            # Service is loaded — but if OTHER services are also loaded, the
            # GPU might be overcommitted. Self-managed services (vram_mb=0)
            # need exclusive GPU access for model switching. Evict others.
            # invoke() is an explicit user action — override all locks.
            others_loaded = [n for n, l in self._loaded.items()
                            if l and n != service]
            for other in others_loaded:
                logger.info("Forge: evicting co-loaded %s (service %s is handling invoke)",
                            other, service)
                await asyncio.to_thread(self._do_unload, other)
            if others_loaded:
                await self._wait_gpu_ready(service)
            svc = self._services[service]
            return await asyncio.to_thread(svc.infer, payload)

        # Service is not loaded — evict anything else holding GPU before loading.
        # Self-managed services (vram_mb=0) bypass _can_fit() so
        # _evict_for is never called for them. Co-loaded subprocess services
        # (LLM, kimodo) hold real GPU memory outside the ledger. Boot them.
        # invoke() is an explicit user action — override all locks.
        others_loaded = [n for n, l in self._loaded.items()
                         if l and n != service]
        for other in others_loaded:
            logger.info("Forge: evicting co-loaded %s before loading %s (invoke)",
                        other, service)
            await asyncio.to_thread(self._do_unload, other)
        if others_loaded:
            await self._wait_gpu_ready(service)

        self._cleanup_stale_allocations()
        if not self._can_fit(service):
            self._evict_for(service, force=True)

        # After evicting subprocess services (LLM, ComfyUI), real GPU memory
        # is freed asynchronously. Self-managed services (vram_mb=0)
        # bypass the ledger check in _can_fit() and go straight to CUDA malloc.
        # Wait for real GPU memory to be available before loading.
        await self._wait_gpu_ready(service)

        await self._load_with_cleanup(service, model, quant, payload)

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

        # Clear stale pipeline locks before eviction — explicit user action
        # overrides any locks left by crashed/abandoned pipeline runs.
        self._persistence_overrides.clear()

        self._cleanup_stale_allocations()
        try:
            if not self._can_fit(service):
                self._evict_for(service, force=True)
        except RuntimeError as e:
            return {"status": "error", "error": str(e)}

        await self._wait_gpu_ready(service)

        try:
            await self._load_with_cleanup(service, model, quant)
        except Exception as exc:
            logger.exception("Forge: preload %s failed", service)
            return {"status": "error", "error": f"Failed to load {service}: {exc}"}

        if not self._loaded.get(service):
            return {"status": "error", "error": f"Failed to load {service}: service reported not loaded after _do_load"}

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
        # Reconcile stale counters: if nothing is loaded but allocations
        # persist (from orphaned loads), reset to ground truth.
        if not any(self._loaded.values()):
            stale = self._total_allocated()
            if stale > 0:
                logger.warning("Forge: resetting %dMB in stale allocations", stale)
                self._vram_allocations.clear()
                self._vram_free_mb = AVAILABLE_MB
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
        # Serialize preload/invoke so concurrent requests don't race on
        # VRAM checks and model loading. Without this, two simultaneous
        # preload calls can both pass _can_fit() before either marks
        # itself as loaded, causing GPU overcommit.
        self._load_lock = asyncio.Lock()
        # Reap any zombie subprocesses left by previous forge replicas
        self._reap_zombies()

    @staticmethod
    def _reap_zombies():
        """Kill and reap any leftover subprocess processes from previous replicas.

        When a forge replica dies without cleanly stopping its subprocess
        (e.g. during a Ray Serve redeploy), the llama-server or ComfyUI
        becomes an orphan. Kill any remaining subprocesses to free GPU memory
        and ports.
        """
        import os, signal
        # Kill known C++ zombie binaries by process name (safe — no -f flag).
        for name in ["llama-server"]:
            try:
                subprocess.run(["pkill", "-9", name],
                               capture_output=True, timeout=5)
            except Exception:
                pass
        # Kill Python zombie subprocesses by reading /proc/cmdline directly.
        # This avoids pkill -f matching command-line arguments inside the
        # raylet (e.g. --runtime_env_agent_command=/usr/bin/python3 ... main.py)
        # which would SIGKILL the raylet and crash the entire worker node.
        _PY_PATTERNS = ["main.py", "llama_cpp", "forge_kohya"]
        # Skip Ray internal processes (their cmdline contains " main.py" in
        # agent/main.py, dashboard scripts, etc. — not forge zombies).
        _SKIP_PATHS = ["/_private/", "/ray/dashboard/", "/ray/autoscaler/"]
        for proc_dir in os.listdir("/proc"):
            if not proc_dir.isdigit():
                continue
            pid = int(proc_dir)
            if pid == os.getpid():
                continue
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
                if not comm.startswith("python"):
                    continue
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="replace").replace(
                        "\0", " "
                    )
                if any(p in cmdline for p in _SKIP_PATHS):
                    continue
                for pat in _PY_PATTERNS:
                    if pat in cmdline:
                        os.kill(pid, signal.SIGKILL)
                        break
            except (IOError, OSError, ValueError):
                pass
        time.sleep(2)

    async def invoke(self, service: str, payload: dict,
                     model: str | None = None, quant: str | None = None) -> dict:
        async with self._load_lock:
            return await self._core.invoke(service, payload, model, quant)

    async def preload(self, service: str, model: str | None = None,
                      quant: str | None = None) -> dict:
        async with self._load_lock:
            return await self._core.preload(service, model, quant)

    async def release(self, service: str | None = None) -> dict:
        async with self._load_lock:
            return await self._core.release(service)

    async def unload_service(self, service: str) -> None:
        await self._core.unload_service(service)

    async def status(self) -> dict:
        return await self._core.status()

    async def run_pipeline(self, pipeline_id: str, params: dict) -> dict:
        """Execute a registered workflow with Forge-aware VRAM tracking.

        Injects a ForgeProxy (instead of bare Wan2GPService) so all
        load/infer calls go through the Forge's VRAM ledger.

        Supports both legacy Python functions and YAML spec-based workflows.
        """
        from gateway.routes.workflows import _WORKFLOW_REGISTRY
        from services.forge_persistence import Persistence
        from services.workflows.base import set_forge_core, clear_forge_core

        fn = _WORKFLOW_REGISTRY.get(pipeline_id)
        if fn is not None:
            # Legacy Python pipeline
            self._core._persistence_overrides["native"] = Persistence.PIPELINE_LOCKED
            set_forge_core(self._core)
            try:
                result = await asyncio.to_thread(fn, **params)
            except TypeError as e:
                return {"status": "error", "error": f"Invalid parameters: {e}"}
            except Exception as e:
                logger.exception("Pipeline %s failed", pipeline_id)
                return {"status": "error", "error": str(e)}
            finally:
                self._core._persistence_overrides.pop("native", None)
                clear_forge_core()
            return result

        # Try YAML spec-based workflow via the workflow engine
        from services.workflows.spec import load_spec, list_specs
        spec_names = list_specs()
        if pipeline_id in spec_names:
            try:
                engine = serve.get_deployment_handle(
                    "workflow_engine", "workflow-engine"
                )
                run_info = await engine.start_run.remote(pipeline_id, params, skip_review=True)
                run_id = run_info["run_id"]

                # Poll until complete (with timeout)
                import time as _time
                t0 = _time.monotonic()
                timeout = 600  # 10 minutes
                while True:
                    run_data = await engine.get_run.remote(run_id)
                    if not run_data:
                        return {"status": "error", "error": "Run not found"}
                    status = run_data.get("status")
                    if status in ("completed", "failed", "cancelled"):
                        break
                    if _time.monotonic() - t0 > timeout:
                        await engine.cancel_run.remote(run_id)
                        return {"status": "error", "error": f"Pipeline timed out after {timeout}s"}
                    await asyncio.sleep(2)

                # Collect outputs from step states
                steps_out = {}
                for sid, ss in (run_data.get("step_states") or {}).items():
                    steps_out[sid] = {
                        "status": ss.get("status"),
                        "outputs": ss.get("outputs"),
                        "duration_ms": ss.get("duration_ms"),
                        "error": ss.get("error"),
                    }

                return {
                    "status": run_data.get("status"),
                    "run_id": run_id,
                    "steps": steps_out,
                    "artifacts": run_data.get("artifacts"),
                }
            except Exception as e:
                logger.exception("YAML pipeline %s failed", pipeline_id)
                return {"status": "error", "error": str(e)}

        return {"status": "error", "error": f"Unknown pipeline: {pipeline_id}"}

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
        model = payload.get("model") or payload.get("model_type")
        quant = payload.get("quant", None)
        result = await self.invoke(service, payload, model, quant)
        return JSONResponse(result)


forge = Forge.bind()
