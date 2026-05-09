"""GPU Scheduler — coordinates load/unload across all GPU deployments.

Ray manages GPU resource allocation (num_gpus=1.0 on every GPU deployment
ensures serialization). This actor just sequences the load/unload calls.

Every GPU service (Docker and non-Docker) exposes:
  - load_model(model_name) — starts the service, claims GPU
  - unload_model() — stops the service, releases GPU
  - is_loaded() — health check

For Docker services (ComfyUI, TRELLIS, AniGen, etc.), these methods
manage the Docker container lifecycle via HTTPToolMixin.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import ray
from ray import serve

logger = logging.getLogger(__name__)

# All GPU services that require exclusive GPU access.
# Service name must match the Ray Serve deployment name.
GPU_SERVICES = {
    "llm":            {"model": "qwen3.6-27b-ud-q4_k_xl", "desc": "LLM (llama.cpp)"},
    "comfyui":        {"model": "comfyui",     "desc": "ComfyUI (image gen)"},
    "trellis":        {"model": "trellis",     "desc": "TRELLIS.2 (image-to-3D)"},
    "anigen":         {"model": "anigen",      "desc": "AniGen (rigged 3D)"},
    "vibevoice_community_tts": {"model": "vibevoice",   "desc": "VibeVoice Community (7B TTS)"},
    "see_through":    {"model": "see-through", "desc": "See-Through (layer decomp)"},
    "ace_step":       {"model": "ace-step",    "desc": "ACE-STEP (music gen)"},
    "hy_motion":      {"model": "hy-motion-1.0","desc": "HY-Motion (3D motion gen)"},
    "gpt_sovits":     {"model": "gpt-sovits",  "desc": "GPT-SoVITS (voice clone)"},
    "index_tts":      {"model": "index-tts",   "desc": "IndexTTS (GPU TTS)"},
    "qwen_tts":       {"model": "qwen-tts",    "desc": "Qwen3-TTS (GPU TTS)"},
    "vibevoice_microsoft":  {"model": "vibevoice-asr","desc": "VibeVoice Microsoft ASR"},
    "qwen_asr":       {"model": "qwen-asr",    "desc": "Qwen ASR (GPU)"},
    "phi4mm":         {"model": "phi4-multimodal","desc": "Phi-4 Multimodal (omni)"},
    "moss_soundeffect": {"model": "moss-soundeffect","desc": "MOSS SoundEffect (text→sound)"},
    "tangoflux":      {"model": "tangoflux",   "desc": "TangoFlux (text→audio)"},
}


@ray.remote
class GPUScheduler:
    """Sequences GPU model swaps. Coordinates load/unload calls.

    Ray's built-in scheduling (num_gpus=1.0) prevents concurrent GPU use.
    This actor just ensures proper unload-before-load ordering so that
    VRAM is freed before the next service starts.

    Usage:
        scheduler = ray.get_actor("gpu_scheduler")
        await scheduler.acquire_gpu.remote("trellis")
    """

    def __init__(self):
        self.current_service: Optional[str] = None
        self.current_model: Optional[str] = None
        self._lock = asyncio.Lock()

    async def acquire_gpu(self, service_name: str, model_name: Optional[str] = None) -> bool:
        """Acquire GPU for a service. Unloads current if different."""
        if service_name not in GPU_SERVICES:
            raise ValueError(f"Unknown GPU service: {service_name}")

        model = model_name or GPU_SERVICES[service_name]["model"]

        async with self._lock:
            if self.current_service == service_name and self.current_model == model:
                handle = self._get_handle(service_name)
                if handle:
                    try:
                        healthy = await handle.is_loaded.remote()
                        if healthy:
                            return True
                    except Exception:
                        pass

            if self.current_service:
                await self._unload_service(self.current_service)

            await self._load_service(service_name, model)

            self.current_service = service_name
            self.current_model = model
            return True

    async def release_gpu(self) -> None:
        """Unload current service, free GPU."""
        async with self._lock:
            if self.current_service:
                await self._unload_service(self.current_service)
                self.current_service = None
                self.current_model = None

    async def status(self) -> dict:
        return {
            "current_service": self.current_service,
            "current_model": self.current_model,
            "all_services": list(GPU_SERVICES.keys()),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _get_handle(service_name: str):
        try:
            return serve.get_deployment_handle(service_name, service_name)
        except Exception:
            return None

    @staticmethod
    async def _load_service(service_name: str, model_name: str) -> None:
        handle = GPUScheduler._get_handle(service_name)
        if not handle:
            raise ValueError(f"No deployment handle for: {service_name}")
        logger.info("Loading %s/%s on GPU", service_name, model_name)
        await handle.load_model.remote(model_name)
        logger.info("GPU running: %s/%s", service_name, model_name)

    @staticmethod
    async def _unload_service(service_name: str) -> None:
        handle = GPUScheduler._get_handle(service_name)
        if not handle:
            return
        logger.info("Unloading %s from GPU", service_name)
        try:
            await handle.unload_model.remote()
        except Exception as e:
            logger.warning("Error unloading %s: %s", service_name, e)
