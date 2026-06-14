"""Native model service — replaces Wan2GP entirely.

Loads any diffusers pipeline with adaptive VRAM optimization.
No mmGP, no handler translation layer, no Wan2GP code.
"""
from __future__ import annotations

import base64
import gc
import io
import logging
import os
import time
from typing import Any, Optional

import torch

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.native.registry import get_model, ModelEntry, ALL_MODELS
from services.native import loader as vram
from services.native.lora import LoRAManager

logger = logging.getLogger(__name__)
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")


class NativeService(ForgeService):
    """Serves any model through native diffusers pipelines."""

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self.pipe = None
        self.entry: Optional[ModelEntry] = None
        self.plan = None
        self.lora_mgr: Optional[LoRAManager] = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ALL_MODELS.keys())}")

        self.entry = entry
        logger.info("Native: loading '%s' (%s)", model_name, entry.pipeline)

        # Resolve path — try local first
        model_path = self._resolve_path(entry)

        # Load pipeline via diffusers
        pipe_cls = self._get_pipeline_class(entry.pipeline)
        load_kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if entry.pipeline == "ModularPipeline":
            load_kwargs["trust_remote_code"] = True

        self.pipe = pipe_cls.from_pretrained(model_path, **load_kwargs)

        # Init LoRA manager
        self.lora_mgr = LoRAManager(self.pipe)

        # Apply adaptive VRAM optimization
        self.plan = vram.plan(self.pipe)
        vram.apply(self.pipe, self.plan)

        self._loaded = True
        self.model_name = model_name
        logger.info("Native: '%s' loaded (%s)", model_name, self.plan.notes)

    def unload(self) -> None:
        if self.lora_mgr:
            self.lora_mgr.unload()
        self.pipe = None
        self.entry = None
        self.plan = None
        self._loaded = False
        self.model_name = None
        vram.release()
        logger.info("Native: unloaded")

    def infer(self, payload: dict) -> dict:
        if not self._loaded or self.pipe is None:
            return {"status": "error", "error": "Not loaded"}
        try:
            return self._generate(payload)
        except torch.cuda.OutOfMemoryError as e:
            vram.release()
            return {"status": "error", "error": f"CUDA OOM: {e}"}
        except Exception as e:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(e)}

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self, payload: dict) -> dict:
        e = self.entry
        prompt = payload.get("prompt") or payload.get("input_prompt") or ""
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        steps = payload.get("steps") or payload.get("sampling_steps") or e.steps
        guidance = payload.get("guidance") or payload.get("guide_scale") or e.guidance
        width = int(payload.get("width", e.width))
        height = int(payload.get("height", e.height))
        seed = payload.get("seed", -1)

        # LoRAs
        self._handle_loras(payload)

        # Generator
        gen = None
        if seed is not None and seed >= 0:
            gen = torch.Generator(device="cuda").manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": gen,
            "width": width,
            "height": height,
        }

        # Video params
        if e.task in ("text2video", "image2video"):
            kwargs["num_frames"] = int(payload.get("num_frames", 121))

        # Negative prompt
        neg = payload.get("n_prompt") or payload.get("negative_prompt")
        if neg:
            kwargs["negative_prompt"] = neg

        # Image input
        img_b64 = payload.get("image_b64")
        if img_b64 and e.task in ("image2video", "edit"):
            from PIL import Image
            kwargs["image"] = Image.open(io.BytesIO(base64.b64decode(img_b64)))

        # Generate
        t0 = time.perf_counter()
        output = self.pipe(**kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return self._format_output(output, elapsed)

    def _handle_loras(self, payload: dict) -> None:
        if not self.lora_mgr:
            return
        loras = payload.get("loras")
        if not loras:
            return
        if isinstance(loras, str):
            loras = [{"path": loras, "scale": 1.0, "name": "default"}]
        names, scales = [], []
        for l in loras:
            n = l.get("name", f"lora_{len(names)}")
            if n not in self.lora_mgr.list():
                self.lora_mgr.load(l.get("path", ""), n)
            names.append(n)
            scales.append(l.get("scale", 1.0))
        if names:
            self.lora_mgr.set_active(names, scales)

    def _format_output(self, output, elapsed: float) -> dict:
        from services.native.registry import is_video
        if is_video(self.model_name):
            frames = output.frames[0] if hasattr(output, "frames") else output[0]
            return {
                "status": "success",
                "output": {"type": "video", "frames": len(frames)},
                "metrics": {
                    "latency_ms": int(elapsed * 1000),
                    "model": self.model_name,
                    "strategy": self.plan.strategy.value if self.plan else "?",
                    "vram_peak_mb": int(torch.cuda.max_memory_allocated(0) / (1024 * 1024)),
                },
                "_video_frames": frames,
            }
        image = output.images[0] if hasattr(output, "images") else output[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "status": "success",
            "output": {"type": "image", "content": base64.b64encode(buf.getvalue()).decode()},
            "metrics": {
                "latency_ms": int(elapsed * 1000),
                "model": self.model_name,
                "strategy": self.plan.strategy.value if self.plan else "?",
                "vram_peak_mb": int(torch.cuda.max_memory_allocated(0) / (1024 * 1024)),
            },
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resolve_path(self, entry: ModelEntry) -> str:
        local = f"/models/native/{entry.name}"
        if os.path.exists(local):
            return local
        # Also check /models/flux-schnell etc (legacy paths)
        legacy = f"/models/{entry.name}"
        if os.path.exists(legacy):
            return legacy
        return entry.repo

    def _get_pipeline_class(self, name: str):
        import diffusers
        cls = getattr(diffusers, name, None)
        if cls is None:
            raise ValueError(f"Pipeline '{name}' not in diffusers. Available: "
                           f"{[x for x in dir(diffusers) if 'Pipeline' in x]}")
        return cls
