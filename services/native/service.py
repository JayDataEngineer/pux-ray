"""Native diffusers ForgeService — replaces Wan2GP handler layer.

Direct diffusers pipeline calls with adaptive VRAM optimization.
This is the main entry point for all image/video generation through native diffusers.

Supported models: Z-Image, Anima, FLUX, Wan, LTX, Qwen-Image
Supported formats: BF16, FP8, GGUF (via adaptive selection)
VRAM optimization: resident, model_cpu_offload, group_offload (adaptive)
LoRA: PEFT-based (load before group_offload)
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
from services.native.models import ModelConfig, get_model_config, MODELS
from services.native.vram import (
    Format, OffloadStrategy, VRAMPlan, plan_vram,
    apply_vram_plan, get_available_vram_mb, release_vram,
)
from services.native.lora import LoRAManager

logger = logging.getLogger(__name__)

# Prevent mmap-induced OOM during loading
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")


class NativeDiffusersService(ForgeService):
    """ForgeService that calls diffusers pipelines directly.

    Replaces the entire Wan2GP handler layer (261K lines) with direct
    from_pretrained() calls + adaptive VRAM optimization.
    """

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0  # Self-managed — reports actual usage after load

    def __init__(self):
        super().__init__()
        self.pipe = None
        self.config: Optional[ModelConfig] = None
        self.plan: Optional[VRAMPlan] = None
        self.lora_manager: Optional[LoRAManager] = None
        self._quant: Optional[str] = None

    # ── ForgeService interface ─────────────────────────────────────────────────

    def load(self, model_name: str, quant: str | None = None) -> None:
        """Load a model into VRAM with adaptive optimization."""
        model_name = model_name or self.default_model
        self._quant = quant

        # Look up model config
        config = get_model_config(model_name)
        if config is None:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

        self.config = config
        logger.info("Native: loading '%s' (%s)", model_name, config.pipeline_class)

        # Resolve model path (local first, then HuggingFace)
        model_path = self._resolve_path(config)

        # Load pipeline
        self.pipe = self._load_pipeline(config, model_path)

        # Initialize LoRA manager
        self.lora_manager = LoRAManager(self.pipe)

        # Apply VRAM optimization
        self._apply_optimization()

        self._loaded = True
        self.model_name = model_name

        # Report actual VRAM usage
        actual = self.actual_vram_mb()
        logger.info("Native: loaded '%s' (%dMB VRAM, strategy=%s)",
                    model_name, actual, self.plan.strategy.value if self.plan else "?")

    def unload(self) -> None:
        """Release model from VRAM."""
        if self.lora_manager:
            self.lora_manager.unload()
            self.lora_manager = None

        self.pipe = None
        self.config = None
        self.plan = None
        self._loaded = False
        self.model_name = None

        release_vram()
        logger.info("Native: unloaded")

    def infer(self, payload: dict) -> dict:
        """Run inference on the loaded model."""
        if not self._loaded or self.pipe is None:
            return {"status": "error", "error": "Model not loaded"}

        try:
            return self._generate(payload)
        except torch.cuda.OutOfMemoryError as e:
            release_vram()
            return {"status": "error", "error": f"CUDA OOM: {e}"}
        except Exception as e:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(e)}

    # ── Pipeline loading ───────────────────────────────────────────────────────

    def _resolve_path(self, config: ModelConfig) -> str:
        """Resolve model path: try local first, fall back to HuggingFace."""
        if os.path.exists(config.repo):
            return config.repo
        if config.repo_diffusers:
            logger.info("Native: local path '%s' not found, using HF: %s",
                        config.repo, config.repo_diffusers)
            return config.repo_diffusers
        return config.repo

    def _load_pipeline(self, config: ModelConfig, model_path: str):
        """Load the diffusers pipeline based on config."""
        # Import the pipeline class dynamically
        pipeline_cls = self._get_pipeline_class(config.pipeline_class)

        # Common loading kwargs
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,  # Prevent double-allocation RAM spike
        }

        # ModularPipeline needs trust_remote_code
        if config.pipeline_class == "ModularPipeline":
            load_kwargs["trust_remote_code"] = True

        logger.info("Native: loading %s from %s", config.pipeline_class, model_path)
        pipe = pipeline_cls.from_pretrained(model_path, **load_kwargs)
        return pipe

    def _get_pipeline_class(self, class_name: str):
        """Dynamically import the pipeline class from diffusers."""
        import diffusers

        cls = getattr(diffusers, class_name, None)
        if cls is None:
            raise ValueError(
                f"Pipeline class '{class_name}' not found in diffusers {diffusers.__version__}. "
                f"Available: {[x for x in dir(diffusers) if 'Pipeline' in x]}"
            )
        return cls

    # ── VRAM optimization ──────────────────────────────────────────────────────

    def _apply_optimization(self) -> None:
        """Decide and apply the VRAM optimization plan."""
        available = get_available_vram_mb()

        # Estimate model sizes (rough estimates based on param counts)
        model_size_mb = self._estimate_model_size()
        te_size_mb = self._estimate_text_encoder_size()

        logger.info("Native: VRAM planning (available=%dMB, model~%dMB, encoder~%dMB)",
                    available, model_size_mb, te_size_mb)

        self.plan = plan_vram(
            available_mb=available,
            model_bf16_size_mb=model_size_mb,
            text_encoder_bf16_size_mb=te_size_mb,
        )
        logger.info("Native: plan = %s", self.plan.notes)

        apply_vram_plan(self.pipe, self.plan, self.config)

    def _estimate_model_size(self) -> int:
        """Estimate transformer size in MB (BF16)."""
        if self.config is None or self.pipe is None:
            return 12000  # Default for 12B model

        if hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            params = sum(p.numel() for p in self.pipe.transformer.parameters())
            return int(params * 2 / (1024 * 1024))  # BF16 = 2 bytes/param

        return 12000

    def _estimate_text_encoder_size(self) -> int:
        """Estimate text encoder size in MB (BF16)."""
        if self.pipe is None:
            return 9500  # T5-XXL default

        total = 0
        for attr in ("text_encoder", "text_encoder_2"):
            if hasattr(self.pipe, attr):
                te = getattr(self.pipe, attr)
                if te is not None:
                    params = sum(p.numel() for p in te.parameters())
                    total += int(params * 2 / (1024 * 1024))
        return total

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self, payload: dict) -> dict:
        """Extract params from payload and run generation."""
        prompt = payload.get("prompt") or payload.get("input_prompt") or ""
        if not prompt:
            return {"status": "error", "error": "No prompt provided"}

        # Extract generation parameters
        config = self.config
        steps = payload.get("steps") or payload.get("sampling_steps") or config.default_steps
        guidance = payload.get("guidance") or payload.get("guide_scale") or config.default_guidance
        width, height = self._parse_size(payload, config)
        seed = payload.get("seed", -1)

        # Handle LoRAs in payload
        self._handle_payload_loras(payload)

        # Create generator for reproducibility
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        # Build call kwargs
        call_kwargs = {
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
        }

        # Task-specific params
        if config.task in ("text2video", "image2video"):
            call_kwargs["width"] = width
            call_kwargs["height"] = height
            num_frames = payload.get("num_frames", 121)
            call_kwargs["num_frames"] = int(num_frames)
        else:
            call_kwargs["width"] = width
            call_kwargs["height"] = height

        # Negative prompt if provided
        neg_prompt = payload.get("n_prompt") or payload.get("negative_prompt")
        if neg_prompt:
            call_kwargs["negative_prompt"] = neg_prompt

        # Image input for image2video / image_edit / img2img
        image_b64 = payload.get("image_b64")
        if image_b64 and config.task in ("image2video", "image_edit", "img2img"):
            from PIL import Image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            if config.task == "image2video":
                call_kwargs["image"] = image
            elif config.task == "image_edit":
                # For editing: use strength parameter if provided
                strength = payload.get("strength", 0.8)
                call_kwargs["image"] = image
                call_kwargs["strength"] = float(strength)
            else:
                # img2img
                call_kwargs["image"] = image

        # Additional reference images for multi-image inputs
        ref_b64 = payload.get("reference_image") or payload.get("reference_image_b64")
        if ref_b64:
            from PIL import Image
            ref = Image.open(io.BytesIO(base64.b64decode(ref_b64)))
            call_kwargs["reference_image"] = ref

        # Run generation
        t0 = time.perf_counter()
        output = self.pipe(**call_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Format output
        result = self._format_output(output, config, elapsed)
        return result

    def _parse_size(self, payload: dict, config: ModelConfig) -> tuple[int, int]:
        """Parse output size from payload or use default."""
        width = payload.get("width", config.default_size[0])
        height = payload.get("height", config.default_size[1])
        return int(width), int(height)

    def _handle_payload_loras(self, payload: dict) -> None:
        """Load and set LoRAs from payload if provided."""
        if not self.lora_manager:
            return

        loras = payload.get("loras") or payload.get("lora_slists")
        if not loras:
            return

        # loras can be a single path, a list of paths, or a list of {path, scale, name}
        if isinstance(loras, str):
            loras = [{"path": loras, "scale": 1.0, "name": "default"}]
        elif isinstance(loras, list) and loras and isinstance(loras[0], str):
            loras = [{"path": p, "scale": 1.0, "name": f"lora_{i}"} for i, p in enumerate(loras)]

        adapter_names = []
        scales = []
        for lora in loras:
            name = lora.get("name", f"lora_{len(adapter_names)}")
            path = lora.get("path", lora.get("slist", ""))
            scale = lora.get("scale", 1.0)

            if name not in self.lora_manager.list_adapters():
                self.lora_manager.load(path, adapter_name=name)
            adapter_names.append(name)
            scales.append(scale)

        if adapter_names:
            self.lora_manager.set_active(adapter_names, scales)

    def _format_output(self, output, config: ModelConfig, elapsed: float) -> dict:
        """Format pipeline output into standard result dict."""
        if config.task in ("text2video", "image2video"):
            # Video output — frames
            frames = output.frames[0] if hasattr(output, "frames") else output[0]
            return {
                "status": "success",
                "output": {
                    "type": "video",
                    "frames": len(frames),
                    "format": "pil_frames",
                },
                "metrics": {
                    "latency_ms": int(elapsed * 1000),
                    "model": config.name,
                    "strategy": self.plan.strategy.value if self.plan else "?",
                    "vram_peak_mb": self.actual_vram_mb(),
                },
                # Store frames for the gateway to serialize
                "_video_frames": frames,
            }
        else:
            # Image output
            image = output.images[0] if hasattr(output, "images") else output[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            return {
                "status": "success",
                "output": {
                    "type": "image",
                    "content": base64.b64encode(image_bytes).decode(),
                    "format": "png",
                },
                "metrics": {
                    "latency_ms": int(elapsed * 1000),
                    "model": config.name,
                    "strategy": self.plan.strategy.value if self.plan else "?",
                    "vram_peak_mb": self.actual_vram_mb(),
                },
            }
