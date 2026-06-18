"""Anima Image Generation Service - Direct integration for Forge.

NOTE: Wan2GP no longer used by Anima. Uses diffusers directly via the
inference pool system. The old wan2gp path references are removed.
"""
import base64
import io
import logging
import os
import sys
import torch
from PIL import Image
from typing import Dict, Any, Optional

# [wan2gp path removed]

logger = logging.getLogger(__name__)


class AnimaService:
    """Anima image generation service - direct integration."""

    def __init__(self):
        self._factory = None
        self._loaded_model = None
        self._vram_mb = 128  # Estimated VRAM usage
        self._forge_core = None  # Set by Forge
        logger.info("AnimaService initialized")

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    def set_forge_core(self, forge_core):
        """Set reference to Forge (called by Forge)."""
        self._forge_core = forge_core

    def load(self, model_name: str = "anima_base", quant: Optional[str] = None) -> None:
        """Load anima model (lazy loading on first generation).

        Args:
            model_name: Must be "anima_base" or "anima/anima_base"
            quant: Ignored (no quantization support)
        """
        if model_name != "anima_base" and model_name != "anima/anima_base":
            raise ValueError(f"AnimaService only supports 'anima_base', got '{model_name}'")

        self._loaded_model = model_name
        logger.info(f"AnimaService: model={model_name} ready for generation")
    
    def unload(self) -> None:
        """Unload anima model."""
        self._factory = None
        self._loaded_model = None
        logger.info("AnimaService: model unloaded")
    
    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate anime image using direct API."""
        if not self._loaded_model:
            return {
                "status": "error",
                "error": "Model not loaded. Call load() first."
            }

        try:
            # Lazy-load model on first generation
            if self._factory is None:
                from models.anima.anima_main import model_factory
                from shared.utils import files_locator as fl
                fl.set_checkpoints_paths(['/mnt/data/models', '/tmp'])

                # Locate the checkpoint files
                transformer_path = fl.locate_file("anima-base-v1.0.safetensors")
                te_path = fl.locate_file("qwen_3_06b_base.safetensors")

                self._factory = model_factory(
                    model_filename=transformer_path,
                    base_model_type='anima_base',
                    text_encoder_filename=te_path,
                )
                logger.info("AnimaService: model loaded")

            # Extract parameters from payload
            prompt = payload.get("prompt") or payload.get("input_prompt") or payload.get("text", "")
            n_prompt = payload.get("n_prompt") or payload.get("negative_prompt")
            width = payload.get("width", 1024)
            height = payload.get("height", 1024)
            steps = payload.get("steps", payload.get("sampling_steps", payload.get("num_inference_steps", 30)))
            seed = payload.get("seed", 42)
            guidance_scale = payload.get("guidance_scale", payload.get("cfg", payload.get("guide_scale", 4.0)))

            logger.info(f"AnimaService generating ({width}x{height}, {steps} steps): {prompt[:50]}...")

            # Generate image
            image_tensor = self._factory.generate(
                seed=seed,
                input_prompt=prompt,
                n_prompt=n_prompt,
                width=width,
                height=height,
                sampling_steps=steps,
                guide_scale=guidance_scale,
            )

            # Convert tensor to PIL Image (result is [C, H, W] float in [-1, 1])
            img = image_tensor.float().clamp(-1, 1)
            img = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            image_np = img.permute(1, 2, 0).cpu().numpy()

            image = Image.fromarray(image_np, 'RGB')

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            logger.info(f"AnimaService generation complete: {len(image_base64)} bytes")

            return {
                "status": "success",
                "media_type": "image/png",
                "data": image_base64,
                "model": self._loaded_model,
                "prompt": prompt,
                "width": width,
                "height": height
            }

        except Exception as e:
            logger.error(f"AnimaService generation failed: {e}")
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# Service factory function for Forge compatibility
def get_anima_service() -> AnimaService:
    """Get AnimaService instance (creates new instance for Forge compatibility)."""
    return AnimaService()

