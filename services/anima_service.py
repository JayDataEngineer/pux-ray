"""Anima Image Generation Service - Direct integration for Forge.

Bypasses Wan2GP handler recursion by using direct API calls.
Integrates working anima generation into Forge service architecture.
"""
import base64
import io
import logging
import os
import sys
import torch
from PIL import Image
from typing import Dict, Any, Optional

# Ensure Wan2GP is in path
sys.path.insert(0, "/opt/wan2gp")
os.environ['WAN2GP_ROOT'] = '/opt/wan2gp'

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
            # Import and create factory
            from models.anima.anima_main import model_factory
            
            self._factory = model_factory(
                checkpoint_dir="/opt/wan2gp/ckpts",
                model_filename=["anima-base-v1.0.safetensors"],
                text_encoder_filename="qwen_3_06b_base.safetensors"
            )
            
            # Extract parameters from payload
            prompt = payload.get("prompt") or payload.get("input_prompt") or payload.get("text", "")
            width = payload.get("width", 512)
            height = payload.get("height", 512)
            steps = payload.get("steps", payload.get("sampling_steps", payload.get("num_inference_steps", 4)))
            seed = payload.get("seed", 42)
            
            logger.info(f"AnimaService generating: {prompt[:50]}...")
            
            # Generate image
            image_tensor = self._factory.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
            
            # Convert tensor to PIL Image
            image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            if image_np.min() < 0:  # Range [-1, 1]
                image_np = ((image_np + 1) / 2 * 255).astype('uint8')
            else:  # Range [0, 1]
                image_np = (image_np * 255).astype('uint8')
            
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

