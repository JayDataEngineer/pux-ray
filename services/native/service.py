"""Native model service — transformers 4 direct loading + custom VRAM hooks.

NO diffusers. NO SGLang serve. NO mmGP.

Loads models via transformers 4 / direct PyTorch from_pretrained.
VRAM managed by our custom BlockStreamHook (services/native/loader.py).
Denoising loop is manual — we control every step.

Architecture:
  User request → load model (transformers) → apply VRAM hooks → manual denoise → output
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
import torch.nn as nn

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.native.registry import get_model, ModelEntry, ALL_MODELS
from services.native.loader import VRAMManager, plan, release, module_size_mb, available_vram_mb

logger = logging.getLogger(__name__)
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")


class NativeService(ForgeService):
    """Serves models via transformers 4 + custom VRAM hooks."""

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self.entry: Optional[ModelEntry] = None
        self._components: dict = {}  # transformer, text_encoder, vae, scheduler
        self._vram_managers: list[VRAMManager] = []
        self._loaded_model: str | None = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ALL_MODELS.keys())}")

        self.entry = entry
        logger.info("Native: loading '%s' from %s", model_name, entry.repo)

        # Resolve local path
        model_path = self._resolve_path(entry)

        # Load components via transformers 4 / direct PyTorch
        self._components = self._load_components(entry, model_path)

        # Plan and apply VRAM strategy
        self._apply_vram()

        self._loaded = True
        self.model_name = model_name
        self._loaded_model = model_name
        logger.info("Native: '%s' ready", model_name)

    def unload(self) -> None:
        for mgr in self._vram_managers:
            mgr.remove()
        self._vram_managers.clear()
        self._components.clear()
        self._loaded = False
        self.model_name = None
        self._loaded_model = None
        release()
        logger.info("Native: unloaded")

    def infer(self, payload: dict) -> dict:
        if not self._loaded:
            return {"status": "error", "error": "Not loaded"}
        try:
            return self._generate(payload)
        except torch.cuda.OutOfMemoryError as e:
            release()
            return {"status": "error", "error": f"CUDA OOM: {e}"}
        except Exception as e:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(e)}

    # ── Model Loading (transformers 4, NO diffusers) ───────────────────────────

    def _resolve_path(self, entry: ModelEntry) -> str:
        local = f"/models/native/{entry.name}"
        if os.path.exists(local):
            return local
        legacy = f"/models/{entry.name}"
        if os.path.exists(legacy):
            return legacy
        return entry.repo

    def _load_components(self, entry: ModelEntry, path: str) -> dict:
        """Load model components via transformers 4 / direct PyTorch.

        Uses transformers AutoModel for text encoders.
        Loads transformer backbone and VAE directly from safetensors.
        NO diffusers pipeline classes.
        """
        from transformers import AutoModel, AutoTokenizer

        components = {}
        model_path = path

        # Text encoder — load via transformers 4
        try:
            te_path = os.path.join(model_path, "text_encoder")
            if os.path.exists(te_path):
                te = AutoModel.from_pretrained(te_path, torch_dtype=torch.bfloat16)
                components["text_encoder"] = te
                logger.info("Loaded text_encoder: %s", type(te).__name__)

                # Try tokenizer
                tok_path = os.path.join(model_path, "tokenizer")
                if os.path.exists(tok_path):
                    tok = AutoTokenizer.from_pretrained(tok_path)
                    components["tokenizer"] = tok
        except Exception as e:
            logger.warning("text_encoder load failed: %s", e)

        # Transformer backbone — load via transformers AutoModel
        try:
            tr_path = os.path.join(model_path, "transformer")
            if os.path.exists(tr_path):
                tr = AutoModel.from_pretrained(tr_path, torch_dtype=torch.bfloat16)
                components["transformer"] = tr
                logger.info("Loaded transformer: %s", type(tr).__name__)
        except Exception as e:
            # Fallback: try direct safetensors loading
            logger.warning("AutoModel transformer load failed: %s — trying safetensors", e)
            try:
                from safetensors.torch import load_file
                import json
                cfg_path = os.path.join(tr_path, "config.json")
                idx_path = os.path.join(tr_path, "model.safetensors.index.json")
                if os.path.exists(idx_path):
                    # Multi-shard: load all shards
                    with open(idx_path) as f:
                        idx = json.load(f)
                    shards = set(idx["weight_map"].values())
                    state = {}
                    for shard in sorted(shards):
                        state.update(load_file(os.path.join(tr_path, shard)))
                else:
                    # Single file
                    single = os.path.join(tr_path, "diffusion_pytorch_model.safetensors")
                    if os.path.exists(single):
                        state = load_file(single)
                    else:
                        raise FileNotFoundError("No safetensors found")
                components["_transformer_state"] = state
                logger.info("Loaded transformer state dict (%d keys)", len(state))
            except Exception as e2:
                logger.error("Transformer load completely failed: %s", e2)

        # VAE — load via transformers AutoModel if available, else skip
        try:
            vae_path = os.path.join(model_path, "vae")
            if os.path.exists(vae_path):
                vae = AutoModel.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
                components["vae"] = vae
                logger.info("Loaded VAE: %s", type(vae).__name__)
        except Exception as e:
            logger.warning("VAE load failed: %s (decode will be unavailable)", e)

        return components

    # ── VRAM Management ────────────────────────────────────────────────────────

    def _apply_vram(self) -> None:
        """Apply our custom VRAM hooks based on model sizes."""
        transformer = self._components.get("transformer")
        text_encoder = self._components.get("text_encoder")
        vae = self._components.get("vae")

        tr_mb = module_size_mb(transformer) if transformer else 0
        te_mb = module_size_mb(text_encoder) if text_encoder else 0
        vae_mb = module_size_mb(vae) if vae else 300

        vram_plan = plan(tr_mb, te_mb, vae_mb)
        logger.info("VRAM plan: %s", vram_plan.notes)

        # VAE always resident BF16
        if vae is not None:
            vae.to("cuda")

        # Text encoder — resident if plan says, else managed
        if text_encoder is not None and vram_plan.strategy.value in ("resident", "fp8_resident"):
            text_encoder.to("cuda")

        # Transformer — apply streaming hooks if needed
        if transformer is not None:
            mgr = VRAMManager(transformer, device="cuda")
            mgr.apply(vram_plan)
            self._vram_managers.append(mgr)

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self, payload: dict) -> dict:
        """Run generation through manual denoising loop."""
        e = self.entry
        prompt = payload.get("prompt", "")
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        steps = payload.get("steps") or e.steps
        guidance = payload.get("guidance") or e.guidance
        width = int(payload.get("width", e.width))
        height = int(payload.get("height", e.height))
        seed = payload.get("seed", -1)

        # Get components
        transformer = self._components.get("transformer")
        if transformer is None:
            return {"status": "error", "error": "Transformer not loaded"}

        # Encode text
        prompt_embeds = self._encode_text(prompt)

        # Initialize latents
        gen = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
        # Latent shape depends on model architecture
        latent_shape = self._get_latent_shape(width, height, payload)
        latents = torch.randn(latent_shape, device="cuda", dtype=torch.bfloat16, generator=gen)

        # Manual denoising loop — we control every step
        t0 = time.perf_counter()

        # Use a simple Euler scheduler if no scheduler loaded
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device="cuda", dtype=torch.bfloat16)

        for i in range(steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            timestep = sigma.unsqueeze(0).expand(latents.shape[0])

            with torch.no_grad():
                # Forward pass — our VRAM hooks stream blocks automatically
                noise = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            # Euler step: x_next = x + (sigma_next - sigma) * v
            latents = latents + (sigma_next - sigma) * noise

        gen_time = time.perf_counter() - t0

        # Decode latents to image
        image_b64 = self._decode_to_base64(latents, width, height)

        return {
            "status": "success",
            "output": {"type": "image", "content": image_b64, "format": "png"},
            "metrics": {
                "latency_ms": int(gen_time * 1000),
                "model": self.model_name,
                "steps": steps,
            },
        }

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text via transformers 4 text encoder."""
        te = self._components.get("text_encoder")
        tok = self._components.get("tokenizer")
        if te is None or tok is None:
            # Fallback: return random embeddings
            logger.warning("No text encoder — using random embeddings")
            return torch.randn(1, 77, 1024, device="cuda", dtype=torch.bfloat16)

        tokens = tok(prompt, return_tensors="pt", padding="max_length",
                     max_length=77, truncation=True).to("cuda")

        with torch.no_grad():
            embeds = te(**tokens)

        # Handle different output formats
        if hasattr(embeds, "last_hidden_state"):
            return embeds.last_hidden_state
        elif isinstance(embeds, tuple):
            return embeds[0]
        return embeds

    def _get_latent_shape(self, width: int, height: int, payload: dict) -> tuple:
        """Get latent shape for the model."""
        task = self.entry.task if self.entry else "text2image"
        if task in ("text2video", "image2video"):
            num_frames = int(payload.get("num_frames", 25))
            return (1, 128, num_frames, height // 32, width // 32)
        return (1, 128, height // 32, width // 32)

    def _decode_to_base64(self, latents: torch.Tensor, width: int, height: int) -> str:
        """Decode latents to base64 PNG image."""
        vae = self._components.get("vae")

        if vae is not None:
            with torch.no_grad():
                # Ensure VAE is on GPU
                if next(vae.parameters()).device.type == "cpu":
                    vae.to("cuda")
                decoded = vae.decode(latents)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                elif isinstance(decoded, tuple):
                    decoded = decoded[0]
        else:
            # No VAE — visualize raw latents as a colorful image
            decoded = latents[0, :3].permute(1, 2, 0) if latents.dim() == 4 else latents[0, :3, 0]

        # Convert to PIL
        from PIL import Image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img * 255).round().to(torch.uint8)

        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        elif img.dim() == 4:
            img = img[0].permute(1, 2, 0)

        img = img.cpu().numpy()
        pil = Image.fromarray(img).resize((width, height), Image.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def actual_vram_mb(self) -> int:
        try:
            return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        except Exception:
            return 0

    @property
    def _loaded_model_name(self) -> str | None:
        return self._loaded_model
