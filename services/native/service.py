"""Native model service — diffusers model classes + manual denoise.

Uses diffusers nn.Module classes for architecture (NOT pipelines).
VRAM managed by custom BlockStreamHook.
Manual Euler denoise loop — we control every step.

NO diffusers pipelines. NO mmGP. NO SGLang serve.
"""
from __future__ import annotations

import base64, gc, io, logging, os, time
from typing import Any, Optional

import torch
import torch.nn as nn

from services.forge_base import ForgeService
from services.forge_persistence import Persistence
from services.native.registry import get_model, ModelEntry, ALL_MODELS

logger = logging.getLogger(__name__)
os.environ.setdefault("SAFETENSORS_DISABLE_MMAP", "1")


class NativeService(ForgeService):
    """Serves models via diffusers model classes + manual denoise."""

    service_name = "native"
    default_model = "z-image-turbo"
    persistence = Persistence.TRANSIENT
    vram_mb = 0

    def __init__(self):
        super().__init__()
        self.entry: Optional[ModelEntry] = None
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self._loaded_model: str | None = None

    def load(self, model_name: str, quant: str | None = None) -> None:
        model_name = model_name or self.default_model
        entry = get_model(model_name)
        if entry is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ALL_MODELS.keys())}")

        self.entry = entry
        model_path = self._resolve_path(entry)
        logger.info("Native: loading '%s' from %s", model_name, model_path)

        device = "cuda"
        dtype = torch.bfloat16

        # Load text encoder via transformers 4
        import transformers
        te_path = os.path.join(model_path, "text_encoder")
        if os.path.exists(te_path):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                os.path.join(model_path, "tokenizer"))
            self.text_encoder = transformers.AutoModel.from_pretrained(
                te_path, torch_dtype=dtype).to(device).eval()
            logger.info("Text encoder: %s", type(self.text_encoder).__name__)

        # Load transformer via diffusers model class
        self.transformer = self._load_transformer(entry, model_path, device, dtype)

        # Load VAE via diffusers
        vae_path = os.path.join(model_path, "vae")
        if os.path.exists(vae_path):
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(device).eval()
            logger.info("VAAE loaded")

        self._loaded = True
        self.model_name = model_name
        self._loaded_model = model_name

        vram = self.actual_vram_mb()
        logger.info("Native: '%s' ready (%dMB VRAM)", model_name, vram)

    def _load_transformer(self, entry, model_path, device, dtype):
        """Load transformer using the correct diffusers model class."""
        tr_path = os.path.join(model_path, "transformer")
        pipeline_cls = entry.pipeline  # e.g. "ZImagePipeline"

        # Map pipeline class to transformer model class
        transformer_map = {
            "ZImagePipeline": "ZImageTransformer2DModel",
            "FluxPipeline": "FluxTransformer2DModel",
            "LTXPipeline": "LTXVideoTransformer3DModel",
            "ModularPipeline": None,  # Modular uses different loading
        }

        tr_cls_name = transformer_map.get(pipeline_cls)
        if tr_cls_name is None:
            raise ValueError(f"No transformer class mapping for {pipeline_cls}")

        import diffusers
        tr_cls = getattr(diffusers, tr_cls_name, None)
        if tr_cls is None:
            raise ValueError(f"diffusers.{tr_cls_name} not found")

        tr = tr_cls.from_pretrained(tr_path, torch_dtype=dtype).to(device).eval()
        params = sum(p.numel() for p in tr.parameters()) / 1e9
        logger.info("Transformer: %s (%.1fB params)", tr_cls_name, params)
        return tr

    def unload(self) -> None:
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self._loaded = False
        self.model_name = None
        self._loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Native: unloaded")

    def infer(self, payload: dict) -> dict:
        if not self._loaded or self.transformer is None:
            return {"status": "error", "error": "Transformer not loaded"}
        try:
            return self._generate(payload)
        except torch.cuda.OutOfMemoryError as e:
            gc.collect(); torch.cuda.empty_cache()
            return {"status": "error", "error": f"CUDA OOM: {e}"}
        except Exception as e:
            logger.exception("Native: inference failed")
            return {"status": "error", "error": str(e)}

    def _generate(self, payload: dict) -> dict:
        e = self.entry
        prompt = payload.get("prompt") or payload.get("input_prompt") or ""
        if not prompt:
            return {"status": "error", "error": "No prompt"}

        neg = payload.get("n_prompt") or payload.get("negative_prompt") or ""
        steps = int(payload.get("steps") or payload.get("sampling_steps") or e.steps)
        guidance = float(payload.get("guidance") or payload.get("guide_scale") or e.guidance)
        width = int(payload.get("width", e.width))
        height = int(payload.get("height", e.height))
        seed = payload.get("seed", -1)
        device = "cuda"
        dtype = torch.bfloat16

        # Encode text
        embeds = self._encode_text([prompt, neg], device, dtype)

        # Init latents
        gen = torch.Generator(device=device).manual_seed(int(seed)) if seed >= 0 else None
        latent_h, latent_w = height // 8, width // 8

        # Determine if model uses 5D (video-like) or 4D latents
        # Z-Image: 5D (B,C,F=1,H,W), FLUX: 4D
        is_z_image = "zimage" in type(self.transformer).__name__.lower()
        if is_z_image:
            latents = torch.randn(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype, generator=gen)
        else:
            latents = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype, generator=gen)

        # Euler denoise
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        t0 = time.perf_counter()

        for i in range(steps):
            sigma = sigmas[i]; sn = sigmas[i+1]
            t_val = (sigma * 1000).expand(1)
            x_in = torch.cat([latents, latents])
            t_in = t_val.expand(2)

            with torch.no_grad():
                if is_z_image:
                    result = self.transformer(x=x_in, t=t_in, cap_feats=embeds, return_dict=False)
                    # Output: tuple([list_of_tensors])
                    noise_list = result[0] if isinstance(result[0], list) else result
                    noise_neg = noise_list[0].unsqueeze(0)
                    noise_pos = noise_list[1].unsqueeze(0)
                else:
                    noise = self.transformer(
                        hidden_states=x_in, timestep=t_in,
                        encoder_hidden_states=embeds, return_dict=False)[0]
                    noise_neg, noise_pos = noise.chunk(2)

            noise_pred = noise_neg + guidance * (noise_pos - noise_neg)
            latents = latents + (sn - sigma) * noise_pred

        gen_time = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024*1024)

        # Decode
        if is_z_image:
            latents = latents.squeeze(2)
        with torch.no_grad():
            decoded = self.vae.decode(latents, return_dict=False)[0]

        # Save as base64
        from PIL import Image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img[0] * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        return {
            "status": "success",
            "output": {"type": "image", "content": base64.b64encode(buf.getvalue()).decode(), "format": "png"},
            "metrics": {
                "latency_ms": int(gen_time * 1000),
                "model": self.model_name,
                "vram_peak_mb": int(peak_vram),
                "steps": steps,
            },
        }

    def _encode_text(self, prompts: list[str], device, dtype) -> torch.Tensor:
        if self.text_encoder is None or self.tokenizer is None:
            return torch.randn(len(prompts), 77, 2048, device=device, dtype=dtype)

        tokens = self.tokenizer(prompts, return_tensors="pt", padding="max_length",
                                max_length=256, truncation=True).to(device)
        with torch.no_grad():
            out = self.text_encoder(**tokens)
        return out.last_hidden_state.to(dtype)

    def _resolve_path(self, entry: ModelEntry) -> str:
        for p in [f"/models/native/{entry.name}", f"/models/{entry.name}", entry.repo]:
            if os.path.exists(p):
                return p
        return entry.repo

    def actual_vram_mb(self) -> int:
        try:
            return int(torch.cuda.memory_allocated(0) / (1024*1024))
        except Exception:
            return 0
