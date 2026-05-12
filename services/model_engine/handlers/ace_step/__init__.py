"""ACE-Step handler — text-to-music generation.

Decomposes ACE-Step into nn.Module components for mmgp VRAM management.
Supports v1, v1.5 (7 variants), and v1.5 XL.

Reference: Wan2GP's models/TTS/ace_step_handler.py decomposition.
Pipe dicts:
    v1:    {"transformer": ACEStepTransformer2DModel, "text_encoder": UMT5EncoderModel, "codec": MusicDCAE}
    v1.5:  {"transformer": AceStepConditionGenerationModel, "text_encoder_2": Qwen3Model, "codec": AutoencoderOobleck}
    v1.5 XL: Same as v1.5 but with AceStepXLTransformer
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from services.model_engine.base_handler import BaseHandler, LoadResult, ModelVariant

logger = logging.getLogger(__name__)


# ── Transformer config variants for v1.5 ──────────────────────────────────────
V15_VARIANTS = {
    "base": "ace_step_v1_5_transformer_config_base.json",
    "sft": "ace_step_v1_5_transformer_config_sft.json",
    "turbo": "ace_step_v1_5_transformer_config_turbo.json",
    "turbo_shift1": "ace_step_v1_5_transformer_config_turbo_shift1.json",
    "turbo_shift3": "ace_step_v1_5_transformer_config_turbo_shift3.json",
    "turbo_continuous": "ace_step_v1_5_transformer_config_turbo_continuous.json",
    "xl_turbo": "ace_step_v1_5_xl_transformer_config_turbo.json",
}


# ── Model type → variant metadata ─────────────────────────────────────────────
VARIANTS = {
    # v1
    "ace_step_v1": ModelVariant(
        name="ace_step_v1",
        family="ace_step",
        display_name="ACE-Step v1",
        vram_estimate_gb=8,
        defaults={
            "steps": 60,
            "guidance_scale": 7.0,
            "sample_solver": "euler",
            "duration_seconds": 30,
        },
    ),
    # v1.5 variants
    "ace_step_v1_5": ModelVariant(
        name="ace_step_v1_5",
        family="ace_step",
        display_name="ACE-Step 1.5 (base)",
        vram_estimate_gb=7,
        defaults={
            "steps": 8,
            "guidance_scale": 1.0,
            "alt_guidance_scale": 2.5,
            "duration_seconds": 30,
            "temperature": 0.85,
            "top_p": 0.9,
        },
    ),
    "ace_step_v1_5_turbo": ModelVariant(
        name="ace_step_v1_5_turbo",
        family="ace_step",
        display_name="ACE-Step 1.5 Turbo",
        vram_estimate_gb=7,
        defaults={
            "steps": 8,
            "guidance_scale": 1.0,
            "alt_guidance_scale": 2.5,
            "duration_seconds": 30,
            "temperature": 0.85,
            "top_p": 0.9,
        },
    ),
    "ace_step_v1_5_sft": ModelVariant(
        name="ace_step_v1_5_sft",
        family="ace_step",
        display_name="ACE-Step 1.5 SFT",
        vram_estimate_gb=7,
        defaults={
            "steps": 8,
            "guidance_scale": 1.0,
            "alt_guidance_scale": 2.5,
            "duration_seconds": 30,
            "temperature": 0.85,
            "top_p": 0.9,
        },
    ),
    # v1.5 XL
    "ace_step_v1_5_xl_turbo": ModelVariant(
        name="ace_step_v1_5_xl_turbo",
        family="ace_step",
        display_name="ACE-Step 1.5 XL Turbo",
        vram_estimate_gb=12,
        defaults={
            "steps": 8,
            "guidance_scale": 1.0,
            "alt_guidance_scale": 2.5,
            "duration_seconds": 30,
            "temperature": 0.85,
            "top_p": 0.9,
        },
    ),
}


# ── Model classes (internalized from Wan2GP reference) ────────────────────────
# Wan2GP ships standalone copies of the ACE-Step model code that don't depend
# on the acestep package. We internalize them into our handler package.

from .models.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
from .models.configuration_acestep_v15 import AceStepConfig

# VAE — available in diffusers >= 0.33
from diffusers import AutoencoderOobleck


# ── Handler ────────────────────────────────────────────────────────────────────

class AceStepHandler(BaseHandler):
    """ACE-Step music generation handler.

    Decomposes ACE-Step models into pipe dicts for mmgp VRAM management.
    Supports v1, v1.5 (base/sft/turbo/turbo_shift variants), and v1.5 XL.

    Pipe dict (v1.5):
        {
            "transformer": AceStepConditionGenerationModel,
            "text_encoder_2": Qwen3Model,
            "codec": AutoencoderOobleck,
            "text_encoder"?: Qwen3ForCausalLM  (optional, for lyrics LM)
        }
    """

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def get_variant(self, model_type: str) -> ModelVariant:
        if model_type not in VARIANTS:
            raise ValueError(
                f"Unknown ACE-Step type: {model_type}. "
                f"Available: {list(VARIANTS.keys())}"
            )
        return VARIANTS[model_type]

    def _is_v1(self, model_type: str) -> bool:
        return model_type == "ace_step_v1"

    def _is_v15_xl(self, model_type: str) -> bool:
        return model_type == "ace_step_v1_5_xl_turbo"

    def _is_v15(self, model_type: str) -> bool:
        return model_type.startswith("ace_step_v1_5")

    def _resolve_transformer_variant(self, model_type: str) -> str:
        """Map model_type to the v1.5 transformer config variant name."""
        if model_type == "ace_step_v1_5":
            return "base"
        if model_type == "ace_step_v1_5_xl_turbo":
            return "xl_turbo"
        # Extract variant: ace_step_v1_5_turbo → turbo
        parts = model_type.replace("ace_step_v1_5_", "")
        if parts in V15_VARIANTS:
            return parts
        return "turbo"  # default fallback

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        quantize_transformer: bool = False,
        **kwargs,
    ) -> LoadResult:
        if self._is_v1(model_type):
            return self._load_v1(model_path, dtype, **kwargs)
        else:
            return self._load_v15(model_type, model_path, dtype, **kwargs)

    def _load_v1(self, model_path: Path, dtype: torch.dtype, **kwargs) -> LoadResult:
        """Load ACE-Step v1.

        Pipe dict: {"transformer": ACEStepTransformer2DModel,
                     "text_encoder": UMT5EncoderModel,
                     "codec": MusicDCAE}
        """
        raise NotImplementedError("ACE-Step v1 loading not yet implemented")

    def _load_v15(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype,
        enable_lm: bool = True,
        **kwargs,
    ) -> LoadResult:
        """Load ACE-Step v1.5 or v1.5 XL.

        Resolves the transformer variant, loads all components,
        builds tokenizers and audio code vocabulary, then constructs
        the pipeline and pipe dict.
        """
        from mmgp import offload

        variant_name = self._resolve_transformer_variant(model_type)
        config_filename = V15_VARIANTS.get(variant_name, V15_VARIANTS["turbo"])

        logger.info("Loading ACE-Step v1.5 variant=%s config=%s", variant_name, config_filename)

        # ── Resolve paths ──────────────────────────────────────────
        # model_path is the ACE-Step root (e.g., models/audio/acestep/)
        # Sub-directories contain individual components
        transformer_dir = self._find_component_dir(
            model_path, ["acestep-v15-turbo", "acestep-v1.5-turbo", variant_name]
        )
        vae_dir = self._find_component_dir(model_path, ["vae"])
        te2_dir = self._find_component_dir(
            model_path, ["Qwen3-Embedding-0.6B", "qwen3-embedding"]
        )
        lm_dir = self._find_component_dir(
            model_path, ["acestep-5Hz-lm-1.7B", "acestep-lm"]
        )
        silence_path = self._find_silence_latent(model_path, transformer_dir)

        # ── Load model classes (internalized in handler package) ───
        TransformerModel = AceStepConditionGenerationModel
        VaeModel = AutoencoderOobleck

        from transformers import Qwen3Model, Qwen3ForCausalLM

        # ── Load transformer ────────────────────────────────────────
        config_path = transformer_dir / config_filename
        if not config_path.exists():
            # Try default config.json
            config_path = transformer_dir / "config.json"

        transformer_weights = self._find_weights(transformer_dir)
        logger.info("Loading transformer from %s (config=%s)", transformer_weights.name, config_path.name)

        transformer = offload.fast_load_transformers_model(
            str(transformer_weights),
            modelClass=TransformerModel,
            defaultConfigPath=str(config_path) if config_path.exists() else None,
            default_dtype=dtype,
        )
        transformer.eval()

        # ── Load audio VAE ──────────────────────────────────────────
        vae_weights = self._find_weights(vae_dir)
        vae_config_path = vae_dir / "config.json"
        logger.info("Loading VAE from %s", vae_weights.name)

        audio_vae = offload.fast_load_transformers_model(
            str(vae_weights),
            modelClass=VaeModel,
            defaultConfigPath=str(vae_config_path) if vae_config_path.exists() else None,
            default_dtype=dtype,
        )
        audio_vae.eval()

        # ── Load text encoder 2 (Qwen3 0.6B embedding) ────────────
        te2_weights = self._find_weights(te2_dir)
        te2_config_path = te2_dir / "config.json"
        logger.info("Loading text encoder 2 from %s", te2_weights.name)

        text_encoder_2 = offload.fast_load_transformers_model(
            str(te2_weights),
            modelClass=Qwen3Model,
            defaultConfigPath=str(te2_config_path) if te2_config_path.exists() else None,
            default_dtype=dtype,
        )
        text_encoder_2.eval()

        # ── Load LM (optional) ─────────────────────────────────────
        lm_model = None
        if enable_lm and lm_dir is not None and lm_dir.is_dir():
            lm_weights = self._find_weights(lm_dir)
            lm_config_path = lm_dir / "config.json"
            logger.info("Loading LM from %s", lm_weights.name)

            lm_model = offload.fast_load_transformers_model(
                str(lm_weights),
                modelClass=Qwen3ForCausalLM,
                defaultConfigPath=str(lm_config_path) if lm_config_path.exists() else None,
                default_dtype=dtype,
            )
            lm_model.eval()

        # ── Load silence latent ─────────────────────────────────────
        silence_latent = None
        if silence_path is not None and silence_path.exists():
            silence_latent = torch.load(str(silence_path), map_location="cpu")
            logger.info("Loaded silence latent: shape=%s", tuple(silence_latent.shape))

        # ── Load tokenizers ─────────────────────────────────────────
        from transformers import AutoTokenizer

        pre_text_tokenizer = AutoTokenizer.from_pretrained(
            str(te2_dir), padding_side="right",
        )
        lm_tokenizer = None
        if lm_model is not None:
            lm_tokenizer = AutoTokenizer.from_pretrained(
                str(lm_dir), padding_side="left",
            )

        # ── Build audio code vocabulary ─────────────────────────────
        audio_code_token_ids = None
        audio_code_token_map = None
        audio_code_mask = None
        if lm_tokenizer is not None:
            from .audio_codes import build_audio_code_vocab

            audio_code_token_ids, audio_code_token_map, audio_code_mask = (
                build_audio_code_vocab(lm_tokenizer)
            )

        # ── Construct pipeline ──────────────────────────────────────
        from .pipeline_v15 import AceStepV15Pipeline

        pipeline = AceStepV15Pipeline(
            transformer=transformer,
            audio_vae=audio_vae,
            text_encoder_2=text_encoder_2,
            lm_model=lm_model,
            pre_text_tokenizer=pre_text_tokenizer,
            lm_tokenizer=lm_tokenizer,
            silence_latent=silence_latent,
            audio_code_token_ids=audio_code_token_ids,
            audio_code_token_map=audio_code_token_map,
            audio_code_mask=audio_code_mask,
        )

        # ── Build pipe dict for mmgp ────────────────────────────────
        pipe = {
            "transformer": transformer,
            "text_encoder_2": text_encoder_2,
            "codec": audio_vae,
        }
        if lm_model is not None:
            pipe["text_encoder"] = lm_model

        # Co-tenants: models that can share VRAM
        # The transformer is the largest component; encoder and codec
        # can be offloaded while it runs
        co_tenants = {
            "transformer": ["codec"],
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info(
            "ACE-Step v1.5 loaded: variant=%s VRAM=%.0fMB pipe_keys=%s lm=%s",
            variant_name, vram, list(pipe.keys()), "yes" if lm_model else "no",
        )

        return LoadResult(
            pipeline=pipeline,
            pipe=pipe,
            co_tenants=co_tenants,
        )

    # ── Path resolution helpers ─────────────────────────────────────

    @staticmethod
    def _find_component_dir(root: Path, candidates: list[str]) -> Path | None:
        """Find a component directory by trying multiple names."""
        for name in candidates:
            path = root / name
            if path.is_dir():
                return path
        # Try case-insensitive
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_dir():
                    for name in candidates:
                        if entry.name.lower() == name.lower():
                            return entry
        return None

    @staticmethod
    def _find_weights(component_dir: Path) -> Path:
        """Find the weights file (safetensors) in a component directory."""
        # Prefer safetensors
        for pattern in ["model.safetensors", "*.safetensors", "diffusion_pytorch_model.safetensors"]:
            matches = list(component_dir.glob(pattern))
            if matches:
                return matches[0]
        # Fallback: any weights file
        for pattern in ["*.bin", "*.pt", "*.pth"]:
            matches = list(component_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No weights file found in {component_dir}")

    @staticmethod
    def _find_silence_latent(root: Path, transformer_dir: Path) -> Path | None:
        """Find the silence latent file."""
        for base in [transformer_dir, root]:
            for name in ["silence_latent.pt", "silence_latents.pt"]:
                path = base / name
                if path.exists():
                    return path
        return None
