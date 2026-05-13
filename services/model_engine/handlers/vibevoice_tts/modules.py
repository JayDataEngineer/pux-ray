"""VibeVoice TTS raw nn.Module loading + mmgp setup.

Decomposes VibeVoiceForConditionalGenerationInference into:
- language_model: Qwen2-based LM
- acoustic_tokenizer: encode/decode speech <-> acoustic latents
- semantic_tokenizer: speech -> semantic features
- acoustic_connector: acoustic -> LM space projection
- semantic_connector: semantic -> LM space projection
- prediction_head: DDPM diffusion head for speech generation
- lm_head: vocabulary projection

Each module managed independently by mmgp.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class VibeVoiceTTSModules:
    """Raw nn.Modules for VibeVoice TTS inference."""

    language_model: Any
    acoustic_tokenizer: Any
    semantic_tokenizer: Any
    acoustic_connector: Any
    semantic_connector: Any
    prediction_head: Any
    lm_head: Any

    # Processor for tokenization
    processor: Any = None
    # Noise scheduler for diffusion
    noise_scheduler: Any = None
    # Speech scaling/bias
    speech_scaling_factor: Any = None
    speech_bias_factor: Any = None

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
    ) -> VibeVoiceTTSModules:
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading VibeVoice TTS from %s", model_path)

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="cpu",
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        # Extract individual nn.Modules
        inner = model.model
        language_model = inner.language_model
        acoustic_tokenizer = inner.acoustic_tokenizer
        semantic_tokenizer = inner.semantic_tokenizer
        acoustic_connector = inner.acoustic_connector
        semantic_connector = inner.semantic_connector
        prediction_head = inner.prediction_head
        lm_head = model.lm_head

        noise_scheduler = getattr(inner, "noise_scheduler", None)
        speech_scaling_factor = getattr(inner, "speech_scaling_factor", None)
        speech_bias_factor = getattr(inner, "speech_bias_factor", None)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move each module independently
        for mod in [language_model, acoustic_tokenizer, semantic_tokenizer,
                    acoustic_connector, semantic_connector, prediction_head, lm_head]:
            mod.to(device)
            mod.eval()

        pipe = {
            "language_model": language_model,
            "acoustic_tokenizer": acoustic_tokenizer,
            "semantic_tokenizer": semantic_tokenizer,
            "acoustic_connector": acoustic_connector,
            "semantic_connector": semantic_connector,
            "prediction_head": prediction_head,
            "lm_head": lm_head,
        }
        co_tenants = {
            "language_model": ["acoustic_tokenizer", "prediction_head"],
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0
        logger.info(
            "VibeVoice TTS decomposed: %s VRAM=%.0fMB",
            list(pipe.keys()), vram,
        )

        return cls(
            language_model=language_model,
            acoustic_tokenizer=acoustic_tokenizer,
            semantic_tokenizer=semantic_tokenizer,
            acoustic_connector=acoustic_connector,
            semantic_connector=semantic_connector,
            prediction_head=prediction_head,
            lm_head=lm_head,
            processor=processor,
            noise_scheduler=noise_scheduler,
            speech_scaling_factor=speech_scaling_factor,
            speech_bias_factor=speech_bias_factor,
            device=device,
            dtype=dtype,
            pipe=pipe,
            co_tenants=co_tenants,
        )
