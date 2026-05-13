"""VibeVoice ASR raw nn.Module loading + mmgp setup.

Decomposes VibeVoiceASRForConditionalGeneration into:
- language_model: Qwen2-based LM (the heaviest module)
- acoustic_tokenizer: speech -> acoustic features
- semantic_tokenizer: speech -> semantic features
- acoustic_connector: acoustic -> LM space projection (linear)
- semantic_connector: semantic -> LM space projection (linear)
- lm_head: vocabulary projection (linear)

Each module is managed independently by mmgp.
The orchestrator calls forward()/encode() directly on each.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class VibeVoiceASRModules:
    """Raw nn.Modules for VibeVoice ASR inference."""

    language_model: Any        # model.model.language_model
    acoustic_tokenizer: Any    # model.model.acoustic_tokenizer
    semantic_tokenizer: Any    # model.model.semantic_tokenizer
    acoustic_connector: Any    # model.model.acoustic_connector
    semantic_connector: Any    # model.model.semantic_connector
    lm_head: Any               # model.lm_head

    # Processor for tokenization + audio preprocessing
    processor: Any = None

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
    ) -> VibeVoiceASRModules:
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading VibeVoice ASR from %s", model_path)

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
        lm_head = model.lm_head

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move each module to device independently
        language_model = language_model.to(device)
        acoustic_tokenizer = acoustic_tokenizer.to(device)
        semantic_tokenizer = semantic_tokenizer.to(device)
        acoustic_connector = acoustic_connector.to(device)
        semantic_connector = semantic_connector.to(device)
        lm_head = lm_head.to(device)

        language_model.eval()
        acoustic_tokenizer.eval()
        semantic_tokenizer.eval()

        # mmgp pipe dict — each module managed independently
        pipe = {
            "language_model": language_model,
            "acoustic_tokenizer": acoustic_tokenizer,
            "semantic_tokenizer": semantic_tokenizer,
            "acoustic_connector": acoustic_connector,
            "semantic_connector": semantic_connector,
            "lm_head": lm_head,
        }
        co_tenants = {
            "language_model": ["acoustic_tokenizer", "semantic_tokenizer"],
        }

        vram = torch.cuda.memory_allocated(0) / (1024**2) if device.type == "cuda" else 0
        logger.info(
            "VibeVoice ASR decomposed: %s VRAM=%.0fMB",
            list(pipe.keys()), vram,
        )

        return cls(
            language_model=language_model,
            acoustic_tokenizer=acoustic_tokenizer,
            semantic_tokenizer=semantic_tokenizer,
            acoustic_connector=acoustic_connector,
            semantic_connector=semantic_connector,
            lm_head=lm_head,
            processor=processor,
            device=device,
            dtype=dtype,
            pipe=pipe,
            co_tenants=co_tenants,
        )
