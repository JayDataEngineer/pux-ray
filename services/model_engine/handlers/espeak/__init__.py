"""eSpeak TTS handler — subprocess-based phoneme synthesis.
 
No nn.Modules — eSpeak is a C library. The handler provides the same
BaseHandler interface with an empty pipe dict. One system, one truth.
"""
from __future__ import annotations

import base64
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import torch

from services.model_engine.base_handler import BaseHandler, ModelVariant

logger = logging.getLogger(__name__)

VARIANTS = {
    "espeak": ModelVariant(
        name="espeak", family="espeak",
        display_name="eSpeak-NG Phoneme TTS",
        vram_estimate_gb=0,
        defaults={"voice": "en", "speed": 175, "pitch": 50},
    ),
}


class EspeakHandler(BaseHandler):

    def supported_types(self) -> list[str]:
        return list(VARIANTS.keys())

    def resolve_path(self, model_type: str, models_root: Path) -> Path:
        return models_root  # no model files needed

    def get_variant(self, model_type: str) -> ModelVariant:
        return VARIANTS["espeak"]

    def load_model(
        self,
        model_type: str,
        model_path: Path,
        **kwargs,
    ) -> tuple:
        from registry.config import Config
        bin_path = Config().get("binaries.espeak_ng", "espeak-ng")

        result = subprocess.run(
            ["which", bin_path], capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng binary not found: {bin_path}")

        from .orchestrator import EspeakOrchestrator
        orchestrator = EspeakOrchestrator(bin_path=bin_path)

        return orchestrator, {}
