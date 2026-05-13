"""Kokoro TTS raw nn.Module loading + mmgp setup.

Decomposes KModel into:
- bert: CustomAlbert text encoder
- bert_encoder: Linear projection of BERT output
- predictor: ProsodyPredictor (duration + F0 + noise prediction)
- text_encoder: TextEncoder (phoneme embedding + CNN + LSTM)
- decoder: Decoder + Generator (audio waveform synthesis)

Each module managed independently by mmgp.
Orchestrator calls forward() directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class KokoroModules:
    """Raw nn.Modules for Kokoro TTS inference."""

    bert: Any            # CustomAlbert — text encoding
    bert_encoder: Any    # Linear — BERT output projection
    predictor: Any       # ProsodyPredictor — duration, F0, noise
    text_encoder: Any    # TextEncoder — phoneme features
    decoder: Any         # Decoder + Generator — audio synthesis

    # Pipeline for text preprocessing (G2P, chunking, voice loading)
    pipeline: Any = None

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    pipe: dict = field(default_factory=dict)
    co_tenants: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> KokoroModules:
        from kokoro import KModel, KPipeline

        logger.info("Loading Kokoro TTS")

        model = KModel()
        pipeline = KPipeline(lang_code="a", model=model)

        # Extract individual nn.Modules from KModel
        bert = model.bert
        bert_encoder = model.bert_encoder
        predictor = model.predictor
        text_encoder = model.text_encoder
        decoder = model.decoder

        device = torch.device("cpu")
        bert.eval()
        predictor.eval()
        text_encoder.eval()
        decoder.eval()

        # mmgp pipe dict — CPU modules, no offloading needed
        pipe = {
            "bert": bert,
            "bert_encoder": bert_encoder,
            "predictor": predictor,
            "text_encoder": text_encoder,
            "decoder": decoder,
        }
        co_tenants = {}

        logger.info("Kokoro decomposed: %s", list(pipe.keys()))

        return cls(
            bert=bert,
            bert_encoder=bert_encoder,
            predictor=predictor,
            text_encoder=text_encoder,
            decoder=decoder,
            pipeline=pipeline,
            device=device,
            dtype=dtype,
            pipe=pipe,
            co_tenants=co_tenants,
        )
