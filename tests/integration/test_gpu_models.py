"""GPU integration tests — load + generate for GPU-requiring custom handlers.

Run on the GPU server: pytest tests/integration/test_gpu_models.py -m gpu

Each test loads a real model, runs inference, and validates the output.
Tests are individually marked @pytest.mark.gpu and @pytest.mark.slow.
Auto-skipped when no CUDA GPU is available.
"""
from __future__ import annotations

import gc
import importlib
import os
import time

import pytest


pytest.skip("Dead vibevoice TTS handler removed — rewrite with current model tests", allow_module_level=True)


def _load_handler(handler_path, model_type):
    """Import handler → load_model → (pipeline, pipe_wrapper)."""
    mod = importlib.import_module(handler_path)
    handler = mod.family_handler
    model_def = handler.query_model_def(model_type, {})
    pipeline, pipe_wrapper = handler.load_model(
        [], model_type, model_type, model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None, VAE_dtype=None, profile=0,
    )
    return pipeline, pipe_wrapper


def _cleanup(pipeline, pipe_wrapper):
    del pipeline
    if isinstance(pipe_wrapper, dict):
        pipe = pipe_wrapper.get("pipe", {})
        if isinstance(pipe, dict):
            for v in pipe.values():
                import torch
                if isinstance(v, torch.nn.Module):
                    del v
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class TestMOSS:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.handler
    def test_moss_generate(self):
        pipeline, pw = _load_handler("models.moss.moss_handler", "moss-soundeffect")
        try:
            result = pipeline.generate(prompt="gentle rain", max_tokens=64)
            assert result["status"] == "success"
        finally:
            _cleanup(pipeline, pw)


class TestVibeVoiceASR:
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.handler
    def test_vibevoice_asr_generate(self, sample_wav_b64):
        pipeline, pw = _load_handler(
            "models.vibevoice_asr.vibevoice_asr_handler", "vibevoice-asr"
        )
        try:
            result = pipeline.generate(
                audio_b64=sample_wav_b64, language="english", max_tokens=128
            )
            assert result["status"] == "success"
        finally:
            _cleanup(pipeline, pw)


