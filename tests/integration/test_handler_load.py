"""GPU integration tests — load_model() contract verification for all handlers.

Run on the GPU server: pytest tests/integration/test_handler_load.py -m gpu

Verifies that load_model() returns (pipeline, {"pipe": dict, "coTenantsMap": dict})
and that the pipeline has a callable generate() method.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ALL_HANDLERS = [
    ("models.kokoro.kokoro_handler", "kokoro"),
    ("models.moss.moss_handler", "moss-soundeffect"),
    ("models.espeak.espeak_handler", "espeak"),
    ("models.faster_whisper.faster_whisper_handler", "faster_whisper"),
    ("models.vibevoice_asr.vibevoice_asr_handler", "vibevoice-asr"),
    ("models.vibevoice_tts.vibevoice_tts_handler", "vibevoice-tts"),
]


_FORK_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "opt" / "wan2gp" / "models"


def _import_handler(import_path: str):
    """Import handler from the fork's models/ package."""
    try:
        mod = importlib.import_module(import_path)
        mod_file = getattr(mod, "__file__", "") or ""
        if "opt/wan2gp/models" in mod_file:
            return mod
    except (ImportError, ModuleNotFoundError):
        pass
    rel_path = import_path
    if rel_path.startswith("models."):
        rel_path = rel_path[len("models."):]
    parts = rel_path.split(".")
    mod_file = _FORK_MODELS_DIR.joinpath(*parts[:-1]) / (parts[-1] + ".py")
    if not mod_file.exists():
        raise ImportError(f"Handler file not found: {mod_file}")
    spec = importlib.util.spec_from_file_location(import_path, str(mod_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=ALL_HANDLERS, ids=[h[1] for h in ALL_HANDLERS])
def handler(request):
    import_path, model_type = request.param
    mod = _import_handler(import_path)
    return import_path, model_type, mod.family_handler


class TestLoadModelContract:
    """load_model() returns (pipeline, {"pipe": dict, "coTenantsMap": dict})."""

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.handler
    @pytest.mark.integration
    def test_load_model_returns_pipeline_and_pipe_dict(self, handler):
        import_path, model_type, fh = handler
        model_def = fh.query_model_def(model_type, {})

        pipeline, pipe_wrapper = fh.load_model(
            [], model_type, model_type, model_def,
            quantizeTransformer=False,
            text_encoder_quantization=None,
            dtype=None, VAE_dtype=None, profile=0,
        )

        assert hasattr(pipeline, "generate"), (
            f"{import_path}: pipeline has no generate()"
        )
        assert callable(pipeline.generate), (
            f"{import_path}: pipeline.generate not callable"
        )
        assert isinstance(pipe_wrapper, dict), (
            f"{import_path}: pipe_wrapper is {type(pipe_wrapper)}, expected dict"
        )
        assert "pipe" in pipe_wrapper, (
            f"{import_path}: missing 'pipe' key"
        )
        assert "coTenantsMap" in pipe_wrapper, (
            f"{import_path}: missing 'coTenantsMap' key"
        )
        assert isinstance(pipe_wrapper["pipe"], dict)
        assert isinstance(pipe_wrapper["coTenantsMap"], dict)

        # Cleanup
        del pipeline
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
