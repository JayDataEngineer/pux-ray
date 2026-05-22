"""MOSS-TTS-Nano handler — lightweight TTS (100M, GPT-2 backbone, 48kHz stereo).

Uses MOSS-Audio-Tokenizer-Nano for 48kHz decode. The model has a
high-level inference() API for end-to-end TTS.
"""
import importlib.util
import io
import logging
import sys
import types
from pathlib import Path

import torch

from models.base_handler import HandlerHooks

logger = logging.getLogger(__name__)

VARIANTS = {
    "moss-tts-nano": {
        "hf_id": "OpenMOSS-Team/MOSS-TTS-Nano-100M",
        "registry": ("audio", "moss-tts-nano"),
        "description": "Lightweight TTS (100M, 48kHz stereo)",
    },
}

NANO_AUDIO_TOKENIZER_REGISTRY = ("audio", "moss-audio-tokenizer-nano")


def _resolve_model_path(model_type):
    from registry.models import ModelRegistry
    reg = ModelRegistry()
    variant = VARIANTS[model_type]
    cat, name = variant["registry"]
    try:
        path = Path(reg.get_path(cat, name))
        if path.is_dir() and (path / "config.json").exists():
            return path
    except (KeyError, FileNotFoundError):
        pass
    from registry.config import Config
    models_root = Path(Config().models_root)
    from huggingface_hub import snapshot_download
    hf_id = variant["hf_id"]
    logger.info("Auto-downloading %s from %s", model_type, hf_id)
    local_dir = str(models_root / cat / name)
    snapshot_download(repo_id=hf_id, local_dir=local_dir)
    return Path(local_dir)


def _resolve_nano_audio_tokenizer_path():
    from registry.models import ModelRegistry
    reg = ModelRegistry()
    cat, name = NANO_AUDIO_TOKENIZER_REGISTRY
    try:
        return Path(reg.get_path(cat, name))
    except (KeyError, FileNotFoundError):
        from registry.config import Config
        models_root = Path(Config().models_root)
        from huggingface_hub import snapshot_download
        logger.info("Auto-downloading MOSS-Audio-Tokenizer-Nano")
        local_dir = str(models_root / cat / name)
        snapshot_download(
            repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
            local_dir=local_dir,
        )
        return Path(local_dir)


def _load_nano_modules(model_path):
    """Load MossTTSNano code as a synthetic package."""
    import torch.nn.init as _init
    sys.modules.setdefault("transformers.initialization", _init)

    import transformers.processing_utils as _pu
    if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
        _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}

    def _load_mod(name, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Required MOSS file not found: {filepath}")
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg_name = "moss_nano_pkg"
    if pkg_name not in sys.modules or not isinstance(sys.modules[pkg_name], types.ModuleType):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(model_path)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    config_mod = _load_mod(
        f"{pkg_name}.configuration_moss_tts_nano",
        model_path / "configuration_moss_tts_nano.py",
    )
    config_mod.__package__ = pkg_name
    setattr(sys.modules[pkg_name], "configuration_moss_tts_nano", config_mod)

    if (model_path / "gpt2_decoder.py").exists():
        gpt2_mod = _load_mod(
            f"{pkg_name}.gpt2_decoder",
            model_path / "gpt2_decoder.py",
        )
        gpt2_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "gpt2_decoder", gpt2_mod)

    if (model_path / "prompting.py").exists():
        prompt_mod = _load_mod(
            f"{pkg_name}.prompting",
            model_path / "prompting.py",
        )
        prompt_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "prompting", prompt_mod)

    modeling_mod = _load_mod(
        f"{pkg_name}.modeling_moss_tts_nano",
        model_path / "modeling_moss_tts_nano.py",
    )
    modeling_mod.__package__ = pkg_name

    if (model_path / "tokenization_moss_tts_nano.py").exists():
        tok_mod = _load_mod(
            f"{pkg_name}.tokenization_moss_tts_nano",
            model_path / "tokenization_moss_tts_nano.py",
        )
        tok_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "tokenization_moss_tts_nano", tok_mod)

    return config_mod, modeling_mod


class _MossNanoHooks(HandlerHooks):
    needs_bf16_autocast = False
    needs_device_patch = False

    def before_generate(self, pipeline, kwargs):
        # Ensure the entire model is on CUDA — the inference() API
        # creates input tensors on device= and if model params are
        # split across devices, embedding lookups fail.
        dev = pipeline.device
        pipeline.model.to(dev)
        if pipeline.audio_tokenizer is not None:
            pipeline.audio_tokenizer.to(dev)
        # Re-tie weights after .to() — .to() can break ties by
        # creating separate CUDA tensors for tied parameter pairs.
        pipeline.model.tie_weights()
        return kwargs


HANDLER_META = {
    "input_type": "text",
    "output_type": "audio",
    "hooks": _MossNanoHooks(),
}


class family_handler:
    @staticmethod
    def query_supported_types():
        return list(VARIANTS)

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "moss"

    @staticmethod
    def query_family_infos():
        return {"moss": (303, "MOSS Audio")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "audio_only": True,
            "image_outputs": False,
            "inference_steps": False,
            "temperature": True,
            "top_k_slider": True,
            "duration_slider": {
                "label": "Max duration (seconds)",
                "min": 1, "max": 600, "increment": 1, "default": 30,
            },
        }

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        model_path = _resolve_model_path(base_model_type)
        config_mod, modeling_mod = _load_nano_modules(model_path)

        MossTTSNanoConfig = config_mod.MossTTSNanoConfig
        MossTTSNanoForCausalLM = modeling_mod.MossTTSNanoForCausalLM
        MossTTSNanoSentencePieceTokenizer = modeling_mod.MossTTSNanoSentencePieceTokenizer

        from transformers import AutoConfig, AutoModel
        AutoConfig.register("moss_tts_nano", MossTTSNanoConfig)
        AutoModel.register(MossTTSNanoConfig, MossTTSNanoForCausalLM)

        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=dtype or torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()

        gpt2_config = getattr(model.config, "gpt2_config", None)
        if gpt2_config is not None and getattr(gpt2_config, "n_layer", None) is not None:
            model.config.num_hidden_layers = gpt2_config.n_layer

        tokenizer = MossTTSNanoSentencePieceTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        nano_audio_tok_path = _resolve_nano_audio_tokenizer_path()
        audio_tokenizer = None
        if nano_audio_tok_path.is_dir():
            audio_tokenizer = AutoModel.from_pretrained(
                str(nano_audio_tok_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            )

        pipe = {"model": model}
        return _Pipeline(model, tokenizer, audio_tokenizer), pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "repeat_generation": 1,
            "temperature": 0.9,
            "top_k": 50,
            "prompt": "Hello, how are you today?",
        })

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty."
        return None




class _Pipeline:
    def __init__(self, model, tokenizer, audio_tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return next(self.model.parameters()).device

    def generate(self, *, input_prompt="", max_tokens=4096, seed=-1, **kw):
        prompt = input_prompt or kw.get("text", "") or kw.get("prompt", "")
        if not prompt:
            raise ValueError("input_prompt or text required")

        dev = self.device

        # Ensure model + audio tokenizer are on CUDA and weight ties are intact.
        # Model was loaded with CUDA default device, but .to() is a safe guard
        # against any intervening device changes.
        self.model.to(dev)
        self.model.tie_weights()
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.to(dev)

        # Set default device to CUDA so model.inference() creates internal
        # tensors (token indices, etc.) on the correct device.
        prev_dev = torch.get_default_device()
        torch.set_default_device(dev)
        output_path = f"/tmp/moss_nano_out_{hash(prompt) % 1000000}.wav"
        try:
            result = self.model.inference(
                text=prompt,
                output_audio_path=output_path,
                mode="continuation",
                max_new_frames=min(max_tokens, 300),
                text_tokenizer=self.tokenizer,
                audio_tokenizer=self.audio_tokenizer,
                device=dev,
            )
        finally:
            torch.set_default_device(prev_dev)

        audio_array = None
        if isinstance(result, dict):
            for key in ("audio", "waveform", "array", "wav"):
                val = result.get(key)
                if val is not None:
                    audio_array = val
                    break
        if audio_array is not None:
            import numpy as np
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            sample_rate = 48000
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=0)
            import scipy.io.wavfile as wavfile
            buf = io.BytesIO()
            wavfile.write(buf, sample_rate, audio_array.astype(np.float32))
            wav_data = buf.getvalue()
            from models.base_handler import audio_response
            return audio_response(wav_data)

        import os as _os
        if _os.path.exists(output_path) and _os.path.getsize(output_path) > 1000:
            with open(output_path, "rb") as f:
                wav_data = f.read()
            from models.base_handler import audio_response
            return audio_response(wav_data)

        raise RuntimeError(f"No audio in result: keys={list(result.keys()) if isinstance(result, dict) else type(result)}")
