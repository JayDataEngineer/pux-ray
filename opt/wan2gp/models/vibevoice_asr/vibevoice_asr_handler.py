"""VibeVoice ASR — Wan2GP-native handler using HF checkpoint weights.

Pipeline: audio → acoustic+semantic encoders → connector → Qwen2 LM → text.
"""
import base64
import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from models.base_handler import BaseFamilyHandler, _make_handler_cls

logger = logging.getLogger(__name__)

AUDIO_SR = 24000
HOP_LENGTH = 3200
TOKENIZER_PATH = "/tmp/vibevoice-tokenizer"


def _normalize_audio(audio: torch.Tensor, target_dB_FS=-25.0, eps=1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(audio ** 2))
    audio = audio * (10 ** (target_dB_FS / 20)) / (rms + eps)
    max_val = torch.max(torch.abs(audio))
    if max_val > 1.0:
        audio = audio / (max_val + eps)
    return audio


def _ensure_tokenizer():
    path = Path(TOKENIZER_PATH)
    if not (path / "tokenizer.json").exists():
        from huggingface_hub import hf_hub_download
        path.mkdir(parents=True, exist_ok=True)
        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            hf_hub_download("microsoft/VibeVoice-ASR-HF", f, local_dir=str(path))
        cfg_path = path / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.pop("extra_special_tokens", None)
        cfg.pop("processor_class", None)
        cfg_path.write_text(json.dumps(cfg))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    for token in ("<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
                  "<|object_ref_end|>", "<|box_start|>", "<|box_end|>"):
        tok.add_special_tokens({"additional_special_tokens": [token]})
    return tok


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["vibevoice-asr"]
    FAMILY = "vibevoice_asr"
    FAMILY_INFOS = {"vibevoice_asr": (304, "VibeVoice ASR")}
    DEFAULTS = {"language": "english"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.models import ModelRegistry
        model_path = Path(ModelRegistry().get_path("asr", "vibevoice-asr"))
        from models.vibevoice_asr.vibevoice_asr.model import VibeVoiceAsrModel
        model = VibeVoiceAsrModel.from_pretrained(model_path, dtype=dtype or torch.bfloat16)

        pipe = {
            "acoustic_encoder": model.acoustic_tokenizer,
            "semantic_encoder": model.semantic_tokenizer,
            "acoustic_connector": model.acoustic_connector,
            "semantic_connector": model.semantic_connector,
            "language_model": model.language_model,
            "lm_head": model.lm_head,
        }
        co_tenants = {
            "acoustic_encoder": ["semantic_encoder"],
            "language_model": ["acoustic_connector", "semantic_connector"],
        }

        return _Pipeline(model), {"pipe": pipe, "coTenantsMap": co_tenants}


class _Pipeline:
    def __init__(self, model):
        self.model = model
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _ensure_tokenizer()
        return self._tokenizer

    @property
    def device(self):
        return self.model.device

    def generate(self, *, audio_b64=None, audio_path=None, language="english",
                 max_tokens=512, seed=-1, **kw):
        import soundfile as sf

        audio_np = self._load_audio(audio_b64, audio_path)
        audio_t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        audio_t = _normalize_audio(audio_t).to(self.device)
        pad = (HOP_LENGTH - audio_t.shape[-1] % HOP_LENGTH) % HOP_LENGTH
        if pad:
            audio_t = torch.nn.functional.pad(audio_t, (0, pad))

        chat = [[{"role": "user", "content": [
            {"type": "audio", "audio": audio_np},
            {"type": "text", "text": f"Transcribe the following audio into {language}."},
        ]}]]
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )[0]
        text = text.replace("<|AUDIO_DURATION|>", f"{audio_np.shape[-1] / AUDIO_SR:.2f}")

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)

        audio_embeds = self.model.get_audio_features(audio_t)
        audio_gen = self.model.generate(input_ids, audio_embeds, max_new_tokens=max_tokens)
        text_out = self.tokenizer.decode(audio_gen[0], skip_special_tokens=True)
        text_out = text_out.removeprefix("assistant").strip()
        try:
            parsed = json.loads(text_out)
            if isinstance(parsed, list):
                texts = [s.get("Content", "") for s in parsed if isinstance(s, dict)]
                return {"status": "success", "text": " ".join(texts).strip(),
                        "segments": parsed}
        except (json.JSONDecodeError, TypeError):
            pass
        return {"status": "success", "text": text_out}

    def _load_audio(self, audio_b64=None, audio_path=None):
        import soundfile as sf
        if audio_b64:
            data, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="float32")
        elif audio_path:
            data, sr = sf.read(audio_path, dtype="float32")
        else:
            raise ValueError("audio_b64 or audio_path required")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != AUDIO_SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=AUDIO_SR)
        return data
