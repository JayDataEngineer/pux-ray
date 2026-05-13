"""VibeVoice Microsoft ASR — speech recognition with native diarization (ForgeService).

Uses microsoft/VibeVoice-ASR 7B model (~16GB VRAM). Conforms to ForgeService
interface: load() / unload() / infer().
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from services.forge_base import ForgeService

logger = logging.getLogger(__name__)

VIBEVOICE_ASR_PATH = os.environ.get("VIBEVOICE_ASR_MODEL_PATH", "/models/asr/vibevoice-asr")


class VibeVoiceMicrosoftService(ForgeService):
    """Forge adapter for VibeVoice Microsoft ASR (~16GB VRAM).

    Uses microsoft/VibeVoice-ASR 7B for speech recognition with speaker diarization.
    """

    vram_mb = 16_384
    service_name = "vibevoice_microsoft"
    default_model = "vibevoice-asr"

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None

    def load(self, model_name: str | None = None) -> None:
        model_name = model_name or self.default_model
        import torch

        # Patch: community fork imports VibeVoiceASRTextTokenizerFast from
        # original vibevoice, but original only has VibeVoiceTextTokenizerFast
        import vibevoice.modular.modular_vibevoice_text_tokenizer as _vtok
        if not hasattr(_vtok, 'VibeVoiceASRTextTokenizerFast'):
            _vtok.VibeVoiceASRTextTokenizerFast = _vtok.VibeVoiceTextTokenizerFast

        from vibevoice_community.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        from vibevoice_community.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

        if not os.path.isdir(VIBEVOICE_ASR_PATH):
            raise FileNotFoundError(f"VibeVoice ASR model not found at {VIBEVOICE_ASR_PATH}")

        self.processor = VibeVoiceASRProcessor.from_pretrained(VIBEVOICE_ASR_PATH)
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            VIBEVOICE_ASR_PATH, torch_dtype=torch.float32, device_map="auto",
        )
        self.model_name = model_name
        self._loaded = True
        logger.info("VibeVoice ASR loaded from %s", VIBEVOICE_ASR_PATH)

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.processor
        self.model = None
        self.processor = None
        super().unload()

    def infer(self, payload: dict) -> dict:
        import base64
        from services.base import _b64_decode

        audio_input = payload.get("audio")
        if isinstance(audio_input, str):
            audio_bytes = _b64_decode(audio_input)
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        else:
            return {"status": "error", "error": "audio (bytes or base64) is required"}

        language = payload.get("language")
        result = self._run_transcribe(audio=audio_bytes, language=language)
        return {"status": "ok", **result}

    def _run_transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        diarize: bool = True,
        num_speakers: int | None = None,
    ) -> dict:
        import torch
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        waveform, sr = sf.read(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        inputs = self.processor(
            audio=waveform, sampling_rate=sr,
            return_tensors="pt", language=language,
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=4096)

        text = self.processor.decode(output[0], skip_special_tokens=True)

        return {
            "text": text,
            "language": language or "auto",
            "segments": [{"start": 0, "end": 0, "text": text, "speaker": "SPEAKER_00"}],
        }
