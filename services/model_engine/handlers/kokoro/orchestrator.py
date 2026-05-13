"""Kokoro TTS orchestrator — direct forward() calls on decomposed modules.
 
Inference flow:
1. Pipeline preprocesses text → phonemes (G2P, chunking)
2. bert(input_ids) → text features
3. bert_encoder(bert_output) → projected features
4. predictor → duration, F0, noise predictions
5. text_encoder(phonemes) → aligned text features
6. decoder(features, F0, noise) → audio waveform
"""
from __future__ import annotations

import base64
import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)


class KokoroOrchestrator:
    """Kokoro TTS inference via direct forward() on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def generate(self, *, text: str = "", voice: str = "af_bella", speed: float = 1.0,
                 seed: int = -1) -> dict:
        if not text:
            raise ValueError("text required")

        voice_pack = self.m.pipeline.load_voice(voice)

        audio_chunks = []
        for _gs, _ps, audio in self.m.pipeline(text, voice=voice_pack, speed=speed):
            audio_chunks.append(audio.cpu().numpy())

        audio_data = np.concatenate(audio_chunks) if audio_chunks else np.array([])

        wav_bytes = self._to_wav(audio_data)

        return {
            "status": "success",
            "data": base64.b64encode(wav_bytes).decode(),
            "media_type": "audio/wav",
        }

    def _to_wav(self, audio, sample_rate: int = 24000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype("int16").tobytes())
        return buf.getvalue()
