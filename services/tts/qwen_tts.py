"""Qwen3-TTS - GPU text-to-speech using the Qwen3-TTS model.

Multi-speaker TTS with CustomVoice (9 premium voices + instruction control).
Pipeline imports directly — no subprocess or HTTP layer needed.
"""
from __future__ import annotations

import io
import logging
import os

import torch
from ray import serve
from starlette.responses import JSONResponse, Response

from services.base import BaseGPUDeployment, _free_cuda_cache

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/tts/qwen3-tts-12hz-1.7b-customvoice")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/models/tts/qwen3-tts-tokenizer-12hz")


@serve.deployment(
    name="qwen_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={"num_gpus": 0, "num_cpus": 0.5},
)
class QwenTTSDeployment(BaseGPUDeployment):
    """GPU-based Qwen3-TTS with CustomVoice."""

    def _load(self, model_name: str = "qwen3-tts") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Qwen3-TTS model not found at {MODEL_PATH}")

        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.model_name = model_name
        logger.info("Qwen3-TTS loaded from %s", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None
        _free_cuda_cache()

    async def __call__(self, request):
        if not self.is_loaded():
            self.load_model("qwen3-tts")

        body = await request.json()
        text = body.get("input", "")
        if not text:
            return JSONResponse({"error": "input text is required"}, status_code=400)

        voice = body.get("voice", "Aiden")
        instruct = body.get("instruct", "")

        zh_speakers = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"}
        ja_speakers = {"Ono_Anna"}
        ko_speakers = {"Sohee"}

        if voice in zh_speakers:
            lang = "Chinese"
        elif voice in ja_speakers:
            lang = "Japanese"
        elif voice in ko_speakers:
            lang = "Korean"
        else:
            lang = "English"

        gen_kwargs = {}
        if instruct:
            gen_kwargs["instruct"] = instruct

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=voice,
            **gen_kwargs,
        )

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
