"""Qwen3-TTS - GPU text-to-speech using the Qwen3-TTS model.

Multi-speaker TTS with CustomVoice (9 premium voices + instruction control).
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/tts/qwen3-tts-12hz-1.7b-customvoice")


@serve.deployment(
    name="qwen_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
            },
        },
    },
)
class QwenTTSDeployment(BaseGPUDeployment):
    """GPU-based Qwen3-TTS with CustomVoice."""

    def _load(self, model_name: str = "qwen3-tts") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Qwen3-TTS model not found at {MODEL_PATH}")

        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model_name = model_name
        logger.info("Qwen3-TTS loaded from %s", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None
        super()._unload()

    def _generate(self, text: str, voice: str, instruct: str) -> tuple:
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

        return self.model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=voice,
            **gen_kwargs,
        )

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, voice, instruct}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "qwen3-tts")

            wavs, sr = await asyncio.to_thread(
                self._generate,
                extracted.get("text", ""),
                extracted.get("voice", "Aiden"),
                extracted.get("instruct", ""),
            )

            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, wavs[0], sr, format="WAV")
            buf.seek(0)
            audio = buf.read()

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("qwen_tts error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)