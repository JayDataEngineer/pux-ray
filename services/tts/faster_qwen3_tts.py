"""Faster Qwen3-TTS — CUDA graph accelerated TTS.

Drop-in replacement for qwen_tts using torch.cuda.CUDAGraph for 5x speedup.
Supports CustomVoice (9 premium speakers), voice cloning, and voice design.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/tts/qwen3-tts-12hz-1.7b-customvoice")

# Default reference audio for voice cloning when no ref provided
DEFAULT_REF = "/models/tts/kokoro/samples/af_heart_0.wav"

SPEAKER_LANG = {
    "Vivian": "Chinese", "Serena": "Chinese", "Uncle_Fu": "Chinese",
    "Dylan": "Chinese", "Eric": "Chinese",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}


@serve.deployment(
    name="faster_qwen3_tts",
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
class FasterQwen3TTSDeployment(BaseGPUDeployment):
    """CUDA-graph accelerated Qwen3-TTS."""

    def _load(self, model_name: str = "qwen3-tts") -> None:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"Qwen3-TTS model not found at {MODEL_PATH}")

        from faster_qwen3_tts import FasterQwen3TTS

        self.model = FasterQwen3TTS.from_pretrained(
            MODEL_PATH,
            device="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model_name = model_name
        logger.info("FasterQwen3-TTS loaded from %s (CUDA graphs)", MODEL_PATH)

    def _unload(self) -> None:
        self.model = None
        super()._unload()

    def _generate_custom_voice(self, text: str, voice: str, instruct: str) -> tuple:
        lang = SPEAKER_LANG.get(voice, "English")
        kwargs = {}
        if instruct:
            kwargs["instruct"] = instruct
        return self.model.generate_custom_voice(
            text=text, speaker=voice, language=lang, **kwargs,
        )

    def _generate_voice_clone(self, text: str, ref_audio: str, ref_text: str, language: str) -> tuple:
        return self.model.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio, ref_text=ref_text,
        )

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, voice, instruct, ref_audio_b64, ref_text, language}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "qwen3-tts")

            text = extracted.get("text", "")
            if not text:
                return JSONResponse(self.handle_error("text is required"), status_code=400)

            import soundfile as sf

            if extracted.get("reference_audio"):
                # Voice cloning mode
                import tempfile
                from services.base import _b64_decode
                ref_bytes = extracted["reference_audio"]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(ref_bytes)
                    ref_path = tmp.name
                ref_text = extracted.get("ref_text", "")
                language = extracted.get("language", "English")
                audio_list, sr = await asyncio.to_thread(
                    self._generate_voice_clone, text, ref_path, ref_text, language,
                )
                os.unlink(ref_path)
            elif extracted.get("ref_audio_b64"):
                import tempfile
                from services.base import _b64_decode
                ref_bytes = _b64_decode(extracted["ref_audio_b64"])
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(ref_bytes)
                    ref_path = tmp.name
                ref_text = extracted.get("ref_text", "")
                language = extracted.get("language", "English")
                audio_list, sr = await asyncio.to_thread(
                    self._generate_voice_clone, text, ref_path, ref_text, language,
                )
                os.unlink(ref_path)
            else:
                # CustomVoice mode (default)
                voice = extracted.get("voice", "Aiden")
                instruct = extracted.get("instruct", "")
                audio_list, sr = await asyncio.to_thread(
                    self._generate_custom_voice, text, voice, instruct,
                )

            buf = io.BytesIO()
            sf.write(buf, audio_list[0], sr, format="WAV")
            buf.seek(0)
            audio = buf.read()

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("faster_qwen3_tts error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)
