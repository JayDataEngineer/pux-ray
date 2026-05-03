"""Qwen3-TTS - GPU text-to-speech using the Qwen3-TTS model.

Multi-speaker TTS with CustomVoice (9 premium voices + instruction control).
Model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (~8GB VRAM).
"""

from __future__ import annotations

import logging

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

# Default HuggingFace model — qwen-tts handles auto-download if not cached locally.
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@serve.deployment(
    name="qwen_tts",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 1.0,
        "num_cpus": 0.5,
    },
)
class QwenTTSDeployment(BaseGPUDeployment):
    """GPU-based Qwen3-TTS with CustomVoice."""

    def _load(self, model_name: str = "qwen3-tts") -> None:
        from qwen_tts import Qwen3TTSModel
        import torch

        # Use HF model ID directly — auto-downloads and caches
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.model_name = model_name
        self._speakers = self.model.get_supported_speakers()
        self._languages = self.model.get_supported_languages()
        logger.info("Qwen3-TTS loaded. Speakers: %s", self._speakers)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def synthesize(
        self,
        text: str,
        voice: str = "Aiden",
        mode: str = "customvoice",
        output_format: str = "wav",
        instruct: str = "",
    ) -> bytes:
        """Synthesize speech.

        For CustomVoice model: text, language, speaker, instruct.
        Voice can be: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
        instruct: natural language instruction for tone/emotion/style control.
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        import soundfile as sf
        import io

        speaker = voice
        # Try to infer language from speaker
        zh_speakers = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"}
        ja_speakers = {"Ono_Anna"}
        ko_speakers = {"Sohee"}

        if speaker in zh_speakers:
            lang = "Chinese"
        elif speaker in ja_speakers:
            lang = "Japanese"
        elif speaker in ko_speakers:
            lang = "Korean"
        else:
            lang = "English"

        gen_kwargs = {}
        if instruct:
            gen_kwargs["instruct"] = instruct

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=speaker,
            **gen_kwargs,
        )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()

    async def __call__(self, request):
        body = await request.json()
        if not self.is_loaded():
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"error": "QwenTTS model not loaded. Use /v1/audio/speech via ingress (port 18080)."},
                status_code=503,
            )
        audio = await self.synthesize(
            text=body.get("input", ""),
            voice=body.get("voice", "Aiden"),
            mode=body.get("mode", "customvoice"),
            output_format=body.get("response_format", "wav"),
            instruct=body.get("instruct", ""),
        )
        from starlette.responses import Response
        return Response(content=audio, media_type="audio/wav")
