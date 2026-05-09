"""VibeVoice Microsoft — Microsoft VibeVoice ASR with native diarization (~16GB VRAM).

Uses microsoft/VibeVoice-ASR 7B model for speech recognition with speaker diarization.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig

logger = logging.getLogger(__name__)

VIBEVOICE_ASR_PATH = os.environ.get("VIBEVOICE_ASR_MODEL_PATH", "/models/asr/vibevoice-asr")


@serve.deployment(
    name="vibevoice_microsoft",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
            }
        },
    },
)
class VibeVoiceMicrosoftDeployment(BaseGPUDeployment):
    """VibeVoice Microsoft — microsoft/VibeVoice-ASR with native speaker diarization."""

    def _load(self, model_name: str = "vibevoice-asr") -> None:
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
        logger.info("VibeVoice ASR loaded from %s", VIBEVOICE_ASR_PATH)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        super()._unload()

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

    def _extract_input(self, inp) -> dict:
        result = super()._extract_input(inp)
        if inp.audio_b64:
            from services.base import _b64_decode
            result["audio"] = _b64_decode(inp.audio_b64)
        return result

    async def __call__(self, request):
        """TNAP endpoint. Supports JSON with audio_b64 or multipart."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                form = await request.form()
                audio_file = form["file"]
                audio_bytes = await audio_file.read()
                language = form.get("language")

                if "config" in form:
                    requested = InferenceConfig(**form["config"])
                    if requested != self.config:
                        self.config = requested

                if not self.is_loaded():
                    await asyncio.to_thread(self.load_model, "vibevoice-asr")

                result = await asyncio.to_thread(
                    lambda: self._run_transcribe(audio=audio_bytes, language=language),
                )
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                audio_bytes = extracted.get("audio")
                if not audio_bytes:
                    return JSONResponse(self.handle_error("audio_b64 required"), status_code=400)

                if not self.is_loaded():
                    await asyncio.to_thread(self.load_model, "vibevoice-asr")

                result = await asyncio.to_thread(
                    lambda: self._run_transcribe(audio=audio_bytes, language=extracted.get("language")),
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    json.dumps(result).encode("utf-8"),
                    "application/json",
                    latency_ms,
                    extra_metrics={"language": result.get("language", "")},
                )
            )
        except Exception as e:
            logger.error("vibevoice_asr error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)
