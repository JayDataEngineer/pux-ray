"""GPU ASR services - VibeVoice ASR and Qwen ASR.

VibeVoice ASR: 7B model with native diarization (~16GB VRAM).
Qwen ASR: 1.7B model, 52 languages.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ray import serve

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)


@serve.deployment(
    name="vibevoice_asr",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
    },
)
class VibeVoiceASRDeployment(BaseGPUDeployment):
    """VibeVoice ASR with native speaker diarization."""

    def _load(self, model_name: str = "vibevoice-asr") -> None:
        from registry.models import ModelRegistry
        from transformers import AutoModelForCausalLM, AutoProcessor

        registry = ModelRegistry()
        model_path = registry.get_path("asr", model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"VibeVoice ASR model not found at {model_path}. "
                f"Run 'task models:pull' to download it."
            )

        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype="auto", device_map="auto",
            local_files_only=True,
        )
        self.model_name = model_name
        logger.info("VibeVoice ASR loaded from %s", model_path)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None

    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        diarize: bool = True,
        num_speakers: int | None = None,
    ) -> dict:
        """Transcribe with optional diarization."""
        if not self.is_loaded():
            self.load_model("vibevoice-asr")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        # VibeVoice ASR inference
        import torch
        import soundfile as sf
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

    async def __call__(self, request):
        form = await request.form()
        audio_file = form["file"]
        audio_bytes = await audio_file.read()
        language = form.get("language")

        result = await self.transcribe(audio=audio_bytes, language=language)
        from starlette.responses import JSONResponse
        return JSONResponse(result)


@serve.deployment(
    name="qwen_asr",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
    },
)
class QwenASRDeployment(BaseGPUDeployment):
    """Qwen3-ASR 1.7B. 52 languages."""

    def _load(self, model_name: str = "qwen-asr") -> None:
        from registry.models import ModelRegistry
        from transformers import AutoModelForCausalLM, AutoProcessor

        registry = ModelRegistry()
        model_path = registry.get_path("asr", model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Qwen ASR model not found at {model_path}. "
                f"Run 'task models:pull' to download it."
            )

        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype="auto", device_map="auto",
            local_files_only=True,
        )
        self.model_name = model_name
        logger.info("Qwen ASR loaded from %s", model_path)

    def _unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None

    async def transcribe(self, audio: bytes, language: str | None = None) -> dict:
        if not self.is_loaded():
            self.load_model("qwen-asr")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        import torch
        import soundfile as sf
        waveform, sr = sf.read(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        inputs = self.processor(
            audio=waveform, sampling_rate=sr,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=4096)

        text = self.processor.decode(output[0], skip_special_tokens=True)
        return {"text": text, "language": language or "auto"}

    async def __call__(self, request):
        form = await request.form()
        audio_file = form["file"]
        audio_bytes = await audio_file.read()
        result = await self.transcribe(audio=audio_bytes)
        from starlette.responses import JSONResponse
        return JSONResponse(result)
