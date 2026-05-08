"""VibeVoice TTS — Long-form multi-speaker speech synthesis.

Uses VibeVoiceForConditionalGenerationInference. Supports multi-speaker voice cloning.
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

MODEL_PATH = os.environ.get("VIBEVOICE_MODEL_PATH", "vibevoice/VibeVoice-7B")
VOICES_DIR = os.environ.get("VIBEVOICE_VOICES_DIR", "/opt/vibevoice-community/voices")


@serve.deployment(
    name="vibevoice",
    num_replicas=1,
    max_ongoing_requests=1,
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
class VibeVoiceDeployment(BaseGPUDeployment):
    """VibeVoice long-form multi-speaker TTS."""

    def _load(self, model_name: str = "vibevoice-tts-7b") -> None:
        # Patch: community fork may reference classes from original vibevoice
        import vibevoice.modular.modular_vibevoice_text_tokenizer as _vtok
        if not hasattr(_vtok, 'VibeVoiceASRTextTokenizerFast'):
            _vtok.VibeVoiceASRTextTokenizerFast = _vtok.VibeVoiceTextTokenizerFast

        from vibevoice_community.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice_community.processor.vibevoice_processor import VibeVoiceProcessor

        self.processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

        self.pipeline = VibeVoiceForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        self.pipeline.eval()
        self.pipeline.set_ddpm_inference_steps(num_steps=10)

        self.model = True
        self.model_name = model_name
        logger.info("VibeVoice loaded: %s", MODEL_PATH)

    def _unload(self) -> None:
        self.pipeline = None
        self.processor = None
        self.model = None
        super()._unload()

    def _find_voice_file(self, speaker_name: str) -> str:
        for ext in (".wav", ".flac", ".mp3"):
            path = os.path.join(VOICES_DIR, f"{speaker_name}{ext}")
            if os.path.isfile(path):
                return path
        if os.path.isdir(VOICES_DIR):
            wavs = [f for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
            if wavs:
                return os.path.join(VOICES_DIR, wavs[0])
        raise FileNotFoundError(f"No voice file found for '{speaker_name}' in {VOICES_DIR}")

    def _generate_audio(self, text: str, voice_samples: list[str]) -> bytes:
        import soundfile as sf

        processed = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in processed.items():
            if torch.is_tensor(v):
                processed[k] = v.to("cuda:0")

        outputs = self.pipeline.generate(
            **processed,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
        )

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("generation produced no audio")

        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().to(torch.float32).numpy()

        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        buf.seek(0)
        return buf.getvalue()

    def _extract_input(self, inp) -> dict:
        result = super()._extract_input(inp)
        if inp.audio_b64:
            from services.base import _b64_decode
            result["reference_audio"] = _b64_decode(inp.audio_b64)
        return result

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {text, voice, audio_b64}, config}."""
        if request.method == "GET":
            return {"status": "ok", "model": self.model_name, "loaded": self.is_loaded()}

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "vibevoice-tts-7b")

            text = extracted.get("text", "")
            if not text:
                return JSONResponse(self.handle_error("text is required"), status_code=400)

            speaker_names = extracted.get("voice", "Andrew")
            if isinstance(speaker_names, str):
                speaker_names = [s.strip() for s in speaker_names.split(",")]

            voice_samples = []
            if extracted.get("reference_audio"):
                import tempfile
                ref_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                ref_path.write(extracted["reference_audio"])
                ref_path.flush()
                voice_samples.append(ref_path.name)
            else:
                for name in speaker_names:
                    try:
                        wav_path = self._find_voice_file(name)
                        voice_samples.append(wav_path)
                    except FileNotFoundError:
                        return JSONResponse(
                            self.handle_error(f"No voice file for speaker '{name}'"),
                            status_code=400,
                        )

            audio = await asyncio.to_thread(self._generate_audio, text, voice_samples)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("vibevoice error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)