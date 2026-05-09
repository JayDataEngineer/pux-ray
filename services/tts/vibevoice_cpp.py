"""VibeVoice.cpp — GGML-accelerated TTS + ASR via vibevoice-cli.

C++ inference engine for Microsoft VibeVoice built on ggml.
Supports quantized GGUF models (Q8_0, Q4_K) for both TTS and ASR.
Runs as subprocess via Ray Serve — no Python model loading needed.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment

logger = logging.getLogger(__name__)

VV_MODELS_DIR = os.environ.get("VIBEVOICE_CPP_MODELS", "/models/vibevoice-cpp")
VV_BIN = os.environ.get("VIBEVOICE_CPP_BIN", "vibevoice-cli")


class _VibeVoiceCppBase(BaseGPUDeployment):
    """Shared logic for vibevoice.cpp TTS + ASR via subprocess calls to vibevoice-cli."""

    def _load(self, model_name: str = "vibevoice-cpp") -> None:
        self.tts_model = os.path.join(VV_MODELS_DIR, "vibevoice-realtime-0.5B-q8_0.gguf")
        self.asr_model = os.path.join(VV_MODELS_DIR, "vibevoice-asr-q8_0.gguf")
        self.tokenizer = os.path.join(VV_MODELS_DIR, "tokenizer.gguf")
        self.voices_dir = VV_MODELS_DIR

        # Verify CLI binary exists
        result = subprocess.run(
            [VV_BIN, "--help"], capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"vibevoice-cli not found or not executable")

        # Verify critical model files
        for path, label in [
            (self.tokenizer, "tokenizer"),
            (self.tts_model, "TTS model"),
            (self.asr_model, "ASR model"),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"vibevoice.cpp {label} not found at {path}")

        # Find a default voice file
        self.default_voice = None
        for f in sorted(Path(self.voices_dir).glob("voice-en-*.gguf")):
            self.default_voice = str(f)
            break

        self.model = True
        self.model_name = model_name
        logger.info("vibevoice.cpp ready (TTS=%s, ASR=%s, voice=%s)",
                     self.tts_model, self.asr_model, self.default_voice)

    def _unload(self) -> None:
        self.model = None
        super()._unload()

    def _find_voice(self, voice_name: str | None) -> str | None:
        if not voice_name:
            return self.default_voice
        pattern = f"voice-*{voice_name}*.gguf"
        matches = list(Path(self.voices_dir).glob(pattern))
        if matches:
            return str(matches[0])
        return self.default_voice

    def _run_tts(self, text: str, voice_name: str | None = None,
                 ref_audio_path: str | None = None, seed: int = 42) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        cmd = [
            VV_BIN, "tts",
            "--model", self.tts_model,
            "--tokenizer", self.tokenizer,
            "--text", text,
            "--out", out_path,
            "--seed", str(seed),
            "--steps", "20",
            "--cfg", "1.3",
        ]

        if ref_audio_path:
            cmd.extend(["--ref-audio", ref_audio_path])
        else:
            voice_path = self._find_voice(voice_name)
            if voice_path:
                cmd.extend(["--voice", voice_path])

        logger.info("vibevoice.cpp TTS: text=%r voice=%s", text[:60], voice_name)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"vibevoice-cli tts failed: {result.stderr}")

        data = Path(out_path).read_bytes()
        Path(out_path).unlink(missing_ok=True)
        return data

    def _run_asr(self, audio_bytes: bytes, language: str | None = None) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        cmd = [
            VV_BIN, "asr",
            "--model", self.asr_model,
            "--tokenizer", self.tokenizer,
            "--audio", audio_path,
            "--max-new-tokens", "8192",
        ]

        logger.info("vibevoice.cpp ASR: audio=%d bytes", len(audio_bytes))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        Path(audio_path).unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"vibevoice-cli asr failed: {result.stderr}")

        # Parse JSON output: [{"Start":0.0,"End":2.8,"Speaker":0,"Content":"..."}]
        output = result.stdout.strip()
        # The CLI prints timing info to stderr, JSON to stdout
        # Find the JSON array in stdout
        if not output.startswith("["):
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("["):
                    output = line
                    break

        try:
            segments = json.loads(output)
        except json.JSONDecodeError:
            segments = [{"text": output, "speaker": 0}]

        text = " ".join(s.get("Content", s.get("text", "")) for s in segments)
        return {
            "text": text,
            "language": language or "auto",
            "segments": segments,
        }

    async def __call__(self, request):
        """TNAP endpoint: TTS if text provided, ASR if audio_b64 provided."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "backend": os.environ.get("VIBEVOICE_BACKEND", "cuda"),
            }

        start = time.perf_counter()

        try:
            body = await request.json()
            tnap_req, extracted = self.handle_request(body)

            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, "vibevoice-cpp")

            # Determine mode: ASR if audio provided, TTS if text provided
            audio_bytes = extracted.get("audio")
            text = extracted.get("text", "")

            if audio_bytes:
                # ASR mode
                result = await asyncio.to_thread(
                    self._run_asr, audio_bytes, extracted.get("language"),
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
            elif text:
                # TTS mode
                ref_audio = None
                if extracted.get("reference_audio"):
                    import tempfile as _tf
                    ref_bytes = extracted["reference_audio"]
                    with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(ref_bytes)
                        ref_audio = tmp.name

                audio = await asyncio.to_thread(
                    self._run_tts,
                    text,
                    extracted.get("voice"),
                    ref_audio,
                    extracted.get("seed", 42),
                )

                if ref_audio:
                    Path(ref_audio).unlink(missing_ok=True)

                latency_ms = int((time.perf_counter() - start) * 1000)
                return JSONResponse(
                    self.handle_response(audio, "audio/wav", latency_ms)
                )
            else:
                return JSONResponse(
                    self.handle_error("text (for TTS) or audio_b64 (for ASR) required"),
                    status_code=400,
                )
        except Exception as e:
            logger.error("vibevoice_cpp error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)


@serve.deployment(
    name="vibevoice_cpp_gpu",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 0.5,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "VIBEVOICE_BACKEND": "cuda",
            },
        },
    },
)
class VibeVoiceCppGpuDeployment(_VibeVoiceCppBase):
    """vibevoice.cpp GPU — GGML quantized TTS + ASR via CUDA backend."""


@serve.deployment(
    name="vibevoice_cpp_cpu",
    num_replicas=1,
    max_ongoing_requests=2,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1.0,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "VIBEVOICE_BACKEND": "cpu",
            },
        },
    },
)
class VibeVoiceCppCpuDeployment(_VibeVoiceCppBase):
    """vibevoice.cpp CPU — GGML quantized TTS + ASR via CPU backend."""
