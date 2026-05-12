"""ACE-STEP — Music generation from text prompts (model engine handler).

Uses the Model Engine's AceStepHandler for mmgp-managed VRAM.
The handler decomposes the model into nn.Module components and
registers them with mmgp for efficient GPU memory management.

Old approach: vendor AceStepHandler → no mmgp, no pipe dict, static VRAM
New approach: model_engine AceStepHandler → mmgp profile, pipe dict, dynamic VRAM
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig
from services.forge_base import ForgeService

logger = logging.getLogger(__name__)


@serve.deployment(
    name="ace_step",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCHAUDIO_USE_BACKEND": "ffmpeg",
                "HF_HOME": "/models/hf_cache",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
    },
)
class ACEStepDeployment(BaseGPUDeployment):
    """ACE-STEP text-to-music via native PyTorch inference."""
    vram_mb = 8_192
    _service_name = "ace_step"

    def __init__(self):
        super().__init__()
        self.handler = None

    def _load(self, model_name: str = "ace-step") -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("audio", model_name)

        if not model_path.is_dir():
            raise FileNotFoundError(
                f"ACE-Step checkpoints not found at {model_path}. "
                f"Check model_registry.yaml 'audio.ace-step' entry."
            )

        vendor = str(Path(cfg.project_root) / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

        ckpts_dir = str(model_path)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = ckpts_dir

        logger.info("Loading ACE-Step handler from %s", ckpts_dir)

        from acestep.handler import AceStepHandler

        self.handler = AceStepHandler()

        quantization = None
        if self.config.low_resource:
            logger.info("ACE-STEP LOW_RESOURCE mode — fp16, int8 quantization")
            self.config.precision = "fp16"
            quantization = "int8_weight_only"

        status_msg, success = self.handler.initialize_service(
            project_root="",
            config_path="acestep-v15-turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            quantization=quantization,
        )

        if not success:
            raise RuntimeError(f"ACE-Step init failed: {status_msg}")

        self.model = True
        self.model_name = model_name
        torch.cuda.empty_cache()
        gc.collect()

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("ACE-Step loaded (precision=%s, low_resource=%s, VRAM=%.0fMB)",
                    self.config.precision, self.config.low_resource, vram)

    def _unload(self) -> None:
        if self.handler is not None:
            del self.handler
            self.handler = None
        self.model = None
        self.model_name = None
        super()._unload()

    async def __call__(self, request):
        """TNAP endpoint: {action, input: {prompt, duration, bpm, seed, instrumental}, config}."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "precision": self.config.precision,
                "low_resource": self.config.low_resource,
            }

        start = time.perf_counter()

        try:
            content_type = request.headers.get("content-type", "")

            if "multipart/form-data" in content_type:
                form = await request.form()
                if "config" in form:
                    requested = InferenceConfig(**json.loads(str(form["config"])))
                    if requested != self.config:
                        self.config = requested
                body = dict(form)
                body["config"] = None
                extracted = {}
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

            import asyncio

            model_name = body.get("model", self.model_name or "ace-step")
            if not self.is_loaded():
                await asyncio.to_thread(self.load_model, model_name)

            payload = {**body, **extracted}
            result = await asyncio.to_thread(self._infer, payload)

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(
                    result["data"],
                    result["media_type"],
                    latency_ms,
                    extra_metrics=result.get("extra", {}),
                )
            )
        except Exception as e:
            logger.error("ACE-Step inference failed: %s", e, exc_info=True)
            return JSONResponse(self.handle_error(str(e)), status_code=500)

    def _infer(self, payload: dict) -> dict:
        import soundfile as sf

        prompt = payload.get("prompt", payload.get("caption", ""))
        if not prompt:
            return {"data": json.dumps({"error": "prompt is required"}).encode(), "media_type": "application/json"}

        duration = float(payload.get("duration", 30))
        bpm = int(payload.get("bpm", 120))
        seed = payload.get("seed", -1)
        audio_format = payload.get("audio_format", "wav")
        instrumental = payload.get("instrumental", True)
        inference_steps = 8 if not self.config.low_resource else 4

        logger.info("ACE-Step generate: prompt=%r dur=%.0f bpm=%d steps=%d",
                    prompt[:60], duration, bpm, inference_steps)

        try:
            result = self.handler.generate_music(
                captions=prompt,
                audio_duration=duration,
                bpm=bpm,
                seed=str(seed),
                use_random_seed=(seed == -1),
                inference_steps=inference_steps,
                task_type="text2music" if instrumental else "text2music",
            )

            if not result.get("success", False):
                error = result.get("error", result.get("status_message", "unknown"))
                return {"data": json.dumps({"error": error}).encode(), "media_type": "application/json"}

            audios = result.get("audios", [])
            if not audios:
                return {"data": json.dumps({"error": "No audio produced"}).encode(), "media_type": "application/json"}

            audio_data = audios[0]
            # Handler returns {"tensor": torch.Tensor, "sample_rate": int}
            if isinstance(audio_data, dict) and "tensor" in audio_data:
                import torch as _torch
                tensor = audio_data["tensor"]
                sample_rate = audio_data.get("sample_rate", 48000)
                waveform = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
                buf = io.BytesIO()
                sf.write(buf, waveform.T, sample_rate, format=audio_format.upper())
                data = buf.getvalue()
            elif isinstance(audio_data, np.ndarray):
                sample_rate = result.get("sample_rate", 48000)
                buf = io.BytesIO()
                sf.write(buf, audio_data.T, sample_rate, format=audio_format.upper())
                data = buf.getvalue()
            elif isinstance(audio_data, bytes):
                data = audio_data
            else:
                return {"data": json.dumps({"error": f"Unexpected audio format: {type(audio_data).__name__}"}).encode(), "media_type": "application/json"}

            media_types = {
                "wav": "audio/wav", "mp3": "audio/mpeg",
                "flac": "audio/flac", "ogg": "audio/ogg",
            }
            logger.info("ACE-Step done: %dKB %s", len(data) // 1024, audio_format)
            return {
                "data": data,
                "media_type": media_types.get(audio_format, "audio/wav"),
                "extra": {"bpm": bpm, "duration": duration, "instrumental": instrumental},
            }

        except Exception as e:
            logger.error("ACE-Step inference failed: %s", e, exc_info=True)
            return {"data": json.dumps({"error": str(e)}).encode(), "media_type": "application/json"}


# ─── Forge Service (model engine handler) ────────────────────────────────────

class ACEStepService(ForgeService):
    """ACE-Step text-to-music via the Model Engine handler + mmgp.

    Uses AceStepHandler to decompose the model into a pipe dict,
    then mmgp manages VRAM. vram_mb=0 signals the Forge that this
    service manages its own memory (via mmgp).
    """
    vram_mb = 0  # mmgp manages VRAM — tell Forge not to track
    service_name = "ace_step"
    default_model = "ace_step_v1_5_turbo"

    def __init__(self):
        super().__init__()
        self._load_result = None
        self._executor = None

    def load(self, model_name: str = "ace_step_v1_5_turbo") -> None:
        from services.model_engine.executor import ModelExecutor
        from services.model_engine.handlers.ace_step import AceStepHandler

        handler = AceStepHandler()

        # Resolve model path from registry
        model_path = self._resolve_model_path()

        # Create executor with mmgp profile 1 (all VRAM, RTX 4090 24GB)
        self._executor = ModelExecutor(
            models_root=model_path.parent,
            mmgp_profile=1,
        )
        self._executor.register_handler("ace_step", handler)

        # Load the model
        self._load_result = self._executor.load(model_name)

        self.model_name = model_name
        self._loaded = True

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("ACE-Step loaded via model engine (VRAM=%.0fMB)", vram)

    def unload(self) -> None:
        if self._executor is not None:
            self._executor.unload()
            self._executor = None
        self._load_result = None
        self.model_name = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, payload: dict) -> dict:
        """Run inference. Pipeline returns tensor audio, we encode to bytes."""
        import base64
        import soundfile as sf

        prompt = payload.get("prompt", payload.get("caption", ""))
        if not prompt:
            return {"status": "error", "error": "prompt is required"}

        # Map old-style payload to pipeline kwargs
        pipeline_payload = {
            "prompt": prompt,
            "duration": float(payload.get("duration", 30)),
            "steps": int(payload.get("steps", 8)),
            "temperature": float(payload.get("temperature", 0.85)),
            "top_p": float(payload.get("top_p", 0.9)),
            "seed": payload.get("seed"),
            "custom_settings": {
                "bpm": int(payload.get("bpm", 120)),
                "keyscale": payload.get("keyscale", "C"),
                "timesignature": int(payload.get("timesignature", 4)),
                "language": payload.get("language", "unknown"),
            },
        }

        result = self._executor.infer(self.model_name, pipeline_payload)

        audio_tensor = result["audio"]
        sample_rate = result["sample_rate"]
        audio_format = payload.get("audio_format", "wav")

        # Tensor → numpy → bytes
        waveform = audio_tensor.cpu().numpy()
        if waveform.ndim == 3:
            waveform = waveform[0]  # remove batch dim
        # waveform shape: [channels, samples] → soundfile wants [samples, channels]
        buf = io.BytesIO()
        sf.write(buf, waveform.T, sample_rate, format=audio_format.upper())
        audio_bytes = buf.getvalue()

        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }

        return {
            "status": "success",
            "data": base64.b64encode(audio_bytes).decode(),
            "media_type": media_types.get(audio_format, "audio/wav"),
            "sample_rate": sample_rate,
            "duration": result.get("duration_seconds", payload.get("duration", 30)),
        }

    def _resolve_model_path(self) -> Path:
        """Resolve model weights directory."""
        try:
            from registry.config import Config
            from registry.models import ModelRegistry

            registry = ModelRegistry()
            model_path = registry.get_path("audio", "acestep")
            if model_path.is_dir():
                return model_path
        except Exception:
            pass

        # Fallback: standard path on the cluster
        fallback = Path("/home/user/Documents/models/audio/acestep")
        if fallback.is_dir():
            return fallback

        raise FileNotFoundError(
            "ACE-Step model weights not found. "
            "Set model_registry.yaml 'audio.acestep' or place weights at "
            "/home/user/Documents/models/audio/acestep/"
        )
