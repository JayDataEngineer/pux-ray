"""ACE-STEP — Music generation from text prompts (Ray-native).

Generates music from text descriptions using the ACE-Step diffusion pipeline.
Conforms to TNAP: unified request/response protocol.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
import time

import numpy as np
import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, InferenceConfig

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