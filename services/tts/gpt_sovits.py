"""GPT-SoVITS — Voice cloning TTS (Ray-native).

Clones voices from reference audio using GPT-SoVITS.
Conforms to TNAP: unified request/response protocol.
Supports both JSON (text only) and multipart (with reference audio).
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
from ray import serve
from starlette.responses import JSONResponse

from services.base import BaseGPUDeployment, _b64_decode

logger = logging.getLogger(__name__)


@serve.deployment(
    name="gpt_sovits",
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={
        "num_gpus": 0,
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": "/models/hf_cache",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        },
    },
)
class GPTSoVITSDeployment(BaseGPUDeployment):
    """GPT-SoVITS voice cloning TTS via native PyTorch inference."""

    def __init__(self):
        super().__init__()
        self.tts_pipeline = None

    def _load(self, model_name: str = "gpt-sovits") -> None:
        from registry.config import Config
        from registry.models import ModelRegistry

        cfg = Config()
        registry = ModelRegistry()
        model_path = registry.get_path("tts", model_name)

        if not model_path.is_dir():
            raise FileNotFoundError(
                f"GPT-SoVITS model not found at {model_path}. "
                f"Check model_registry.yaml 'tts.gpt-sovits' entry."
            )

        gptsovits_root = Path("/opt/gpt-sovits")
        if not gptsovits_root.is_dir():
            gptsovits_root = model_path

        gptsovits_pkg = gptsovits_root / "GPT_SoVITS"
        for p in (str(gptsovits_root), str(gptsovits_pkg)):
            if p not in sys.path:
                sys.path.insert(0, p)
        for subdir in gptsovits_pkg.iterdir():
            if subdir.is_dir() and str(subdir) not in sys.path:
                sys.path.insert(0, str(subdir))

        # audio_sr.py imports models.model from tools/AP_BWE_main
        tools_apbwe = gptsovits_root / "tools" / "AP_BWE_main"
        if tools_apbwe.is_dir() and str(tools_apbwe) not in sys.path:
            sys.path.insert(0, str(tools_apbwe))

        # Pretrained models live on PVC at /models/tts/gpt-sovits/
        # Config references GPT_SoVITS/pretrained_models/ as relative paths
        # Rewrite config with absolute paths to a temp file
        pretrained_dir = registry.get_path("tts", model_name)
        orig_config = gptsovits_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
        if not orig_config.exists():
            raise FileNotFoundError(f"TTS config not found: {orig_config}")

        import yaml
        import tempfile
        with open(orig_config) as f:
            cfg = yaml.safe_load(f)
        # Replace relative pretrained model paths with absolute paths
        for version_key in cfg:
            if isinstance(cfg[version_key], dict):
                for k, v in cfg[version_key].items():
                    if isinstance(v, str) and "pretrained_models" in v:
                        parts = v.split("pretrained_models/")
                        if len(parts) == 2:
                            cfg[version_key][k] = str(pretrained_dir / parts[1])

        tmp_config = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
        yaml.dump(cfg, tmp_config)
        tmp_config.close()
        config_path = tmp_config.name
        logger.info("Rewrote GPT-SoVITS config with pretrained dir %s", pretrained_dir)
        if not Path(config_path).exists():
            raise FileNotFoundError(f"TTS config not found: {config_path}")

        if self.config.low_resource:
            logger.info("GPT-SoVITS LOW_RESOURCE mode — fp16, CPU offload")
            self.config.precision = "fp16"

        logger.info("Loading GPT-SoVITS from %s (config=%s)", gptsovits_root, config_path)

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        tts_config = TTS_Config(config_path)

        if torch.cuda.is_available():
            for version_key in ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4"):
                if version_key in tts_config.default_configs:
                    tts_config.default_configs[version_key]["device"] = "cuda"
                    tts_config.default_configs[version_key]["is_half"] = self.config.precision in ("fp16", "bf16")
            tts_config.device = "cuda"
            tts_config.is_half = self.config.precision in ("fp16", "bf16")

        self.tts_pipeline = TTS(tts_config)

        self.model = True
        self.model_name = model_name
        torch.cuda.empty_cache()
        gc.collect()

        vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0
        logger.info("GPT-SoVITS loaded (precision=%s, low_resource=%s, VRAM=%.0fMB)",
                    self.config.precision, self.config.low_resource, vram)

    def _unload(self) -> None:
        if self.tts_pipeline is not None:
            del self.tts_pipeline
            self.tts_pipeline = None
        self.model = None
        self.model_name = None
        super()._unload()

    def _infer(
        self, text: str, text_language: str, prompt_text: str,
        prompt_language: str, refer_bytes: bytes,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            refer_path = Path(tmpdir) / "reference.wav"
            refer_path.write_bytes(refer_bytes)

            logger.info("GPT-SoVITS synthesize: text=%r lang=%s ref=%s",
                        text[:60], text_language, refer_path.name)

            result = self.tts_pipeline.run({
                "text": text,
                "text_lang": text_language,
                "ref_audio_path": str(refer_path),
                "prompt_text": prompt_text,
                "prompt_lang": prompt_language,
                "media_type": "wav",
                "batch_size": 1,
            })

            audio_data = None
            for sampling_rate, audio_chunk, _ in result:
                if audio_chunk is not None:
                    audio_data = audio_chunk
                    break

            if audio_data is None:
                raise RuntimeError("No audio produced")

            import soundfile as sf
            import numpy as np

            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            if isinstance(audio_data, np.ndarray):
                buf = io.BytesIO()
                sf.write(buf, audio_data.T if audio_data.ndim > 1 else audio_data, 32000, format="WAV")
                return buf.getvalue()
            elif isinstance(audio_data, bytes):
                return audio_data
            else:
                raise RuntimeError(f"Unexpected audio format: {type(audio_data)}")

    async def __call__(self, request):
        """TNAP endpoint. Accepts JSON with audio_b64 or multipart with refer_wav."""
        if request.method == "GET":
            return {
                "status": "ok",
                "model": self.model_name,
                "loaded": self.is_loaded(),
                "precision": self.config.precision,
                "low_resource": self.config.low_resource,
            }

        start = time.perf_counter()
        content_type = request.headers.get("content-type", "")

        try:
            if "multipart/form-data" in content_type:
                form = await request.form()
                config_json = form.get("config")
                if config_json:
                    body = json.loads(str(config_json))
                    tnap_req, extracted = self.handle_request(body)
                else:
                    body = {"input": dict(form)}
                    tnap_req, extracted = self.handle_request(body)

                text = form.get("text", "") or extracted.get("text", "")
                text_language = str(form.get("text_language", "en"))
                prompt_text = str(form.get("prompt_text", ""))
                prompt_language = str(form.get("prompt_language", "en"))

                refer_file = form.get("refer_wav") or form.get("file")
                if not refer_file:
                    return JSONResponse(self.handle_error("refer_wav required"), status_code=400)
                refer_bytes = await refer_file.read()

                if not self.is_loaded():
                    await asyncio.to_thread(self.load_model, "gpt-sovits")

                audio = await asyncio.to_thread(
                    self._infer, text, text_language, prompt_text, prompt_language, refer_bytes
                )
            else:
                body = await request.json()
                tnap_req, extracted = self.handle_request(body)

                text = extracted.get("text", "")
                if not text:
                    return JSONResponse(self.handle_error("text is required"), status_code=400)

                if not self.is_loaded():
                    await asyncio.to_thread(self.load_model, "gpt-sovits")

                text_language = extracted.get("language", "en")
                prompt_text = extracted.get("prompt_text", "")
                prompt_language = extracted.get("prompt_language", "en")

                if extracted.get("audio"):
                    refer_bytes = extracted["audio"]
                else:
                    default_ref = "/models/tts/kokoro/samples/af_heart_0.wav"
                    if os.path.isfile(default_ref):
                        refer_bytes = Path(default_ref).read_bytes()
                    else:
                        return JSONResponse(
                            self.handle_error("reference audio (audio_b64) required"),
                            status_code=400,
                        )

                audio = await asyncio.to_thread(
                    self._infer, text, text_language, prompt_text, prompt_language, refer_bytes
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            return JSONResponse(
                self.handle_response(audio, "audio/wav", latency_ms)
            )
        except Exception as e:
            logger.error("gpt_sovits error: %s", e)
            return JSONResponse(self.handle_error(str(e)), status_code=500)